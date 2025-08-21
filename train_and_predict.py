#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train simple models to predict double-perturbation expression profiles from singles,
with:
  - noise-robust baselines (trimmed-median control, winsorization)
  - compact, symmetric feature sets (basic or expanded)
  - per-gene adaptive ensembling (learned blend of additive and ridge)
  - optional scaled-additive baseline (global or per-gene alpha)
  - k-fold evaluation (logs RMSD for additive, ridge, ensemble, adaptive)
  - per-PAIR and per-GENE RMSD logs
  - a compact summary metrics CSV (--metrics_csv) for quick reading/plotting
  - support for test_pairs as CSV
  - switchable modes: eval (k-fold) vs predict (write prediction/prediction.csv)
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

PAIR_RE = re.compile(r"^(g\d{4})\+(g\d{4}|ctrl)$", re.IGNORECASE)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "predict"], required=True,
                    help="'eval' = k-fold CV on observed doubles; 'predict' = generate prediction.csv for provided pairs")
    ap.add_argument("--train_csv", required=True,
                    help="Matrix CSV (rows=genes, cols=perturbations). Index col=gene id.")
    ap.add_argument("--test_pairs", default=None,
                    help="CSV file containing test pairs. If omitted in 'predict' mode, will error. "
                         "Accepted columns: 'perturbation' or first column.")
    ap.add_argument("--out_csv", default="prediction/prediction.csv",
                    help="Output path for predictions in predict mode.")
    ap.add_argument("--template_csv", default=None,
                    help="Optional 3-col template (gene, perturbation, expression) to copy row order from")

    # Models and features
    ap.add_argument("--no_ridge", action="store_true",
                    help="Skip per-gene ridge; use additive baseline only")
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1.0,10.0",
                    help="Comma list for RidgeCV alphas")
    ap.add_argument("--ensemble_weight", type=float, default=0.5,
                    help="Weight on ridge in [0,1]; (1-w) on additive (used when --adaptive_ensemble is OFF)")
    ap.add_argument("--features", choices=["basic","expanded"], default="basic",
                    help="Feature set for ridge. basic: 7 feats; expanded: adds min/max/quadratics.")

    # Scaled additive options
    ap.add_argument("--scaled_additive", choices=["none","global","per_gene"], default="none",
                    help="If enabled, fit alpha so y = alpha*(add) + (1-alpha)*ctrl. Fit on training folds (eval) or all doubles (predict).")
    ap.add_argument("--alpha_clip", type=float, default=1.0,
                    help="Clip |alpha| <= alpha_clip (default 1.0).")

    # Adaptive ensembling options
    ap.add_argument("--adaptive_ensemble", action="store_true",
                    help="Learn per-gene weights to blend additive and ridge on training doubles.")

    # Robust baseline options
    ap.add_argument("--robust_ctrl", action="store_true",
                    help="Use trimmed-median across singles as control instead of median.")
    ap.add_argument("--trim_pct", type=float, default=0.10,
                    help="Trim proportion per tail for trimmed-median control (e.g., 0.10 = 10%).")
    ap.add_argument("--winsorize_pct", type=float, default=0.00,
                    help="Winsorize singles per-gene at this two-tailed percent (e.g., 0.01). 0 disables.")

    # Eval logging
    ap.add_argument("--k_folds", type=int, default=5,
                    help="Number of folds for eval mode")
    ap.add_argument("--metrics_csv", default="metrics/eval_metrics.csv",
                    help="Summary metrics CSV (overall & per-fold means) in eval mode")
    ap.add_argument("--metrics_per_pair_csv", default="metrics/metrics_per_pair.csv",
                    help="Where to write per-pair eval metrics CSV (eval mode)")
    ap.add_argument("--metrics_per_gene_csv", default="metrics/metrics_per_gene.csv",
                    help="Where to write per-gene eval metrics CSV (eval mode)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffling")
    return ap.parse_args()

# ----------------------------
# Helpers
# ----------------------------
def is_single(col: str) -> bool:
    m = PAIR_RE.match(col)
    return bool(m and m.group(2).lower() == "ctrl")

def is_double(col: str) -> bool:
    m = PAIR_RE.match(col)
    return bool(m and m.group(2).lower() != "ctrl")

def parse_pair(pair: str) -> Tuple[str, str]:
    pair = pair.strip()
    m = PAIR_RE.match(pair)
    if not m:
        raise ValueError(f"Bad pair format: {pair}")
    a, b = m.group(1), m.group(2)
    return (a, b)

def load_matrix(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = [i.strip() for i in df.index]
    if len(df.columns) != len(set(df.columns)):
        df = df.groupby(axis=1, level=0).mean()
    return df

def winsorize_singles(singles: Dict[str, np.ndarray], pct: float) -> Dict[str, np.ndarray]:
    """Winsorize per gene across singles; returns new dict."""
    if pct <= 0:
        return singles
    keys = list(singles.keys())
    S = np.stack([singles[k] for k in keys], axis=1)  # [genes, num_singles]
    lo = np.nanpercentile(S, 100*pct, axis=1)
    hi = np.nanpercentile(S, 100*(1-pct), axis=1)
    S_clip = np.clip(S, lo[:,None], hi[:,None])
    out = {}
    for j,k in enumerate(keys):
        out[k] = S_clip[:, j]
    return out

def trimmed_median_ctrl(singles: Dict[str, np.ndarray], trim_pct: float) -> np.ndarray:
    """Per-gene trimmed median across singles."""
    S = np.stack(list(singles.values()), axis=1)  # [genes, num_singles]
    if trim_pct <= 0 or S.shape[1] < 4:
        return np.median(S, axis=1)
    lo_q = 100 * trim_pct
    hi_q = 100 * (1 - trim_pct)
    lo = np.nanpercentile(S, lo_q, axis=1)
    hi = np.nanpercentile(S, hi_q, axis=1)
    S_mask = np.clip(S, lo[:,None], hi[:,None])
    return np.median(S_mask, axis=1)

def extract_singles(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    singles = {}
    for c in df.columns:
        if is_single(c):
            g = c.split("+")[0]
            singles[g] = df[c].values.astype(float)
    return singles

def find_ctrl_vector(df: pd.DataFrame, singles: Dict[str, np.ndarray],
                     robust_ctrl: bool, trim_pct: float) -> np.ndarray:
    ctrl_cols = [c for c in df.columns if c.lower() in ("ctrl+ctrl", "ctrl")]
    if ctrl_cols:
        return df[ctrl_cols[0]].values.astype(float)
    if len(singles) == 0:
        raise RuntimeError("No singles found; cannot derive control.")
    if robust_ctrl:
        return trimmed_median_ctrl(singles, trim_pct)
    S = np.stack(list(singles.values()), axis=1)
    return np.median(S, axis=1)

def build_training_pairs(df: pd.DataFrame) -> List[Tuple[str, str, np.ndarray]]:
    pairs = []
    for c in df.columns:
        if is_double(c):
            a, b = c.split("+", 1)
            pairs.append((a, b, df[c].values.astype(float)))
    return pairs

# ----------------------------
# Features
# ----------------------------
def make_feature_vector(ea: float, eb: float, e0: float, expanded: bool) -> np.ndarray:
    if not expanded:
        return np.array([1.0, ea, eb, ea*eb, abs(ea-eb), ea+eb, e0], dtype=float)
    # expanded: compact, symmetric, nonlinear
    return np.array([
        1.0,
        ea, eb,
        ea + eb,
        ea * eb,
        abs(ea - eb),
        min(ea, eb),
        max(ea, eb),
        ea*ea, eb*eb,
        (ea - eb)*(ea - eb),
        e0
    ], dtype=float)

def num_features(expanded: bool) -> int:
    return 12 if expanded else 7

# ----------------------------
# Scaled additive & adaptive weights
# ----------------------------
def fit_scaled_additive_alpha(y_add: np.ndarray, y_true: np.ndarray, ctrl_vec: np.ndarray,
                              mode: str, alpha_clip: float) -> np.ndarray:
    """
    Fit alpha so y ≈ alpha*(add) + (1-alpha)*ctrl.
    mode: 'global' returns scalar alpha broadcasted; 'per_gene' returns vector per gene.
    """
    add_minus_ctrl = y_add - ctrl_vec
    if mode == "global":
        num = np.dot(add_minus_ctrl, (y_true - ctrl_vec))
        den = np.dot(add_minus_ctrl, add_minus_ctrl) + 1e-12
        alpha = np.clip(num / den, -alpha_clip, alpha_clip)
        return np.full_like(ctrl_vec, alpha, dtype=float)
    else:
        # per_gene
        num = add_minus_ctrl * (y_true - ctrl_vec)
        den = add_minus_ctrl * add_minus_ctrl + 1e-12
        alpha = np.clip(num / den, -alpha_clip, alpha_clip)
        return alpha

def apply_scaled_additive(ea_vec: np.ndarray, eb_vec: np.ndarray, ctrl_vec: np.ndarray,
                          alpha_vec: Optional[np.ndarray]) -> np.ndarray:
    base = ea_vec + eb_vec - ctrl_vec
    if alpha_vec is None:
        return np.clip(base, 0.0, None)
    y = alpha_vec * base + (1.0 - alpha_vec) * ctrl_vec
    return np.clip(y, 0.0, None)

def fit_per_gene_adaptive_weights(y_add_M: np.ndarray, y_ridge_M: np.ndarray, y_true_M: np.ndarray) -> np.ndarray:
    """
    Closed-form per-gene optimal w in y ≈ (1-w)*y_add + w*y_ridge minimizing MSE over columns/samples.
    Inputs are matrices [genes, samples].
    Returns w per gene in [0,1].
    """
    D = y_ridge_M - y_add_M
    num = np.sum(D * (y_true_M - y_add_M), axis=1)
    den = np.sum(D * D, axis=1) + 1e-12
    w = num / den
    w = np.clip(w, 0.0, 1.0)
    return w

# ----------------------------
# Modeling
# ----------------------------
def fit_per_gene_ridge(
    df: pd.DataFrame,
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    alphas: List[float],
    train_pair_indices: Optional[List[int]],
    expanded_features: bool
) -> np.ndarray:
    genes = list(df.index)
    gN = len(genes)
    F = num_features(expanded_features)
    coefs = np.zeros((gN, F), dtype=float)

    all_pairs = build_training_pairs(df)
    train_pairs = all_pairs if train_pair_indices is None else [all_pairs[i] for i in train_pair_indices]

    for i_g in range(gN):
        X_list, y_list = [], []
        e0 = float(ctrl_vec[i_g])

        for a, b, y_vec in train_pairs:
            ea_vec = singles.get(a)
            eb_vec = singles.get(b)
            if ea_vec is None or eb_vec is None:
                continue
            ea, eb = float(ea_vec[i_g]), float(eb_vec[i_g])
            X_list.append(make_feature_vector(ea, eb, e0, expanded_features))
            y_list.append(float(y_vec[i_g]))

        if len(X_list) >= 10:
            X = np.vstack(X_list)
            y = np.array(y_list)
            model = RidgeCV(alphas=alphas, fit_intercept=False)
            model.fit(X, y)
            coefs[i_g, :] = model.coef_

    return coefs

def additive_prediction(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    alpha_vec: Optional[np.ndarray] = None
) -> np.ndarray:
    a, b = pair
    ea = singles.get(a)
    eb = singles.get(b)
    if ea is None:
        ea = ctrl_vec
    if eb is None:
        eb = ctrl_vec
    return apply_scaled_additive(ea, eb, ctrl_vec, alpha_vec)

def ridge_prediction_for_pair(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    coefs: np.ndarray,
    expanded_features: bool
) -> np.ndarray:
    a, b = pair
    ea_vec = singles.get(a)
    eb_vec = singles.get(b)
    if ea_vec is None:
        ea_vec = ctrl_vec
    if eb_vec is None:
        eb_vec = ctrl_vec

    y = np.zeros_like(ctrl_vec, dtype=float)
    for i in range(len(ctrl_vec)):
        fv = make_feature_vector(float(ea_vec[i]), float(eb_vec[i]), float(ctrl_vec[i]), expanded_features)
        y[i] = float(np.dot(coefs[i, :], fv))
    return np.clip(y, 0.0, None)

# ----------------------------
# Evaluation (k-fold on doubles)
# ----------------------------
def eval_kfold(
    df: pd.DataFrame,
    singles_in: Dict[str, np.ndarray],
    robust_ctrl: bool,
    trim_pct: float,
    winsorize_pct: float,
    alphas: List[float],
    k_folds: int,
    ensemble_weight: float,
    seed: int,
    use_ridge: bool,
    expanded_features: bool,
    scaled_additive_mode: str,
    alpha_clip: float,
    adaptive_ensemble: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(df.index)
    all_pairs = build_training_pairs(df)

    # Preprocess singles (winsorize if requested)
    singles = winsorize_singles(singles_in, winsorize_pct)
    ctrl_vec = find_ctrl_vector(df, singles, robust_ctrl, trim_pct)

    # Keep only doubles whose singles exist
    usable_idx = []
    for i, (a, b, _y) in enumerate(all_pairs):
        if singles.get(a) is not None and singles.get(b) is not None:
            usable_idx.append(i)

    if len(usable_idx) < k_folds:
        raise RuntimeError(f"Not enough usable doubles ({len(usable_idx)}) for k={k_folds} folds.")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    rows_pair = []
    rows_gene = []

    fold_id = 0
    for train_idx_rel, test_idx_rel in kf.split(usable_idx):
        fold_id += 1
        train_idx_abs = [usable_idx[i] for i in train_idx_rel]
        test_idx_abs  = [usable_idx[i] for i in test_idx_rel]

        # Fit ridge on training folds
        if use_ridge:
            coefs = fit_per_gene_ridge(df, singles, ctrl_vec, alphas, train_pair_indices=train_idx_abs,
                                       expanded_features=expanded_features)
        else:
            coefs = None

        # ----- Fit scaled-additive alpha on training folds -----
        alpha_vec = None
        if scaled_additive_mode != "none":
            # Build training matrices for alpha fit
            train_pairs = [all_pairs[i] for i in train_idx_abs]
            # Use a pooled approach: concatenate samples for stable fit
            y_add_cols, y_true_cols = [], []
            for a,b,y_true in train_pairs:
                y_add = additive_prediction(singles, ctrl_vec, (a,b), alpha_vec=None)  # base (alpha None)
                y_add_cols.append(y_add)
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)   # [genes, samples]
            y_true_M = np.stack(y_true_cols, axis=1) # [genes, samples]
            # Fit alpha per-gene or global
            if scaled_additive_mode == "global":
                # flatten
                alpha_vec = fit_scaled_additive_alpha(y_add_M.flatten(), y_true_M.flatten(),
                                                      np.tile(ctrl_vec, y_add_M.shape[1]),
                                                      mode="global", alpha_clip=alpha_clip)
            else:
                # per-gene using means across samples
                alpha_vec = fit_scaled_additive_alpha(np.mean(y_add_M, axis=1),
                                                      np.mean(y_true_M, axis=1),
                                                      ctrl_vec, mode="per_gene", alpha_clip=alpha_clip)

        # ----- If adaptive ensemble: fit per-gene weights on training folds -----
        w_adapt = None
        if adaptive_ensemble and use_ridge:
            y_add_cols, y_ridge_cols, y_true_cols = [], [], []
            for a,b,y_true in [all_pairs[i] for i in train_idx_abs]:
                y_add = additive_prediction(singles, ctrl_vec, (a,b), alpha_vec)
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a,b), coefs, expanded_features)
                y_add_cols.append(y_add)
                y_ridge_cols.append(y_ridge)
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)
            y_ridge_M = np.stack(y_ridge_cols, axis=1)
            y_true_M = np.stack(y_true_cols, axis=1)
            w_adapt = fit_per_gene_adaptive_weights(y_add_M, y_ridge_M, y_true_M)  # [genes]

        # Per-gene accumulators for this fold
        gN = len(genes)
        sse_add = np.zeros(gN, dtype=float)
        sse_ridge = np.zeros(gN, dtype=float)
        sse_ens = np.zeros(gN, dtype=float)
        sse_adapt = np.zeros(gN, dtype=float)
        counts = np.zeros(gN, dtype=int)

        # Evaluate on held-out fold pairs
        for idx_abs in test_idx_rel:
            # Note: idx_abs is relative to usable_idx; convert to absolute
            idx_abs_true = usable_idx[idx_abs]
            a, b, y_true = all_pairs[idx_abs_true]
            pair_name = f"{a}+{b}"

            y_add = additive_prediction(singles, ctrl_vec, (a, b), alpha_vec)
            rmsd_add = mean_squared_error(y_true, y_add, squared=False)

            if use_ridge:
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a, b), coefs, expanded_features)
                rmsd_ridge = mean_squared_error(y_true, y_ridge, squared=False)
                # fixed global-weight ensemble for continuity
                y_ens = (1.0 - ensemble_weight) * y_add + ensemble_weight * y_ridge
                y_ens = np.clip(y_ens, 0.0, None)
                rmsd_ens = mean_squared_error(y_true, y_ens, squared=False)
            else:
                y_ridge = y_add
                rmsd_ridge = None
                y_ens = y_add
                rmsd_ens = rmsd_add

            # adaptive per-gene
            if adaptive_ensemble and use_ridge and w_adapt is not None:
                y_adapt = np.clip((1.0 - w_adapt) * y_add + w_adapt * y_ridge, 0.0, None)
                rmsd_adapt = mean_squared_error(y_true, y_adapt, squared=False)
            else:
                y_adapt = y_ens
                rmsd_adapt = rmsd_ens

            rows_pair.append({
                "fold": fold_id,
                "pair": pair_name,
                "rmsd_additive": rmsd_add,
                "rmsd_ridge": rmsd_ridge,
                "rmsd_ensemble": rmsd_ens,
                "rmsd_adaptive": rmsd_adapt if adaptive_ensemble and use_ridge else None
            })

            # accumulate per-gene SSE
            sse_add += (y_add - y_true) ** 2
            sse_ens += (y_ens - y_true) ** 2
            if adaptive_ensemble and use_ridge:
                sse_adapt += (y_adapt - y_true) ** 2
            if use_ridge:
                sse_ridge += (y_ridge - y_true) ** 2
            counts += 1

        # fold-level per-gene RMSD (avg over pairs)
        mask = counts > 0
        rmsd_add_gene = np.zeros_like(sse_add); rmsd_add_gene[mask] = np.sqrt(sse_add[mask] / counts[mask])
        rmsd_ens_gene = np.zeros_like(sse_add); rmsd_ens_gene[mask] = np.sqrt(sse_ens[mask] / counts[mask])
        if adaptive_ensemble and use_ridge:
            rmsd_adapt_gene = np.zeros_like(sse_add); rmsd_adapt_gene[mask] = np.sqrt(sse_adapt[mask] / counts[mask])
        else:
            rmsd_adapt_gene = np.array([np.nan]*gN)
        if use_ridge:
            rmsd_ridge_gene = np.zeros_like(sse_add); rmsd_ridge_gene[mask] = np.sqrt(sse_ridge[mask] / counts[mask])
        else:
            rmsd_ridge_gene = np.array([np.nan]*gN)

        for i_g, gname in enumerate(genes):
            rows_gene.append({
                "fold": fold_id,
                "gene": gname,
                "rmsd_additive": float(rmsd_add_gene[i_g]) if counts[i_g] > 0 else np.nan,
                "rmsd_ridge": float(rmsd_ridge_gene[i_g]) if counts[i_g] > 0 else np.nan,
                "rmsd_ensemble": float(rmsd_ens_gene[i_g]) if counts[i_g] > 0 else np.nan,
                "rmsd_adaptive": float(rmsd_adapt_gene[i_g]) if counts[i_g] > 0 else np.nan,
                "pairs_evaluated": int(counts[i_g])
            })

    metrics_per_pair_df = pd.DataFrame(rows_pair)
    metrics_per_gene_df = pd.DataFrame(rows_gene)

    # Console summary
    print("\n=== K-FOLD SUMMARY ===")
    print(f"folds: {k_folds}, pairs evaluated: {len(metrics_per_pair_df)}")
    print(f"Mean RMSD (additive): {metrics_per_pair_df['rmsd_additive'].mean():.6f}")
    if use_ridge:
        print(f"Mean RMSD (ridge):    {metrics_per_pair_df['rmsd_ridge'].mean():.6f}")
        print(f"Mean RMSD (ensemble): {metrics_per_pair_df['rmsd_ensemble'].mean():.6f}")
        if adaptive_ensemble:
            print(f"Mean RMSD (adaptive): {metrics_per_pair_df['rmsd_adaptive'].mean():.6f}")
    else:
        print("Ridge disabled (--no_ridge). Ensemble/adaptive equal additive.")

    return metrics_per_pair_df, metrics_per_gene_df

# ----------------------------
# Test pairs CSV reader
# ----------------------------
def read_test_pairs_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "perturbation" in df.columns:
        col = df["perturbation"]
    else:
        col = df[df.columns[0]]
    pairs = []
    for v in col.astype(str).tolist():
        v = v.strip()
        if not v:
            continue
        _ = parse_pair(v)  # validate
        pairs.append(v)
    return pairs

# ----------------------------
# Output formatting
# ----------------------------
def make_output_frame(
    genes: List[str],
    pairs: List[str],
    predictions: Dict[str, np.ndarray],
    template_csv: Optional[str]
) -> pd.DataFrame:
    if template_csv:
        tpl = pd.read_csv(template_csv)
        assert set(["gene", "perturbation"]).issubset(set(tpl.columns[:3]))
        out_gene = tpl["gene"].tolist()
        out_pair = tpl["perturbation"].tolist()
        vals = []
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        for g, p in zip(out_gene, out_pair):
            vals.append(float(predictions[p][gene_to_idx[g]]))
        return pd.DataFrame({"gene": out_gene, "perturbation": out_pair, "expression": vals})
    else:
        rows = []
        for p in pairs:
            y = predictions[p]
            for i, g in enumerate(genes):
                rows.append((g, p, float(y[i])))
        return pd.DataFrame(rows, columns=["gene", "perturbation", "expression"])

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    if args.mode == "predict":
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
        os.makedirs(os.path.dirname(args.metrics_per_pair_csv), exist_ok=True)
        os.makedirs(os.path.dirname(args.metrics_per_gene_csv), exist_ok=True)

    alphas = [float(x) for x in args.ridge_alphas.split(",") if x]

    df = load_matrix(args.train_csv)
    genes = list(df.index)

    singles_raw = extract_singles(df)
    # Winsorize singles if requested
    singles = winsorize_singles(singles_raw, args.winsorize_pct)

    if args.mode == "eval":
        metrics_pair_df, metrics_gene_df = eval_kfold(
            df=df,
            singles_in=singles,
            robust_ctrl=args.robust_ctrl,
            trim_pct=float(args.trim_pct),
            winsorize_pct=float(args.winsorize_pct),
            alphas=alphas,
            k_folds=args.k_folds,
            ensemble_weight=float(args.ensemble_weight),
            seed=args.seed,
            use_ridge=(not args.no_ridge),
            expanded_features=(args.features == "expanded"),
            scaled_additive_mode=args.scaled_additive,
            alpha_clip=float(args.alpha_clip),
            adaptive_ensemble=bool(args.adaptive_ensemble),
        )
        # write long-form logs
        metrics_pair_df.to_csv(args.metrics_per_pair_csv, index=False)
        metrics_gene_df.to_csv(args.metrics_per_gene_csv, index=False)
        print(f"\nWrote per-pair metrics to   {args.metrics_per_pair_csv}")
        print(f"Wrote per-gene metrics to   {args.metrics_per_gene_csv}")

        # build compact summary table for --metrics_csv
        summary_rows = []
        overall = {
            "scope": "overall",
            "fold": "all",
            "mean_rmsd_additive": metrics_pair_df["rmsd_additive"].mean(),
            "mean_rmsd_ridge": metrics_pair_df["rmsd_ridge"].mean() if not args.no_ridge else np.nan,
            "mean_rmsd_ensemble": metrics_pair_df["rmsd_ensemble"].mean(),
            "mean_rmsd_adaptive": metrics_pair_df["rmsd_adaptive"].mean() if (not args.no_ridge and args.adaptive_ensemble) else np.nan,
            "pairs": len(metrics_pair_df)
        }
        summary_rows.append(overall)
        for f in sorted(metrics_pair_df["fold"].unique()):
            sub = metrics_pair_df[metrics_pair_df["fold"] == f]
            summary_rows.append({
                "scope": "per_fold",
                "fold": int(f),
                "mean_rmsd_additive": sub["rmsd_additive"].mean(),
                "mean_rmsd_ridge": sub["rmsd_ridge"].mean() if not args.no_ridge else np.nan,
                "mean_rmsd_ensemble": sub["rmsd_ensemble"].mean(),
                "mean_rmsd_adaptive": sub["rmsd_adaptive"].mean() if (not args.no_ridge and args.adaptive_ensemble) else np.nan,
                "pairs": len(sub)
            })
        pd.DataFrame(summary_rows).to_csv(args.metrics_csv, index=False)
        print(f"Wrote summary metrics to    {args.metrics_csv}")

        recap = {
            "overall_add": float(overall["mean_rmsd_additive"]),
            "overall_ridge": None if args.no_ridge else float(overall["mean_rmsd_ridge"]),
            "overall_ens": float(overall["mean_rmsd_ensemble"]),
            "overall_adapt": None if (args.no_ridge or not args.adaptive_ensemble) else float(overall["mean_rmsd_adaptive"])
        }
        print("\nOVERALL:", recap)

    else:  # predict
        if not args.test_pairs:
            raise ValueError("In 'predict' mode, --test_pairs (CSV) is required.")
        test_pairs_list = read_test_pairs_csv(args.test_pairs)

        # Control (robust if requested)
        ctrl_vec = find_ctrl_vector(df, singles, args.robust_ctrl, float(args.trim_pct))

        # Fit ridge on ALL usable doubles (max training signal)
        expanded = (args.features == "expanded")
        if args.no_ridge:
            coefs = None
        else:
            coefs = fit_per_gene_ridge(df, singles, ctrl_vec, alphas, train_pair_indices=None,
                                       expanded_features=expanded)

        # Fit scaled-additive alpha on all doubles (optional)
        alpha_vec = None
        if args.scaled_additive != "none":
            y_add_cols, y_true_cols = [], []
            for (a,b,y_true) in build_training_pairs(df):
                y_add_cols.append(additive_prediction(singles, ctrl_vec, (a,b), alpha_vec=None))
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)
            y_true_M = np.stack(y_true_cols, axis=1)
            if args.scaled_additive == "global":
                alpha_vec = fit_scaled_additive_alpha(y_add_M.flatten(), y_true_M.flatten(),
                                                      np.tile(ctrl_vec, y_add_M.shape[1]),
                                                      mode="global", alpha_clip=float(args.alpha_clip))
            else:
                alpha_vec = fit_scaled_additive_alpha(np.mean(y_add_M, axis=1),
                                                      np.mean(y_true_M, axis=1),
                                                      ctrl_vec, mode="per_gene", alpha_clip=float(args.alpha_clip))

        # Adaptive ensemble weights on all doubles (optional)
        w_adapt = None
        if args.adaptive_ensemble and not args.no_ridge:
            y_add_cols, y_ridge_cols, y_true_cols = [], [], []
            for (a,b,y_true) in build_training_pairs(df):
                y_add_cols.append(additive_prediction(singles, ctrl_vec, (a,b), alpha_vec))
                y_ridge_cols.append(ridge_prediction_for_pair(singles, ctrl_vec, (a,b), coefs, expanded))
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)
            y_ridge_M = np.stack(y_ridge_cols, axis=1)
            y_true_M = np.stack(y_true_cols, axis=1)
            w_adapt = fit_per_gene_adaptive_weights(y_add_M, y_ridge_M, y_true_M)

        # Produce predictions
        final_preds: Dict[str, np.ndarray] = {}
        for p in test_pairs_list:
            a, b = parse_pair(p)
            y_add = additive_prediction(singles, ctrl_vec, (a, b), alpha_vec)
            if args.no_ridge:
                y_final = y_add
            else:
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a, b), coefs, expanded)
                if args.adaptive_ensemble and w_adapt is not None:
                    y_final = np.clip((1.0 - w_adapt) * y_add + w_adapt * y_ridge, 0.0, None)
                else:
                    w = float(args.ensemble_weight)
                    y_final = np.clip((1.0 - w) * y_add + w * y_ridge, 0.0, None)
            final_preds[p] = y_final

        out_df = make_output_frame(genes, test_pairs_list, final_preds, args.template_csv)
        if out_df.isnull().any().any():
            raise RuntimeError("Output contains NaNs; please check inputs.")
        out_df = out_df[["gene", "perturbation", "expression"]].astype({"gene":str,"perturbation":str,"expression":float})
        out_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(out_df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()