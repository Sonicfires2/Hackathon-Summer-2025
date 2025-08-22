#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict double-perturbation expression from singles with robust baselines,
compact symmetric features, optional adaptive/ridge ensemble, and
a lightweight residual learner to capture non-additive effects.

Layout:
  - Train matrix: data/train_matrix.csv  (rows=genes, cols=perturbations; index=gene id)
  - Test pairs:   data/test_pairs.csv    (single column of 'g####+g####' or 'g####+ctrl')
  - Output:       prediction/prediction.csv

Modes:
  --mode eval     : k-fold CV across observed doubles, logs RMSD (add/ridge/ens/adapt/resid)
  --mode predict  : write prediction/prediction.csv for pairs in data/test_pairs.csv

Upgrades (opt-in via flags; defaults preserve old behavior):
  - Per-gene standardization of ridge features/targets
  - Residual HGB with early stopping + L2 regularization
  - Residual target clipping (MAD-based)
  - PCA pair-context features for residual model
  - Residual gating via inner-CV OOF R^2
  - Bayesian shrinkage of adaptive weights
  - Weighted RMSD (per-gene noise) for eval
  - Symmetry averaging for (a,b)/(b,a) predictions
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

PAIR_RE = re.compile(r"^(g\d{4})\+(g\d{4}|ctrl)$", re.IGNORECASE)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "predict"], required=True)
    ap.add_argument("--train_csv", default="data/train_matrix.csv")
    ap.add_argument("--test_pairs", default="data/test_pairs.csv")
    ap.add_argument("--out_csv", default="prediction/prediction.csv")
    ap.add_argument("--template_csv", default=None)

    # Models and features
    ap.add_argument("--no_ridge", action="store_true")
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1.0,10.0")
    ap.add_argument("--ensemble_weight", type=float, default=0.5)
    ap.add_argument("--features", choices=["basic","expanded"], default="expanded")

    # Scaled additive & adaptive blend
    ap.add_argument("--scaled_additive", choices=["none","global","per_gene"], default="none")
    ap.add_argument("--alpha_clip", type=float, default=1.0)
    ap.add_argument("--adaptive_ensemble", action="store_true")

    # Robust baseline options
    ap.add_argument("--robust_ctrl", action="store_true")
    ap.add_argument("--trim_pct", type=float, default=0.10)
    ap.add_argument("--winsorize_pct", type=float, default=0.00)

    # Residual learner (small non-linear correction)
    ap.add_argument("--residual_learner", choices=["none","hgb"], default="hgb",
                    help="Learn residual y_true - y_base from pair+gene features.")
    ap.add_argument("--resid_on", choices=["additive","ensemble"], default="ensemble",
                    help="Which base to correct with residual learner.")
    ap.add_argument("--resid_max_depth", type=int, default=3)
    ap.add_argument("--resid_learning_rate", type=float, default=0.05)
    ap.add_argument("--resid_max_iter", type=int, default=300)
    ap.add_argument("--resid_min_samples_leaf", type=int, default=20,
                    help="Min samples per leaf for HGB residual models.")

    # ---- NEW: safer/faster residual options (defaults chosen to be safe) ----
    ap.add_argument("--resid_l2", type=float, default=1e-3, help="L2 regularization for HGB residuals.")
    ap.add_argument("--resid_early_stop", action="store_true",
                    help="Enable early stopping for HGB residuals (val_fraction=0.15, n_iter_no_change=20).")
    ap.add_argument("--resid_clip_mad_mult", type=float, default=0.0,
                    help="Clip residual targets to ±mult*MAD per gene (0=off).")
    ap.add_argument("--resid_pca_k", type=int, default=0,
                    help="If >0, append PCA pair-context features (K PCs) to residual features.")
    ap.add_argument("--resid_gate_r2", type=float, default=0.0,
                    help="If >0, apply residuals only for genes with inner-CV OOF R^2 >= threshold.")

    # Eval logging
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--metrics_csv", default="metrics/eval_metrics.csv")
    ap.add_argument("--metrics_per_pair_csv", default="metrics/metrics_per_pair.csv")
    ap.add_argument("--metrics_per_gene_csv", default="metrics/metrics_per_gene.csv")
    ap.add_argument("--seed", type=int, default=42)

    # ---- NEW: extras that do not break old flags ----
    ap.add_argument("--ridge_standardize", action="store_true",
                    help="Standardize per-gene ridge features/targets (better alpha selection).")
    ap.add_argument("--weighted_rmsd", action="store_true",
                    help="Eval: weight per-gene errors by inverse noise (MAD from singles).")
    ap.add_argument("--adapt_shrink", type=float, default=0.0,
                    help="Bayesian shrinkage strength λ for adaptive per-gene weights (0=off).")
    ap.add_argument("--symmetry_average", action="store_true",
                    help="Predict: if both a+b and b+a present, average their predictions symmetrically.")

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
    return (m.group(1), m.group(2))

def load_matrix(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = [i.strip() for i in df.index]
    # de-duplicate columns by averaging if necessary
    if len(df.columns) != len(set(df.columns)):
        df = df.groupby(axis=1, level=0).mean()
    return df

def winsorize_singles(singles: Dict[str, np.ndarray], pct: float) -> Dict[str, np.ndarray]:
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
    S = np.stack(list(singles.values()), axis=1)
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
    return trimmed_median_ctrl(singles, trim_pct) if robust_ctrl else np.median(
        np.stack(list(singles.values()), axis=1), axis=1
    )

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

def pair_global_features(ea_vec: np.ndarray, eb_vec: np.ndarray) -> np.ndarray:
    """
    Global pair stats across the whole transcriptome: cosine similarity, norms, dot product.
    Same values are used for all target genes in the pair.
    """
    a = ea_vec; b = eb_vec
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    cos = float(dot / (na * nb))
    return np.array([cos, na, nb, dot], dtype=float)

# ----------------------------
# Scaled additive & adaptive weights
# ----------------------------
def fit_scaled_additive_alpha(y_add: np.ndarray, y_true: np.ndarray, ctrl_vec: np.ndarray,
                              mode: str, alpha_clip: float) -> np.ndarray:
    add_minus_ctrl = y_add - ctrl_vec
    if mode == "global":
        num = np.dot(add_minus_ctrl, (y_true - ctrl_vec))
        den = np.dot(add_minus_ctrl, add_minus_ctrl) + 1e-12
        alpha = np.clip(num / den, -alpha_clip, alpha_clip)
        return np.full_like(ctrl_vec, alpha, dtype=float)
    else:
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
    D = y_ridge_M - y_add_M
    num = np.sum(D * (y_true_M - y_add_M), axis=1)
    den = np.sum(D * D, axis=1) + 1e-12
    w = num / den
    return np.clip(w, 0.0, 1.0)

# ----------------------------
# Noise scales, weighted RMSE
# ----------------------------
def per_gene_mad(vecs: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-gene MAD across provided vectors (list of [genes]-length vectors).
    """
    if len(vecs) == 0:
        return None
    M = np.stack(vecs, axis=1)  # [genes, n]
    med = np.median(M, axis=1)
    mad = np.median(np.abs(M - med[:, None]), axis=1) + 1e-12
    return mad

def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray]) -> float:
    """
    Weighted RMSE across genes. w should be weights per gene (higher => more weight).
    If w is None, falls back to unweighted RMSE.
    """
    if w is None:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # normalize weights to mean 1 for comparability
    w = w / (np.mean(w) + 1e-12)
    err2 = w * (y_true - y_pred) ** 2
    return float(np.sqrt(np.mean(err2)))

# ----------------------------
# Modeling: Ridge with optional standardization (per gene)
# ----------------------------
def fit_per_gene_ridge(
    df: pd.DataFrame,
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    alphas: List[float],
    train_pair_indices: Optional[List[int]],
    expanded_features: bool,
    standardize: bool = False
) -> Tuple[np.ndarray, Optional[List[StandardScaler]], Optional[List[StandardScaler]]]:
    """
    Returns (coefs, x_scalers, y_scalers).
    If standardize=False, scalers are None and behavior matches old code.
    """
    genes = list(df.index)
    gN = len(genes)
    F = num_features(expanded_features)
    coefs = np.zeros((gN, F), dtype=float)
    xscalers = [None] * gN
    yscalers = [None] * gN

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

            if standardize:
                xsc = StandardScaler()
                ysc = StandardScaler()
                Xs = xsc.fit_transform(X)
                ys = ysc.fit_transform(y.reshape(-1,1)).ravel()
                model = RidgeCV(alphas=alphas, fit_intercept=False)
                model.fit(Xs, ys)
                coefs[i_g, :] = model.coef_
                xscalers[i_g] = xsc
                yscalers[i_g] = ysc
            else:
                model = RidgeCV(alphas=alphas, fit_intercept=False)
                model.fit(X, y)
                coefs[i_g, :] = model.coef_

    return coefs, (xscalers if standardize else None), (yscalers if standardize else None)

def ridge_prediction_for_pair(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    coefs: np.ndarray,
    expanded_features: bool,
    xscalers: Optional[List[StandardScaler]] = None,
    yscalers: Optional[List[StandardScaler]] = None
) -> np.ndarray:
    a, b = pair
    ea_vec = singles.get(a, ctrl_vec)
    eb_vec = singles.get(b, ctrl_vec)

    y = np.zeros_like(ctrl_vec, dtype=float)
    for i in range(len(ctrl_vec)):
        fv = make_feature_vector(float(ea_vec[i]), float(eb_vec[i]), float(ctrl_vec[i]), expanded_features)
        if xscalers is not None and xscalers[i] is not None and yscalers is not None and yscalers[i] is not None:
            fv = xscalers[i].transform(fv.reshape(1,-1)).ravel()
            y_scaled = float(np.dot(coefs[i, :], fv))
            # unscale
            y[i] = y_scaled * (yscalers[i].scale_[0] if hasattr(yscalers[i], "scale_") else 1.0) + \
                   (yscalers[i].mean_[0] if hasattr(yscalers[i], "mean_") else 0.0)
        else:
            y[i] = float(np.dot(coefs[i, :], fv))
    return np.clip(y, 0.0, None)

# ----------------------------
# PCA pair-context
# ----------------------------
def build_pca_context(singles: Dict[str, np.ndarray], k: int):
    if k <= 0:
        return None
    keys = list(singles.keys())
    if len(keys) == 0:
        return None
    M = np.stack([singles[k] for k in keys], axis=1)  # [genes, n_singles]
    # We want components in genes-space; TruncatedSVD on genes x samples works well
    svd = TruncatedSVD(n_components=min(k, min(M.shape)-1), random_state=0)
    U = svd.fit(M).components_.T  # [n_singles, k] in this orientation; we will use U_gene afterward
    # To project a gene-level vector, we need gene PCs. Recompute on M^T to get gene loadings:
    svd2 = TruncatedSVD(n_components=min(k, min(M.T.shape)-1), random_state=0)
    Ugene = svd2.fit(M.T).components_.T  # [genes, k]
    return Ugene  # project gene vectors via Ugene.T @ vec  -> [k]

def project_gene_vec(vec: np.ndarray, Ugene: np.ndarray) -> np.ndarray:
    # vec: [genes,], Ugene: [genes, k]
    return Ugene.T @ vec  # [k]

# ----------------------------
# Residual learner
# ----------------------------
def build_pair_gene_features(
    ea_vec: np.ndarray, eb_vec: np.ndarray,
    ctrl_val: float,
    ea: float, eb: float,
    expanded: bool,
    pca_proj_a: Optional[np.ndarray] = None,
    pca_proj_b: Optional[np.ndarray] = None
) -> np.ndarray:
    """Concatenate per-gene compact features with global pair features (+ optional PCA context)."""
    f_gene = make_feature_vector(ea, eb, float(ctrl_val), expanded)
    f_pair = pair_global_features(ea_vec, eb_vec)
    if pca_proj_a is None or pca_proj_b is None:
        return np.concatenate([f_gene, f_pair], axis=0)
    # add symmetric PCA context: sum and abs-diff
    sum_pc = pca_proj_a + pca_proj_b
    diff_pc = np.abs(pca_proj_a - pca_proj_b)
    return np.concatenate([f_gene, f_pair, sum_pc, diff_pc], axis=0)

def _clip_residuals_mad(resids: np.ndarray, mult: float) -> np.ndarray:
    if mult <= 0:
        return resids
    med = np.median(resids)
    mad = np.median(np.abs(resids - med)) + 1e-12
    lo, hi = med - mult*mad, med + mult*mad
    return np.clip(resids, lo, hi)

def _inner_cv_oof_r2(X: np.ndarray, y: np.ndarray, seed: int, params: dict) -> Tuple[np.ndarray, float]:
    """
    Simple inner 5-fold OOF R^2 for residuals (per gene). Returns (y_oof_pred, r2).
    """
    if X.shape[0] < 40:
        # too small, return trivial predictions
        return np.zeros_like(y), -np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    yhat = np.zeros_like(y)
    for tr, te in kf.split(X):
        xsc = StandardScaler()
        Xt = xsc.fit_transform(X[tr])
        Xv = xsc.transform(X[te])
        md = HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("max_iter", 300),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 1e-3),
            early_stopping=params.get("early_stopping", False),
            validation_fraction=0.15 if params.get("early_stopping", False) else 0.1,
            n_iter_no_change=20 if params.get("early_stopping", False) else None,
            random_state=seed
        )
        md.fit(Xt, y[tr])
        yhat[te] = md.predict(Xv)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return yhat, float(r2)

def train_residual_models(
    df: pd.DataFrame,
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    base_on: str,
    coefs: Optional[np.ndarray],
    expanded_features: bool,
    alphas: List[float],
    train_pair_indices: List[int],
    resid_params: dict,
    random_state: int,
    pca_Ugene: Optional[np.ndarray],
    gate_r2_thresh: float
):
    """
    Train a small HGBR per-gene to predict residuals (y_true - y_base).
    Returns (scalers, models, gate_mask_per_gene (bool)).
    """
    all_pairs = build_training_pairs(df)
    genes = list(df.index)
    gN = len(genes)
    models = [None] * gN
    scalers = [None] * gN
    gate_mask = np.array([True] * gN)  # default allow

    if base_on == "ensemble" and coefs is None:
        raise ValueError("Residual base 'ensemble' requires ridge coefficients.")

    # Precompute PCA pair projections (same for all genes per pair)
    pair_to_pca = {}
    for idx in train_pair_indices:
        a, b, _ = all_pairs[idx]
        if pca_Ugene is not None:
            ea_vec = singles.get(a, ctrl_vec)
            eb_vec = singles.get(b, ctrl_vec)
            pca_a = project_gene_vec(ea_vec, pca_Ugene)
            pca_b = project_gene_vec(eb_vec, pca_Ugene)
        else:
            pca_a = pca_b = None
        pair_to_pca[idx] = (pca_a, pca_b)

    for i_g, _gname in enumerate(genes):
        Xg = []
        yg = []
        for idx in train_pair_indices:
            a, b, y_true = all_pairs[idx]
            ea_vec = singles.get(a, ctrl_vec)
            eb_vec = singles.get(b, ctrl_vec)

            # base predictions
            y_add = additive_prediction(singles, ctrl_vec, (a,b), alpha_vec=None)
            if base_on == "additive":
                y_base = y_add
            else:
                # ridge prediction needs coefs
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a,b), coefs, expanded_features)
                y_base = 0.5 * (y_add + y_ridge)

            resid = float(y_true[i_g] - y_base[i_g])

            pca_a, pca_b = pair_to_pca[idx]
            f = build_pair_gene_features(
                ea_vec, eb_vec, ctrl_vec[i_g], float(ea_vec[i_g]), float(eb_vec[i_g]),
                expanded_features, pca_a, pca_b
            )
            Xg.append(f); yg.append(resid)

        if len(Xg) >= 20:  # need enough doubles to train non-linear model
            Xg = np.vstack(Xg); yg = np.array(yg)

            # optional target clipping
            clip_mult = resid_params.get("clip_mad_mult", 0.0)
            if clip_mult > 0:
                yg = _clip_residuals_mad(yg, clip_mult)

            # inner OOF R^2
            yhat_oof, r2 = _inner_cv_oof_r2(
                Xg, yg, random_state,
                dict(
                    max_depth=resid_params.get("max_depth", 3),
                    learning_rate=resid_params.get("learning_rate", 0.05),
                    max_iter=resid_params.get("max_iter", 300),
                    min_samples_leaf=resid_params.get("min_samples_leaf", 20),
                    l2_regularization=resid_params.get("l2_regularization", 1e-3),
                    early_stopping=resid_params.get("early_stopping", False),
                )
            )
            if gate_r2_thresh > 0 and r2 < gate_r2_thresh:
                gate_mask[i_g] = False  # do not apply residuals for this gene

            # final fit on all (store scaler + model)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xg)
            model = HistGradientBoostingRegressor(
                max_depth=resid_params.get("max_depth", 3),
                learning_rate=resid_params.get("learning_rate", 0.05),
                max_iter=resid_params.get("max_iter", 300),
                min_samples_leaf=resid_params.get("min_samples_leaf", 20),
                l2_regularization=resid_params.get("l2_regularization", 1e-3),
                early_stopping=resid_params.get("early_stopping", False),
                validation_fraction=0.15 if resid_params.get("early_stopping", False) else 0.1,
                n_iter_no_change=20 if resid_params.get("early_stopping", False) else None,
                random_state=random_state
            )
            model.fit(Xs, yg)
            models[i_g] = model
            scalers[i_g] = scaler
        else:
            models[i_g] = None
            scalers[i_g] = None
            gate_mask[i_g] = False if gate_r2_thresh > 0 else True

    return scalers, models, gate_mask

def predict_with_residual(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    base_pred: np.ndarray,
    expanded_features: bool,
    scalers, models,
    pca_Ugene: Optional[np.ndarray],
    gate_mask: Optional[np.ndarray]
) -> np.ndarray:
    a, b = pair
    ea_vec = singles.get(a, ctrl_vec)
    eb_vec = singles.get(b, ctrl_vec)

    # per-pair PCA projections (once)
    if pca_Ugene is not None:
        pca_a = project_gene_vec(ea_vec, pca_Ugene)
        pca_b = project_gene_vec(eb_vec, pca_Ugene)
    else:
        pca_a = pca_b = None

    y = base_pred.copy()
    for i in range(len(ctrl_vec)):
        if gate_mask is not None and gate_mask.shape[0] == len(ctrl_vec) and not gate_mask[i]:
            continue  # gated off
        f = build_pair_gene_features(
            ea_vec, eb_vec, ctrl_vec[i], float(ea_vec[i]), float(eb_vec[i]),
            expanded_features, pca_a, pca_b
        )
        sc = scalers[i]; md = models[i]
        if md is not None and sc is not None:
            fv = sc.transform(f.reshape(1,-1))
            y[i] += float(md.predict(fv)[0])
    return np.clip(y, 0.0, None)

# ----------------------------
# Modeling helpers (must be defined before eval_kfold)
# ----------------------------
def additive_prediction(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    alpha_vec: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Additive baseline with optional scaled-additive mixing toward control.
    y = alpha * (ea + eb - ctrl) + (1 - alpha) * ctrl
    """
    a, b = pair
    ea = singles.get(a, ctrl_vec)
    eb = singles.get(b, ctrl_vec)
    base = ea + eb - ctrl_vec
    if alpha_vec is None:
        return np.clip(base, 0.0, None)
    y = alpha_vec * base + (1.0 - alpha_vec) * ctrl_vec
    return np.clip(y, 0.0, None)

def ridge_prediction_for_pair(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    pair: Tuple[str, str],
    coefs: np.ndarray,
    expanded_features: bool,
    xscalers: Optional[List[StandardScaler]] = None,
    yscalers: Optional[List[StandardScaler]] = None
) -> np.ndarray:
    """
    Per-gene ridge prediction using the compact/expanded feature vector.
    Supports optional per-gene standardization via xscalers/yscalers.
    """
    a, b = pair
    ea_vec = singles.get(a, ctrl_vec)
    eb_vec = singles.get(b, ctrl_vec)

    y = np.zeros_like(ctrl_vec, dtype=float)
    for i in range(len(ctrl_vec)):
        fv = make_feature_vector(float(ea_vec[i]), float(eb_vec[i]), float(ctrl_vec[i]), expanded_features)
        if xscalers is not None and xscalers[i] is not None and yscalers is not None and yscalers[i] is not None:
            fv = xscalers[i].transform(fv.reshape(1, -1)).ravel()
            y_scaled = float(np.dot(coefs[i, :], fv))
            # unscale back to original target scale
            y[i] = y_scaled * (yscalers[i].scale_[0] if hasattr(yscalers[i], "scale_") else 1.0) + \
                   (yscalers[i].mean_[0] if hasattr(yscalers[i], "mean_") else 0.0)
        else:
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
    adaptive_ensemble: bool,
    residual_learner: str,
    resid_on: str,
    resid_params: dict,
    ridge_standardize: bool,
    weighted_rmsd_flag: bool,
    adapt_shrink: float,
    resid_pca_k: int,
    resid_gate_r2: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genes = list(df.index)
    all_pairs = build_training_pairs(df)

    # Preprocess singles
    singles = winsorize_singles(singles_in, winsorize_pct)
    ctrl_vec = find_ctrl_vector(df, singles, robust_ctrl, trim_pct)

    # Optional per-gene noise (from singles) for weighted RMSD
    noise = per_gene_mad(list(singles.values())) if weighted_rmsd_flag else None
    weights = None
    if noise is not None:
        weights = 1.0 / (noise + 1e-12)

    # Keep only doubles with both singles present
    usable_idx = []
    for i, (a, b, _y) in enumerate(all_pairs):
        if singles.get(a) is not None and singles.get(b) is not None:
            usable_idx.append(i)

    if len(usable_idx) < k_folds:
        raise RuntimeError(f"Not enough usable doubles ({len(usable_idx)}) for k={k_folds} folds.")

    # PCA context (once)
    pca_Ugene = build_pca_context(singles, resid_pca_k) if resid_pca_k > 0 else None

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
            coefs, xsc, ysc = fit_per_gene_ridge(
                df, singles, ctrl_vec, alphas, train_pair_indices=train_idx_abs,
                expanded_features=expanded_features, standardize=ridge_standardize
            )
        else:
            coefs = xsc = ysc = None

        # ----- Fit scaled-additive alpha on training folds -----
        alpha_vec = None
        if scaled_additive_mode != "none":
            y_add_cols, y_true_cols = [], []
            for a,b,y_true in [all_pairs[i] for i in train_idx_abs]:
                y_add_cols.append(additive_prediction(singles, ctrl_vec, (a,b), alpha_vec=None))
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)
            y_true_M = np.stack(y_true_cols, axis=1)
            if scaled_additive_mode == "global":
                alpha_vec = fit_scaled_additive_alpha(y_add_M.flatten(), y_true_M.flatten(),
                                                      np.tile(ctrl_vec, y_add_M.shape[1]),
                                                      mode="global", alpha_clip=alpha_clip)
            else:
                alpha_vec = fit_scaled_additive_alpha(np.mean(y_add_M, axis=1),
                                                      np.mean(y_true_M, axis=1),
                                                      ctrl_vec, mode="per_gene", alpha_clip=alpha_clip)

        # ----- If adaptive ensemble: fit per-gene weights on training folds -----
        w_adapt = None
        if adaptive_ensemble and use_ridge:
            y_add_cols, y_ridge_cols, y_true_cols = [], [], []
            for a,b,y_true in [all_pairs[i] for i in train_idx_abs]:
                y_add = additive_prediction(singles, ctrl_vec, (a,b), alpha_vec)
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a,b), coefs, expanded_features, xsc, ysc)
                y_add_cols.append(y_add); y_ridge_cols.append(y_ridge); y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1)
            y_ridge_M = np.stack(y_ridge_cols, axis=1)
            y_true_M = np.stack(y_true_cols, axis=1)
            w_adapt_raw = fit_per_gene_adaptive_weights(y_add_M, y_ridge_M, y_true_M)
            if adapt_shrink > 0:
                n = y_add_M.shape[1]
                w_global = ensemble_weight
                w_adapt = (n/(n+adapt_shrink))*w_adapt_raw + (adapt_shrink/(n+adapt_shrink))*w_global
            else:
                w_adapt = w_adapt_raw

        # ----- Residual models on training folds (optional) -----
        scalers = models = gate_mask = None
        if residual_learner != "none":
            scalers, models, gate_mask = train_residual_models(
                df, singles, ctrl_vec, base_on=resid_on, coefs=coefs,
                expanded_features=expanded_features, alphas=alphas,
                train_pair_indices=train_idx_abs,
                resid_params=dict(
                    max_depth=resid_params.get("max_depth", 3),
                    learning_rate=resid_params.get("learning_rate", 0.05),
                    max_iter=resid_params.get("max_iter", 300),
                    min_samples_leaf=resid_params.get("min_samples_leaf", 20),
                    l2_regularization=resid_params.get("l2_regularization", 1e-3),
                    early_stopping=resid_params.get("early_stopping", False),
                    clip_mad_mult=resid_params.get("clip_mad_mult", 0.0),
                ),
                random_state=seed,
                pca_Ugene=pca_Ugene,
                gate_r2_thresh=resid_gate_r2
            )

        # Per-gene accumulators for this fold
        gN = len(genes)
        sse_add = np.zeros(gN, dtype=float)
        sse_ridge = np.zeros(gN, dtype=float)
        sse_ens = np.zeros(gN, dtype=float)
        sse_adapt = np.zeros(gN, dtype=float)
        sse_resid = np.zeros(gN, dtype=float)
        counts = np.zeros(gN, dtype=int)

        # Evaluate on held-out fold pairs
        for idx_abs in test_idx_abs:
            a, b, y_true = all_pairs[idx_abs]
            pair_name = f"{a}+{b}"

            y_add = additive_prediction(singles, ctrl_vec, (a, b), alpha_vec)
            rmsd_add = weighted_rmse(y_true, y_add, weights) if weighted_rmsd_flag else mean_squared_error(y_true, y_add, squared=False)

            if use_ridge:
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a, b), coefs, expanded_features, xsc, ysc)
                y_ens = np.clip((1.0 - ensemble_weight) * y_add + ensemble_weight * y_ridge, 0.0, None)
                rmsd_ridge = weighted_rmse(y_true, y_ridge, weights) if weighted_rmsd_flag else mean_squared_error(y_true, y_ridge, squared=False)
                rmsd_ens = weighted_rmse(y_true, y_ens, weights) if weighted_rmsd_flag else mean_squared_error(y_true, y_ens, squared=False)
            else:
                y_ridge = y_add; y_ens = y_add
                rmsd_ridge = None; rmsd_ens = rmsd_add

            # adaptive per-gene
            if adaptive_ensemble and use_ridge and w_adapt is not None:
                y_adapt = np.clip((1.0 - w_adapt) * y_add + w_adapt * y_ridge, 0.0, None)
                rmsd_adapt = weighted_rmse(y_true, y_adapt, weights) if weighted_rmsd_flag else mean_squared_error(y_true, y_adapt, squared=False)
            else:
                y_adapt = y_ens; rmsd_adapt = rmsd_ens

            # residual correction (optional)
            if residual_learner != "none" and models is not None:
                base_for_resid = y_add if resid_on == "additive" else y_ens
                y_resid = predict_with_residual(singles, ctrl_vec, (a,b), base_for_resid,
                                                expanded_features, scalers, models, pca_Ugene, gate_mask)
                rmsd_resid = weighted_rmse(y_true, y_resid, weights) if weighted_rmsd_flag else mean_squared_error(y_true, y_resid, squared=False)
            else:
                y_resid = y_adapt; rmsd_resid = rmsd_adapt

            rows_pair.append({
                "fold": fold_id,
                "pair": pair_name,
                "rmsd_additive": rmsd_add,
                "rmsd_ridge": rmsd_ridge,
                "rmsd_ensemble": rmsd_ens,
                "rmsd_adaptive": rmsd_adapt if adaptive_ensemble and use_ridge else None,
                "rmsd_residual": rmsd_resid if residual_learner != "none" else None
            })

            # accumulate per-gene SSE (unweighted, for per-gene RMSD like before)
            sse_add += (y_add - y_true) ** 2
            if use_ridge: sse_ridge += (y_ridge - y_true) ** 2
            sse_ens  += (y_ens  - y_true) ** 2
            if adaptive_ensemble and use_ridge: sse_adapt += (y_adapt - y_true) ** 2
            if residual_learner != "none": sse_resid += (y_resid - y_true) ** 2
            counts += 1

        # fold-level per-gene RMSD
        mask = counts > 0
        def fold_rmsd(sse):
            out = np.zeros(gN); out[mask] = np.sqrt(sse[mask] / counts[mask]); return out
        rmsd_add_gene = fold_rmsd(sse_add)
        rmsd_ens_gene = fold_rmsd(sse_ens)
        rmsd_ridge_gene = fold_rmsd(sse_ridge) if use_ridge else np.full(gN, np.nan)
        rmsd_adapt_gene = fold_rmsd(sse_adapt) if adaptive_ensemble and use_ridge else np.full(gN, np.nan)
        rmsd_resid_gene = fold_rmsd(sse_resid) if residual_learner != "none" else np.full(gN, np.nan)

        for i_g, gname in enumerate(genes):
            rows_gene.append({
                "fold": fold_id,
                "gene": gname,
                "rmsd_additive": float(rmsd_add_gene[i_g]),
                "rmsd_ridge": float(rmsd_ridge_gene[i_g]) if use_ridge else np.nan,
                "rmsd_ensemble": float(rmsd_ens_gene[i_g]),
                "rmsd_adaptive": float(rmsd_adapt_gene[i_g]) if adaptive_ensemble and use_ridge else np.nan,
                "rmsd_residual": float(rmsd_resid_gene[i_g]) if residual_learner != "none" else np.nan,
                "pairs_evaluated": int(counts[i_g])
            })

    metrics_per_pair_df = pd.DataFrame(rows_pair)
    metrics_per_gene_df = pd.DataFrame(rows_gene)

    print("\n=== K-FOLD SUMMARY ===")
    print(f"folds: {k_folds}, pairs evaluated: {len(metrics_per_pair_df)}")
    print(f"Mean RMSD (additive): {metrics_per_pair_df['rmsd_additive'].mean():.6f}")
    if use_ridge:
        print(f"Mean RMSD (ridge):    {metrics_per_pair_df['rmsd_ridge'].mean():.6f}")
    print(f"Mean RMSD (ensemble): {metrics_per_pair_df['rmsd_ensemble'].mean():.6f}")
    if adaptive_ensemble and use_ridge:
        print(f"Mean RMSD (adaptive): {metrics_per_pair_df['rmsd_adaptive'].mean():.6f}")
    if residual_learner != "none":
        print(f"Mean RMSD (residual): {metrics_per_pair_df['rmsd_residual'].mean():.6f}")

    return metrics_per_pair_df, metrics_per_gene_df

# ----------------------------
# Test pairs CSV reader (robust to header being a valid pair)
# ----------------------------
def read_test_pairs_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "perturbation" in df.columns:
        series = df["perturbation"].astype(str)
        header_candidate = None
    else:
        first_col_name = str(df.columns[0]) if len(df.columns) > 0 else ""
        series = df.iloc[:, 0].astype(str) if len(df.columns) > 0 else pd.Series([], dtype=str)
        header_candidate = first_col_name.strip()

    pairs: List[str] = []
    if header_candidate and PAIR_RE.match(header_candidate):
        pairs.append(header_candidate)

    for v in series.tolist():
        v = v.strip()
        if not v:
            continue
        _ = parse_pair(v)
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
    singles = winsorize_singles(singles_raw, args.winsorize_pct)

    if args.mode == "eval":
        resid_params = dict(
            max_depth=args.resid_max_depth,
            learning_rate=args.resid_learning_rate,
            max_iter=args.resid_max_iter,
            min_samples_leaf=args.resid_min_samples_leaf,
            l2_regularization=args.resid_l2,
            early_stopping=bool(args.resid_early_stop),
            clip_mad_mult=float(args.resid_clip_mad_mult),
        )
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
            residual_learner=args.residual_learner,
            resid_on=args.resid_on,
            resid_params=resid_params,
            ridge_standardize=bool(args.ridge_standardize),
            weighted_rmsd_flag=bool(args.weighted_rmsd),
            adapt_shrink=float(args.adapt_shrink),
            resid_pca_k=int(args.resid_pca_k),
            resid_gate_r2=float(args.resid_gate_r2)
        )
        metrics_pair_df.to_csv(args.metrics_per_pair_csv, index=False)
        metrics_gene_df.to_csv(args.metrics_per_gene_csv, index=False)
        print(f"\nWrote per-pair metrics to   {args.metrics_per_pair_csv}")
        print(f"Wrote per-gene metrics to   {args.metrics_per_gene_csv}")

        # summary CSV
        summary_rows = []
        overall = {
            "scope": "overall",
            "fold": "all",
            "mean_rmsd_additive": metrics_pair_df["rmsd_additive"].mean(),
            "mean_rmsd_ridge": metrics_pair_df["rmsd_ridge"].mean() if not args.no_ridge else np.nan,
            "mean_rmsd_ensemble": metrics_pair_df["rmsd_ensemble"].mean(),
            "mean_rmsd_adaptive": metrics_pair_df["rmsd_adaptive"].mean() if (not args.no_ridge and args.adaptive_ensemble) else np.nan,
            "mean_rmsd_residual": metrics_pair_df["rmsd_residual"].mean() if args.residual_learner != "none" else np.nan,
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
                "mean_rmsd_residual": sub["rmsd_residual"].mean() if args.residual_learner != "none" else np.nan,
                "pairs": len(sub)
            })
        pd.DataFrame(summary_rows).to_csv(args.metrics_csv, index=False)
        print(f"Wrote summary metrics to    {args.metrics_csv}")
    else:
        # predict
        test_pairs_list = read_test_pairs_csv(args.test_pairs)
        print(f"[predict] loaded pairs: {len(test_pairs_list)}")
        ctrl_vec = find_ctrl_vector(df, singles, args.robust_ctrl, float(args.trim_pct))
        expanded = (args.features == "expanded")

        # PCA context (optional)
        pca_Ugene = build_pca_context(singles, int(args.resid_pca_k)) if int(args.resid_pca_k) > 0 else None

        # ridge coefs on all doubles if enabled
        coefs = xsc = ysc = None
        if not args.no_ridge:
            coefs, xsc, ysc = fit_per_gene_ridge(
                df, singles, ctrl_vec, [float(x) for x in args.ridge_alphas.split(",") if x],
                train_pair_indices=None, expanded_features=expanded, standardize=bool(args.ridge_standardize)
            )

        # scaled-additive alpha on all doubles (optional)
        alpha_vec = None
        if args.scaled_additive != "none":
            y_add_cols, y_true_cols = [], []
            for (a,b,y_true) in build_training_pairs(df):
                y_add_cols.append(additive_prediction(singles, ctrl_vec, (a,b), alpha_vec=None))
                y_true_cols.append(y_true)
            y_add_M = np.stack(y_add_cols, axis=1); y_true_M = np.stack(y_true_cols, axis=1)
            if args.scaled_additive == "global":
                alpha_vec = fit_scaled_additive_alpha(y_add_M.flatten(), y_true_M.flatten(),
                                                      np.tile(ctrl_vec, y_add_M.shape[1]),
                                                      mode="global", alpha_clip=float(args.alpha_clip))
            else:
                alpha_vec = fit_scaled_additive_alpha(np.mean(y_add_M, axis=1),
                                                      np.mean(y_true_M, axis=1),
                                                      ctrl_vec, mode="per_gene", alpha_clip=float(args.alpha_clip))

        # residual learner on ALL doubles (optional)
        scalers = models = gate_mask = None
        if args.residual_learner != "none":
            resid_params = dict(
                max_depth=args.resid_max_depth,
                learning_rate=args.resid_learning_rate,
                max_iter=args.resid_max_iter,
                min_samples_leaf=args.resid_min_samples_leaf,
                l2_regularization=args.resid_l2,
                early_stopping=bool(args.resid_early_stop),
                clip_mad_mult=float(args.resid_clip_mad_mult),
            )
            base_on = args.resid_on
            all_idx = list(range(len(build_training_pairs(df))))
            scalers, models, gate_mask = train_residual_models(
                df, singles, ctrl_vec, base_on=base_on, coefs=coefs,
                expanded_features=expanded, alphas=alphas,
                train_pair_indices=all_idx,
                resid_params=resid_params, random_state=args.seed,
                pca_Ugene=pca_Ugene, gate_r2_thresh=float(args.resid_gate_r2)
            )

        # Produce predictions
        final_preds: Dict[str, np.ndarray] = {}
        for p in test_pairs_list:
            a, b = parse_pair(p)
            y_add = additive_prediction(singles, ctrl_vec, (a, b), alpha_vec)
            if args.no_ridge:
                y_base = y_add
            else:
                y_ridge = ridge_prediction_for_pair(singles, ctrl_vec, (a, b), coefs, expanded, xsc, ysc)
                w = float(args.ensemble_weight)
                y_base = np.clip((1.0 - w) * y_add + w * y_ridge, 0.0, None)

            # optional residual correction
            if args.residual_learner != "none" and models is not None:
                y_final = predict_with_residual(
                    singles, ctrl_vec, (a,b),
                    y_base if args.resid_on == "ensemble" else y_add,
                    expanded, scalers, models, pca_Ugene, gate_mask
                )
            else:
                y_final = y_base

            final_preds[p] = y_final

        # Optional symmetry averaging if both orders present
        if args.symmetry_average:
            # find pairs that have both a+b and b+a
            pair_set = set(final_preds.keys())
            visited = set()
            for p in list(pair_set):
                if p in visited:
                    continue
                a,b = parse_pair(p)
                q = f"{b}+{a}"
                if q in pair_set and q not in visited:
                    yavg = 0.5*(final_preds[p] + final_preds[q])
                    final_preds[p] = yavg
                    final_preds[q] = yavg
                    visited.add(p); visited.add(q)

        out_df = make_output_frame(genes, test_pairs_list, final_preds, args.template_csv)
        if out_df.isnull().any().any():
            raise RuntimeError("Output contains NaNs; please check inputs.")
        out_df = out_df[["gene", "perturbation", "expression"]].astype({"gene":str,"perturbation":str,"expression":float})
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(out_df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()