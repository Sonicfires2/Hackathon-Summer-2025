#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train simple models to predict double-perturbation expression profiles from singles,
then write prediction/prediction.csv in the exact hackathon format.

Usage:
  python train_and_predict.py \
      --train_csv data/train_matrix.csv \
      --test_pairs data/test_pairs.txt \
      --out_csv prediction/prediction.csv \
      [--template_csv data/template_prediction.csv] \
      [--no_ridge]  # if you want baseline-only
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

PAIR_RE = re.compile(r"^(g\d{4})\+(g\d{4}|ctrl)$", re.IGNORECASE)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="Matrix CSV (rows=genes, cols=perturbations). Index col=gene id.")
    ap.add_argument("--test_pairs", required=True, help="Text file with one pair per line, e.g. g0160+g0495")
    ap.add_argument("--out_csv", required=True, help="Output: prediction/prediction.csv")
    ap.add_argument("--template_csv", default=None, help="Optional 3-col template to copy (gene,perturbation) order from")
    ap.add_argument("--no_ridge", action="store_true", help="Skip per-gene ridge; use additive baseline only")
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1.0,10.0", help="Comma list for RidgeCV")
    ap.add_argument("--ensemble_weight", type=float, default=0.5, help="Weight on ridge model in [0,1]; (1-w) on additive")
    return ap.parse_args()

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
    # The file you pasted shows a leading comma in the header → index column is the first.
    df = pd.read_csv(csv_path, index_col=0)
    # Normalize column labels (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    df.index = [i.strip() for i in df.index]
    # De-duplicate columns if necessary by averaging
    if len(df.columns) != len(set(df.columns)):
        df = df.groupby(axis=1, level=0).mean()
    return df

def extract_singles(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    singles = {}
    for c in df.columns:
        if is_single(c):
            g = c.split("+")[0]
            singles[g] = df[c].values.astype(float)
    return singles

def find_ctrl_vector(df: pd.DataFrame, singles: Dict[str, np.ndarray]) -> np.ndarray:
    # Try explicit control column(s): 'ctrl+ctrl' or 'ctrl'
    ctrl_cols = [c for c in df.columns if c.lower() in ("ctrl+ctrl", "ctrl")]
    if ctrl_cols:
        return df[ctrl_cols[0]].values.astype(float)
    # Fallback: per-gene median across singles (robust)
    if len(singles) == 0:
        raise RuntimeError("No singles found; cannot derive control.")
    S = np.stack(list(singles.values()), axis=1)  # shape: [num_genes, num_single_perturbations]
    return np.median(S, axis=1)

def build_training_pairs(df: pd.DataFrame) -> List[Tuple[str, str, np.ndarray]]:
    """
    Return list of (a, b, y_vec) for all double columns present in df.
    y_vec is the 1000-dim expression vector across genes.
    """
    pairs = []
    for c in df.columns:
        if is_double(c):
            a, b = c.split("+", 1)
            pairs.append((a, b, df[c].values.astype(float)))
    return pairs

def make_feature_vector(ea: float, eb: float, e0: float) -> np.ndarray:
    # Symmetric, low-dimensional features for a given target gene value
    return np.array([
        1.0,                # intercept
        ea,
        eb,
        ea * eb,
        abs(ea - eb),
        ea + eb,
        e0
    ], dtype=float)

def fit_per_gene_ridge(
    df: pd.DataFrame,
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    alphas: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a separate RidgeCV for each target gene across available double columns.
    Returns (coef_matrix, intercepts)
      - coef_matrix shape: [num_genes, num_features]
      - intercepts shape:  [num_genes]
    If a gene has too few doubles or missing singles for a pair, it falls back to zeros (intercept-only), so that
    prediction reduces toward baseline/additive ensemble.
    """
    genes = list(df.index)
    num_genes = len(genes)
    num_features = 7  # as in make_feature_vector
    coefs = np.zeros((num_genes, num_features), dtype=float)
    intercepts = np.zeros((num_genes,), dtype=float)

    # Pre-cache single vectors for speed
    # singles_map[g_perturb][row_gene_index] = expression
    # ctrl_vec[row_gene_index] is baseline
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    train_pairs = build_training_pairs(df)

    # For each target gene g*, build a dataset across doubles
    for i_g, g_star in enumerate(genes):
        X_list = []
        y_list = []
        e0 = ctrl_vec[i_g]

        for a, b, y_vec in train_pairs:
            # Need both singles present
            ea_vec = singles.get(a, None)
            eb_vec = singles.get(b, None)
            if ea_vec is None or eb_vec is None:
                continue
            ea = float(ea_vec[i_g])
            eb = float(eb_vec[i_g])
            fv = make_feature_vector(ea, eb, e0)
            X_list.append(fv)
            y_list.append(float(y_vec[i_g]))

        if len(X_list) >= 10:
            X = np.vstack(X_list)
            y = np.array(y_list)
            model = RidgeCV(alphas=alphas, fit_intercept=False)
            model.fit(X, y)
            coefs[i_g, :] = model.coef_
            intercepts[i_g] = 0.0  # already in feature 1
        else:
            # Too few training samples → keep zeros, acts like fallback
            pass

    return coefs, intercepts

def additive_prediction(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    genes_order: List[str],
    pair: Tuple[str, str],
) -> np.ndarray:
    a, b = pair
    ea = singles.get(a, None)
    eb = singles.get(b, None)
    if ea is None or eb is None:
        # Backfill with per-gene median if a single is missing (rare)
        # But we already used per-gene median as ctrl; use that as flat vector here
        if ea is None and eb is None:
            return ctrl_vec.copy()
        if ea is None:
            ea = ctrl_vec
        if eb is None:
            eb = ctrl_vec
    y = ea + eb - ctrl_vec
    y = np.clip(y, 0.0, None)
    return y

def ridge_prediction_for_pair(
    singles: Dict[str, np.ndarray],
    ctrl_vec: np.ndarray,
    genes_order: List[str],
    pair: Tuple[str, str],
    coefs: np.ndarray,
) -> np.ndarray:
    a, b = pair
    ea_vec = singles.get(a, None)
    eb_vec = singles.get(b, None)
    if ea_vec is None or eb_vec is None:
        # If a single is missing, use ctrl as backfill (same logic as additive)
        if ea_vec is None and eb_vec is None:
            ea_vec = ctrl_vec
            eb_vec = ctrl_vec
        elif ea_vec is None:
            ea_vec = ctrl_vec
        else:
            eb_vec = ctrl_vec

    # Build features per-gene and multiply by learned coefs
    # coefs shape: [num_genes, 7], features: [7]
    y = np.zeros_like(ctrl_vec, dtype=float)
    for i in range(len(genes_order)):
        fv = make_feature_vector(float(ea_vec[i]), float(eb_vec[i]), float(ctrl_vec[i]))
        y[i] = float(np.dot(coefs[i, :], fv))
    y = np.clip(y, 0.0, None)
    return y

def read_test_pairs(path: str) -> List[str]:
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Accept either "g0001+g0002" or "g0001+ctrl" in case organizers test singles too
            _ = parse_pair(line)  # validates format
            pairs.append(line)
    return pairs

def make_output_frame(
    genes: List[str],
    pairs: List[str],
    predictions: Dict[str, np.ndarray],
    template_csv: Optional[str]
) -> pd.DataFrame:
    if template_csv:
        # Copy row order from template (must match gene & pair names found there)
        tpl = pd.read_csv(template_csv)
        assert set(tpl.columns[:3]) >= set(["gene", "perturbation", "expression"])
        out_gene = tpl["gene"].tolist()
        out_pair = tpl["perturbation"].tolist()
        vals = []
        for g, p in zip(out_gene, out_pair):
            vals.append(predictions[p][genes.index(g)])
        return pd.DataFrame({"gene": out_gene, "perturbation": out_pair, "expression": vals})
    else:
        # Default order: all genes (as in matrix) × all pairs (as in test list)
        rows = []
        for p in pairs:
            y = predictions[p]
            for i, g in enumerate(genes):
                rows.append((g, p, float(y[i])))
        return pd.DataFrame(rows, columns=["gene", "perturbation", "expression"])

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    alphas = [float(x) for x in args.ridge_alphas.split(",") if x]

    df = load_matrix(args.train_csv)
    genes = list(df.index)

    # Build singles dictionary and ctrl vector
    singles = extract_singles(df)
    ctrl_vec = find_ctrl_vector(df, singles)

    # Read test pairs
    test_pairs = read_test_pairs(args.test_pairs)

    # Additive predictions for all pairs
    add_preds = {}
    for p in test_pairs:
        pair = parse_pair(p)
        add_preds[p] = additive_prediction(singles, ctrl_vec, genes, pair)

    if args.no_ridge:
        final_preds = add_preds
    else:
        # Fit per-gene ridge across observed doubles
        coefs, _ = fit_per_gene_ridge(df, singles, ctrl_vec, alphas)
        ridge_preds = {}
        for p in test_pairs:
            pair = parse_pair(p)
            ridge_preds[p] = ridge_prediction_for_pair(singles, ctrl_vec, genes, pair, coefs)

        # Simple fixed-weight ensemble
        w = float(args.ensemble_weight)
        final_preds = {}
        for p in test_pairs:
            final_preds[p] = (1.0 - w) * add_preds[p] + w * ridge_preds[p]
            final_preds[p] = np.clip(final_preds[p], 0.0, None)

    out_df = make_output_frame(genes, test_pairs, final_preds, args.template_csv)
    # Final sanity checks
    if out_df.isnull().any().any():
        raise RuntimeError("Output contains NaNs; please check inputs.")
    # Enforce column order and types
    out_df = out_df[["gene", "perturbation", "expression"]]
    out_df["gene"] = out_df["gene"].astype(str)
    out_df["perturbation"] = out_df["perturbation"].astype(str)
    out_df["expression"] = out_df["expression"].astype(float)

    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()