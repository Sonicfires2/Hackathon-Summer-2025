# Double Perturbation Expression Prediction

## Overview

This project predicts **double-perturbation** gene expression profiles from **single-perturbation** data (Perturb-seq style).  
It combines a robust additive baseline with per-gene **ridge regression**, optional **adaptive ensembling**, and a lightweight **residual learner** (HistGradientBoostingRegressor) to capture non-additive effects (synergy/antagonism). It’s designed to run comfortably on a laptop (e.g., MacBook M3 Pro, 36 GB RAM).

### Key ideas

- **Additive baseline (scaled):**  
  \( \hat{y}_{add} = \alpha \cdot (y_a + y_b - y_{ctrl}) + (1-\alpha)\cdot y_{ctrl} \)  
  with \(\alpha\) fitted **globally** or **per-gene**, and optional **robust control** via trimmed median.
- **Ridge regression per gene:**  
  Compact, symmetric features per gene (basic or expanded) trained on observed doubles.
- **Adaptive ensemble (optional):**  
  Learns a per-gene weight \( w \in [0,1] \) to blend additive and ridge on training doubles.
- **Residual learner (optional):**  
  A small **HistGradientBoostingRegressor** per gene fits residuals on top of additive/ensemble using per-gene features + **global pair features** (cosine similarity, norms, dot of single profiles).

The script is robust to common CSV quirks (e.g., a **first row** like `g0037+g0083` being parsed as a header).

---

## Repo layout (expected)

```
.
├── train_and_predict.py
├── data/
│   ├── train_matrix.csv        # rows = genes; cols = perturbations (e.g., g0001+ctrl, g0001+g0123, ...)
│   └── test_pairs.csv          # single column; each row "g####+g####" or "g####+ctrl"
├── prediction/
│   └── prediction.csv          # (generated) 3-column output: gene, perturbation, expression
├── metrics/                    # (generated) evaluation logs/CSVs
└── predict.sh                  # convenience script to run prediction
```

---

## Install & setup

1) **Python**: 3.10–3.12 recommended  
2) **Create a venv & install deps**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn
```

> No GPU/CUDA is required. Everything runs on CPU.

---

## Data format

### `data/train_matrix.csv`
- **Index (row labels):** gene IDs (e.g., `g0001`, `g0002`, …)
- **Columns:** perturbations like:
  - Singles: `g0001+ctrl`, `g0123+ctrl`, …
  - Doubles: `g0001+g0123`, `g0456+g0789`, …
  - Control (optional): `ctrl` or `ctrl+ctrl`  
- **Values:** expression (float). Missing values are not expected.

### `data/test_pairs.csv`
- **Single column** containing strings like `g0001+g0123` or `g0123+ctrl`.  
- Header can be either `perturbation` or **absent**.  
- If the first line is itself a valid pair (e.g., `g0037+g0083`), the script **recovers it** even if pandas swallowed it as the column header.

---

## Core algorithm (quick technical notes)

1. **Control vector**  
   - If a control column `ctrl` / `ctrl+ctrl` exists, use it.  
   - Otherwise, compute control per gene across **singles** via **median** or **trimmed median** (`--robust_ctrl`, `--trim_pct`).

2. **Winsorization (optional)**  
   - Per-gene across singles to damp extreme values (`--winsorize_pct`).

3. **Additive baseline**  
   - Predict \( y_a + y_b - y_{ctrl} \) with optional **scaled** version using \(\alpha\) (`--scaled_additive {none,global,per_gene}`; `--alpha_clip`).

4. **Ridge per gene**  
   - Feature sets:
     - `basic` (7 features): `[1, ea, eb, ea*eb, |ea-eb|, ea+eb, e0]`
     - `expanded` (12 features): adds min/max and quadratics  
   - Fit `RidgeCV` across observed doubles; predict per gene.

5. **Adaptive ensemble (optional)**  
   - Learn per-gene weight \( w \) on training doubles s.t.  
     \( \hat{y} = (1-w)\cdot \hat_{add} + w\cdot \hat{y}_{ridge} \).

6. **Residual learner (optional)**  
   - Train small **HGB** per gene to predict residuals on top of:
     - additive (`--resid_on additive`) or
     - ensemble (`--resid_on ensemble`)  
   - Uses per-gene features + **global pair features** (cosine similarity, norms of `ea_vec`/`eb_vec`, pair dot product).

---

## Command-line options

### Major switches

- `--mode {eval,predict}`  
  - `eval`: k-fold CV over observed doubles, produces metrics in `metrics/`.  
  - `predict`: read `data/test_pairs.csv` and write `prediction/prediction.csv`.

- `--features {basic,expanded}`  
  Choose feature size for ridge/residual features. `expanded` is usually better.

- `--no_ridge`  
  Disable ridge entirely (pure additive + optional residual on additive).

- `--adaptive_ensemble`  
  Learn per-gene blend between additive and ridge (used in eval; in predict you can still blend via `--ensemble_weight`).

- `--scaled_additive {none,global,per_gene}`  
  Fit \(\alpha\) to scale additive towards control.

- `--robust_ctrl --trim_pct 0.10`  
  Use trimmed median (10% per tail) to estimate control from singles if explicit control column missing.

- `--winsorize_pct 0.01`  
  Winsorize single profiles per gene at 1% per tail.

### Ridge settings

- `--ridge_alphas` (comma list): e.g., `1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100`  
- `--ensemble_weight` (float): fixed blend \( \hat{y} = (1-w)\cdot add + w\cdot ridge \) when `--adaptive_ensemble` is **off** (and used as base in predict).

### Residual learner (HGB)

- `--residual_learner {none,hgb}`: turn on/off
- `--resid_on {additive,ensemble}`: which base to correct
- `--resid_max_depth` (int): tree depth (default 3)
- `--resid_learning_rate` (float): small LR (default 0.05)
- `--resid_max_iter` (int): boosting iterations (default 300)
- `--resid_min_samples_leaf` (int): min samples per leaf (default 20)

> Tip: For small training sets, keep trees **shallow** and leaves **not too small** to avoid overfit.

### Evaluation logging

- `--k_folds` (int): default 5  
- `--metrics_csv`, `--metrics_per_pair_csv`, `--metrics_per_gene_csv`  
  Summary and detailed CSVs under `metrics/`.

---

## Quickstart

### Evaluate (5-fold CV)
```bash
mkdir -p metrics
python train_and_predict.py \
  --mode eval \
  --train_csv data/train_matrix.csv \
  --k_folds 5 \
  --features expanded \
  --robust_ctrl --trim_pct 0.10 \
  --winsorize_pct 0.01 \
  --scaled_additive per_gene \
  --adaptive_ensemble \
  --ensemble_weight 0.5 \
  --ridge_alphas 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100 \
  --residual_learner hgb --resid_on ensemble \
  --resid_max_depth 2 --resid_learning_rate 0.05 --resid_max_iter 300 \
  --resid_min_samples_leaf 25 \
  --metrics_csv metrics/eval_metrics.csv \
  --metrics_per_pair_csv metrics/metrics_per_pair.csv \
  --metrics_per_gene_csv metrics/metrics_per_gene.csv
```

### Predict (from `data/test_pairs.csv`)
```bash
python train_and_predict.py \
  --mode predict \
  --train_csv data/train_matrix.csv \
  --test_pairs data/test_pairs.csv \
  --out_csv prediction/prediction.csv \
  --features expanded \
  --robust_ctrl --trim_pct 0.10 \
  --winsorize_pct 0.01 \
  --scaled_additive per_gene \
  --ensemble_weight 0.5 \
  --ridge_alphas 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100 \
  --residual_learner hgb --resid_on ensemble \
  --resid_max_depth 2 --resid_learning_rate 0.05 --resid_max_iter 300 \
  --resid_min_samples_leaf 25
```

(See script below.)

---

## Output format

`prediction/prediction.csv` is a **3-column** CSV:

| gene | perturbation | expression |
|------|---------------|------------|
| g0001 | g0001+g0123 | 0.12345 |
| g0002 | g0001+g0123 | 0.06789 |
| … | … | … |

Ordering is `(gene, pair)` by nested loops (or a custom template if you pass `--template_csv` with `gene, perturbation, expression` columns).

---

## Tuning tips (to push RMSD lower)

- **Alpha mode**: `--scaled_additive per_gene` typically helps.
- **Features**: `--features expanded` is recommended.
- **Ridge**: Use a **dense alpha grid** and keep `fit_intercept=False` (already set).
- **Residuals**:
  - Start with `--resid_on ensemble` if ridge is on.
  - Keep HGB **shallow** (`--resid_max_depth 2–3`) and moderate `--resid_min_samples_leaf 20–50`.
  - Increase `--resid_max_iter` gradually; stop if CV RMSD stalls.
- **Robustness**: `--winsorize_pct 0.01` and `--robust_ctrl --trim_pct 0.10` often stabilize training.
- **Adaptive ensemble**: Turn on during `eval` to learn principled per-gene weights; in `predict`, the script blends using `--ensemble_weight` unless residual models are applied.

---

## Troubleshooting
- **NaNs in output**: The script raises if any NaNs appear. Check train matrix for missing/invalid numbers.
- **Not enough doubles for k-fold**: Reduce `--k_folds` or ensure both singles exist for the doubles you want to use.

---

## License / Attribution
- Built for the **Rochester Biomedical DS Hackathon (Summer 2025)** task: predicting double over-expression gene expression from Perturb-seq-like data.
- Based on classic additive baselines and lightweight ML on top.

---

## Run for predictions

```bash
# Ensure output dir exists
mkdir -p prediction

python train_and_predict.py \
  --mode predict \
  --train_csv data/train_matrix.csv \
  --test_pairs data/test_pairs.csv \
  --out_csv prediction/prediction.csv \
  --features expanded \
  --robust_ctrl --trim_pct 0.10 \
  --winsorize_pct 0.01 \
  --scaled_additive per_gene \
  --ensemble_weight 0.5 \
  --ridge_alphas 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100 \
  --residual_learner hgb --resid_on ensemble \
  --resid_max_depth 2 --resid_learning_rate 0.05 --resid_max_iter 300 \
  --resid_min_samples_leaf 25

```
