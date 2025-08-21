### 0 The structure
Hackathon-Summer-2025/
├─ data/
│  ├─ train_matrix.csv          # Provided training matrix
│  ├─ test_pairs.csv            # Provided test pairs
├─ prediction/
│  └─ prediction.csv            # Model outputs (to submit)
├─ metrics/                     # Evaluation logs (created on eval mode)
│
├─ train_and_predict.py         # Main script (training + eval + predict)
└─ requirements.txt             # Python dependencies

### 1 Build the env
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 Build predictions (basic ridge+additive ensemble)
```bash
python train_and_predict.py \
  --train_csv data/train_matrix.csv \
  --test_pairs data/test_pairs.csv \
  --out_csv prediction/prediction.csv
```

### 2.1 K-fold with logs on train dataset
```bash
mkdir -p metrics
python train_and_predict.py \
  --mode eval \
  --train_csv data/train_matrix.csv \
  --k_folds 5 \
  --ensemble_weight 0.5 \
  --metrics_csv metrics/eval_metrics.csv
```
Run this to get 
- Console summary (mean RMSD for additive / ridge / ensemble).
- Per-pair per-fold metrics in metrics/eval_metrics.csv

### 2.2 K-fold per genes with logs on train dataset
```bash
mkdir -p metrics
python train_and_predict.py \
  --mode eval \
  --train_csv data/train_matrix.csv \
  --k_folds 5 \
  --ensemble_weight 0.5 \
  --metrics_per_pair_csv metrics/metrics_per_pair.csv \
  --metrics_per_gene_csv metrics/metrics_per_gene.csv
```

### 2.3 Disable ridge to see baseline ceiling:
```bash
python train_and_predict.py --mode eval --train_csv data/train_matrix.csv --no_ridge
```

### 2.4 F-fold eval with robust control, expanded features, scaled additive (per-gene), and adaptive ensembling 
```bash
mkdir -p metrics
python train_and_predict.py \
  --mode eval \
  --train_csv data/train_matrix.csv \
  --k_folds 5 \
  --features expanded \
  --robust_ctrl --trim_pct 0.1 \
  --winsorize_pct 0.01 \
  --scaled_additive per_gene \
  --adaptive_ensemble \
  --ensemble_weight 0.5 \
  --metrics_csv metrics/eval_metrics.csv
```

### 2.5 Same as 2.4 but for scoreboard
```bash
python train_and_predict.py \
  --mode predict \
  --train_csv data/train_matrix.csv \
  --test_pairs data/test_pairs.csv \
  --features expanded \
  --robust_ctrl --trim_pct 0.1 \
  --winsorize_pct 0.01 \
  --scaled_additive per_gene \
  --adaptive_ensemble \
  --ensemble_weight 0.5 \
  --out_csv prediction/prediction.csv
```