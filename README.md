### 0 The structure
Hackathon-Summer-2025/
├─ data/
│  ├─ train_matrix.csv        
│  ├─ test_pairs.cvs           
├─ prediction/
│  └─ prediction.csv           
├─ train_and_predict.py
└─ requirements.txt

### 1 Build the env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### 2 Build predictions (ridge+additive ensemble)
python train_and_predict.py \
  --train_csv data/train_matrix.csv \
  --test_pairs data/test_pairs.csv \
  --out_csv prediction/prediction.csv
