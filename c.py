# how many pairs did we read?
python - <<'PY'
import pandas as pd, re
df = pd.read_csv("data/test_pairs.csv")
col = df["perturbation"] if "perturbation" in df.columns else df[df.columns[0]]
pairs = []
pat = re.compile(r"^(g\d{4})\+(g\d{4}|ctrl)$", re.I)
for v in col.astype(str):
    v=v.strip()
    if v and pat.match(v): pairs.append(v)
print("pairs_total:", len(pairs), "unique:", len(set(pairs)))
print("bad_rows:", df.shape[0] - len(pairs))
PY

# how many genes?
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/train_matrix.csv", index_col=0)
print("genes:", df.shape[0])
PY