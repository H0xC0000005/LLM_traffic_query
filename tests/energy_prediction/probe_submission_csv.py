import pandas as pd

path = r"submissions/exp1_b_s42.csv"  # change if needed

# Read only the first few rows (fast even for huge files)
df_head = pd.read_csv(path, nrows=10)
print(df_head)

# Basic sanity checks
print("\nshape(head):", df_head.shape)
print("columns:", list(df_head.columns))
print("dtypes:\n", df_head.dtypes)

# Quick checks on the prediction column (assumes 2-column Kaggle-style file)
pred_col = df_head.columns[1]
print(f"\n{pred_col} summary (head):")
print(df_head[pred_col].describe())

# Optional: ensure row_id is increasing in the head
id_col = df_head.columns[0]
print(f"\n{id_col} increasing in head:", df_head[id_col].is_monotonic_increasing)

# Fast row count without loading the whole file into memory
with open(path, "rb") as f:
    n_rows = sum(1 for _ in f) - 1  # minus header

print("rows:", n_rows)

# Optional: also print columns (from header)
print("columns:", list(pd.read_csv(path, nrows=0).columns))
