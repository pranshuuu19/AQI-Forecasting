import pandas as pd # type: ignore
import os

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "/Users/pranshu/AQI_project/data/processed/master_processed.csv"   # ← change to your actual file path
OUTPUT_DIR  = "data"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remaining 15% automatically

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"Total rows      : {len(df)}")
print(f"Date range      : {df['datetime'].min()} → {df['datetime'].max()}")
print(f"Cities          : {df['city'].unique().tolist()}")

# ── Split indices ─────────────────────────────────────────────────────────────
n        = len(df)
train_end = int(n * TRAIN_RATIO)
val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

train = df.iloc[:train_end].copy()
val   = df.iloc[train_end:val_end].copy()
test  = df.iloc[val_end:].copy()

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUTPUT_DIR}/val.csv",     index=False)
test.to_csv(f"{OUTPUT_DIR}/test.csv",   index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
Split Summary
=============
Total rows : {n}
Train      : {len(train)} rows  ({len(train)/n*100:.1f}%)
             {train['datetime'].min()} → {train['datetime'].max()}
Validation : {len(val)} rows  ({len(val)/n*100:.1f}%)
             {val['datetime'].min()} → {val['datetime'].max()}
Test       : {len(test)} rows  ({len(test)/n*100:.1f}%)
             {test['datetime'].min()} → {test['datetime'].max()}
"""
print(summary)
with open(f"{OUTPUT_DIR}/split_summary.txt", "w") as f:
    f.write(summary)

print("✓ Saved to data/train.csv, data/val.csv, data/test.csv")