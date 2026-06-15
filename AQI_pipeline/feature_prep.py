import pandas as pd # type: ignore
import json

# ── Load train only (never look at val/test to make decisions) ───────────────
train = pd.read_csv("data/train.csv")
print(f"Train columns ({len(train.columns)}): loaded")

# ── Columns to DROP ───────────────────────────────────────────────────────────
DROP_COLS = [
    "datetime",       # not a numeric feature
    "city",           # replaced by city_encoded
    "aqi_label",      # derived from pm2_5 → target leakage
    # duplicate hour_bucket cols (the .1 versions were already removed in your cleaning)
    # if any remain, add them here e.g. "hour_bucket_daytime.1"
]

# ── Target columns ────────────────────────────────────────────────────────────
# We train 3 separate models, one per horizon
TARGETS = ["target_1h", "target_24h", "target_48h"]

# ── Feature columns = everything else ─────────────────────────────────────────
all_cols     = list(train.columns)
feature_cols = [c for c in all_cols if c not in DROP_COLS + TARGETS]

print(f"\nFeature columns ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  {c}")

print(f"\nTarget columns : {TARGETS}")
print(f"Dropped columns: {DROP_COLS}")

# ── Check for any remaining NaNs in features ─────────────────────────────────
nan_counts = train[feature_cols].isnull().sum()
nan_cols   = nan_counts[nan_counts > 0]
if len(nan_cols):
    print(f"\n⚠ NaN found in features — handle before modelling:")
    print(nan_cols)
else:
    print("\n✓ No NaN in feature columns")

# ── Save config ───────────────────────────────────────────────────────────────
config = {
    "feature_cols" : feature_cols,
    "target_cols"  : TARGETS,
    "drop_cols"    : DROP_COLS,
}
with open("data/feature_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n✓ Saved data/feature_config.json")