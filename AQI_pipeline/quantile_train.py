
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import xgboost as xgb # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore

# ── Load data + config ────────────────────────────────────────────────────────
train  = pd.read_csv("data/train.csv")
val    = pd.read_csv("data/val.csv")

with open("data/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = config["target_cols"]  # ["target_1h", "target_24h", "target_48h"]

os.makedirs("models", exist_ok=True)

X_train = train[FEATURES]
X_val   = val[FEATURES]

# ── Quantile params — same as your tuned model defaults ──────────────────────
BASE_PARAMS = {
    "n_estimators"        : 500,
    "learning_rate"       : 0.05,
    "max_depth"           : 6,
    "subsample"           : 0.8,
    "colsample_bytree"    : 0.8,
    "min_child_weight"    : 5,
    "random_state"        : 42,
    "n_jobs"              : -1,
    "verbosity"           : 0,
    "early_stopping_rounds": 50,
}

QUANTILES = {
    "lower": 0.10,  # 10th percentile — lower bound
    "upper": 0.90,  # 90th percentile — upper bound
}

# ── Train per target per quantile ─────────────────────────────────────────────
for target in TARGETS:
    horizon   = target.replace("target_", "")
    y_train   = train[target]
    y_val     = val[target]

    print(f"\n{'='*50}")
    print(f" Training quantile models for: {target}")
    print(f"{'='*50}")

    for q_name, alpha in QUANTILES.items():
        print(f"\n  Quantile: {q_name} (alpha={alpha})")

        params = {
            **BASE_PARAMS,
            "objective"     : "reg:quantileerror",
            "quantile_alpha": alpha,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validate — lower should be below point predictions, upper above
        val_preds = model.predict(X_val)
        coverage  = np.mean(
            (y_val >= val_preds) if q_name == "lower" else (y_val <= val_preds)
        )
        mae = mean_absolute_error(y_val, val_preds)

        print(f"  Best round : {model.best_iteration}")
        print(f"  Val MAE    : {mae:.4f}")
        print(f"  Coverage   : {coverage*100:.1f}% "
              f"({'actuals above lower' if q_name == 'lower' else 'actuals below upper'})")

        path = f"models/quantile_{horizon}_{q_name}.json"
        model.save_model(path)
        print(f"  ✓ Saved to {path}")

print("\n✓ All 6 quantile models trained and saved.")
print("\nModel files created:")
for target in TARGETS:
    horizon = target.replace("target_", "")
    for q in ["lower", "upper"]:
        print(f"  models/quantile_{horizon}_{q}.json")