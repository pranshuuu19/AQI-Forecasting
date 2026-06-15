import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import xgboost as xgb # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore

# ── Load data & config ────────────────────────────────────────────────────────
train  = pd.read_csv("data/train.csv")
val    = pd.read_csv("data/val.csv")

with open("data/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = config["target_cols"]   # ["target_1h", "target_24h", "target_48h"]

os.makedirs("models", exist_ok=True)

# ── XGBoost hyperparameters (good starting defaults) ─────────────────────────
# These are reasonable defaults — you will tune these in Step 4
XGB_PARAMS = {
    "objective"        : "reg:squarederror",
    "n_estimators"     : 1000,      # high — early stopping will cut this down
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "subsample"        : 0.8,       # use 80% of rows per tree (reduces overfitting)
    "colsample_bytree" : 0.8,       # use 80% of features per tree
    "min_child_weight" : 5,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "early_stopping_rounds": 50,    # stop if val loss doesn't improve for 50 rounds
    "verbosity"        : 0,
}

X_train = train[FEATURES]
X_val   = val[FEATURES]

log_lines = []

# ── Train one model per target ─────────────────────────────────────────────────
for target in TARGETS:
    horizon = target.replace("target_", "")
    print(f"\n{'='*50}")
    print(f" Training model for: {target}  (horizon: {horizon})")
    print(f"{'='*50}")

    y_train = train[target]
    y_val   = val[target]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,   # print every 100 rounds
    )

    best_round = model.best_iteration
    print(f"  Best round (early stopping): {best_round}")

    # ── Validation metrics ────────────────────────────────────────────────────
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae  = mean_absolute_error(y_val, val_preds)
    r2   = r2_score(y_val, val_preds)

    result = (
        f"\nModel: {target}\n"
        f"  Best round : {best_round}\n"
        f"  Val RMSE   : {rmse:.4f}\n"
        f"  Val MAE    : {mae:.4f}\n"
        f"  Val R²     : {r2:.4f}\n"
    )
    print(result)
    log_lines.append(result)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = f"models/model_{horizon}.json"
    model.save_model(model_path)
    print(f"  ✓ Saved to {model_path}")

# ── Save training log ─────────────────────────────────────────────────────────
with open("models/training_log.txt", "w") as f:
    f.writelines(log_lines)

print("\n✓ All 3 models trained and saved.")