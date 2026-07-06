import json
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xgboost as xgb # type: ignore
import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data & config (identical to model.py) ───────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

with open(f"{DATA_DIR}/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = config["target_cols"]  # ["target_1h", "target_24h", "target_48h"]

X_train = train[FEATURES]
X_test  = test[FEATURES]

print(f"Train rows: {len(train)} | Test rows: {len(test)} | Features: {len(FEATURES)}")

# ── Random Forest hyperparameters ────────────────────────────────────────────
# Modest settings on purpose — this is a baseline comparison, not a second
# tuning project. n_jobs=-1 to use all cores; random_state for reproducibility.
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
}

results = []

for target in TARGETS:
    horizon = target.replace("target_", "")
    y_train = train[target]
    y_test  = test[target]

    print(f"\n{'='*50}\n Horizon: {horizon}\n{'='*50}")

    # ── Linear Regression ─────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # ── Existing tuned XGBoost model (already trained, just load + predict) ─
    # NOTE: tuned_model_*.json was trained on an older 49-feature set, while
    # feature_config.json now reflects the newer 61-feature set (added for the
    # quantile models). Rather than trust feature_config.json, we read the
    # model's own expected feature names — same pattern dashboard.py uses —
    # so this always matches whatever that specific model file was trained on.
    xgb_path = f"{MODELS_DIR}/tuned_model_{horizon}.json"
    if not os.path.exists(xgb_path):
        xgb_path = f"{MODELS_DIR}/model_{horizon}.json"
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    xgb_features = xgb_model.get_booster().feature_names
    missing = [f for f in xgb_features if f not in X_test.columns]
    if missing:
        raise ValueError(f"Test set is missing features the XGBoost model expects: {missing}")
    xgb_preds = xgb_model.predict(X_test[xgb_features])
    print(f"  (XGBoost model uses {len(xgb_features)} features; LR/RF use {len(FEATURES)})")


    for model_name, preds in [
        ("Linear Regression", lr_preds),
        ("Random Forest", rf_preds),
        ("XGBoost (tuned)", xgb_preds),
    ]:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        print(f"  {model_name:<20} RMSE={rmse:7.3f}  MAE={mae:7.3f}  R2={r2:.4f}")
        results.append({
            "horizon": horizon,
            "model": model_name,
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3),
            "R2": round(r2, 4),
        })

results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
print(f"\n✓ Saved metrics table to {RESULTS_DIR}/model_comparison.csv")

# ── Chart: grouped RMSE bar chart, one group per horizon ─────────────────────
horizons = results_df["horizon"].unique().tolist()
models_order = ["Linear Regression", "Random Forest", "XGBoost (tuned)"]
colors = {"Linear Regression": "#94A3B8", "Random Forest": "#F59E0B", "XGBoost (tuned)": "#10B981"}

fig, ax = plt.subplots(figsize=(9, 5))
bar_width = 0.25
x = np.arange(len(horizons))

for i, model_name in enumerate(models_order):
    subset = results_df[results_df["model"] == model_name].set_index("horizon").loc[horizons]
    ax.bar(x + i * bar_width, subset["RMSE"], width=bar_width,
           label=model_name, color=colors[model_name])
    for xi, val in zip(x + i * bar_width, subset["RMSE"]):
        ax.text(xi, val + 0.3, f"{val:.2f}", ha="center", fontsize=9)

ax.set_xticks(x + bar_width)
ax.set_xticklabels(horizons)
ax.set_ylabel("RMSE (μg/m³, lower is better)")
ax.set_title("Model Comparison — Test Set RMSE by Horizon")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
print(f"✓ Saved chart to {RESULTS_DIR}/model_comparison.png")