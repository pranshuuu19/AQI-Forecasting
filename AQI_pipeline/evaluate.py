import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import argparse
import xgboost as xgb # type: ignore
from sklearn.metrics import (mean_squared_error, mean_absolute_error, # type: ignore
                              r2_score, confusion_matrix) 

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", action="store_true",
                    help="Use non-tuned baseline models from step 3")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
test = pd.read_csv("data/test.csv")
with open("data/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = config["target_cols"]
X_test   = test[FEATURES]

os.makedirs("results", exist_ok=True)

# ── AQI bucketing (India CPCB pm2_5 thresholds) ──────────────────────────────
def pm25_to_aqi_label(val):
    if val <= 30:  return "Good"
    elif val <= 60: return "Satisfactory"
    elif val <= 90: return "Moderate"
    elif val <= 120: return "Poor"
    elif val <= 250: return "Very Poor"
    else:           return "Severe"

AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ── Evaluate each model ───────────────────────────────────────────────────────
all_metrics = []

for target in TARGETS:
    horizon = target.replace("target_", "")
    prefix  = "model" if args.baseline else "tuned_model"
    path    = f"models/{prefix}_{horizon}.json"

    if not os.path.exists(path):
        print(f"⚠  {path} not found — skipping")
        continue

    model = xgb.XGBRegressor()
    model.load_model(path)

    y_true = test[target].values
    y_pred = model.predict(X_test)

    # ── Regression metrics ────────────────────────────────────────────────────
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # ── Within-category accuracy: % predictions in same AQI bucket ───────────
    true_labels = [pm25_to_aqi_label(v) for v in y_true]
    pred_labels = [pm25_to_aqi_label(v) for v in y_pred]
    cat_acc     = np.mean([t == p for t, p in zip(true_labels, pred_labels)])

    metrics = {
        "horizon"      : horizon,
        "RMSE"         : round(rmse, 4),
        "MAE"          : round(mae, 4),
        "R2"           : round(r2, 4),
        "Category_Acc" : round(cat_acc * 100, 2),
    }
    all_metrics.append(metrics)

    print(f"\n── {target} ──────────────────────────────")
    print(f"  RMSE            : {rmse:.4f} μg/m³")
    print(f"  MAE             : {mae:.4f} μg/m³")
    print(f"  R²              : {r2:.4f}")
    print(f"  AQI category acc: {cat_acc*100:.1f}%")

    # ── Save predictions ──────────────────────────────────────────────────────
    preds_df = pd.DataFrame({
        "datetime"    : test["datetime"],
        "city"        : test["city"],
        "actual_pm25" : y_true,
        "pred_pm25"   : y_pred.round(2),
        "actual_label": true_labels,
        "pred_label"  : pred_labels,
        "error"       : (y_pred - y_true).round(2),
    })
    preds_df.to_csv(f"results/predictions_{horizon}.csv", index=False)

    # ── Confusion matrix (1h only — most useful for portfolio) ───────────────
    if horizon == "1h":
        labels_present = sorted(set(true_labels + pred_labels),
                                key=lambda x: AQI_ORDER.index(x))
        cm = confusion_matrix(true_labels, pred_labels, labels=labels_present)
        cm_df = pd.DataFrame(cm, index=labels_present, columns=labels_present)
        cm_df.to_csv("results/aqi_confusion_1h.csv")
        print(f"\n  AQI Confusion Matrix (1h):\n{cm_df}")

# ── Summary table ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/test_metrics.csv", index=False)

print("\n\n" + "="*50)
print("  FINAL TEST SET RESULTS")
print("="*50)
print(metrics_df.to_string(index=False))
print("\n✓ Saved to results/")