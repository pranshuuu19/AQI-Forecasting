import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import xgboost as xgb # type: ignore
import shap # type: ignore
import matplotlib # type: ignore
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt # type: ignore

# ── Load ──────────────────────────────────────────────────────────────────────
test = pd.read_csv("data/test.csv")
with open("data/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = config["target_cols"]
X_test   = test[FEATURES]

os.makedirs("results", exist_ok=True)

# ── SHAP per model ────────────────────────────────────────────────────────────
for target in TARGETS:
    horizon = target.replace("target_", "")
    path    = f"models/tuned_model_{horizon}.json"
    if not os.path.exists(path):
        path = f"models/model_{horizon}.json"   # fallback to baseline
    if not os.path.exists(path):
        print(f"⚠  No model found for {target}, skipping")
        continue

    print(f"\nComputing SHAP for {target} ...")

    model   = xgb.XGBRegressor()
    model.load_model(path)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)   # returns Explanation object

    # ── 1. Summary bar plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, ax=ax, show=False)
    ax.set_title(f"Top 20 Features by Mean |SHAP| — {target}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/shap_summary_{horizon}.png", dpi=150)
    plt.close()

    # ── 2. Beeswarm plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Beeswarm — {target}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/shap_beeswarm_{horizon}.png", dpi=150)
    plt.close()

    # ── 3. Save raw SHAP values (top 20 features) ────────────────────────────
    shap_df = pd.DataFrame(shap_values.values, columns=FEATURES)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top20    = mean_abs.head(20).index.tolist()
    shap_df[top20].to_csv(f"results/shap_values_{horizon}.csv", index=False)

    print(f"  ✓ Plots saved: results/shap_*_{horizon}.png")
    print(f"  Top 5 features:")
    for feat, val in mean_abs.head(5).items():
        print(f"    {feat:<35} mean|SHAP| = {val:.4f}")

print("\n✓ SHAP analysis complete.")
print("\nHOW TO READ THE PLOTS:")
print("  Bar plot   : longer bar = more important feature overall")
print("  Beeswarm   : red = high feature value, blue = low feature value")
print("               right = pushes pm2_5 prediction UP, left = DOWN")