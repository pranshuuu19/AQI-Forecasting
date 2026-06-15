import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import argparse
import optuna # type: ignore
import xgboost as xgb # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default=None,
                    help="Tune only one target e.g. target_1h")
parser.add_argument("--trials", type=int, default=50,
                    help="Number of Optuna trials (default 50)")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
val   = pd.read_csv("data/val.csv")

with open("data/feature_config.json") as f:
    config = json.load(f)

FEATURES = config["feature_cols"]
TARGETS  = [args.target] if args.target else config["target_cols"]

X_train = train[FEATURES]
X_val   = val[FEATURES]

os.makedirs("models", exist_ok=True)

# ── Objective function Optuna will optimise ───────────────────────────────────
def make_objective(y_train, y_val):
    def objective(trial):
        params = {
            "objective"            : "reg:squarederror",
            "n_estimators"         : 1000,
            "learning_rate"        : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth"            : trial.suggest_int("max_depth", 3, 10),
            "subsample"            : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight"     : trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha"            : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"           : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state"         : 42,
            "n_jobs"               : -1,
            "early_stopping_rounds": 30,
            "verbosity"            : 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    return objective

# ── Run tuning per target ─────────────────────────────────────────────────────
for target in TARGETS:
    horizon = target.replace("target_", "")
    print(f"\nTuning {target} — {args.trials} trials ...")

    y_train = train[target]
    y_val   = val[target]

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(y_train, y_val), n_trials=args.trials,
                   show_progress_bar=True)

    best = study.best_params
    best_rmse = study.best_value
    print(f"  Best RMSE : {best_rmse:.4f}")
    print(f"  Best params: {best}")

    # Save best params
    with open(f"models/best_params_{horizon}.json", "w") as f:
        json.dump(best, f, indent=2)

    # Retrain final model with best params on train+val combined
    print(f"  Retraining on train+val combined with best params ...")
    train_val  = pd.concat([train, val], ignore_index=True)
    X_trainval = train_val[FEATURES]
    y_trainval = train_val[target]

    final_params = {
        "objective"       : "reg:squarederror",
        "n_estimators"    : 500,  # best n_estimators from early stopping
        "random_state"    : 42,
        "n_jobs"          : -1,
        "verbosity"       : 0,
        **best
    }
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_trainval, y_trainval)
    final_model.save_model(f"models/tuned_model_{horizon}.json")
    print(f"  ✓ Saved models/tuned_model_{horizon}.json")

print("\n✓ Tuning complete.")