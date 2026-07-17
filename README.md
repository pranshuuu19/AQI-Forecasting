# 🌫️ AQI Forecasting — India

An end-to-end machine learning system that forecasts PM2.5 air quality 1, 24, and 48 hours ahead for 5 major Indian cities, served through a live Streamlit dashboard with confidence intervals, SHAP-based explainability, and a persistent self-evaluation loop.

**Live demo:** [aqi-forecasting-2b3fjukvhb3o7jzhwz2ckc.streamlit.app](https://aqi-forecasting-2b3fjukvhb3o7jzhwz2ckc.streamlit.app)

---

## What it does

- Fetches live weather + air quality data on demand (no stored dataset needed at inference time)
- Engineers 40–61 features (lags, rolling stats, interactions, festival/seasonal flags)
- Predicts PM2.5 at three horizons (1h / 24h / 48h) using separate tuned XGBoost models per horizon
- Wraps each prediction with an 80% confidence interval from quantile regression models
- Explains every prediction with SHAP, translated into a plain-English narrative
- Benchmarked against Linear Regression and Random Forest baselines on an identical test set — XGBoost is the best or tied-best model at every horizon, with its advantage growing at longer horizons where relationships are more nonlinear (see `results/model_comparison.csv`)
- Self-logs every prediction to a persistent PostgreSQL database (Supabase) and automatically resolves it against the real observed outcome once the target time passes — a lightweight, always-on model-monitoring loop that survives app restarts/redeploys
- Exposes the same predictions through a small FastAPI backend (`GET /predict/{city}`), independent of the Streamlit UI

## Cities covered

Delhi · Mumbai · Bengaluru · Chennai · Kolkata

## Model performance (held-out test set)

| Horizon | RMSE | MAE | R² | AQI Category Accuracy |
|---|---|---|---|---|
| 1h | 5.18 | 2.35 | 0.9744 | 93.5% |
| 24h | 17.42 | 10.26 | 0.7119 | 72.4% |
| 48h | 19.70 | 12.02 | 0.6349 | 68.2% |

Quantile interval coverage: ~87% (1h), ~88% (24h/48h) for the target 80% band.

## Tech stack

Python · XGBoost · SHAP · Optuna · Streamlit · FastAPI · PostgreSQL (Supabase) · Plotly · scikit-learn · Open-Meteo API (free, no API key)

## Architecture

```
Live weather + AQI data (Open-Meteo)
        │
        ▼
Feature engineering (lags, rolling stats, interactions, festival flags)
        │
        ├──► 3 point-estimate XGBoost models (1h / 24h / 48h)
        │
        └──► 6 quantile regression models (lower/upper × 3 horizons)
        │
        ▼
Streamlit dashboard: forecast cards, 48h trend chart with confidence band,
SHAP waterfall explanation
        │
        ▼
PostgreSQL (Supabase): every prediction logged persistently
        │
        ▼
Lazy evaluation: on every dashboard visit, live data for all 5 cities is
fetched and any prediction whose target time has passed is automatically
resolved against the real observed value — no external cron/scheduler
needed, the visit itself is the trigger
```

Each horizon is modeled independently rather than as a single multi-output model, since the driving features shift meaningfully with horizon — lag features dominate at 1h, while city identity and month dominate at 48h as the model falls back toward a climatological baseline.

## Why PostgreSQL instead of a local file

The live prediction log originally lived in a local CSV. On Streamlit Community Cloud, the app's filesystem is ephemeral — every restart (inactivity sleep, or any redeploy) wipes local files, since they were never part of the git repo to begin with. Moving the log to a hosted Postgres database (Supabase, free tier) means predictions and their resolved outcomes persist independently of the app container's lifecycle.

## Why lazy evaluation instead of a scheduled job

An earlier version of this project used a GitHub Actions cron job to periodically refresh predictions and resolve actuals. It was dropped in favor of a simpler design: every dashboard visit fetches live data for all 5 cities and resolves any prediction whose target time has passed, before rendering the page. A visit *is* the trigger — no external scheduler to configure, monitor, or trust. `fetch_live_data()` is cached (10 min TTL) and the resolution check is throttled to once per 10 minutes per session, so this stays cheap even under frequent traffic.

## Repo structure

```
AQI_project/
├── dashboard.py                 # Main Streamlit app (5 pages)
├── main.py                      # FastAPI backend (GET /health, GET /predict/{city})
├── predictor.py                 # Shared prediction logic (used by dashboard + API)
├── prediction_log.py            # Postgres-backed prediction logging + actuals resolution
├── src/                         # Data fetching + feature engineering
├── AQI_pipeline/                # Training pipeline (split, tune, train, evaluate, SHAP, model comparison)
├── models/                      # Trained XGBoost models (point + quantile)
├── data/                        # Feature config, split summary
└── results/                     # Test metrics, predictions, SHAP plots, model comparison
```

## Running locally

**Dashboard:**
```bash
git clone https://github.com/pranshuuu19/AQI-Forecasting.git
cd AQI-Forecasting
pip install -r requirements.txt
cp .env.example .env   # then add your real Supabase DATABASE_URL
streamlit run dashboard.py
```

**API (optional, runs independently of the dashboard):**
```bash
uvicorn main:app --reload
```
Then visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Dashboard pages

1. **Live Forecast** — pick a city, get real-time PM2.5 forecasts with confidence ranges, a 48h trend chart, pollutant breakdown, and an on-demand SHAP explanation
2. **Model Performance** — test-set metrics, actual-vs-predicted scatter, AQI confusion matrix
3. **Historical Analysis** — time series, hour × day-of-week heatmap, seasonal and city comparisons
4. **Real-Time Eval** — persistently logged live predictions, automatically resolved against actual outcomes, with live RMSE/MAE tracked against test-set benchmarks
5. **About** — pipeline summary and key findings

## Known limitations

- Delhi shows higher live prediction error than the other cities. This reflects genuine PM2.5 volatility (dust storms, sudden shifts not captured by lag features) rather than a modeling flaw — the quantile intervals widen accordingly for Delhi, which is the model correctly signaling its own uncertainty.
- Actuals resolution relies on a dashboard visit happening within ~6 days of a prediction's target time (the window `fetch_live_data` pulls from Open-Meteo). A prediction that goes unresolved longer than that may stay unresolved. Fine for normal/periodic use; not a hard scheduling guarantee.
- No automated retraining pipeline yet — models are retrained manually as new data accumulates.

## Author

Pranshu — BTech, MIT Manipal
