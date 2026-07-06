# 🌫️ AQI Forecasting — India

An end-to-end machine learning system that forecasts PM2.5 air quality 1, 24, and 48 hours ahead for 5 major Indian cities, served through a live Streamlit dashboard with confidence intervals and SHAP-based explainability.

**Live demo:** _add your Streamlit Cloud link here once deployed_

---

## What it does

- Fetches live weather + air quality data on demand (no stored dataset needed at inference time)
- Engineers 40–60 features (lags, rolling stats, interactions, festival/seasonal flags)
- Predicts PM2.5 at three horizons (1h / 24h / 48h) using separate tuned XGBoost models per horizon
- Wraps each prediction with an 80% confidence interval from quantile regression models
- Explains every prediction with SHAP, translated into a plain-English narrative
- Self-logs every prediction it makes and automatically checks itself against the real outcome once the target time passes — a lightweight live model-monitoring loop

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

Python · XGBoost · SHAP · Optuna · Streamlit · Plotly · Open-Meteo API (free, no API key)

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
SHAP waterfall explanation, self-logging live evaluation
```

Each horizon is modeled independently rather than as a single multi-output model, since the driving features shift meaningfully with horizon — lag features dominate at 1h, while city identity and month dominate at 48h as the model falls back toward a climatological baseline.

## Repo structure

```
AQI_project/
├── dashboard.py                 # Main Streamlit app (5 pages)
├── src/                         # Data fetching + feature engineering
├── AQI_pipeline/                # Training pipeline (split, tune, train, evaluate, SHAP)
├── models/                      # Trained XGBoost models (point + quantile)
├── data/                        # Feature config, split summary
└── results/                     # Test metrics, predictions, SHAP plots
```

## Running locally

```bash
git clone https://github.com/pranshuuu19/AQI-Forecasting.git
cd AQI-Forecasting
pip install -r requirements.txt
streamlit run dashboard.py
```

## Dashboard pages

1. **Live Forecast** — pick a city, get real-time PM2.5 forecasts with confidence ranges, a 48h trend chart, pollutant breakdown, and an on-demand SHAP explanation
2. **Model Performance** — test-set metrics, actual-vs-predicted scatter, AQI confusion matrix
3. **Historical Analysis** — time series, hour × day-of-week heatmap, seasonal and city comparisons
4. **Real-Time Eval** — self-logged live predictions checked against actual outcomes once resolved, with live RMSE/MAE tracked against test-set benchmarks
5. **About** — pipeline summary and key findings

## Known limitations

- Delhi shows higher live prediction error than the other cities. This reflects genuine PM2.5 volatility (dust storms, sudden shifts not captured by lag features) rather than a modeling flaw — the quantile intervals widen accordingly for Delhi, which is the model correctly signaling its own uncertainty.
- Live evaluation only updates when the dashboard is opened/refreshed; there's no background scheduler running independently.
- No automated retraining pipeline yet — models are retrained manually as new data accumulates.

## Author

Pranshu — BTech, MIT Manipal
