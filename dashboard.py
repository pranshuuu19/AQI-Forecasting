"""
AQI Forecasting Dashboard — On-Demand Architecture
====================================================
No background processes. Fetches live data, engineers features, predicts,
and self-logs evaluation history entirely on-demand when the dashboard
is opened or interacted with.

HOW TO RUN:
  pip install streamlit plotly shap matplotlib pandas numpy xgboost requests
  streamlit run dashboard.py

FOLDER STRUCTURE:
  models/tuned_model_1h.json, tuned_model_24h.json, tuned_model_48h.json
  data/feature_config.json
  data/live_predictions_log.csv   (auto-created, self-logging)
  results/test_metrics.csv
  results/aqi_confusion_1h.csv
  results/predictions_1h.csv, predictions_24h.csv, predictions_48h.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AQI Forecast India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E8EAF0;
}
.stApp { background-color: #0A0E1A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }

.card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-sm {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 16px;
}
.metric-card {
    background: #111827;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px 0;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-sub { font-size: 12px; color: #9CA3AF; margin-top: 2px; }

.forecast-card {
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.forecast-horizon {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; opacity: 0.7; margin-bottom: 8px;
}
.forecast-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px; font-weight: 700; line-height: 1; margin-bottom: 4px;
}
.forecast-unit { font-size: 12px; opacity: 0.6; margin-bottom: 12px; }
.forecast-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; background: rgba(255,255,255,0.15);
}
.aqi-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 24px; font-size: 13px; font-weight: 600;
}
.section-header {
    font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em; color: #4B5563;
    margin-bottom: 12px; margin-top: 4px;
}
.divider { height: 1px; background: #1E2A3A; margin: 20px 0; }
.live-dot {
    display: inline-block; width: 8px; height: 8px; background: #10B981;
    border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.alert-banner {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 12px;
}
.trend-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 10px; border-radius: 16px; font-size: 12px; font-weight: 600;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
AQI_LEVELS = [
    ("Good",         0,   30,  "#10B981", "#064E3B", "😊"),
    ("Satisfactory", 31,  60,  "#84CC16", "#1A2E05", "🙂"),
    ("Moderate",     61,  90,  "#F59E0B", "#451A03", "😐"),
    ("Poor",         91,  120, "#EF4444", "#450A0A", "😷"),
    ("Very Poor",    121, 250, "#8B5CF6", "#2E1065", "🤢"),
    ("Severe",       251, 999, "#EC4899", "#500724", "☠️"),
]
AQI_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

CITIES = {
    "Delhi"    : (28.6139, 77.2090),
    "Mumbai"   : (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai"  : (13.0827, 80.2707),
    "Kolkata"  : (22.5726, 88.3639),
}

LIVE_LOG_PATH = "data/live_predictions_log.csv"

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9CA3AF", family="Inter"),
    xaxis=dict(gridcolor="#1E2A3A", linecolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    yaxis=dict(gridcolor="#1E2A3A", linecolor="#1E2A3A", zerolinecolor="#1E2A3A"),
)

FEATURE_LABELS = {
    "pm2_5_lag1": "recent PM2.5 levels",
    "pm2_5_lag24": "PM2.5 levels 24h ago",
    "pm2_5_lag48": "PM2.5 levels 48h ago",
    "pm2_5_rolling6": "short-term pollution trend",
    "pm2_5_rolling24": "daily pollution average",
    "pm2_5_change_1h": "recent rate of change",
    "pm2_5_change_24h": "24-hour change trend",
    "wind_speed": "wind speed",
    "wind_pm25_ratio": "wind dispersal effect",
    "humidity": "humidity",
    "humidity_pm25": "humidity trapping pollutants",
    "temperature": "temperature",
    "temp_wind_interaction": "temperature-wind interaction",
    "hour": "time of day",
    "hour_bucket_evening_rush": "evening rush hour traffic",
    "hour_bucket_morning_rush": "morning rush hour traffic",
    "month": "seasonal patterns",
    "is_stubble_burning": "stubble burning season",
    "is_weekend": "weekend traffic patterns",
    "city_encoded": "city-specific pollution baseline",
    "no2": "nitrogen dioxide levels",
    "co": "carbon monoxide levels",
    "pm10": "PM10 particulate levels",
    "rainfall": "rainfall",
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_aqi_level(val):
    for label, lo, hi, color, bg, emoji in AQI_LEVELS:
        if lo <= val <= hi:
            return label, color, bg, emoji
    return "Severe", "#EC4899", "#500724", "☠️"

def aqi_severity_rank(label):
    try:
        return AQI_ORDER.index(label)
    except ValueError:
        return 0

def aqi_gauge(value):
    label, color, bg, emoji = get_aqi_level(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": " μg/m³", "font": {"size": 28, "color": color, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#374151",
                     "tickfont": {"color": "#6B7280", "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#111827",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   30],  "color": "rgba(6,78,59,0.2)"},
                {"range": [30,  60],  "color": "rgba(26,46,5,0.2)"},
                {"range": [60,  90],  "color": "rgba(69,26,3,0.2)"},
                {"range": [90,  120], "color": "rgba(69,10,10,0.2)"},
                {"range": [120, 250], "color": "rgba(46,16,101,0.2)"},
                {"range": [250, 300], "color": "rgba(80,7,36,0.2)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                         "thickness": 0.8, "value": value},
        },
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9CA3AF", family="Inter"),
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig

@st.cache_resource
def load_models():
    models = {}
    for h in ["1h", "24h", "48h"]:
        for prefix in ["tuned_model", "model"]:
            path = f"models/{prefix}_{h}.json"
            if os.path.exists(path):
                m = xgb.XGBRegressor()
                m.load_model(path)
                models[h] = m
                break
    return models

@st.cache_data(ttl=300)
def load_static_data():
    with open("data/feature_config.json") as f:
        config = json.load(f)
    metrics   = pd.read_csv("results/test_metrics.csv")
    confusion = pd.read_csv("results/aqi_confusion_1h.csv", index_col=0)
    p1h  = pd.read_csv("results/predictions_1h.csv",  parse_dates=["datetime"])
    p24h = pd.read_csv("results/predictions_24h.csv", parse_dates=["datetime"])
    p48h = pd.read_csv("results/predictions_48h.csv", parse_dates=["datetime"])
    return config, metrics, confusion, p1h, p24h, p48h

def fetch_live_data(city):
    """Fetch current conditions from Open-Meteo forecast API (supports past + current)."""
    lat, lon = CITIES[city]
    now_ist    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    end_date   = (now_ist + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (now_ist - timedelta(days=6)).strftime("%Y-%m-%d")

    try:
        aqi_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
            f"&timezone=Asia/Kolkata"
        )
        aqi_r = requests.get(aqi_url, timeout=15).json()
        if "hourly" not in aqi_r:
            raise ValueError(f"AQI API error: {aqi_r.get('reason', aqi_r)}")
        aqi_df = pd.DataFrame({
            "datetime": pd.to_datetime(aqi_r["hourly"]["time"]),
            "pm2_5": aqi_r["hourly"]["pm2_5"],
            "pm10" : aqi_r["hourly"]["pm10"],
            "co"   : aqi_r["hourly"]["carbon_monoxide"],
            "no2"  : aqi_r["hourly"]["nitrogen_dioxide"],
            "o3"   : aqi_r["hourly"]["ozone"],
            "so2"  : aqi_r["hourly"]["sulphur_dioxide"],
        })

        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
            f"&timezone=Asia/Kolkata"
        )
        wx_r = requests.get(wx_url, timeout=15).json()
        if "hourly" not in wx_r:
            raise ValueError(f"Weather API error: {wx_r.get('reason', wx_r)}")
        wx_df = pd.DataFrame({
            "datetime"   : pd.to_datetime(wx_r["hourly"]["time"]),
            "temperature": wx_r["hourly"]["temperature_2m"],
            "humidity"   : wx_r["hourly"]["relative_humidity_2m"],
            "wind_speed" : wx_r["hourly"]["wind_speed_10m"],
            "rainfall"   : wx_r["hourly"]["precipitation"],
        })

        df = pd.merge(aqi_df, wx_df, on="datetime", how="inner")
        df = df[df["datetime"] <= now_ist].dropna().sort_values("datetime").reset_index(drop=True)
        df["city"] = city
        return df, None

    except Exception as e:
        return None, str(e)

def engineer_features(df):
    df = df.copy()
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    for col in ["pm2_5","pm10","no2","co","o3","so2"]:
        df[f"{col}_lag1"]  = df[col].shift(1)
        df[f"{col}_lag24"] = df[col].shift(24)
        df[f"{col}_lag48"] = df[col].shift(48)
    df["pm2_5_rolling6"]      = df["pm2_5"].rolling(6,  min_periods=1).mean()
    df["pm2_5_rolling24"]     = df["pm2_5"].rolling(24, min_periods=1).mean()
    df["pm2_5_change_1h"]     = df["pm2_5"].diff(1)
    df["pm2_5_change_24h"]    = df["pm2_5"].diff(24)
    df["pm2_5_rolling6_std"]  = df["pm2_5"].rolling(6,  min_periods=1).std().fillna(0)
    df["pm2_5_rolling24_max"] = df["pm2_5"].rolling(24, min_periods=1).max()
    df["pm2_5_rolling24_min"] = df["pm2_5"].rolling(24, min_periods=1).min()
    df["humidity_pm25"]         = df["humidity"] * df["pm2_5"]
    df["wind_pm25_ratio"]       = df["pm2_5"] / (df["wind_speed"] + 1)
    df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"]
    city_mean = df["pm2_5"].mean()
    city_std  = df["pm2_5"].std()
    df["is_pollution_spike"]  = (df["pm2_5"] > city_mean + 2 * city_std).astype(int)
    df["date_str"] = df["datetime"].dt.date.astype(str)
    df["is_diwali"]          = df["date_str"].isin(["2025-10-20","2025-10-21"]).astype(int)
    df["is_diwali_window"]   = df["date_str"].isin(["2025-10-17","2025-10-18","2025-10-19",
                                "2025-10-20","2025-10-21","2025-10-22","2025-10-23","2025-10-24"]).astype(int)
    df["is_stubble_burning"] = df["month"].isin([10,11]).astype(int)
    df = df.drop(columns=["date_str"])
    city_order = {"Bengaluru":0,"Chennai":1,"Delhi":2,"Kolkata":3,"Mumbai":4}
    df["city_encoded"] = city_order.get(df["city"].iloc[0], 0)
    h = df["hour"].iloc[-1]
    df["hour_bucket_morning_rush"] = int(6 <= h <= 9)
    df["hour_bucket_daytime"]      = int(10 <= h <= 16)
    df["hour_bucket_evening_rush"] = int(17 <= h <= 20)
    df["hour_bucket_night"]        = int(h <= 5 or h >= 21)
    return df

def make_shap_narrative(shap_values, features, feature_names, top_n=3):
    """Convert SHAP values into a plain-English narrative."""
    vals = shap_values.values[0]
    pairs = list(zip(feature_names, vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    rising_factors = [FEATURE_LABELS.get(f, f) for f, v in pairs[:top_n] if v > 0]
    falling_factors = [FEATURE_LABELS.get(f, f) for f, v in pairs[:top_n] if v < 0]

    sentences = []
    if rising_factors:
        sentences.append(f"PM2.5 is being pushed **higher** mainly due to {', '.join(rising_factors)}.")
    if falling_factors:
        sentences.append(f"PM2.5 is being pulled **lower** by {', '.join(falling_factors)}.")
    if not sentences:
        sentences.append("No single factor dominates — prediction reflects a balance of current conditions.")
    return " ".join(sentences)

def build_48h_forecast(featured_df, models, features, current_pm25, pred_1h, pred_24h, pred_48h):
    """
    Build a smooth 48h trend by anchoring on the actual 1h/24h/48h model outputs
    (each model was specifically trained for that horizon) and interpolating
    between them, rather than recursively chaining the 1h model 48 times —
    which compounds error and collapses to a flat rolling-mean reversion.
    """
    last_dt = featured_df.iloc[-1]["datetime"]

    anchors_t = [0, 1, 24, 48]
    anchors_v = [current_pm25, pred_1h or current_pm25,
                 pred_24h or current_pm25, pred_48h or current_pm25]

    # Add a touch of natural hourly variation around the interpolated trend
    # using the diurnal pattern learned from the last 24h of real data,
    # so the line doesn't look artificially smooth/robotic.
    recent = featured_df.tail(24).copy()
    if len(recent) >= 24 and recent["pm2_5"].std() > 0:
        diurnal = recent.groupby(recent["datetime"].dt.hour)["pm2_5"].mean()
        diurnal_norm = diurnal - diurnal.mean()
    else:
        diurnal_norm = pd.Series(0, index=range(24))

    hours = np.arange(0, 49)
    base_curve = np.interp(hours, anchors_t, anchors_v)

    future_rows = []
    for h_ahead in range(1, 49):
        future_time = last_dt + timedelta(hours=h_ahead)
        diurnal_adj = diurnal_norm.get(future_time.hour, 0) * 0.3  # damped influence
        val = max(0, base_curve[h_ahead] + diurnal_adj)
        future_rows.append({"datetime": future_time, "pm2_5": val})

    return pd.DataFrame(future_rows)

# ── Self-logging evaluation system ───────────────────────────────────────────
def log_prediction(city, current_dt, current_pm25, pred_1h, pred_24h, pred_48h):
    """Log a new prediction. Called every time dashboard fetches live data for a city."""
    os.makedirs("data", exist_ok=True)
    new_row = pd.DataFrame([{
        "prediction_time" : (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        "city"             : city,
        "current_datetime" : current_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "current_pm25"     : round(current_pm25, 2),
        "target_time_1h"   : (current_dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "target_time_24h"  : (current_dt + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
        "target_time_48h"  : (current_dt + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S"),
        "pred_1h"  : round(pred_1h, 2)  if pred_1h  else None,
        "pred_24h" : round(pred_24h, 2) if pred_24h else None,
        "pred_48h" : round(pred_48h, 2) if pred_48h else None,
        "actual_1h" : None, "actual_24h": None, "actual_48h": None,
        "error_1h"  : None, "error_24h" : None, "error_48h" : None,
    }])

    if os.path.exists(LIVE_LOG_PATH):
        existing = pd.read_csv(LIVE_LOG_PATH)
        # Avoid duplicate logging for same city + same current_datetime
        dup_mask = (existing["city"] == city) & (existing["current_datetime"] == new_row["current_datetime"].iloc[0])
        if dup_mask.any():
            return existing
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(LIVE_LOG_PATH, index=False)
    return combined

def evaluate_past_predictions(live_df_by_city):
    """Check logged predictions, fill in actuals for any whose target time has passed."""
    if not os.path.exists(LIVE_LOG_PATH):
        return

    log_df = pd.read_csv(LIVE_LOG_PATH)
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    changed = False

    for idx, row in log_df.iterrows():
        city = row["city"]
        if city not in live_df_by_city:
            continue
        city_data = live_df_by_city[city].copy()
        city_data["datetime"] = pd.to_datetime(city_data["datetime"])

        for horizon in ["1h", "24h", "48h"]:
            if pd.notna(row.get(f"actual_{horizon}")) or pd.isna(row.get(f"pred_{horizon}")):
                continue
            target_time = pd.to_datetime(row[f"target_time_{horizon}"])
            if target_time > now_ist:
                continue

            city_data["time_diff"] = (city_data["datetime"] - target_time).abs()
            closest = city_data.nsmallest(1, "time_diff")
            if len(closest) > 0 and closest.iloc[0]["time_diff"] < timedelta(minutes=61):
                actual = round(float(closest.iloc[0]["pm2_5"]), 2)
                error  = round(float(row[f"pred_{horizon}"]) - actual, 2)
                log_df.at[idx, f"actual_{horizon}"] = actual
                log_df.at[idx, f"error_{horizon}"]  = error
                changed = True

    if changed:
        log_df.to_csv(LIVE_LOG_PATH, index=False)

@st.cache_data(ttl=60)
def load_live_log():
    if os.path.exists(LIVE_LOG_PATH):
        df = pd.read_csv(LIVE_LOG_PATH)
        df["prediction_time"]  = pd.to_datetime(df["prediction_time"])
        df["current_datetime"] = pd.to_datetime(df["current_datetime"])
        return df
    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════
models = load_models()
config, metrics, confusion, p1h, p24h, p48h = load_static_data()
FEATURES = config["feature_cols"]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER + NAV
# ══════════════════════════════════════════════════════════════════════════════
now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

col_logo, col_nav, col_time = st.columns([1.2, 4, 1.2])
with col_logo:
    st.markdown("""
    <div style='padding-top:8px'>
        <span style='font-family:Space Grotesk;font-size:20px;font-weight:700;
                     color:#E8EAF0;letter-spacing:-0.03em;'>🌫️ AQI<span
                     style='color:#1D4ED8'>India</span></span>
    </div>""", unsafe_allow_html=True)

with col_time:
    st.markdown(f"""
    <div style='text-align:right;padding-top:10px;'>
        <span class='live-dot'></span>
        <span style='font-size:12px;color:#6B7280;'>
            {now_ist.strftime('%d %b, %I:%M %p')} IST
        </span>
    </div>""", unsafe_allow_html=True)

pages = ["🔮 Live Forecast", "📊 Model Performance", "📈 Historical Analysis", "⚡ Real-Time Eval", "ℹ️ About"]
if "page" not in st.session_state:
    st.session_state.page = pages[0]

with col_nav:
    cols = st.columns(len(pages))
    for i, (col, page) in enumerate(zip(cols, pages)):
        with col:
            if st.button(page, key=f"nav_{i}", use_container_width=True,
                        type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Live Forecast":

    if "city" not in st.session_state:
        st.session_state.city = "Delhi"

    st.markdown('<div class="section-header">Select City</div>', unsafe_allow_html=True)
    city_cols = st.columns(len(CITIES))
    for col, city_name in zip(city_cols, CITIES.keys()):
        with col:
            active = "primary" if st.session_state.city == city_name else "secondary"
            if st.button(city_name, key=f"city_{city_name}", use_container_width=True, type=active):
                st.session_state.city = city_name
                st.rerun()

    selected_city = st.session_state.city

    with st.spinner(f"Fetching live data for {selected_city}..."):
        live_df, error = fetch_live_data(selected_city)

    if error or live_df is None or len(live_df) == 0:
        st.error(f"Could not fetch live data: {error}")
        st.stop()

    featured = engineer_features(live_df)
    latest   = featured.iloc[-1]
    for f in FEATURES:
        if f not in featured.columns:
            featured[f] = 0
    x_input = pd.DataFrame([featured.iloc[-1][FEATURES]]).fillna(0)

    pred_1h  = float(models["1h"].predict(x_input)[0])  if "1h"  in models else None
    pred_24h = float(models["24h"].predict(x_input)[0]) if "24h" in models else None
    pred_48h = float(models["48h"].predict(x_input)[0]) if "48h" in models else None

    current_pm25 = float(latest["pm2_5"])
    cur_label, cur_color, cur_bg, cur_emoji = get_aqi_level(current_pm25)

    # ── Self-log this prediction + evaluate past ones ────────────────────────
    log_prediction(selected_city, latest["datetime"], current_pm25, pred_1h, pred_24h, pred_48h)
    evaluate_past_predictions({selected_city: live_df})

    # ── Pollution spike alert ─────────────────────────────────────────────────
    worst_future_label = cur_label
    worst_future_val = current_pm25
    for pred in [pred_1h, pred_24h]:
        if pred:
            flabel, _, _, _ = get_aqi_level(pred)
            if aqi_severity_rank(flabel) > aqi_severity_rank(worst_future_label):
                worst_future_label = flabel
                worst_future_val = pred

    if aqi_severity_rank(worst_future_label) > aqi_severity_rank(cur_label):
        _, alert_color, alert_bg, alert_emoji = get_aqi_level(worst_future_val)
        st.markdown(f"""
        <div class="alert-banner" style="background:{alert_bg};border-color:{alert_color};">
            <span style="font-size:24px;">⚠️</span>
            <div>
                <div style="font-weight:700;color:{alert_color};font-size:14px;">
                    Pollution Spike Expected — {selected_city}
                </div>
                <div style="font-size:13px;color:#9CA3AF;">
                    AQI predicted to reach <b style="color:{alert_color};">{worst_future_label}</b>
                    ({worst_future_val:.0f} μg/m³) within the next 24 hours
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Trend direction ───────────────────────────────────────────────────────
    lag1_val = featured.iloc[-1].get("pm2_5_lag1", current_pm25)
    if pd.notna(lag1_val):
        diff = current_pm25 - lag1_val
        if diff > 2:
            trend_icon, trend_text, trend_color = "↑", "Rising", "#EF4444"
        elif diff < -2:
            trend_icon, trend_text, trend_color = "↓", "Falling", "#10B981"
        else:
            trend_icon, trend_text, trend_color = "→", "Stable", "#9CA3AF"
    else:
        trend_icon, trend_text, trend_color = "→", "Stable", "#9CA3AF"

    # ── Row 1: Gauge + Weather + Forecast cards ──────────────────────────────
    col_gauge, col_weather, col_forecasts = st.columns([1.2, 1, 2.8])

    with col_gauge:
        st.markdown('<div class="card" style="text-align:center;padding:16px 12px;">'
                    '<div class="section-header">Current PM2.5</div>', unsafe_allow_html=True)
        st.plotly_chart(aqi_gauge(current_pm25), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"""
            <div style="margin-top:-10px;text-align:center;">
                <span class="aqi-pill" style="background:{cur_bg};color:{cur_color};border:1px solid {cur_color}40;">
                    {cur_emoji} {cur_label}
                </span>
                <span class="trend-pill" style="background:{trend_color}20;color:{trend_color};margin-left:6px;">
                    {trend_icon} {trend_text}
                </span>
            </div>
            <div style="font-size:11px;color:#4B5563;text-align:center;margin-top:8px;">
                as of {latest['datetime'].strftime('%I:%M %p')} IST
            </div>
        </div>""", unsafe_allow_html=True)

    with col_weather:
        temp = latest.get("temperature", 0)
        hum  = latest.get("humidity", 0)
        ws   = latest.get("wind_speed", 0)
        rain = latest.get("rainfall", 0)
        st.markdown(f"""
        <div class="card" style="height:100%;">
            <div class="section-header">Weather Now</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px;">
                <div class="card-sm" style="text-align:center;">
                    <div style="font-size:22px;">🌡️</div>
                    <div style="font-size:20px;font-weight:700;color:#F59E0B;font-family:'Space Grotesk',sans-serif;">{temp:.1f}°</div>
                    <div style="font-size:10px;color:#6B7280;">TEMP</div>
                </div>
                <div class="card-sm" style="text-align:center;">
                    <div style="font-size:22px;">💧</div>
                    <div style="font-size:20px;font-weight:700;color:#3B82F6;font-family:'Space Grotesk',sans-serif;">{hum:.0f}%</div>
                    <div style="font-size:10px;color:#6B7280;">HUMIDITY</div>
                </div>
                <div class="card-sm" style="text-align:center;">
                    <div style="font-size:22px;">💨</div>
                    <div style="font-size:20px;font-weight:700;color:#10B981;font-family:'Space Grotesk',sans-serif;">{ws:.1f}</div>
                    <div style="font-size:10px;color:#6B7280;">WIND m/s</div>
                </div>
                <div class="card-sm" style="text-align:center;">
                    <div style="font-size:22px;">🌧️</div>
                    <div style="font-size:20px;font-weight:700;color:#60A5FA;font-family:'Space Grotesk',sans-serif;">{rain:.1f}</div>
                    <div style="font-size:10px;color:#6B7280;">RAIN mm</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_forecasts:
        st.markdown('<div class="section-header">PM2.5 Forecast</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        for col, pred, horizon, label in zip(
            [fc1, fc2, fc3], [pred_1h, pred_24h, pred_48h],
            ["1h", "24h", "48h"], ["1 Hour", "24 Hours", "48 Hours"]
        ):
            with col:
                if pred:
                    flabel, fcolor, fbg, femoji = get_aqi_level(pred)
                    st.markdown(f"""
                    <div class="forecast-card" style="background:{fbg};border-color:{fcolor}30;">
                        <div class="forecast-horizon" style="color:{fcolor};">⏱ {label}</div>
                        <div class="forecast-value" style="color:{fcolor};">{pred:.1f}</div>
                        <div class="forecast-unit" style="color:{fcolor};">μg/m³</div>
                        <div class="forecast-badge" style="color:{fcolor};">{femoji} {flabel}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Row 2: 48h forecast graph + Pollutants ───────────────────────────────
    col_chart, col_poll = st.columns([2.5, 1])

    with col_chart:
        st.markdown('<div class="section-header">48-Hour PM2.5 Forecast Trend</div>', unsafe_allow_html=True)

        future_df = build_48h_forecast(featured, models, FEATURES, current_pm25, pred_1h, pred_24h, pred_48h)
        hist_df = live_df.tail(24)[["datetime","pm2_5"]].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["datetime"], y=hist_df["pm2_5"],
            name="Historical", mode="lines", line=dict(color="#3B82F6", width=2),
        ))
        connect_x = [hist_df["datetime"].iloc[-1]] + list(future_df["datetime"])
        connect_y = [hist_df["pm2_5"].iloc[-1]]    + list(future_df["pm2_5"])
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y, name="Forecast", mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.05)",
        ))
        for val, lbl, clr in [(30,"Good","#10B981"),(60,"Satisfactory","#84CC16"),
                               (90,"Moderate","#F59E0B"),(120,"Poor","#EF4444")]:
            fig.add_hline(y=val, line_dash="dash", line_color=clr, line_width=1, opacity=0.4,
                         annotation_text=lbl, annotation_position="right",
                         annotation_font=dict(color=clr, size=10))
        fig.add_vline(x=str(latest["datetime"]), line_dash="solid", line_color="#6B7280",
                     line_width=1, opacity=0.5)
        all_y_vals = list(hist_df["pm2_5"]) + list(future_df["pm2_5"])
        y_min = max(0, min(all_y_vals) - 15)
        y_max = max(all_y_vals) + 15

        fig.update_layout(
            height=280, showlegend=True,
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
            xaxis_title="", yaxis_title="PM2.5 (μg/m³)",
            yaxis_range=[y_min, y_max], **PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_poll:
        st.markdown('<div class="section-header">Pollutant Levels</div>', unsafe_allow_html=True)
        pollutants = {
            "PM10": (latest.get("pm10",0), 100, "#3B82F6"),
            "CO"  : (latest.get("co",0)/10, 100, "#F59E0B"),
            "NO₂" : (latest.get("no2",0), 80,  "#EF4444"),
            "O₃"  : (latest.get("o3",0),  180, "#10B981"),
            "SO₂" : (latest.get("so2",0), 80,  "#8B5CF6"),
        }
        for name, (val, max_val, color) in pollutants.items():
            pct = min(100, (val / max_val) * 100)
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
                    <span style="color:#9CA3AF;">{name}</span>
                    <span style="color:{color};font-weight:600;">{val:.1f}</span>
                </div>
                <div style="background:#1E2A3A;border-radius:4px;height:6px;">
                    <div style="background:{color};width:{pct}%;height:6px;border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SHAP explanation with narrative — selectable horizon ────────────────
    with st.expander("🔍 Why this prediction? (SHAP Explanation)", expanded=False):
        shap_horizon = st.radio(
            "Explain forecast for:",
            ["1 Hour", "24 Hours", "48 Hours"],
            horizontal=True, key="shap_horizon_select",
        )
        horizon_map = {"1 Hour": "1h", "24 Hours": "24h", "48 Hours": "48h"}
        shap_h = horizon_map[shap_horizon]
        pred_map = {"1h": pred_1h, "24h": pred_24h, "48h": pred_48h}
        selected_pred = pred_map[shap_h]

        if shap_h in models:
            with st.spinner("Computing explanation..."):
                explainer = shap.TreeExplainer(models[shap_h])
                sv = explainer(x_input)

                narrative = make_shap_narrative(sv, FEATURES, FEATURES)
                flabel, fcolor, fbg, femoji = get_aqi_level(selected_pred) if selected_pred else (cur_label, cur_color, cur_bg, cur_emoji)
                st.markdown(f"""
                <div class="card-sm" style="background:#1E2A3A;border-left:3px solid {fcolor};margin-bottom:16px;">
                    <div style="font-size:13px;color:#9CA3AF;margin-bottom:6px;">
                        Predicted PM2.5 in {shap_horizon.lower()}: 
                        <b style="color:{fcolor};">{selected_pred:.1f} μg/m³ ({flabel})</b>
                    </div>
                    <div style="font-size:13px;color:#E8EAF0;line-height:1.6;">
                        💡 {narrative}
                    </div>
                </div>""", unsafe_allow_html=True)

                fig2, ax = plt.subplots(figsize=(10, 4))
                fig2.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                shap.plots.waterfall(sv[0], max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig2, transparent=True)
                plt.close()
                st.caption("Red bars push PM2.5 prediction up. Blue bars push it down.")
        else:
            st.info(f"Model for {shap_horizon} not available.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="section-header">Test Set Performance — Held-out data the model never saw</div>', unsafe_allow_html=True)

    for _, row in metrics.iterrows():
        h = row["horizon"]
        colors = {"1h": "#10B981", "24h": "#F59E0B", "48h": "#EF4444"}
        c = colors.get(h, "#3B82F6")
        st.markdown(f"""<div class="card" style="border-left:3px solid {c};">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;color:{c};margin-bottom:12px;">
                ⏱ {h.upper()} HORIZON
            </div>""", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">RMSE</div>
                <div class="metric-value" style="color:{c};">{row['RMSE']:.2f}</div>
                <div class="metric-sub">μg/m³</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">MAE</div>
                <div class="metric-value" style="color:{c};">{row['MAE']:.2f}</div>
                <div class="metric-sub">μg/m³</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">R²</div>
                <div class="metric-value" style="color:{c};">{row['R2']:.4f}</div>
                <div class="metric-sub">variance explained</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">AQI Category Acc</div>
                <div class="metric-value" style="color:{c};">{row['Category_Acc']:.1f}%</div>
                <div class="metric-sub">correct category</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_scatter, col_cm = st.columns([1.5, 1])

    with col_scatter:
        st.markdown('<div class="section-header">Actual vs Predicted</div>', unsafe_allow_html=True)
        h_sel = st.selectbox("Horizon", ["1h","24h","48h"], key="scatter_h", label_visibility="collapsed")
        pdf   = {"1h": p1h, "24h": p24h, "48h": p48h}[h_sel]
        if "city" in pdf.columns:
            city_filter = st.selectbox("City", ["All"] + list(pdf["city"].unique()), key="scatter_c", label_visibility="collapsed")
            pdf = pdf if city_filter == "All" else pdf[pdf["city"] == city_filter]

        color_map = {"Good":"#10B981","Satisfactory":"#84CC16","Moderate":"#F59E0B",
                     "Poor":"#EF4444","Very Poor":"#8B5CF6","Severe":"#EC4899"}
        fig = px.scatter(pdf, x="actual_pm25", y="pred_pm25",
                        color="actual_label" if "actual_label" in pdf.columns else None,
                        color_discrete_map=color_map, opacity=0.5, height=380,
                        labels={"actual_pm25":"Actual PM2.5","pred_pm25":"Predicted PM2.5"})
        max_v = max(pdf["actual_pm25"].max(), pdf["pred_pm25"].max()) * 1.05
        fig.add_trace(go.Scatter(x=[0,max_v], y=[0,max_v], mode="lines",
                                name="Perfect", line=dict(color="#374151",dash="dash",width=1.5)))
        fig.update_layout(**PLOT_THEME, height=380, legend=dict(orientation="h",y=-0.15,font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_cm:
        st.markdown('<div class="section-header">AQI Confusion Matrix (1h)</div>', unsafe_allow_html=True)
        fig = px.imshow(confusion, text_auto=True, aspect="auto",
                       color_continuous_scale=[[0,"#0A0E1A"],[0.5,"#1E3A5F"],[1,"#1D4ED8"]],
                       labels=dict(x="Predicted",y="Actual",color="Count"))
        fig.update_layout(**{k:v for k,v in PLOT_THEME.items() if k not in ["xaxis","yaxis"]},
                         height=380, coloraxis_showscale=False,
                         xaxis=dict(tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.caption("Diagonal = correct predictions.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown('<div class="section-header">RMSE by Horizon</div>', unsafe_allow_html=True)
        fig = px.bar(metrics, x="horizon", y="RMSE", text="RMSE", color="horizon",
                    color_discrete_map={"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"})
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont=dict(color="#E8EAF0"))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False, yaxis_title="RMSE (μg/m³)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with col_b2:
        st.markdown('<div class="section-header">R² by Horizon</div>', unsafe_allow_html=True)
        fig = px.bar(metrics, x="horizon", y="R2", text="R2", color="horizon",
                    color_discrete_map={"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"})
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside", textfont=dict(color="#E8EAF0"))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False, yaxis_title="R²", yaxis_range=[0,1.05])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Historical Analysis":

    if "hist_city" not in st.session_state:
        st.session_state.hist_city = "Delhi"

    st.markdown('<div class="section-header">Select City</div>', unsafe_allow_html=True)
    hc = st.columns(len(CITIES))
    for col, city_name in zip(hc, CITIES.keys()):
        with col:
            if st.button(city_name, key=f"hcity_{city_name}", use_container_width=True,
                        type="primary" if st.session_state.hist_city == city_name else "secondary"):
                st.session_state.hist_city = city_name
                st.rerun()

    hcity = st.session_state.hist_city

    st.markdown('<div class="section-header">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    h_sel = st.selectbox("Horizon", ["1h","24h","48h"], key="hist_h", label_visibility="collapsed")
    ts_df = {"1h":p1h,"24h":p24h,"48h":p48h}[h_sel]
    ts_df = ts_df[ts_df["city"]==hcity] if "city" in ts_df.columns else ts_df

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["actual_pm25"], name="Actual",
                            line=dict(color="#3B82F6",width=1.5)))
    fig.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["pred_pm25"], name=f"Predicted ({h_sel})",
                            line=dict(color="#F59E0B",width=1.5,dash="dot")))
    for val, lbl, clr in [(30,"Good","#10B981"),(60,"Satisfactory","#84CC16"),
                           (90,"Moderate","#F59E0B"),(120,"Poor","#EF4444")]:
        fig.add_hline(y=val, line_dash="dash", line_color=clr, line_width=1, opacity=0.3,
                     annotation_text=lbl, annotation_font=dict(color=clr,size=9))
    fig.update_layout(**PLOT_THEME, height=300, hovermode="x unified",
                     legend=dict(orientation="h",y=1.1,font=dict(size=11)), yaxis_title="PM2.5 (μg/m³)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_heat, col_month = st.columns(2)

    with col_heat:
        st.markdown('<div class="section-header">Avg PM2.5 — Hour × Day of Week</div>', unsafe_allow_html=True)
        if "hour" in ts_df.columns and "day_of_week" in ts_df.columns:
            pivot = ts_df.pivot_table(values="actual_pm25",index="hour",columns="day_of_week",aggfunc="mean")
            pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot.columns)]
            fig = px.imshow(pivot, aspect="auto",
                           color_continuous_scale=[[0,"#0A0E1A"],[0.4,"#1E3A5F"],[0.7,"#F59E0B"],[1,"#EF4444"]],
                           labels=dict(x="Day",y="Hour",color="PM2.5"))
            fig.update_layout(**{k:v for k,v in PLOT_THEME.items() if k not in ["xaxis","yaxis"]},
                             height=320, coloraxis_showscale=True,
                             coloraxis_colorbar=dict(thickness=10,len=0.8,tickfont=dict(size=9,color="#6B7280")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Hour/day columns not available in predictions file")

    with col_month:
        st.markdown('<div class="section-header">Monthly Average PM2.5</div>', unsafe_allow_html=True)
        if "datetime" in ts_df.columns:
            ts_df2 = ts_df.copy()
            ts_df2["month"] = ts_df2["datetime"].dt.month
            monthly = ts_df2.groupby("month")["actual_pm25"].mean().reset_index()
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["month_name"] = monthly["month"].map(month_names)
            fig = px.bar(monthly, x="month_name", y="actual_pm25", color="actual_pm25",
                        color_continuous_scale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#EF4444"]], text_auto=".1f")
            fig.update_traces(textfont=dict(color="#E8EAF0",size=10))
            fig.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False,
                             yaxis_title="Avg PM2.5 (μg/m³)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">City Comparison — Average PM2.5</div>', unsafe_allow_html=True)
    if "city" in p1h.columns:
        city_avg = p1h.groupby("city")["actual_pm25"].agg(["mean","std"]).reset_index()
        city_avg.columns = ["City","Mean","Std"]
        city_avg = city_avg.sort_values("Mean", ascending=False)
        fig = px.bar(city_avg, x="City", y="Mean", error_y="Std", color="Mean",
                    color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]], text_auto=".1f")
        fig.update_traces(textfont=dict(color="#E8EAF0",size=11))
        fig.update_layout(**PLOT_THEME, height=300, coloraxis_showscale=False,
                         yaxis_title="Avg PM2.5 (μg/m³)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — REAL-TIME EVALUATION (self-logging, no background process)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Real-Time Eval":
    st.markdown("""
    <div class="card-sm" style="background:#1E2A3A;border-left:3px solid #3B82F6;margin-bottom:16px;">
        <div style="font-size:12px;color:#9CA3AF;">
            ℹ️ This log grows automatically every time you open the Live Forecast page for any city.
            Visit a few cities on Page 1, then return here to see predictions accumulate and resolve over time.
        </div>
    </div>""", unsafe_allow_html=True)

    log_df = load_live_log()

    if log_df.empty:
        st.info("No predictions logged yet. Visit the 🔮 Live Forecast page and select a city to start logging.")
        st.stop()

    total_preds = len(log_df)
    evaluated   = log_df["actual_1h"].notna().sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Predictions Logged</div>
            <div class="metric-value" style="color:#3B82F6;">{total_preds}</div>
            <div class="metric-sub">across all cities</div></div>""", unsafe_allow_html=True)
    with c2:
        cities_logged = log_df["city"].nunique()
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Cities Tracked</div>
            <div class="metric-value" style="color:#10B981;">{cities_logged}</div>
            <div class="metric-sub">of 5 total</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">1h Evaluated</div>
            <div class="metric-value" style="color:#F59E0B;">{evaluated}</div>
            <div class="metric-sub">predictions resolved</div></div>""", unsafe_allow_html=True)
    with c4:
        if evaluated > 0:
            live_rmse = np.sqrt((log_df["error_1h"].dropna()**2).mean())
            test_rmse = float(metrics[metrics["horizon"]=="1h"]["RMSE"].values[0])
            delta = live_rmse - test_rmse
            delta_color = "#10B981" if delta < 2 else "#EF4444"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Live 1h RMSE</div>
                <div class="metric-value" style="color:{delta_color};">{live_rmse:.2f}</div>
                <div class="metric-sub">Test: {test_rmse:.2f} | Δ {delta:+.2f}</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Live 1h RMSE</div>
                <div class="metric-value" style="color:#4B5563;">—</div>
                <div class="metric-sub">visit again in 1hr+</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Latest Logged Predictions</div>', unsafe_allow_html=True)
    latest_preds = log_df.sort_values("prediction_time").groupby("city").tail(1)[
        ["city","current_datetime","current_pm25","pred_1h","pred_24h","pred_48h","actual_1h","error_1h"]
    ].sort_values("current_pm25", ascending=False)

    def color_pm25(val):
        if pd.isna(val): return ""
        _, color, _, _ = get_aqi_level(val)
        return f"color: {color}; font-weight: 600;"

    st.dataframe(
        latest_preds.style.map(color_pm25, subset=["current_pm25","pred_1h","pred_24h","pred_48h"]),
        use_container_width=True, hide_index=True,
    )

    evaluated_df = log_df[log_df["actual_1h"].notna()].copy()
    if len(evaluated_df) > 0:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">1h Prediction vs Actual — Over Time</div>', unsafe_allow_html=True)

        city_sel = st.selectbox("City", ["All"] + list(log_df["city"].unique()), key="rt_city")
        plot_df  = evaluated_df if city_sel == "All" else evaluated_df[evaluated_df["city"]==city_sel]

        fig = go.Figure()
        for city_name, grp in plot_df.groupby("city"):
            fig.add_trace(go.Scatter(x=grp["current_datetime"], y=grp["actual_1h"],
                                    name=f"{city_name} Actual", mode="lines+markers",
                                    marker=dict(size=4), line=dict(width=1.5)))
            fig.add_trace(go.Scatter(x=grp["current_datetime"], y=grp["pred_1h"],
                                    name=f"{city_name} Predicted", mode="lines",
                                    line=dict(width=1.5, dash="dot")))
        fig.update_layout(**PLOT_THEME, height=320, hovermode="x unified", yaxis_title="PM2.5 (μg/m³)",
                         legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="section-header">Per City — Live Accuracy</div>', unsafe_allow_html=True)
        city_acc = evaluated_df.groupby("city").apply(
            lambda x: pd.Series({
                "Predictions": len(x),
                "Live RMSE"  : round(np.sqrt((x["error_1h"]**2).mean()), 2),
                "Live MAE"   : round(x["error_1h"].abs().mean(), 2),
            })
        ).reset_index()
        st.dataframe(city_acc, use_container_width=True, hide_index=True)
    else:
        st.info("⏳ Actuals fill in once an hour has passed since a prediction was logged. Keep visiting Page 1 periodically.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    col_about, col_stats = st.columns([1.8, 1])

    with col_about:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;margin-bottom:8px;">
                AQI Forecasting System
            </div>
            <div style="color:#6B7280;font-size:14px;line-height:1.7;margin-bottom:20px;">
                An end-to-end machine learning system for on-demand PM2.5 air quality
                forecasting across 5 major Indian cities. Predicts pollution levels 1, 24,
                and 48 hours ahead using XGBoost with SHAP explainability — all computed
                live when you open the dashboard, no background infrastructure required.
            </div>
            <div class="divider"></div>
            <div class="section-header">Pipeline</div>
            <div style="font-size:13px;color:#9CA3AF;line-height:2;">
                1. <b style='color:#E8EAF0;'>Data Collection</b> — Hourly weather + AQI from Open-Meteo API<br>
                2. <b style='color:#E8EAF0;'>Feature Engineering</b> — Lag features, rolling stats, interaction terms, festival flags<br>
                3. <b style='color:#E8EAF0;'>Modelling</b> — Separate XGBoost regressors per forecast horizon<br>
                4. <b style='color:#E8EAF0;'>Tuning</b> — Optuna hyperparameter search (50 trials per model)<br>
                5. <b style='color:#E8EAF0;'>Evaluation</b> — Chronological train/val/test split to prevent leakage<br>
                6. <b style='color:#E8EAF0;'>Explainability</b> — SHAP values translated into plain-English narratives<br>
                7. <b style='color:#E8EAF0;'>Live Monitoring</b> — Self-logging on-demand evaluation, no scheduler needed
            </div>
            <div class="divider"></div>
            <div class="section-header">Key Findings</div>
            <div style="font-size:13px;color:#9CA3AF;line-height:2;">
                • <b style='color:#E8EAF0;'>1h model</b> is momentum-driven — current PM2.5 and lag features dominate<br>
                • <b style='color:#E8EAF0;'>24h model</b> transitions to seasonal and city-level patterns<br>
                • <b style='color:#E8EAF0;'>48h model</b> relies on city identity and month — climatological baseline<br>
                • Most misclassifications occur between <b style='color:#E8EAF0;'>adjacent AQI categories</b> only
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="card"><div class="section-header">Test Set Results</div>', unsafe_allow_html=True)
        for _, row in metrics.iterrows():
            colors = {"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"}
            c = colors.get(row["horizon"],"#3B82F6")
            st.markdown(f"""
            <div class="card-sm" style="margin-bottom:10px;border-left:2px solid {c};">
                <div style="color:{c};font-weight:700;font-size:12px;margin-bottom:6px;">
                    {row['horizon'].upper()} HORIZON
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div><div style="font-size:10px;color:#6B7280;">RMSE</div>
                        <div style="font-weight:700;color:#E8EAF0;">{row['RMSE']:.2f} μg/m³</div></div>
                    <div><div style="font-size:10px;color:#6B7280;">R²</div>
                        <div style="font-weight:700;color:#E8EAF0;">{row['R2']:.4f}</div></div>
                    <div style="grid-column:span 2;"><div style="font-size:10px;color:#6B7280;">AQI Category Accuracy</div>
                        <div style="font-weight:700;color:{c};">{row['Category_Acc']:.1f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div><div class="section-header">Tech Stack</div>'
                    '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">', unsafe_allow_html=True)
        for tech in ["Python","XGBoost","SHAP","Optuna","Streamlit","Plotly","Pandas","Open-Meteo API"]:
            st.markdown(f"""<span style="background:#1E2A3A;border:1px solid #374151;border-radius:6px;
                          padding:4px 10px;font-size:11px;color:#9CA3AF;">{tech}</span>""", unsafe_allow_html=True)

        st.markdown("""</div><div class="divider"></div>
            <div class="section-header">Cities Covered</div>
            <div style="font-size:13px;color:#9CA3AF;line-height:2;">
                🏙️ Delhi &nbsp;|&nbsp; 🌆 Mumbai &nbsp;|&nbsp; 🌇 Bengaluru<br>
                🌃 Chennai &nbsp;|&nbsp; 🌉 Kolkata
            </div></div>""", unsafe_allow_html=True)
