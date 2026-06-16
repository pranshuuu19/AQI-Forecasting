"""
AQI Forecasting Dashboard
==========================
HOW TO RUN:
  pip install streamlit plotly shap matplotlib pandas numpy xgboost requests
  streamlit run dashboard.py

FOLDER STRUCTURE:
  models/tuned_model_1h.json, tuned_model_24h.json, tuned_model_48h.json
  data/feature_config.json
  data/predictions_log.csv
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

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E1A;
    color: #E8EAF0;
}
.stApp { background-color: #0A0E1A; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.02em;
}

/* ── Nav tabs ── */
.nav-container {
    display: flex;
    gap: 4px;
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 2rem;
    border: 1px solid #1E2A3A;
}
.nav-tab {
    flex: 1;
    text-align: center;
    padding: 10px 8px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #6B7280;
    transition: all 0.2s;
    text-decoration: none;
}
.nav-tab.active {
    background: #1E40AF;
    color: white;
    font-weight: 600;
}

/* ── Cards ── */
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

/* ── Metric cards ── */
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
.metric-sub {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 2px;
}

/* ── Forecast cards ── */
.forecast-card {
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.forecast-horizon {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-bottom: 8px;
}
.forecast-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.forecast-unit {
    font-size: 12px;
    opacity: 0.6;
    margin-bottom: 12px;
}
.forecast-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
}

/* ── AQI badge ── */
.aqi-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 600;
}

/* ── City selector ── */
.city-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.city-btn {
    padding: 8px 18px;
    border-radius: 24px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid #1E2A3A;
    background: #111827;
    color: #9CA3AF;
    cursor: pointer;
}
.city-btn.active {
    background: #1E40AF;
    border-color: #1E40AF;
    color: white;
    font-weight: 600;
}

/* ── Section header ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4B5563;
    margin-bottom: 12px;
    margin-top: 4px;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: #1E2A3A;
    margin: 20px 0;
}

/* ── Live badge ── */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #10B981;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Scrollbar ── */
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

CITIES = {
    "Delhi"    : (28.6139, 77.2090),
    "Mumbai"   : (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai"  : (13.0827, 80.2707),
    "Kolkata"  : (22.5726, 88.3639),
}

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9CA3AF", family="Inter"),
    xaxis=dict(gridcolor="#1E2A3A", linecolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    yaxis=dict(gridcolor="#1E2A3A", linecolor="#1E2A3A", zerolinecolor="#1E2A3A"),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_aqi_level(val):
    for label, lo, hi, color, bg, emoji in AQI_LEVELS:
        if lo <= val <= hi:
            return label, color, bg, emoji
    return "Severe", "#EC4899", "#500724", "☠️"

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
    fig.update_layout(height=200,
                  paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(color="#9CA3AF", family="Inter"),
                  margin=dict(l=20, r=20, t=20, b=10))
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

@st.cache_data(ttl=60)
def load_predictions_log():
    if os.path.exists("data/predictions_log.csv"):
        df = pd.read_csv("data/predictions_log.csv")
        df["prediction_time"]  = pd.to_datetime(df["prediction_time"])
        df["current_datetime"] = pd.to_datetime(df["current_datetime"])
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_live_data(city):
    """Fetch current conditions from Open-Meteo for selected city."""
    lat, lon = CITIES[city]
    now_ist    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    end_date   = now_ist.strftime("%Y-%m-%d")
    start_date = (now_ist - timedelta(days=6)).strftime("%Y-%m-%d")

    try:
        # AQI
        aqi_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
            f"&timezone=Asia/Kolkata"
        )
        aqi_r = requests.get(aqi_url, timeout=15).json()
        aqi_df = pd.DataFrame({
            "datetime": pd.to_datetime(aqi_r["hourly"]["time"]),
            "pm2_5": aqi_r["hourly"]["pm2_5"],
            "pm10" : aqi_r["hourly"]["pm10"],
            "co"   : aqi_r["hourly"]["carbon_monoxide"],
            "no2"  : aqi_r["hourly"]["nitrogen_dioxide"],
            "o3"   : aqi_r["hourly"]["ozone"],
            "so2"  : aqi_r["hourly"]["sulphur_dioxide"],
        })

        # Weather (archive)
        wx_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
            f"&timezone=Asia/Kolkata"
        )
        wx_r = requests.get(wx_url, timeout=15).json()
        wx_df = pd.DataFrame({
            "datetime"   : pd.to_datetime(wx_r["hourly"]["time"]),
            "temperature": wx_r["hourly"]["temperature_2m"],
            "humidity"   : wx_r["hourly"]["relative_humidity_2m"],
            "wind_speed" : wx_r["hourly"]["wind_speed_10m"],
            "rainfall"   : wx_r["hourly"]["precipitation"],
        })

        df = pd.merge(aqi_df, wx_df, on="datetime", how="inner")
        df = df[df["datetime"] <= now_ist].dropna()
        df = df.sort_values("datetime").reset_index(drop=True)
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
    df["humidity_pm25"]        = df["humidity"] * df["pm2_5"]
    df["wind_pm25_ratio"]      = df["pm2_5"] / (df["wind_speed"] + 1)
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

# Navigation
pages = ["🔮 Live Forecast", "📊 Model Performance", "📈 Historical Analysis", "⚡ Real-Time Eval", "ℹ️ About"]
if "page" not in st.session_state:
    st.session_state.page = pages[0]

with col_nav:
    cols = st.columns(len(pages))
    for i, (col, page) in enumerate(zip(cols, pages)):
        with col:
            if st.button(page, key=f"nav_{i}",
                        use_container_width=True,
                        type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Live Forecast":

    # City selector
    if "city" not in st.session_state:
        st.session_state.city = "Delhi"

    st.markdown('<div class="section-header">Select City</div>', unsafe_allow_html=True)
    city_cols = st.columns(len(CITIES))
    for col, city_name in zip(city_cols, CITIES.keys()):
        with col:
            active = "primary" if st.session_state.city == city_name else "secondary"
            if st.button(city_name, key=f"city_{city_name}",
                        use_container_width=True, type=active):
                st.session_state.city = city_name
                st.cache_data.clear()
                st.rerun()

    selected_city = st.session_state.city

    # Fetch live data
    with st.spinner(f"Fetching live data for {selected_city}..."):
        live_df, error = fetch_live_data(selected_city)

    if error or live_df is None or len(live_df) == 0:
        st.error(f"Could not fetch live data: {error}")
        st.stop()

    # Engineer features + predict
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

    # ── Row 1: Gauge + Current conditions + Forecast cards ──────────────────
    col_gauge, col_weather, col_forecasts = st.columns([1.2, 1, 2.8])

    with col_gauge:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:16px 12px;">
            <div class="section-header">Current PM2.5</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(aqi_gauge(current_pm25), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"""
            <div style="margin-top:-10px;text-align:center;">
                <span class="aqi-pill" style="background:{cur_bg};color:{cur_color};border:1px solid {cur_color}40;">
                    {cur_emoji} {cur_label}
                </span>
            </div>
            <div style="font-size:11px;color:#4B5563;text-align:center;margin-top:8px;">
                as of {latest['datetime'].strftime('%I:%M %p')} IST
            </div>
        </div>""", unsafe_allow_html=True)

    with col_weather:
        temp = latest.get("temperature", "—")
        hum  = latest.get("humidity", "—")
        ws   = latest.get("wind_speed", "—")
        rain = latest.get("rainfall", "—")
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
            [fc1, fc2, fc3],
            [pred_1h, pred_24h, pred_48h],
            ["1h", "24h", "48h"],
            ["1 Hour", "24 Hours", "48 Hours"]
        ):
            with col:
                if pred:
                    flabel, fcolor, fbg, femoji = get_aqi_level(pred)
                    st.markdown(f"""
                    <div class="forecast-card" style="background:{fbg};border-color:{fcolor}30;">
                        <div class="forecast-horizon" style="color:{fcolor};">⏱ {label}</div>
                        <div class="forecast-value" style="color:{fcolor};">{pred:.1f}</div>
                        <div class="forecast-unit" style="color:{fcolor};">μg/m³</div>
                        <div class="forecast-badge" style="color:{fcolor};">
                            {femoji} {flabel}
                        </div>
                    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Row 2: 48h forecast graph + Pollutants ───────────────────────────────
    col_chart, col_poll = st.columns([2.5, 1])

    with col_chart:
        st.markdown('<div class="section-header">48-Hour PM2.5 Forecast Trend</div>', unsafe_allow_html=True)

        # Build rolling 48h prediction by predicting each future hour
        future_rows = []
        df_copy = featured.copy()
        for h_ahead in range(1, 49):
            row = df_copy.iloc[-1].copy()
            for f in FEATURES:
                if f not in row.index:
                    row[f] = 0
            x = pd.DataFrame([row[FEATURES]]).fillna(0)
            pred_val = float(models["1h"].predict(x)[0]) if "1h" in models else current_pm25
            future_time = latest["datetime"] + timedelta(hours=h_ahead)
            future_rows.append({"datetime": future_time, "pm2_5": pred_val})
            # Shift lag features for next iteration
            if "pm2_5_lag1" in df_copy.columns:
                new_row = df_copy.iloc[-1].copy()
                new_row["datetime"] = future_time
                new_row["pm2_5"] = pred_val
                new_row["pm2_5_lag1"] = df_copy.iloc[-1]["pm2_5"]
                df_copy = pd.concat([df_copy, pd.DataFrame([new_row])], ignore_index=True)

        future_df = pd.DataFrame(future_rows)

        # Historical last 24h
        hist_df = live_df.tail(24)[["datetime","pm2_5"]].copy()

        fig = go.Figure()
        # Historical
        fig.add_trace(go.Scatter(
            x=hist_df["datetime"], y=hist_df["pm2_5"],
            name="Historical", mode="lines",
            line=dict(color="#3B82F6", width=2),
        ))
        # Forecast
        connect_x = [hist_df["datetime"].iloc[-1], future_df["datetime"].iloc[0]]
        connect_y = [hist_df["pm2_5"].iloc[-1],    future_df["pm2_5"].iloc[0]]
        fig.add_trace(go.Scatter(
            x=connect_x + list(future_df["datetime"]),
            y=connect_y + list(future_df["pm2_5"]),
            name="Forecast", mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.05)",
        ))
        # AQI threshold lines
        thresholds = [(30,"Good","#10B981"),(60,"Satisfactory","#84CC16"),
                      (90,"Moderate","#F59E0B"),(120,"Poor","#EF4444")]
        for val, lbl, clr in thresholds:
            fig.add_hline(y=val, line_dash="dash", line_color=clr,
                         line_width=1, opacity=0.4,
                         annotation_text=lbl,
                         annotation_position="right",
                         annotation_font=dict(color=clr, size=10))
        # Now line
        fig.add_vline(x=str(latest["datetime"]), line_dash="solid",
                     line_color="#6B7280", line_width=1, opacity=0.5)
        fig.add_annotation(x=latest["datetime"], y=current_pm25*1.1,
                          text="NOW", font=dict(color="#6B7280",size=10),
                          showarrow=False)
        fig.update_layout(
            height=280, showlegend=True,
            legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
            xaxis_title="", yaxis_title="PM2.5 (μg/m³)",
            **PLOT_THEME
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
                <div style="display:flex;justify-content:space-between;
                            font-size:12px;margin-bottom:4px;">
                    <span style="color:#9CA3AF;">{name}</span>
                    <span style="color:{color};font-weight:600;">{val:.1f}</span>
                </div>
                <div style="background:#1E2A3A;border-radius:4px;height:6px;">
                    <div style="background:{color};width:{pct}%;height:6px;
                                border-radius:4px;transition:width 0.3s;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SHAP explanation ─────────────────────────────────────────────────────
    with st.expander("🔍 Why this prediction? (SHAP Explanation for 1h forecast)", expanded=False):
        if "1h" in models:
            with st.spinner("Computing explanation..."):
                explainer = shap.TreeExplainer(models["1h"])
                sv = explainer(x_input)
                fig2, ax = plt.subplots(figsize=(10, 4))
                fig2.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                shap.plots.waterfall(sv[0], max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig2, transparent=True)
                plt.close()
                st.caption("Red bars push PM2.5 prediction up. Blue bars push it down.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<div class="section-header">Test Set Performance — Held-out data the model never saw</div>', unsafe_allow_html=True)

    # KPI row per horizon
    for _, row in metrics.iterrows():
        h = row["horizon"]
        colors = {"1h": "#10B981", "24h": "#F59E0B", "48h": "#EF4444"}
        c = colors.get(h, "#3B82F6")
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {c};">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                        font-weight:700;color:{c};margin-bottom:12px;">
                ⏱ {h.upper()} HORIZON
            </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value" style="color:{c};">{row['RMSE']:.2f}</div>
                <div class="metric-sub">μg/m³</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value" style="color:{c};">{row['MAE']:.2f}</div>
                <div class="metric-sub">μg/m³</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">R²</div>
                <div class="metric-value" style="color:{c};">{row['R2']:.4f}</div>
                <div class="metric-sub">variance explained</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">AQI Category Acc</div>
                <div class="metric-value" style="color:{c};">{row['Category_Acc']:.1f}%</div>
                <div class="metric-sub">correct category</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Charts row
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
                        color_discrete_map=color_map,
                        opacity=0.5, height=380,
                        labels={"actual_pm25":"Actual PM2.5","pred_pm25":"Predicted PM2.5"})
        max_v = max(pdf["actual_pm25"].max(), pdf["pred_pm25"].max()) * 1.05
        fig.add_trace(go.Scatter(x=[0,max_v], y=[0,max_v], mode="lines",
                                name="Perfect", line=dict(color="#374151",dash="dash",width=1.5)))
        fig.update_layout(**PLOT_THEME, height=380,
                         legend=dict(orientation="h",y=-0.15,font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_cm:
        st.markdown('<div class="section-header">AQI Category Confusion Matrix (1h)</div>', unsafe_allow_html=True)
        fig = px.imshow(confusion, text_auto=True, aspect="auto",
                       color_continuous_scale=[[0,"#0A0E1A"],[0.5,"#1E3A5F"],[1,"#1D4ED8"]],
                       labels=dict(x="Predicted",y="Actual",color="Count"))
        fig.update_layout(**{k:v for k,v in PLOT_THEME.items() if k not in ["xaxis","yaxis"]},
                         height=380, coloraxis_showscale=False,
                         xaxis=dict(tickfont=dict(size=10)),
                         yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.caption("Diagonal = correct predictions. Off-diagonal = misclassified.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Metrics comparison bars
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown('<div class="section-header">RMSE by Horizon</div>', unsafe_allow_html=True)
        fig = px.bar(metrics, x="horizon", y="RMSE", text="RMSE",
                    color="horizon",
                    color_discrete_map={"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"})
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                         textfont=dict(color="#E8EAF0"))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False,
                         yaxis_title="RMSE (μg/m³)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_b2:
        st.markdown('<div class="section-header">R² by Horizon</div>', unsafe_allow_html=True)
        fig = px.bar(metrics, x="horizon", y="R2", text="R2",
                    color="horizon",
                    color_discrete_map={"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"})
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside",
                         textfont=dict(color="#E8EAF0"))
        fig.update_layout(**PLOT_THEME, height=280, showlegend=False,
                         yaxis_title="R²", yaxis_range=[0,1.05])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Historical Analysis":

    # City selector
    if "hist_city" not in st.session_state:
        st.session_state.hist_city = "Delhi"

    st.markdown('<div class="section-header">Select City</div>', unsafe_allow_html=True)
    hc = st.columns(len(CITIES))
    for col, city_name in zip(hc, CITIES.keys()):
        with col:
            if st.button(city_name, key=f"hcity_{city_name}",
                        use_container_width=True,
                        type="primary" if st.session_state.hist_city == city_name else "secondary"):
                st.session_state.hist_city = city_name
                st.rerun()

    hcity = st.session_state.hist_city
    pdf_city = p1h[p1h["city"] == hcity] if "city" in p1h.columns else p1h

    # ── Actual vs Predicted time series ──────────────────────────────────────
    st.markdown('<div class="section-header">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    h_sel = st.selectbox("Horizon", ["1h","24h","48h"], key="hist_h", label_visibility="collapsed")
    ts_df = {"1h":p1h,"24h":p24h,"48h":p48h}[h_sel]
    ts_df = ts_df[ts_df["city"]==hcity] if "city" in ts_df.columns else ts_df

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["actual_pm25"],
                            name="Actual", line=dict(color="#3B82F6",width=1.5)))
    fig.add_trace(go.Scatter(x=ts_df["datetime"], y=ts_df["pred_pm25"],
                            name=f"Predicted ({h_sel})",
                            line=dict(color="#F59E0B",width=1.5,dash="dot")))
    for val, lbl, clr in [(30,"Good","#10B981"),(60,"Satisfactory","#84CC16"),
                           (90,"Moderate","#F59E0B"),(120,"Poor","#EF4444")]:
        fig.add_hline(y=val, line_dash="dash", line_color=clr, line_width=1,
                     opacity=0.3, annotation_text=lbl,
                     annotation_font=dict(color=clr,size=9))
    fig.update_layout(**PLOT_THEME, height=300, hovermode="x unified",
                     legend=dict(orientation="h",y=1.1,font=dict(size=11)),
                     yaxis_title="PM2.5 (μg/m³)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Heatmap + Monthly ─────────────────────────────────────────────────────
    col_heat, col_month = st.columns(2)

    with col_heat:
        st.markdown('<div class="section-header">Avg PM2.5 — Hour × Day of Week</div>', unsafe_allow_html=True)
        if "hour" in ts_df.columns and "day_of_week" in ts_df.columns:
            pivot = ts_df.pivot_table(values="actual_pm25",index="hour",
                                      columns="day_of_week",aggfunc="mean")
            pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot.columns)]
            fig = px.imshow(pivot, aspect="auto",
                           color_continuous_scale=[[0,"#0A0E1A"],[0.4,"#1E3A5F"],
                                                   [0.7,"#F59E0B"],[1,"#EF4444"]],
                           labels=dict(x="Day",y="Hour",color="PM2.5"))
            fig.update_layout(**{k:v for k,v in PLOT_THEME.items() if k not in ["xaxis","yaxis"]},
                             height=320, coloraxis_showscale=True,
                             coloraxis_colorbar=dict(thickness=10,len=0.8,
                                                     tickfont=dict(size=9,color="#6B7280")))
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
            fig = px.bar(monthly, x="month_name", y="actual_pm25",
                        color="actual_pm25",
                        color_continuous_scale=[[0,"#10B981"],[0.4,"#F59E0B"],[1,"#EF4444"]],
                        text_auto=".1f")
            fig.update_traces(textfont=dict(color="#E8EAF0",size=10))
            fig.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False,
                             yaxis_title="Avg PM2.5 (μg/m³)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── City comparison ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">City Comparison — Average PM2.5</div>', unsafe_allow_html=True)
    if "city" in p1h.columns:
        city_avg = p1h.groupby("city")["actual_pm25"].agg(["mean","std"]).reset_index()
        city_avg.columns = ["City","Mean","Std"]
        city_avg = city_avg.sort_values("Mean", ascending=False)
        fig = px.bar(city_avg, x="City", y="Mean", error_y="Std",
                    color="Mean",
                    color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                    text_auto=".1f")
        fig.update_traces(textfont=dict(color="#E8EAF0",size=11))
        fig.update_layout(**PLOT_THEME, height=300, coloraxis_showscale=False,
                         yaxis_title="Avg PM2.5 (μg/m³)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — REAL-TIME EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Real-Time Eval":
    log_df = load_predictions_log()

    if log_df.empty:
        st.info("No predictions logged yet. The fetcher runs every hour automatically.")
        st.stop()

    # ── Header stats ──────────────────────────────────────────────────────────
    total_preds = len(log_df)
    evaluated   = log_df["actual_1h"].notna().sum()
    hours_running = int((log_df["prediction_time"].max() - log_df["prediction_time"].min()).total_seconds() / 3600) + 1

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Predictions Made</div>
            <div class="metric-value" style="color:#3B82F6;">{total_preds}</div>
            <div class="metric-sub">across all cities</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Hours Running</div>
            <div class="metric-value" style="color:#10B981;">{hours_running}</div>
            <div class="metric-sub">since deployment</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">1h Actuals Evaluated</div>
            <div class="metric-value" style="color:#F59E0B;">{evaluated}</div>
            <div class="metric-sub">predictions resolved</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        if evaluated > 0:
            live_rmse = np.sqrt((log_df["error_1h"].dropna()**2).mean())
            test_rmse = float(metrics[metrics["horizon"]=="1h"]["RMSE"].values[0])
            delta = live_rmse - test_rmse
            delta_color = "#10B981" if delta < 2 else "#EF4444"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Live 1h RMSE</div>
                <div class="metric-value" style="color:{delta_color};">{live_rmse:.2f}</div>
                <div class="metric-sub">Test RMSE: {test_rmse:.2f} | Δ {delta:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Live 1h RMSE</div>
                <div class="metric-value" style="color:#4B5563;">—</div>
                <div class="metric-sub">check back in 1 hour</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Latest predictions table ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Latest Predictions</div>', unsafe_allow_html=True)
    latest_preds = log_df.sort_values("prediction_time").groupby("city").tail(1)[
        ["city","current_datetime","current_pm25","pred_1h","pred_24h","pred_48h",
         "actual_1h","error_1h"]
    ].sort_values("current_pm25", ascending=False)

    # Color code by AQI
    def color_pm25(val):
        if pd.isna(val): return ""
        _, color, _, _ = get_aqi_level(val)
        return f"color: {color}; font-weight: 600;"

    st.dataframe(
        latest_preds.style.applymap(color_pm25, subset=["current_pm25","pred_1h","pred_24h","pred_48h"]),
        use_container_width=True, hide_index=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Prediction vs Actual chart ────────────────────────────────────────────
    evaluated_df = log_df[log_df["actual_1h"].notna()].copy()
    if len(evaluated_df) > 0:
        st.markdown('<div class="section-header">1h Prediction vs Actual — Over Time</div>', unsafe_allow_html=True)

        city_sel = st.selectbox("City", ["All"] + list(log_df["city"].unique()), key="rt_city")
        plot_df  = evaluated_df if city_sel == "All" else evaluated_df[evaluated_df["city"]==city_sel]

        fig = go.Figure()
        for city_name, grp in plot_df.groupby("city"):
            fig.add_trace(go.Scatter(
                x=grp["current_datetime"], y=grp["actual_1h"],
                name=f"{city_name} Actual", mode="lines+markers",
                marker=dict(size=4), line=dict(width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=grp["current_datetime"], y=grp["pred_1h"],
                name=f"{city_name} Predicted", mode="lines",
                line=dict(width=1.5, dash="dot")
            ))
        fig.update_layout(**PLOT_THEME, height=320, hovermode="x unified",
                         yaxis_title="PM2.5 (μg/m³)",
                         legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        # ── Per city accuracy ─────────────────────────────────────────────────
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
        st.info("⏳ Actuals will start filling in 1 hour after the first prediction. Check back soon.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    col_about, col_stats = st.columns([1.8, 1])

    with col_about:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;
                        font-weight:700;margin-bottom:8px;">
                AQI Forecasting System
            </div>
            <div style="color:#6B7280;font-size:14px;line-height:1.7;margin-bottom:20px;">
                An end-to-end machine learning pipeline for real-time PM2.5 air quality 
                forecasting across 5 major Indian cities. Predicts pollution levels 1, 24, 
                and 48 hours ahead using XGBoost with SHAP explainability.
            </div>
            <div class="divider"></div>
            <div class="section-header">Pipeline</div>
            <div style="font-size:13px;color:#9CA3AF;line-height:2;">
                1. <b style='color:#E8EAF0;'>Data Collection</b> — Hourly weather + AQI from Open-Meteo API<br>
                2. <b style='color:#E8EAF0;'>Feature Engineering</b> — Lag features (1h/24h/48h), rolling stats, interaction terms, festival flags<br>
                3. <b style='color:#E8EAF0;'>Modelling</b> — Separate XGBoost regressors per forecast horizon<br>
                4. <b style='color:#E8EAF0;'>Tuning</b> — Optuna hyperparameter search (50 trials per model)<br>
                5. <b style='color:#E8EAF0;'>Evaluation</b> — Chronological train/val/test split to prevent leakage<br>
                6. <b style='color:#E8EAF0;'>Explainability</b> — SHAP values for per-prediction explanations<br>
                7. <b style='color:#E8EAF0;'>Automation</b> — GitHub Actions runs fetcher every hour, 24/7
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
        st.markdown("""
        <div class="card">
            <div class="section-header">Test Set Results</div>
        """, unsafe_allow_html=True)
        for _, row in metrics.iterrows():
            colors = {"1h":"#10B981","24h":"#F59E0B","48h":"#EF4444"}
            c = colors.get(row["horizon"],"#3B82F6")
            st.markdown(f"""
            <div class="card-sm" style="margin-bottom:10px;border-left:2px solid {c};">
                <div style="color:{c};font-weight:700;font-size:12px;margin-bottom:6px;">
                    {row['horizon'].upper()} HORIZON
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                    <div>
                        <div style="font-size:10px;color:#6B7280;">RMSE</div>
                        <div style="font-weight:700;color:#E8EAF0;">{row['RMSE']:.2f} μg/m³</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#6B7280;">R²</div>
                        <div style="font-weight:700;color:#E8EAF0;">{row['R2']:.4f}</div>
                    </div>
                    <div style="grid-column:span 2;">
                        <div style="font-size:10px;color:#6B7280;">AQI Category Accuracy</div>
                        <div style="font-weight:700;color:{c};">{row['Category_Acc']:.1f}%</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
            <div class="divider"></div>
            <div class="section-header">Tech Stack</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
        """, unsafe_allow_html=True)

        for tech in ["Python","XGBoost","SHAP","Optuna","Streamlit",
                     "Plotly","Pandas","GitHub Actions","Open-Meteo API"]:
            st.markdown(f"""
            <span style="background:#1E2A3A;border:1px solid #374151;
                          border-radius:6px;padding:4px 10px;
                          font-size:11px;color:#9CA3AF;">{tech}</span>
            """, unsafe_allow_html=True)

        st.markdown("""
            </div>
            <div class="divider"></div>
            <div class="section-header">Cities Covered</div>
            <div style="font-size:13px;color:#9CA3AF;line-height:2;">
                🏙️ Delhi &nbsp;|&nbsp; 🌆 Mumbai &nbsp;|&nbsp; 🌇 Bengaluru<br>
                🌃 Chennai &nbsp;|&nbsp; 🌉 Kolkata
            </div>
        </div>
        """, unsafe_allow_html=True)