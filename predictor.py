import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import requests # type: ignore
import xgboost as xgb # type: ignore
from datetime import datetime, timedelta

CITIES = {
    "Delhi"    : (28.6139, 77.2090),
    "Mumbai"   : (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai"  : (13.0827, 80.2707),
    "Kolkata"  : (22.5726, 88.3639),
}


def load_models(models_dir: str = "models"):
    """Load point-estimate models (tuned_model_*/model_*) and quantile models
    (quantile_{h}_{lower,upper}) from disk. Returns a dict keyed by horizon
    ("1h", "24h", "48h") and by "{horizon}_{lower,upper}" for quantile models.
    """
    models = {}
    for h in ["1h", "24h", "48h"]:
        for prefix in ["tuned_model", "model"]:
            path = os.path.join(models_dir, f"{prefix}_{h}.json")
            if os.path.exists(path):
                m = xgb.XGBRegressor()
                m.load_model(path)
                models[h] = m
                break
        for q in ["lower", "upper"]:
            path = os.path.join(models_dir, f"quantile_{h}_{q}.json")
            if os.path.exists(path):
                m = xgb.XGBRegressor()
                m.load_model(path)
                models[f"{h}_{q}"] = m
    return models


def fetch_live_data(city: str):
    """Fetch the last ~6 days of hourly AQI + weather data for a city from
    Open-Meteo. Returns (dataframe, error_message). error_message is None
    on success.
    """
    if city not in CITIES:
        return None, f"Unknown city: {city}"

    lat, lon = CITIES[city]
    now_ist    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    end_date   = (now_ist + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (now_ist - timedelta(days=6)).strftime("%Y-%m-%d")
    try:
        aqi_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide&timezone=Asia/Kolkata"
        )
        aqi_r = requests.get(aqi_url, timeout=15).json()
        if "hourly" not in aqi_r:
            raise ValueError(f"AQI API error: {aqi_r.get('reason', aqi_r)}")
        aqi_df = pd.DataFrame({
            "datetime": pd.to_datetime(aqi_r["hourly"]["time"]),
            "pm2_5": aqi_r["hourly"]["pm2_5"], "pm10": aqi_r["hourly"]["pm10"],
            "co": aqi_r["hourly"]["carbon_monoxide"], "no2": aqi_r["hourly"]["nitrogen_dioxide"],
            "o3": aqi_r["hourly"]["ozone"], "so2": aqi_r["hourly"]["sulphur_dioxide"],
        })
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"wind_direction_10m,precipitation,surface_pressure,boundary_layer_height&timezone=Asia/Kolkata"
        )
        wx_r = requests.get(wx_url, timeout=15).json()
        if "hourly" not in wx_r:
            raise ValueError(f"Weather API error: {wx_r.get('reason', wx_r)}")
        wx_df = pd.DataFrame({
            "datetime": pd.to_datetime(wx_r["hourly"]["time"]),
            "temperature": wx_r["hourly"]["temperature_2m"], "humidity": wx_r["hourly"]["relative_humidity_2m"],
            "wind_speed": wx_r["hourly"]["wind_speed_10m"], "wind_direction": wx_r["hourly"]["wind_direction_10m"],
            "rainfall": wx_r["hourly"]["precipitation"], "surface_pressure": wx_r["hourly"]["surface_pressure"],
            "boundary_layer_height": wx_r["hourly"]["boundary_layer_height"],
        })
        df = pd.merge(aqi_df, wx_df, on="datetime", how="inner")
        df = df[df["datetime"] <= now_ist].dropna().sort_values("datetime").reset_index(drop=True)
        df["city"] = city
        return df, None
    except Exception as e:
        return None, str(e)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline — identical to dashboard.py's
    engineer_features(), so predictions from the API and the dashboard
    always match.
    """
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    for col in ["pm2_5", "pm10", "no2", "co", "o3", "so2"]:
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
    if "wind_direction" in df.columns:
        df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction"]))
        df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction"]))
    else:
        df["wind_dir_sin"] = 0.0
        df["wind_dir_cos"] = 0.0
    if "boundary_layer_height" in df.columns:
        df["blh_inverse"]          = 1 / (df["boundary_layer_height"] + 1)
        df["blh_pm25_interaction"] = df["pm2_5"] / (df["boundary_layer_height"] + 1)
        df["blh_rolling6_min"]     = df["boundary_layer_height"].rolling(6, min_periods=1).min()
    else:
        df["blh_inverse"] = 0.0
        df["blh_pm25_interaction"] = 0.0
        df["blh_rolling6_min"] = 0.0
    if "surface_pressure" in df.columns:
        df["pressure_change_1h"] = df["surface_pressure"].diff(1).fillna(0)
        df["pressure_change_6h"] = df["surface_pressure"].diff(6).fillna(0)
    else:
        df["pressure_change_1h"] = 0.0
        df["pressure_change_6h"] = 0.0
    city_mean = df["pm2_5"].mean()
    city_std  = df["pm2_5"].std()
    df["is_pollution_spike"] = (df["pm2_5"] > city_mean + 2 * city_std).astype(int)
    df["date_str"] = df["datetime"].dt.date.astype(str)
    df["is_diwali"]        = df["date_str"].isin(["2025-10-20", "2025-10-21"]).astype(int)
    df["is_diwali_window"] = df["date_str"].isin([
        "2025-10-17", "2025-10-18", "2025-10-19", "2025-10-20",
        "2025-10-21", "2025-10-22", "2025-10-23", "2025-10-24",
    ]).astype(int)
    df["is_stubble_burning"] = df["month"].isin([10, 11]).astype(int)
    df = df.drop(columns=["date_str"])
    city_order = {"Bengaluru": 0, "Chennai": 1, "Delhi": 2, "Kolkata": 3, "Mumbai": 4}
    df["city_encoded"] = city_order.get(df["city"].iloc[0], 0)
    h = df["hour"].iloc[-1]
    df["hour_bucket_morning_rush"] = int(6 <= h <= 9)
    df["hour_bucket_daytime"]      = int(10 <= h <= 16)
    df["hour_bucket_evening_rush"] = int(17 <= h <= 20)
    df["hour_bucket_night"]        = int(h <= 5 or h >= 21)
    return df


def _predict_point(models: dict, horizon: str, featured: pd.DataFrame):
    """Point-estimate prediction for a single horizon ("1h"/"24h"/"48h")."""
    if horizon not in models:
        return None
    mf = models[horizon].get_booster().feature_names
    last_row = featured.iloc[-1].copy()
    row_dict = {f: (last_row[f] if f in last_row.index else 0.0) for f in mf}
    x = pd.DataFrame([row_dict])[mf].fillna(0)
    return float(models[horizon].predict(x)[0])


def get_quantile_pred(models: dict, h: str, q: str, featured: pd.DataFrame):
    """Quantile bound prediction — mirrors dashboard.py's get_quantile_pred(),
    including its behavior of returning None (rather than raising) when the
    model is missing or a genuine prediction error occurs.
    """
    key = f"{h}_{q}"
    if key not in models:
        return None
    try:
        mf = models[key].get_booster().feature_names
        last_row = featured.iloc[-1].copy()
        row_dict = {f: (last_row[f] if f in last_row.index else 0.0) for f in mf}
        x = pd.DataFrame([row_dict])[mf].fillna(0)
        val = float(models[key].predict(x)[0])
        return round(val, 2)
    except Exception:
        return None


def predict_city(models: dict, city: str):
    """
    End-to-end prediction for a city: fetches live data, engineers features,
    and returns point + quantile predictions for all three horizons.

    Returns a dict on success, or {"error": "..."} on failure — callers
    (FastAPI routes, Streamlit pages) decide how to surface that.
    """
    live_df, err = fetch_live_data(city)
    if err or live_df is None or len(live_df) == 0:
        return {"error": err or "No live data returned"}

    featured = engineer_features(live_df)
    latest = featured.iloc[-1]
    current_pm25 = float(latest["pm2_5"])

    result = {
        "city": city,
        "as_of": latest["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
        "current_pm25": round(current_pm25, 2),
        "forecast": {},
    }

    for h in ["1h", "24h", "48h"]:
        pred = _predict_point(models, h, featured)
        lower = get_quantile_pred(models, h, "lower", featured)
        upper = get_quantile_pred(models, h, "upper", featured)
        result["forecast"][h] = {
            "pm2_5": round(pred, 2) if pred is not None else None,
            "range_lower": lower,
            "range_upper": upper,
        }

    return result