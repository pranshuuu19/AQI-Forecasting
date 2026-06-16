"""
Real-Time Fetcher v2 — AQI Forecasting System
===============================================
FIXES:
  - current_datetime always shows the most recent PAST hour correctly
  - No duplicate rows for same hour
  - Actuals filled correctly
  - Kolkata timeout handled with retry
  - Clean 5 rows per run, every hour
"""

import requests # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import xgboost as xgb # type: ignore
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CITIES = {
    "Delhi"    : (28.6139, 77.2090),
    "Mumbai"   : (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai"  : (13.0827, 80.2707),
    "Kolkata"  : (22.5726, 88.3639),
}

PREDICTIONS_LOG = "data/predictions_log.csv"
os.makedirs("data", exist_ok=True)

# ── Fetch with retry ──────────────────────────────────────────────────────────
def fetch_with_retry(url, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
    
# ── Fetch AQI ─────────────────────────────────────────────────────────────────
def fetch_aqi(city, lat, lon, start_date, end_date):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
        f"&timezone=Asia/Kolkata"
    )
    d = fetch_with_retry(url)
    return pd.DataFrame({
        "datetime": pd.to_datetime(d["hourly"]["time"]),
        "city"    : city,
        "pm2_5"   : d["hourly"]["pm2_5"],
        "pm10"    : d["hourly"]["pm10"],
        "co"      : d["hourly"]["carbon_monoxide"],
        "no2"     : d["hourly"]["nitrogen_dioxide"],
        "o3"      : d["hourly"]["ozone"],
        "so2"     : d["hourly"]["sulphur_dioxide"],
    })

# ── Fetch Weather ─────────────────────────────────────────────────────────────
def fetch_weather(city, lat, lon, start_date, end_date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&timezone=Asia/Kolkata"
    )
    d = fetch_with_retry(url)
    return pd.DataFrame({
        "datetime"   : pd.to_datetime(d["hourly"]["time"]),
        "city"       : city,
        "temperature": d["hourly"]["temperature_2m"],
        "humidity"   : d["hourly"]["relative_humidity_2m"],
        "wind_speed" : d["hourly"]["wind_speed_10m"],
        "rainfall"   : d["hourly"]["precipitation"],
    })

# ── Fetch all cities ──────────────────────────────────────────────────────────
def fetch_all_cities():
    # Current time in IST
    now_ist    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    # Most recent completed hour in IST
    current_hour_ist = now_ist.replace(minute=0, second=0, microsecond=0)
    
    # Fetch 6 days back to cover lag48 + rolling24
    end_date = (current_hour_ist + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (current_hour_ist - timedelta(days=6)).strftime("%Y-%m-%d")

    print(f"  Fetching data from {start_date} to {end_date}")
    print(f"  Current IST hour: {current_hour_ist}")

    aqi_frames     = []
    weather_frames = []

    for city, (lat, lon) in CITIES.items():
        print(f"  Fetching {city}...")
        try:
            aqi_df     = fetch_aqi(city, lat, lon, start_date, end_date)
            weather_df = fetch_weather(city, lat, lon, start_date, end_date)
            aqi_frames.append(aqi_df)
            weather_frames.append(weather_df)
        except Exception as e:
            print(f"  ⚠ Failed for {city}: {e}")

    if not aqi_frames:
        raise Exception("All city fetches failed")

    aqi     = pd.concat(aqi_frames,     ignore_index=True)
    weather = pd.concat(weather_frames, ignore_index=True)

    # Merge on city + datetime
    master = pd.merge(aqi, weather, on=["datetime", "city"], how="inner")
    master = master.sort_values(["city", "datetime"]).reset_index(drop=True)

    # ── KEY FIX: Keep only rows UP TO current completed hour ─────────────────
    # Archive API has ~5hr lag so weather only goes up to ~5hrs ago
    # AQI API goes up to current hour
    # Take the minimum latest timestamp across both to ensure both have data
    latest_weather = master.groupby("city")["datetime"].max().min()
    cutoff = min(latest_weather, current_hour_ist)
    
    print(f"  Data cutoff: {cutoff}")
    master = master[master["datetime"] <= cutoff].copy()

    return master, cutoff

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    for col in ["pm2_5", "pm10", "no2", "co", "o3", "so2"]:
        df[f"{col}_lag1"]  = df.groupby("city")[col].shift(1)
        df[f"{col}_lag24"] = df.groupby("city")[col].shift(24)
        df[f"{col}_lag48"] = df.groupby("city")[col].shift(48)

    df["pm2_5_rolling6"]  = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(6,  min_periods=1).mean())
    df["pm2_5_rolling24"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).mean())
    df["pm2_5_change_1h"]  = df.groupby("city")["pm2_5"].diff(1)
    df["pm2_5_change_24h"] = df.groupby("city")["pm2_5"].diff(24)
    df["pm2_5_rolling6_std"]  = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(6,  min_periods=1).std().fillna(0))
    df["pm2_5_rolling24_max"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).max())
    df["pm2_5_rolling24_min"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).min())

    df["humidity_pm25"]         = df["humidity"] * df["pm2_5"]
    df["wind_pm25_ratio"]       = df["pm2_5"] / (df["wind_speed"] + 1)
    df["temp_wind_interaction"]  = df["temperature"] * df["wind_speed"]

    city_means = df.groupby("city")["pm2_5"].transform("mean")
    city_stds  = df.groupby("city")["pm2_5"].transform("std")
    df["is_pollution_spike"] = (df["pm2_5"] > city_means + 2 * city_stds).astype(int)

    df["date_str"] = df["datetime"].dt.date.astype(str)
    diwali_exact  = ["2025-10-20", "2025-10-21"]
    diwali_window = ["2025-10-17","2025-10-18","2025-10-19",
                     "2025-10-20","2025-10-21",
                     "2025-10-22","2025-10-23","2025-10-24"]
    df["is_diwali"]          = df["date_str"].isin(diwali_exact).astype(int)
    df["is_diwali_window"]   = df["date_str"].isin(diwali_window).astype(int)
    df["is_stubble_burning"] = df["month"].isin([10, 11]).astype(int)
    df = df.drop(columns=["date_str"])

    city_order = {"Bengaluru": 0, "Chennai": 1, "Delhi": 2, "Kolkata": 3, "Mumbai": 4}
    df["city_encoded"] = df["city"].map(city_order)

    def hour_bucket(h):
        if 6 <= h <= 9:     return "morning_rush"
        elif 10 <= h <= 16: return "daytime"
        elif 17 <= h <= 20: return "evening_rush"
        else:               return "night"

    df["hour_bucket"] = df["hour"].apply(hour_bucket)
    df = pd.get_dummies(df, columns=["hour_bucket"], dtype=int)

    for bucket in ["hour_bucket_daytime","hour_bucket_evening_rush",
                   "hour_bucket_morning_rush","hour_bucket_night"]:
        if bucket not in df.columns:
            df[bucket] = 0

    return df

# ── Load models ───────────────────────────────────────────────────────────────
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

def load_features():
    with open("data/feature_config.json") as f:
        return json.load(f)["feature_cols"]

# ── Make predictions ──────────────────────────────────────────────────────────
def make_predictions(df, models, features, cutoff):
    # Take EXACTLY the cutoff hour row per city
    latest = df[df["datetime"] == cutoff].copy()
    
    # If cutoff hour missing for some city, take their latest available
    cities_missing = set(CITIES.keys()) - set(latest["city"].values)
    if cities_missing:
        for city in cities_missing:
            city_rows = df[df["city"] == city]
            if len(city_rows) > 0:
                latest = pd.concat([latest, city_rows.iloc[[-1]]], ignore_index=True)

    prediction_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")

    results = []
    for _, row in latest.iterrows():
        # Ensure all features present
        for f in features:
            if f not in row.index:
                row[f] = 0
        x_row = pd.DataFrame([row[features]]).fillna(0)

        pred_row = {
            "prediction_time" : (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            "city"            : row["city"],
            "current_pm25"    : round(row["pm2_5"], 2),
            "current_datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_1h"  : (row["datetime"] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_24h" : (row["datetime"] + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_48h" : (row["datetime"] + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S"),
            "pred_1h"  : round(float(models["1h"].predict(x_row)[0]),  2) if "1h"  in models else None,
            "pred_24h" : round(float(models["24h"].predict(x_row)[0]), 2) if "24h" in models else None,
            "pred_48h" : round(float(models["48h"].predict(x_row)[0]), 2) if "48h" in models else None,
            "actual_1h" : None, "actual_24h": None, "actual_48h": None,
            "error_1h"  : None, "error_24h" : None, "error_48h" : None,
        }
        results.append(pred_row)
        print(f"  {row['city']:10} | {row['datetime']} | "
              f"Current: {row['pm2_5']:.1f} | "
              f"1h: {pred_row['pred_1h']} | "
              f"24h: {pred_row['pred_24h']} | "
              f"48h: {pred_row['pred_48h']} μg/m³")

    return pd.DataFrame(results)

# ── Fill actuals retrospectively ──────────────────────────────────────────────
def fill_actuals(log_df, fresh_df):
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)

    for idx, row in log_df.iterrows():
        city      = row["city"]
        city_data = fresh_df[fresh_df["city"] == city].copy()

        for horizon in ["1h", "24h", "48h"]:
            if pd.notna(row.get(f"actual_{horizon}")) or pd.isna(row.get(f"pred_{horizon}")):
                continue

            target_time = pd.to_datetime(row[f"target_time_{horizon}"])
            if target_time > now_ist:
                continue

            # Find closest actual within 30 min
            city_data["time_diff"] = (city_data["datetime"] - target_time).abs()
            closest = city_data.nsmallest(1, "time_diff")

            if len(closest) > 0 and closest.iloc[0]["time_diff"] < timedelta(minutes=31):
                actual = round(float(closest.iloc[0]["pm2_5"]), 2)
                error  = round(float(row[f"pred_{horizon}"]) - actual, 2)
                log_df.at[idx, f"actual_{horizon}"] = actual
                log_df.at[idx, f"error_{horizon}"]  = error

    return log_df

# ── Deduplicate log ───────────────────────────────────────────────────────────
def deduplicate_log(log_df):
    """Remove duplicate rows for same city + current_datetime — keep latest."""
    log_df = log_df.sort_values("prediction_time")
    log_df = log_df.drop_duplicates(
        subset=["city", "current_datetime"], keep="last"
    ).reset_index(drop=True)
    return log_df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    print(f"\n{'='*55}")
    print(f"  AQI Real-Time Fetcher v2 — {now_ist.strftime('%Y-%m-%d %H:%M')} IST")
    print(f"{'='*55}")

    print("\n[1/5] Fetching latest data...")
    fresh_df, cutoff = fetch_all_cities()
    print(f"  {len(fresh_df)} rows fetched, cutoff hour: {cutoff}")

    print("\n[2/5] Engineering features...")
    featured_df = engineer_features(fresh_df)

    print("\n[3/5] Loading models...")
    models   = load_models()
    features = load_features()
    print(f"  Models loaded: {list(models.keys())}")

    print("\n[4/5] Making predictions for cutoff hour...")
    new_preds = make_predictions(featured_df, models, features, cutoff)

    print("\n[5/5] Updating log + filling actuals + deduplicating...")
    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        log_df   = pd.concat([existing, new_preds], ignore_index=True)
    else:
        log_df = new_preds

    log_df = deduplicate_log(log_df)
    log_df = fill_actuals(log_df, fresh_df)
    log_df = deduplicate_log(log_df)  # dedupe again after actuals filled
    log_df.to_csv(PREDICTIONS_LOG, index=False)

    total     = len(log_df)
    evaluated = log_df["actual_1h"].notna().sum()
    print(f"\n  Total rows in log : {total}")
    print(f"  1h actuals filled : {evaluated}")
    if evaluated > 0:
        errors = log_df["error_1h"].dropna()
        rmse   = np.sqrt((errors**2).mean())
        print(f"  Live 1h RMSE      : {rmse:.4f} μg/m³")

    print(f"\n✓ Done. Log saved to {PREDICTIONS_LOG}")