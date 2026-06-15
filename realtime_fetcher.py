import requests # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import joblib # type: ignore
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

# ── Step 1: Fetch latest 3 days from Open-Meteo ───────────────────────────────
def fetch_aqi(city, lat, lon, start_date, end_date):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
        f"&timezone=Asia/Kolkata"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()
    return pd.DataFrame({
        "datetime": d["hourly"]["time"],
        "city"    : city,
        "pm2_5"   : d["hourly"]["pm2_5"],
        "pm10"    : d["hourly"]["pm10"],
        "co"      : d["hourly"]["carbon_monoxide"],
        "no2"     : d["hourly"]["nitrogen_dioxide"],
        "o3"      : d["hourly"]["ozone"],
        "so2"     : d["hourly"]["sulphur_dioxide"],
    })

def fetch_weather(city, lat, lon, start_date, end_date):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&timezone=Asia/Kolkata"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()
    return pd.DataFrame({
        "datetime"   : d["hourly"]["time"],
        "city"       : city,
        "temperature": d["hourly"]["temperature_2m"],
        "humidity"   : d["hourly"]["relative_humidity_2m"],
        "wind_speed" : d["hourly"]["wind_speed_10m"],
        "rainfall"   : d["hourly"]["precipitation"],
    })

def fetch_all_cities():
    # Fetch 5 days back to ensure enough history for lag48
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    aqi_frames     = []
    weather_frames = []

    for city, (lat, lon) in CITIES.items():
        print(f"  Fetching {city}...")
        try:
            aqi_frames.append(fetch_aqi(city, lat, lon, start_date, end_date))
            weather_frames.append(fetch_weather(city, lat, lon, start_date, end_date))
        except Exception as e:
            print(f"  ⚠ Failed for {city}: {e}")

    aqi     = pd.concat(aqi_frames,     ignore_index=True)
    weather = pd.concat(weather_frames, ignore_index=True)

    # Merge exactly like your merger script
    master = pd.merge(aqi, weather, on=["datetime", "city"], how="inner")
    master = master.sort_values(["city", "datetime"]).reset_index(drop=True)
    master["datetime"] = pd.to_datetime(master["datetime"])
    return master

# ── Step 2: Feature engineering (mirrors your feature engineering script) ─────
def engineer_features(df):
    df = df.copy()

    # Time features
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Lag features — computed per city
    for col in ["pm2_5", "pm10", "no2", "co", "o3", "so2"]:
        df[f"{col}_lag1"]  = df.groupby("city")[col].shift(1)
        df[f"{col}_lag24"] = df.groupby("city")[col].shift(24)
        df[f"{col}_lag48"] = df.groupby("city")[col].shift(48)

    # Rolling features
    df["pm2_5_rolling6"]  = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(6,  min_periods=1).mean())
    df["pm2_5_rolling24"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).mean())

    # Rate of change
    df["pm2_5_change_1h"]  = df.groupby("city")["pm2_5"].diff(1)
    df["pm2_5_change_24h"] = df.groupby("city")["pm2_5"].diff(24)

    # Rolling std/max/min
    df["pm2_5_rolling6_std"]  = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(6,  min_periods=1).std().fillna(0))
    df["pm2_5_rolling24_max"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).max())
    df["pm2_5_rolling24_min"] = df.groupby("city")["pm2_5"].transform(
        lambda x: x.rolling(24, min_periods=1).min())

    # Interaction features
    df["humidity_pm25"]        = df["humidity"] * df["pm2_5"]
    df["wind_pm25_ratio"]      = df["pm2_5"] / (df["wind_speed"] + 1)
    df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"]

    # Pollution spike flag
    city_means = df.groupby("city")["pm2_5"].transform("mean")
    city_stds  = df.groupby("city")["pm2_5"].transform("std")
    df["is_pollution_spike"] = (df["pm2_5"] > city_means + 2 * city_stds).astype(int)

    # Festival flags
    df["date_str"] = df["datetime"].dt.date.astype(str)
    diwali_exact  = ["2025-10-20", "2025-10-21"]
    diwali_window = ["2025-10-17","2025-10-18","2025-10-19",
                     "2025-10-20","2025-10-21",
                     "2025-10-22","2025-10-23","2025-10-24"]
    df["is_diwali"]         = df["date_str"].isin(diwali_exact).astype(int)
    df["is_diwali_window"]  = df["date_str"].isin(diwali_window).astype(int)
    df["is_stubble_burning"] = df["month"].isin([10, 11]).astype(int)
    df = df.drop(columns=["date_str"])

    # City encoding — same order as training
    city_order = {"Bengaluru": 0, "Chennai": 1, "Delhi": 2, "Kolkata": 3, "Mumbai": 4}
    df["city_encoded"] = df["city"].map(city_order)

    # Hour buckets
    def hour_bucket(h):
        if 6 <= h <= 9:     return "morning_rush"
        elif 10 <= h <= 16: return "daytime"
        elif 17 <= h <= 20: return "evening_rush"
        else:               return "night"

    df["hour_bucket"] = df["hour"].apply(hour_bucket)
    df = pd.get_dummies(df, columns=["hour_bucket"], dtype=int)

    # Ensure all hour bucket columns exist even if some buckets absent in small window
    for bucket in ["hour_bucket_daytime","hour_bucket_evening_rush",
                   "hour_bucket_morning_rush","hour_bucket_night"]:
        if bucket not in df.columns:
            df[bucket] = 0

    return df

# ── Step 3: Load models + feature config ──────────────────────────────────────
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

# ── Step 4: Make predictions for latest row per city ─────────────────────────
def make_predictions(df, models, features):
    # Take the most recent row per city
    latest = df.sort_values("datetime").groupby("city").tail(1).copy()

    # Align feature columns
    for f in features:
        if f not in latest.columns:
            latest[f] = 0
    X = latest[features].fillna(0)

    results = []
    for _, row in latest.iterrows():
        x_row = pd.DataFrame([row[features].fillna(0)])
        pred_row = {
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city"           : row["city"],
            "current_pm25"   : round(row["pm2_5"], 2),
            "current_datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_1h" : (row["datetime"] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_24h": (row["datetime"] + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),
            "target_time_48h": (row["datetime"] + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S"),
            "pred_1h"        : round(float(models["1h"].predict(x_row)[0]), 2) if "1h"  in models else None,
            "pred_24h"       : round(float(models["24h"].predict(x_row)[0]), 2) if "24h" in models else None,
            "pred_48h"       : round(float(models["48h"].predict(x_row)[0]), 2) if "48h" in models else None,
            "actual_1h"      : None,  # filled in retrospectively
            "actual_24h"     : None,
            "actual_48h"     : None,
            "error_1h"       : None,
            "error_24h"      : None,
            "error_48h"      : None,
        }
        results.append(pred_row)
        print(f"  {row['city']:10} | Current: {row['pm2_5']:.1f} | "
              f"1h: {pred_row['pred_1h']} | 24h: {pred_row['pred_24h']} | "
              f"48h: {pred_row['pred_48h']} μg/m³")

    return pd.DataFrame(results)

# ── Step 5: Retrospective evaluation ─────────────────────────────────────────
def fill_actuals(log_df, fresh_df):
    """
    For each past prediction whose target_time has now passed,
    find the actual pm2_5 from fresh data and compute the error.
    """
    now = datetime.now()

    for idx, row in log_df.iterrows():
        city      = row["city"]
        city_data = fresh_df[fresh_df["city"] == city].copy()

        for horizon in ["1h", "24h", "48h"]:
            # Skip if already filled or prediction missing
            if pd.notna(row.get(f"actual_{horizon}")) or pd.isna(row.get(f"pred_{horizon}")):
                continue

            target_time = pd.to_datetime(row[f"target_time_{horizon}"])

            # Only evaluate if target time has passed
            if target_time > now:
                continue

            # Find actual value — closest row within 30 min of target time
            city_data["time_diff"] = (city_data["datetime"] - target_time).abs()
            closest = city_data.nsmallest(1, "time_diff")

            if len(closest) > 0 and closest.iloc[0]["time_diff"] < timedelta(minutes=30):
                actual = round(closest.iloc[0]["pm2_5"], 2)
                error  = round(float(row[f"pred_{horizon}"]) - actual, 2)
                log_df.at[idx, f"actual_{horizon}"] = actual
                log_df.at[idx, f"error_{horizon}"]  = error

    return log_df

# ── Step 6: Save updated log ──────────────────────────────────────────────────
def update_log(new_preds):
    if os.path.exists(PREDICTIONS_LOG):
        existing = pd.read_csv(PREDICTIONS_LOG)
        updated  = pd.concat([existing, new_preds], ignore_index=True)
    else:
        updated = new_preds
    updated.to_csv(PREDICTIONS_LOG, index=False)
    return updated

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  AQI Real-Time Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    print("\n[1/5] Fetching latest data from Open-Meteo...")
    fresh_df = fetch_all_cities()
    print(f"  Fetched {len(fresh_df)} rows across {fresh_df['city'].nunique()} cities")

    print("\n[2/5] Engineering features...")
    featured_df = engineer_features(fresh_df)
    print("  Done")

    print("\n[3/5] Loading models...")
    models   = load_models()
    features = load_features()
    print(f"  Loaded models: {list(models.keys())}")

    print("\n[4/5] Making predictions...")
    new_preds = make_predictions(featured_df, models, features)

    print("\n[5/5] Updating prediction log + filling actuals...")
    log_df  = update_log(new_preds)
    log_df  = fill_actuals(log_df, fresh_df)
    log_df.to_csv(PREDICTIONS_LOG, index=False)

    # Summary
    total      = len(log_df)
    evaluated  = log_df["actual_1h"].notna().sum()
    if evaluated > 0:
        rmse_live = np.sqrt((log_df["error_1h"].dropna() ** 2).mean())
        print(f"\n  Predictions logged : {total}")
        print(f"  1h predictions evaluated so far: {evaluated}")
        print(f"  Live 1h RMSE so far: {rmse_live:.4f} μg/m³")
    else:
        print(f"\n  Predictions logged: {total}")
        print(f"  No actuals available yet — check back in 1 hour")

    print(f"\n✓ Done. Log saved to {PREDICTIONS_LOG}")