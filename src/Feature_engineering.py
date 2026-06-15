import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import joblib # type: ignore
import os

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("data/processed/master_processed.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
print("Loaded shape:", df.shape)

# ── ADD 1: Rate of change ─────────────────────────────────────
df["pm2_5_change_1h"]  = df.groupby("city")["pm2_5"].diff(1)
df["pm2_5_change_24h"] = df.groupby("city")["pm2_5"].diff(24)
print("✅ Rate of change added")

# ── ADD 2: Rolling std, max, min ──────────────────────────────
df["pm2_5_rolling6_std"]  = df.groupby("city")["pm2_5"].transform(
    lambda x: x.rolling(6).std())
df["pm2_5_rolling24_max"] = df.groupby("city")["pm2_5"].transform(
    lambda x: x.rolling(24).max())
df["pm2_5_rolling24_min"] = df.groupby("city")["pm2_5"].transform(
    lambda x: x.rolling(24).min())
print("✅ Rolling std/max/min added")

# ── ADD 3: Interaction features ───────────────────────────────
df["humidity_pm25"]         = df["humidity"] * df["pm2_5"]
df["wind_pm25_ratio"]       = df["pm2_5"] / (df["wind_speed"] + 1)
df["temp_wind_interaction"]  = df["temperature"] * df["wind_speed"]
print("✅ Interaction features added")

# ── ADD 4: Pollution spike flag ───────────────────────────────
city_means = df.groupby("city")["pm2_5"].transform("mean")
city_stds  = df.groupby("city")["pm2_5"].transform("std")
df["is_pollution_spike"] = (
    df["pm2_5"] > city_means + 2 * city_stds
).astype(int)
print("✅ Pollution spike flag added")

# ── ADD 5: Festival flags ─────────────────────────────────────

# Diwali exact dates
diwali_exact = [
    "2025-10-20", "2025-10-21",  # Diwali 2025
]

# 3 days before and after Diwali (the full pollution window)
diwali_window = [
    "2025-10-17", "2025-10-18", "2025-10-19",  # 3 days before
    "2025-10-20", "2025-10-21",                 # Diwali itself
    "2025-10-22", "2025-10-23", "2025-10-24",  # 3 days after
]

# create a date string column to compare against
df["date_str"] = df["datetime"].dt.date.astype(str)

# 1 if date is exactly Diwali, else 0
df["is_diwali"] = df["date_str"].isin(diwali_exact).astype(int)

# 1 if date is within 3 days of Diwali, else 0
df["is_diwali_window"] = df["date_str"].isin(diwali_window).astype(int)

# 1 if month is October or November (stubble burning season), else 0
df["is_stubble_burning"] = df["month"].isin([10, 11]).astype(int)

# drop the helper column — no longer needed
df = df.drop(columns=["date_str"])

print("✅ Festival flags added")

# ── ADD 6: City encoding ──────────────────────────────────────
le = LabelEncoder()
df["city_encoded"] = le.fit_transform(df["city"])
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/city_encoder.pkl")
print("✅ City encoding added")

# ── ADD 7: Hour buckets ───────────────────────────────────────
def hour_bucket(hour):
    if 6 <= hour <= 9:     return "morning_rush"
    elif 10 <= hour <= 16: return "daytime"
    elif 17 <= hour <= 20: return "evening_rush"
    else:                  return "night"

df["hour_bucket"] = df["hour"].apply(hour_bucket)
df = pd.get_dummies(df, columns=["hour_bucket"], dtype=int)
print("✅ Hour buckets added")

# ── Drop nulls from diff/rolling ─────────────────────────────
before = len(df)
df = df.dropna().reset_index(drop=True)
after = len(df)
print(f"✅ Dropped {before - after} null rows from diff/rolling")

# ── Save ──────────────────────────────────────────────────────
df.to_csv("data/processed/master_processed.csv", index=False)

print("\n── Final output ──────────────────────────────────────")
print("Shape:", df.shape)
print("Total features:", len(df.columns))
print("Columns:", df.columns.tolist())