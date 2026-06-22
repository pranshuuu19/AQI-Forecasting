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
df["humidity_pm25"]          = df["humidity"] * df["pm2_5"]
df["wind_pm25_ratio"]        = df["pm2_5"] / (df["wind_speed"] + 1)
df["temp_wind_interaction"]  = df["temperature"] * df["wind_speed"]
print("✅ Interaction features added")

# ── ADD 4: NEW — Wind direction circular encoding ─────────────
# Wind direction is circular (359° and 1° are close, not far apart)
# so we encode it as sin and cos components instead of raw degrees
if "wind_direction" in df.columns:
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction"]))
    print("✅ Wind direction circular encoding added")
else:
    df["wind_dir_sin"] = 0.0
    df["wind_dir_cos"] = 0.0
    print("⚠ wind_direction column not found — defaulting to 0")

# ── ADD 5: NEW — Boundary layer height features ───────────────
# Low boundary layer = pollutants trapped near surface = high PM2.5
# This is especially important for Delhi evening/night spikes
if "boundary_layer_height" in df.columns:
    # Inverse — lower BLH = higher pollution trapping
    df["blh_inverse"] = 1 / (df["boundary_layer_height"] + 1)
    # Interaction with pm2_5 — captures trapping effect
    df["blh_pm25_interaction"] = df["pm2_5"] / (df["boundary_layer_height"] + 1)
    # Rolling min BLH over last 6h — sustained low BLH = sustained trapping
    df["blh_rolling6_min"] = df.groupby("city")["boundary_layer_height"].transform(
        lambda x: x.rolling(6, min_periods=1).min())
    print("✅ Boundary layer height features added")
else:
    df["blh_inverse"]         = 0.0
    df["blh_pm25_interaction"] = 0.0
    df["blh_rolling6_min"]    = 0.0
    print("⚠ boundary_layer_height column not found — defaulting to 0")

# ── ADD 6: NEW — Surface pressure features ────────────────────
# Low pressure = poor ventilation = pollution builds up
if "surface_pressure" in df.columns:
    # Pressure change over last hour — sudden drops signal bad air quality
    df["pressure_change_1h"] = df.groupby("city")["surface_pressure"].diff(1)
    # Pressure change over last 6h — sustained low pressure trend
    df["pressure_change_6h"] = df.groupby("city")["surface_pressure"].diff(6)
    print("✅ Surface pressure features added")
else:
    df["pressure_change_1h"] = 0.0
    df["pressure_change_6h"] = 0.0
    print("⚠ surface_pressure column not found — defaulting to 0")

# ── ADD 7: Pollution spike flag ───────────────────────────────
city_means = df.groupby("city")["pm2_5"].transform("mean")
city_stds  = df.groupby("city")["pm2_5"].transform("std")
df["is_pollution_spike"] = (
    df["pm2_5"] > city_means + 2 * city_stds
).astype(int)
print("✅ Pollution spike flag added")

# ── ADD 8: Festival flags ─────────────────────────────────────
diwali_exact = ["2025-10-20", "2025-10-21"]
diwali_window = [
    "2025-10-17", "2025-10-18", "2025-10-19",
    "2025-10-20", "2025-10-21",
    "2025-10-22", "2025-10-23", "2025-10-24",
]
df["date_str"]          = df["datetime"].dt.date.astype(str)
df["is_diwali"]         = df["date_str"].isin(diwali_exact).astype(int)
df["is_diwali_window"]  = df["date_str"].isin(diwali_window).astype(int)
df["is_stubble_burning"] = df["month"].isin([10, 11]).astype(int)
df = df.drop(columns=["date_str"])
print("✅ Festival flags added")

# ── ADD 9: City encoding ──────────────────────────────────────
le = LabelEncoder()
df["city_encoded"] = le.fit_transform(df["city"])
os.makedirs("model", exist_ok=True)
joblib.dump(le, "model/city_encoder.pkl")
print("✅ City encoding added")

# ── ADD 10: Hour buckets ──────────────────────────────────────
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