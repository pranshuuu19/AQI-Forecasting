import pandas as pd # type: ignore
import numpy as np # type: ignore

df = pd.read_csv("data/raw/master.csv")

# Step 1: fix datetime
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

# Step 2: confirm no missing values
print("Missing values:\n", df.isnull().sum())

# Step 3: time features
df["hour"]        = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"]       = df["datetime"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Step 4: lag features
for col in ["pm2_5", "pm10", "no2", "co", "o3", "so2"]:
    df[f"{col}_lag1"]  = df.groupby("city")[col].shift(1)
    df[f"{col}_lag24"] = df.groupby("city")[col].shift(24)
    df[f"{col}_lag48"] = df.groupby("city")[col].shift(48)

# Step 5: rolling features
df["pm2_5_rolling6"]  = df.groupby("city")["pm2_5"].transform(lambda x: x.rolling(6).mean())
df["pm2_5_rolling24"] = df.groupby("city")["pm2_5"].transform(lambda x: x.rolling(24).mean())

# Step 6: AQI label
def aqi_category(pm25):
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"

df["aqi_label"] = df["pm2_5"].apply(aqi_category)

# Step 7: forecast targets
df["target_1h"]  = df.groupby("city")["pm2_5"].shift(-1)
df["target_24h"] = df.groupby("city")["pm2_5"].shift(-24)
df["target_48h"] = df.groupby("city")["pm2_5"].shift(-48)

# Step 8: drop rows with nulls from lag/target creation
df = df.dropna().reset_index(drop=True)

# Step 9: save
import os
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/master_processed.csv", index=False)

print("\nShape after preprocessing:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nAQI label distribution:\n", df["aqi_label"].value_counts())
print("\nSample:")
print(df[["datetime", "city", "pm2_5", "aqi_label", "target_1h", "target_24h", "target_48h"]].head(10))