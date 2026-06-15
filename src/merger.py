import pandas as pd # type: ignore

# load both CSVs
weather = pd.read_csv("data/raw/weather_historical_5cities_hourly.csv")
aqi     = pd.read_csv("data/raw/aqi_historical_5cities_hourly.csv")

# merge on datetime + city
master = pd.merge(aqi, weather, on=["datetime", "city"], how="inner")

# sort by city and datetime
master = master.sort_values(["city", "datetime"]).reset_index(drop=True)

# save
master.to_csv("data/raw/master.csv", index=False)

print("Shape:", master.shape)
print("Columns:", master.columns.tolist())
print(master.head(10))