import requests # type: ignore
import pandas as pd # type: ignore

cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}

all_data = []

for city, (lat, lon) in cities.items():

    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date=2025-01-01"
        f"&end_date=2026-06-07"
        f"&hourly="
        f"pm2_5,"
        f"pm10,"
        f"carbon_monoxide,"
        f"nitrogen_dioxide,"
        f"ozone,"
        f"sulphur_dioxide"
        f"&timezone=Asia/Kolkata"
    )

    response = requests.get(url)

    print(f"{city} Status:", response.status_code)

    data = response.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "city": city,
        "pm2_5": data["hourly"]["pm2_5"],
        "pm10": data["hourly"]["pm10"],
        "co": data["hourly"]["carbon_monoxide"],
        "no2": data["hourly"]["nitrogen_dioxide"],
        "o3": data["hourly"]["ozone"],
        "so2": data["hourly"]["sulphur_dioxide"]
    })

    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv(
    "data/raw/aqi_historical_5cities_hourly.csv",
    index=False
)

print(final_df.head())
print("Shape:", final_df.shape)
print("AQI Historical Hourly CSV Created Successfully")