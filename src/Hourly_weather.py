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
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date=2025-01-01"
        f"&end_date=2026-06-07"
        f"&hourly="
        f"temperature_2m,"
        f"relative_humidity_2m,"
        f"wind_speed_10m,"
        f"precipitation"
        f"&timezone=Asia/Kolkata"
    )

    response = requests.get(url)

    print(f"{city} Status:", response.status_code)

    data = response.json()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "city": city,
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "rainfall": data["hourly"]["precipitation"]
    })

    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

final_df.to_csv(
    "data/raw/weather_historical_5cities_hourly.csv",
    index=False
)

print(final_df.head())
print("Shape:", final_df.shape)
print("Hourly Weather CSV Created Successfully")