"""
main.py

FastAPI backend, Exposes the same
prediction logic used by dashboard.py, factored out into predictor.py,
as a small JSON API.

Run locally:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from typing import Optional

from predictor import load_models, predict_city, CITIES

app = FastAPI(
    title="AQI Forecasting API",
    description="On-demand PM2.5 forecasts (1h / 24h / 48h) with 80% confidence intervals, for 5 Indian cities.",
    version="1.0.0",
)

# Models are loaded once at startup, not on every request.
models = load_models()


class HorizonForecast(BaseModel):
    pm2_5: Optional[float]
    range_lower: Optional[float]
    range_upper: Optional[float]


class PredictionResponse(BaseModel):
    city: str
    as_of: str
    current_pm25: float
    forecast: dict[str, HorizonForecast]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "cities_supported": list(CITIES.keys()),
    }


@app.get("/predict/{city}", response_model=PredictionResponse)
def predict(city: str):
    city = city.strip().title()
    if city not in CITIES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown city '{city}'. Supported cities: {list(CITIES.keys())}",
        )

    result = predict_city(models, city)
    if "error" in result:
        raise HTTPException(status_code=502, detail=f"Failed to fetch/predict for {city}: {result['error']}")

    return result