from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

app = FastAPI(title="NER Landslide Early Warning System")
app = FastAPI(title="NER Landslide Early Warning System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "landslide_model.pkl")
model = joblib.load(MODEL_PATH)

class RiskInput(BaseModel):
    rainfall_mm: float
    rainfall_intensity_mmhr: float
    rainfall_3day_cumulative: float
    rainfall_15day_cumulative: float
    temperature_c: float
    wind_speed_kmh: float
    soil_moisture: float
    groundwater_level: float
    slope_angle: float
    elevation_m: float
    distance_to_river_m: float
    seismic_activity_score: float
    vegetation_ndvi: float
    soil_weakness_score: float

@app.post("/predict")
def predict_risk(data: RiskInput):
    features = np.array([[
        data.rainfall_mm,
        data.rainfall_intensity_mmhr,
        data.rainfall_3day_cumulative,
        data.rainfall_15day_cumulative,
        data.temperature_c,
        data.wind_speed_kmh,
        data.soil_moisture,
        data.groundwater_level,
        data.slope_angle,
        data.elevation_m,
        data.distance_to_river_m,
        data.seismic_activity_score,
        data.vegetation_ndvi,
        data.soil_weakness_score
    ]])

    probability = model.predict_proba(features)[0]
    high_prob = float(probability[1]) * 100  # probability of class 1 (landslide risk)

    if high_prob >= 35:
        risk_level = "HIGH"
    elif high_prob >= 15:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "prediction": int(high_prob >= 35),
        "risk_level": risk_level,
        "confidence_percent": round(high_prob, 2)
    }
