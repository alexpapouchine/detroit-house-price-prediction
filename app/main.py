"""
Detroit House Price Prediction API
-----------------------------------
FastAPI application that loads a pre-trained scikit-learn / XGBoost pipeline
ONCE at startup and serves real-time predictions via a POST endpoint.

Per assignment restrictions:
- The model is never trained at request time (model.fit() is never called here).
- The model + preprocessing pipeline are unpickled (joblib.load) at startup.
- Access is protected with a simple API key checked via request header.
"""

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "detroit_house_price_model.joblib"
SCHEMA_PATH = BASE_DIR / "model" / "feature_schema.json"

# In production this should come from a secret manager (GCP Secret Manager).
# For this assignment it is read from an environment variable with a
# development fallback so the app is runnable out-of-the-box locally/in Docker.
API_KEY = os.environ.get("API_KEY", "dev-detroit-housing-key-2026")

# ---------------------------------------------------------------------------
# Load model + schema ONCE at application startup (not per-request)
# ---------------------------------------------------------------------------
model_pipeline = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

VALID_NEIGHBORHOODS = set(schema["neighborhoods"])
NUMERIC_FEATURES = schema["numeric_features"]
CATEGORICAL_FEATURES = schema["categorical_features"]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Detroit House Price Prediction API",
    description="Predicts residential sale price for a Detroit property given "
                 "structural, location, and condition features.",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    neighborhood: str = Field(..., description="Detroit neighborhood name, e.g. 'Midtown'")
    sqft: float = Field(..., gt=0, description="Above-grade square footage")
    bedrooms: int = Field(..., ge=0, le=15)
    bathrooms: float = Field(..., ge=0, le=10)
    lot_size_sqft: float = Field(..., gt=0)
    year_built: int = Field(..., ge=1800, le=2026)
    stories: float = Field(..., ge=0.5, le=5)
    garage_spaces: Optional[float] = Field(0, ge=0, le=6)
    has_basement: int = Field(..., ge=0, le=1)
    has_porch: int = Field(..., ge=0, le=1)
    renovated_last_10yrs: int = Field(..., ge=0, le=1)
    fireplace: int = Field(..., ge=0, le=1)
    condition_score: Optional[float] = Field(None, ge=1, le=10)
    distance_to_downtown_km: float = Field(..., ge=0, le=100)
    crime_index: float = Field(..., ge=0, le=100)
    school_rating: Optional[float] = Field(None, ge=1, le=10)
    walk_score: Optional[float] = Field(None, ge=0, le=100)
    property_tax_annual: float = Field(..., ge=0)
    vacant_lot_nearby: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "neighborhood": "Midtown",
                "sqft": 1450,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "lot_size_sqft": 4200,
                "year_built": 1965,
                "stories": 2,
                "garage_spaces": 1,
                "has_basement": 1,
                "has_porch": 1,
                "renovated_last_10yrs": 1,
                "fireplace": 0,
                "condition_score": 7.2,
                "distance_to_downtown_km": 2.5,
                "crime_index": 35,
                "school_rating": 6.5,
                "walk_score": 78,
                "property_tax_annual": 2100,
                "vacant_lot_nearby": 0,
            }
        }


class PredictionResponse(BaseModel):
    predicted_sale_price: float
    currency: str = "USD"
    model_version: str = "xgboost-v1.0"


def verify_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it via the 'x-api-key' header.",
        )


@app.get("/")
def root():
    return {
        "service": "Detroit House Price Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Liveness/readiness probe for Cloud Run."""
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    if features.neighborhood not in VALID_NEIGHBORHOODS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown neighborhood '{features.neighborhood}'. "
                   f"Valid options: {sorted(VALID_NEIGHBORHOODS)}",
        )

    input_dict = features.model_dump()
    input_df = pd.DataFrame([input_dict])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    prediction = model_pipeline.predict(input_df)[0]

    return PredictionResponse(predicted_sale_price=round(float(prediction), 2))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
