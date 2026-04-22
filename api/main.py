from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import TrafficForecastService


class TrafficRow(BaseModel):
    period: Optional[str] = Field(default=None, description="Optional date in ISO format")
    direct: float = Field(ge=0)
    search: float = Field(ge=0)
    social: float = Field(ge=0)
    referral: float = Field(ge=0)


class PredictRequest(BaseModel):
    history: list[TrafficRow]


class PredictResponse(BaseModel):
    prediction: float
    target: str
    model_version: str


@lru_cache(maxsize=1)
def get_service() -> TrafficForecastService:
    artifacts_dir = os.getenv("MODEL_DIR", "artifacts")
    return TrafficForecastService(artifacts_dir=artifacts_dir)


app = FastAPI(
    title="Traffic Forecast API",
    version="1.0.0",
    description="Inference API for the Conv1D traffic forecasting model.",
)


@app.get("/health")
def health() -> dict:
    artifacts_dir = os.getenv("MODEL_DIR", "artifacts")
    required = ["model.keras", "x_scaler.joblib", "y_scaler.joblib", "metadata.json"]
    missing = [name for name in required if not os.path.exists(os.path.join(artifacts_dir, name))]

    if missing:
        return {
            "status": "not_ready",
            "message": "Model artifacts are missing. Run `python train.py` first.",
            "missing_files": missing,
            "artifacts_dir": artifacts_dir,
        }

    service = get_service()
    return {
        "status": "ok",
        "model_version": service.model_version,
        "target": service.target_column,
        "window_size": service.window_size,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        service = get_service()
        result = service.predict_from_records([row.model_dump() for row in request.history])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(**result.__dict__)
