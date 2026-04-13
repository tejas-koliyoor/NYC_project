import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from joblib import load
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.iot.features import build_sensor_features
from src.iot.sensor_data import validate_sensor_df

app = FastAPI(
    title="ML Inference API",
    description="NYC Taxi and IoT Sensor Failure Prediction service",
)

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

# Optional: load sensor failure model at startup (path configurable via env)
_SENSOR_MODEL_PATH = Path("models/sensor_failure_model.pkl")
_sensor_model = None
if _SENSOR_MODEL_PATH.exists():
    _sensor_model = load(_SENSOR_MODEL_PATH)


# -----------------------------------------------------------------
# Legacy NYC Taxi endpoint (kept for backwards compatibility)
# -----------------------------------------------------------------

class Payload(BaseModel):
    feature1: float = Field(ge=0)
    feature2: float = Field(ge=0)


@app.get("/health")
def health():
    REQUESTS.labels("/health").inc()
    return {
        "status": "ok",
        "sensor_model_loaded": _sensor_model is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(p: Payload):
    start = time.time()
    REQUESTS.labels("/predict").inc()
    score = float(p.feature1 * 0.1 + p.feature2 * 0.2)
    LATENCY.labels("/predict").observe(time.time() - start)
    return {"score": score}


# -----------------------------------------------------------------
# IoT Sensor Failure Prediction endpoint
# -----------------------------------------------------------------

class SensorReading(BaseModel):
    """A single sensor telemetry reading."""

    timestamp: str = Field(
        description="ISO-8601 datetime string (UTC preferred)",
        examples=["2026-03-12T11:05:21Z"],
    )
    machine_id: str = Field(description="Unique machine identifier")
    temperature: float = Field(description="Temperature in °C")
    vibration: float = Field(ge=0.0, description="Vibration in g")
    pressure: float = Field(ge=0.0, description="Pressure in bar")
    rpm: float = Field(ge=0.0, description="Rotations per minute")
    voltage: float = Field(ge=0.0, description="Voltage in V")
    error_code: int = Field(default=0, ge=0, description="Error code (0 = no error)")


class SensorBatch(BaseModel):
    """A batch of sensor readings for one or more machines."""

    readings: list[SensorReading]


class SensorPrediction(BaseModel):
    machine_id: str
    timestamp: str
    failure_probability: float
    will_fail: bool
    model_available: bool
    message: Optional[str] = None


@app.post("/predict/sensor-failure", response_model=list[SensorPrediction])
def predict_sensor_failure(batch: SensorBatch):
    """Predict whether each machine will fail within the next time window.

    Accepts a batch of sensor readings and returns a failure probability for
    each reading.  If no trained model is available, a heuristic score based
    on the error_code is returned instead.
    """
    start = time.time()
    REQUESTS.labels("/predict/sensor-failure").inc()

    if not batch.readings:
        raise HTTPException(status_code=400, detail="readings list must not be empty")

    # Build raw DataFrame from request
    raw = pd.DataFrame([r.model_dump() for r in batch.readings])

    try:
        df = validate_sensor_df(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    results: list[SensorPrediction] = []

    if _sensor_model is not None:
        X = build_sensor_features(df)
        probs = _sensor_model.predict_proba(X.values)[:, 1]
        for i, (_, row) in enumerate(df.iterrows()):
            prob = float(probs[i])
            results.append(
                SensorPrediction(
                    machine_id=str(row["machine_id"]),
                    timestamp=str(row["timestamp"]),
                    failure_probability=round(prob, 4),
                    will_fail=bool(prob >= 0.5),
                    model_available=True,
                )
            )
    else:
        # Heuristic fallback: use error_code > 0 as a simple signal
        for _, row in df.iterrows():
            prob = min(1.0, float(row["error_code"]) * 0.4)
            results.append(
                SensorPrediction(
                    machine_id=str(row["machine_id"]),
                    timestamp=str(row["timestamp"]),
                    failure_probability=round(prob, 4),
                    will_fail=bool(prob >= 0.5),
                    model_available=False,
                    message="No trained model found; using heuristic score based on error_code.",
                )
            )

    LATENCY.labels("/predict/sensor-failure").observe(time.time() - start)
    return results
