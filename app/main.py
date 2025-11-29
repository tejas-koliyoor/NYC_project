# --- imports at top ---
import time
import pandas as pd
from src.privacy import scrub_pii_from_record
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from .schemas import HealthResponse, PredictRequest, PredictResponse, Prediction
from .model import load_pipeline, ModelNotFound

# --- app + metrics objects ---
APP_VERSION = "v1"
app = FastAPI(title="NYC Taxi Duration API", version=APP_VERSION)

PREDICTION_REQUESTS = Counter("prediction_requests_total", "Number of /predict calls")
PREDICTION_ERRORS   = Counter("prediction_errors_total", "Number of failed /predict calls")
PREDICTION_LATENCY  = Histogram(
    "prediction_request_latency_seconds", "Latency of /predict",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
MODEL_LOADED        = Gauge("model_loaded", "Whether model is loaded (1=yes,0=no)")

# --- health updates the gauge ---
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _ = load_pipeline(); loaded = True
    except Exception:
        loaded = False
    MODEL_LOADED.set(1.0 if loaded else 0.0)
    return HealthResponse(status="ok", model_loaded=loaded, version=APP_VERSION)

# --- /metrics endpoint (this is what was missing) ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- /predict (instrumented) ---
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    PREDICTION_REQUESTS.inc()
    try:
        pipeline = load_pipeline()
        if not payload.records:
            raise HTTPException(status_code=400, detail="No records provided")
        clean_records = [scrub_pii_from_record(r) for r in payload.records]
        df = pd.DataFrame(clean_records)
        yhat = pipeline.predict(df)
        return PredictResponse(predictions=[
            Prediction(duration_min_predicted=float(v)) for v in yhat
        ])
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
