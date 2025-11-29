import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from src.privacy import scrub_pii_from_record
from .schemas import HealthResponse, PredictRequest, PredictResponse, Prediction
from .model import (
    load_pipeline,
    MODEL,
    FEATURE_NAMES,
    MODEL_PATH,
    FEAT_NAMES_PATH,
)

app = FastAPI(title="NYC Taxi Model API")

# -------------------- Prometheus Metrics --------------------

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total number of prediction requests"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total", "Total number of failed predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)


# -------------------- HEALTH ENDPOINT --------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _ = load_pipeline()
        loaded = True
    except Exception:
        loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        version="v1",
        extra_info={
            "model_path": MODEL_PATH,
            "feat_names_path": FEAT_NAMES_PATH,
            "n_features_expected": len(FEATURE_NAMES) if FEATURE_NAMES else None,
            "model_n_features_in_": getattr(MODEL, "n_features_in_", None),
            "feature_names": FEATURE_NAMES,
        },
    )


# -------------------- PREDICT ENDPOINT --------------------


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start_time = time.perf_counter()
    PREDICTION_REQUESTS.inc()

    try:
        pipeline = load_pipeline()

        # privacy filter for incoming records
        clean_records = [scrub_pii_from_record(r) for r in payload.records]

        df = pd.DataFrame(clean_records)
        yhat = pipeline.predict(df)

        return PredictResponse(
            predictions=[Prediction(duration_min_predicted=float(v)) for v in yhat]
        )

    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    finally:
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)


# -------------------- METRICS ENDPOINT --------------------


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
