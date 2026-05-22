import time
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from src.privacy import scrub_pii_from_record
from .model import load_pipeline
from .schemas import HealthResponse, PredictRequest, PredictResponse, Prediction

app = FastAPI(title="NYC Taxi Prediction API", version="v1")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Prometheus metrics
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Prediction errors")
PREDICTION_LATENCY = Histogram(
    "prediction_request_latency_seconds", "Latency of prediction requests"
)


def _expected_num_cols(pipeline) -> List[str]:
    """
    Inspect the pipeline's ColumnTransformer to find which columns are treated as numeric.
    This lets us map store_and_fwd_flag correctly depending on how the model was trained.
    """
    pre = getattr(pipeline, "named_steps", {}).get("preprocess")
    if pre is None or not hasattr(pre, "transformers"):
        return []

    for name, _transformer, cols in pre.transformers:
        if name == "num":
            return list(cols)
    return []


def _to_dict(record: Any) -> Dict[str, Any]:
    """Convert a request record into a plain dict."""
    if hasattr(record, "model_dump"):  # Pydantic v2
        return record.model_dump()
    if isinstance(record, dict):
        return record
    return dict(record)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health + model load check."""
    try:
        _ = load_pipeline()
        loaded = True
    except Exception:
        loaded = False

    return HealthResponse(status="ok", model_loaded=loaded, version="v1")


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    PREDICTION_REQUESTS.inc()

    try:
        pipeline = load_pipeline()

        if not payload.records:
            raise HTTPException(status_code=400, detail="No records provided")

        # Convert typed records -> dicts
        raw_records: List[Dict[str, Any]] = [_to_dict(r) for r in payload.records]

        # Apply privacy scrubber (expects dict)
        clean_records = [scrub_pii_from_record(r) for r in raw_records]

        df = pd.DataFrame(clean_records)

        # --- Align store_and_fwd_flag with what the trained pipeline expects ---
        if "store_and_fwd_flag" in df.columns:
            num_cols = _expected_num_cols(pipeline)

            # If pipeline expects numeric, map N/Y -> 0/1
            if "store_and_fwd_flag" in num_cols:
                df["store_and_fwd_flag"] = (
                    df["store_and_fwd_flag"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(
                        {
                            "N": 0,
                            "Y": 1,
                            "0": 0,
                            "1": 1,
                            "NO": 0,
                            "YES": 1,
                            "FALSE": 0,
                            "TRUE": 1,
                        }
                    )
                    .fillna(0)
                    .astype(int)
                )
            else:
                # If pipeline expects categorical, keep it as "N"/"Y"
                df["store_and_fwd_flag"] = (
                    df["store_and_fwd_flag"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"0": "N", "1": "Y", "NO": "N", "YES": "Y", "FALSE": "N", "TRUE": "Y"})
                    .fillna(df["store_and_fwd_flag"].astype(str).str.strip().str.upper())
                )

        # Run inference
        yhat = pipeline.predict(df)

        return PredictResponse(
            predictions=[Prediction(duration_min_predicted=float(v)) for v in yhat]
        )

    except HTTPException:
        PREDICTION_ERRORS.inc()
        raise
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/example")
def example_payload():
    # Serves artifacts/example_payload.json to the UI
    p = Path("artifacts/example_payload.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="example_payload.json not found")
    return Response(p.read_text(encoding="utf-8"), media_type="application/json")

