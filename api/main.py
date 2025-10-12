from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="NYC Taxi Inference API (skeleton)")

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

class Payload(BaseModel):
    # placeholder schema; adjust after you define features
    feature1: float = Field(ge=0)
    feature2: float = Field(ge=0)

@app.get("/health")
def health():
    REQUESTS.labels("/health").inc()
    return {"status": "ok", "model_loaded": False}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(p: Payload):
    import time
    start = time.time()
    REQUESTS.labels("/predict").inc()
    # placeholder score for skeleton
    score = float(p.feature1 * 0.1 + p.feature2 * 0.2)
    LATENCY.labels("/predict").observe(time.time() - start)
    return {"score": score}
