from typing import List, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service status")
    model_loaded: bool
    version: str = Field("v1", description="API version")

class PredictRequest(BaseModel):
    # Batch of records (feature name -> value); must match training columns used in save_model.py
    records: List[Dict[str, Any]]

class Prediction(BaseModel):
    duration_min_predicted: float

class PredictResponse(BaseModel):
    predictions: List[Prediction]
