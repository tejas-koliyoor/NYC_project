from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service status")
    model_loaded: bool
    version: str = Field("v1", description="API version")


class TripRecord(BaseModel):
    # ---- Numerical features ----
    trip_distance: float
    fare_amount: float
    passenger_count: int

    pickup_hour: int
    pickup_dow: int
    is_weekend: int

    # ---- ID / categorical-like features ----
    PULocationID: int
    DOLocationID: int
    payment_type: int
    RatecodeID: int

    # This is categorical in training; allow either "N"/"Y" or 0/1 (or truthy strings)
    store_and_fwd_flag: Union[str, int] = Field(
        ..., description='Store and forward flag ("N"/"Y" or 0/1)'
    )

    @field_validator("store_and_fwd_flag")
    @classmethod
    def normalize_store_and_fwd_flag(cls, v: Union[str, int]) -> str:
        """
        Normalize common representations to "N" or "Y" so the model pipeline
        consistently sees a categorical string (required for OneHotEncoder).

        Accepts:
        - "N"/"Y", "n"/"y"
        - 0/1
        - "true"/"false", "yes"/"no"
        """
        if isinstance(v, int):
            return "Y" if v == 1 else "N"

        s = str(v).strip().upper()

        if s in {"0", "N", "NO", "FALSE", "F"}:
            return "N"
        if s in {"1", "Y", "YES", "TRUE", "T"}:
            return "Y"

        # If it's some unexpected category, keep it as a string
        # (OneHotEncoder(handle_unknown="ignore") will handle unseen values)
        return s


class PredictRequest(BaseModel):
    # Batch of records; must match training columns used in save_model.py
    records: List[TripRecord]


class Prediction(BaseModel):
    duration_min_predicted: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
