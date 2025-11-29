from __future__ import annotations

import json
import os
import pickle
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from .model import (
    load_pipeline,
    MODEL,
    FEATURE_NAMES,
    MODEL_PATH,
    FEAT_NAMES_PATH,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEAT_NAMES_PATH = os.getenv("FEAT_NAMES_PATH", "models/feature_names.json")

app = FastAPI(title="NYC Taxi Model API")

_model = None
_feature_names: List[str] | None = None


def _ensure_loaded() -> None:
    """Load model + feature names on demand (NOT at startup)."""
    global _model, _feature_names

    if _model is None:
        # try joblib first (works for many sklearn dumps), fall back to pickle
        try:
            from joblib import load as joblib_load
            _model = joblib_load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)

    if _feature_names is None:
        with open(FEAT_NAMES_PATH, "r", encoding="utf-8") as f:
            names = json.load(f)
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise RuntimeError("feature_names.json must be a JSON list of strings")
        _feature_names = names


class PredictRequest(BaseModel):
    features: List[float]

@app.get("/meta")
def meta():
    return {
        "model_path": MODEL_PATH,
        "feat_names_path": FEAT_NAMES_PATH,
        "n_features_expected": len(FEATURE_NAMES) if FEATURE_NAMES else None,
        "model_n_features_in_": getattr(MODEL, "n_features_in_", None),
        "feature_names": FEATURE_NAMES,
    }