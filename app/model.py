import os
from functools import lru_cache
from typing import Any
import joblib

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")

class ModelNotFound(Exception):
    pass

@lru_cache(maxsize=1)
def load_pipeline(path: str = DEFAULT_MODEL_PATH) -> Any:
    if not os.path.exists(path):
        raise ModelNotFound(f"Model file not found at {path}")
    return joblib.load(path)
