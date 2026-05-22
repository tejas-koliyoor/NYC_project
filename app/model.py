"""
Model loading utilities for NYC Taxi API.

We ship the Docker image with the trained artifact in:
  artifacts/model.joblib

So the API must load that by default.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import joblib

# -----------------------------
# Default artifact paths (match your Docker/demo setup)
# -----------------------------
DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_FEATURE_NAMES_PATH = Path("models/feature_names.json")


@lru_cache(maxsize=1)
def load_pipeline(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load and cache the scikit-learn pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_feature_names(path: Path = DEFAULT_FEATURE_NAMES_PATH) -> List[str]:
    """Load the list of feature names used during training (optional helper)."""
    if not path.exists():
        raise FileNotFoundError(f"Feature names file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
