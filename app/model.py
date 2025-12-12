"""
Model loading utilities for NYC Taxi API.
Keeps global MODEL + FEATURE_NAMES available for FastAPI routes.
"""

import json
from pathlib import Path
import joblib

# -----------------------------
# Default model artifact paths
# -----------------------------
DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_FEATURE_NAMES_PATH = Path("models/feature_names.json")


# -----------------------------
# Loader functions
# -----------------------------
def load_pipeline(model_path: Path = DEFAULT_MODEL_PATH):
    """Load the scikit-learn pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_feature_names(path: Path = DEFAULT_FEATURE_NAMES_PATH):
    """Load the list of feature names used during training."""
    if not path.exists():
        raise FileNotFoundError(f"Feature names file missing: {path}")
    return json.load(open(path, "r", encoding="utf-8"))


# -----------------------------
# Global singleton model + features
# -----------------------------
try:
    MODEL = load_pipeline()
except Exception:
    MODEL = None  # health endpoint will check this

try:
    FEATURE_NAMES = load_feature_names()
except Exception:
    FEATURE_NAMES = None
