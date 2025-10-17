# TODO: implement loaders & schema checks based on the contract
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import pandas as pd

from .validator import validate_dataframe

# ---- Public API (used by training & batch scoring) ----


def load_month(
    path: str | Path,
    month: Optional[str] = None,  # "YYYY-MM" for freshness window
    taxi_zone_ids: Optional[Set[int]] = None,  # pass known IDs if you have them
) -> pd.DataFrame:
    """
    Load one month of NYC Yellow Taxi data (CSV or Parquet),
    validate against the contract, and return a clean frame with
    duration + is_anomaly columns preserved.
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    df, report = validate_dataframe(raw, month=month, taxi_zone_ids=taxi_zone_ids)

    # You can log/report here if desired; for now, keep simple:
    # print("Validation:", report)

    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple classification target suitable for Day-5 baseline.
    We’ll define: HIGH_TOTAL = 1 if total_amount >= 40.0 else 0.
    (Easy to compute, available at train time only; at serve-time you won’t have total_amount, so
    make sure features below don’t include label-only fields.)
    """
    out = df.copy()
    out["HIGH_TOTAL"] = (out["total_amount"].fillna(0.0) >= 40.0).astype(int)
    return out
