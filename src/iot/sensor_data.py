"""IoT sensor telemetry data loading and validation.

Expected schema:
    timestamp   : datetime-like string or datetime
    machine_id  : str
    temperature : float  (°C)
    vibration   : float  (g)
    pressure    : float  (bar)
    rpm         : float  (rotations per minute)
    voltage     : float  (V)
    error_code  : int    (0 = no error)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------
# Schema constants
# -----------------------------------------------------------------
SENSOR_COLUMNS = [
    "timestamp",
    "machine_id",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "voltage",
    "error_code",
]

NUMERIC_SENSORS = ["temperature", "vibration", "pressure", "rpm", "voltage"]

# Sanity-range bounds for each numeric sensor (inclusive)
SENSOR_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (-50.0, 500.0),
    "vibration": (0.0, 100.0),
    "pressure": (0.0, 1000.0),
    "rpm": (0.0, 50_000.0),
    "voltage": (0.0, 10_000.0),
}


# -----------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------

def load_sensor_csv(path: str | Path) -> pd.DataFrame:
    """Load sensor telemetry from a CSV or Parquet file."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return validate_sensor_df(df)


def validate_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a raw sensor telemetry DataFrame.

    Steps:
      1. Check required columns are present.
      2. Parse ``timestamp`` to UTC-aware datetime.
      3. Coerce numeric columns; out-of-bounds readings become NaN.
      4. Fill remaining NaNs with column medians (sensor drift / gaps).
      5. Coerce ``error_code`` to int (0 for NaN).
      6. Sort by (machine_id, timestamp).

    Returns a clean DataFrame ready for feature engineering.
    """
    df = df.copy()

    # 1. Required columns
    missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}")

    # 2. Timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        df = df.loc[~bad_ts].copy()

    # 3. Numeric coercion + range clipping
    for col in NUMERIC_SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = SENSOR_BOUNDS[col]
        out_of_bounds = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        df.loc[out_of_bounds, col] = float("nan")

    # 4. Impute with column median (handles sensor gaps / drift)
    for col in NUMERIC_SENSORS:
        median = df[col].median()
        if pd.isna(median):
            median = 0.0
        df[col] = df[col].fillna(median)

    # 5. Error code
    df["error_code"] = pd.to_numeric(df["error_code"], errors="coerce").fillna(0).astype(int)

    # 6. Sort
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    return df


def make_synthetic_sensor_data(
    n_machines: int = 5,
    n_rows_per_machine: int = 200,
    failure_rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic sensor telemetry dataset for testing and demos.

    The label column ``will_fail`` is set to 1 for ``failure_rate`` fraction of
    rows (randomly) and 0 otherwise. This simulates a labelled training set.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2026-01-01", tz="UTC")

    for m in range(n_machines):
        machine_id = f"machine_{m:02d}"
        for i in range(n_rows_per_machine):
            ts = base_time + pd.Timedelta(minutes=5 * i)
            rows.append(
                {
                    "timestamp": ts,
                    "machine_id": machine_id,
                    "temperature": float(rng.normal(80, 10)),
                    "vibration": float(rng.exponential(0.05)),
                    "pressure": float(rng.normal(100, 5)),
                    "rpm": float(rng.normal(3500, 200)),
                    "voltage": float(rng.normal(220, 5)),
                    "error_code": int(rng.choice([0, 1, 2], p=[0.95, 0.04, 0.01])),
                    "will_fail": int(rng.random() < failure_rate),
                }
            )

    return pd.DataFrame(rows)


def add_failure_label(
    df: pd.DataFrame,
    label_col: str = "will_fail",
    horizon_minutes: int = 60,
) -> pd.DataFrame:
    """Add a binary failure label based on future ``error_code`` values.

    For each row, ``will_fail = 1`` if any ``error_code > 0`` occurs for the
    same machine within the next ``horizon_minutes`` minutes.  Rows near the
    end of a machine's history (where the full horizon cannot be observed) are
    marked with ``will_fail = -1`` and should be excluded from training.

    If a ``will_fail`` column already exists in ``df`` (e.g. synthetic data),
    it is returned unchanged.
    """
    if label_col in df.columns:
        return df

    df = df.copy()
    df[label_col] = -1  # default: unknown

    for machine_id, grp in df.groupby("machine_id"):
        idx = grp.index
        ts = grp["timestamp"]
        horizon = pd.Timedelta(minutes=horizon_minutes)

        for i in idx:
            t0 = ts.loc[i]
            future = grp[(ts > t0) & (ts <= t0 + horizon)]
            df.loc[i, label_col] = int((future["error_code"] > 0).any())

    return df
