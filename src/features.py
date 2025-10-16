# TODO: implement feature builders mirroring training & serving
from __future__ import annotations
import pandas as pd
import numpy as np

# Categorical “vocabulary” is fixed to avoid train/serve skew
VENDOR_VOCAB = [1, 2, 6, 7]
RATECODE_VOCAB = [1, 2, 3, 4, 5, 6, 99]
PAYMENT_VOCAB = [0, 1, 2, 3, 4, 5, 6]

def _one_hot(series: pd.Series, vocab: list[int], prefix: str) -> pd.DataFrame:
    # normalize unknowns to a reserved bucket if needed
    s = series.copy()
    s = s.where(s.isin(vocab), np.nan)   # unknowns -> NaN (drop columns later)
    dummies = pd.get_dummies(s, prefix=prefix, dtype=int)
    # ensure all expected columns exist (even if absent in this batch)
    for v in vocab:
        col = f"{prefix}_{v}"
        if col not in dummies.columns:
            dummies[col] = 0
    # return in vocab order (stable column order)
    return dummies[[f"{prefix}_{v}" for v in vocab]]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform validated trips into serving-safe numeric features.
    NOTE: We purposely avoid any columns that won't exist at inference time.
    Allowed source columns include pickup time, PU/DO zones, vendor/ratecode/payment,
    passenger_count, trip_distance. We DO NOT use total_amount or tip for features.
    """
    x = pd.DataFrame(index=df.index)

    # Basic numerics
    x["trip_distance"] = df["trip_distance"].fillna(0.0).astype(float).clip(lower=0.0, upper=200.0)
    x["passenger_count"] = df["passenger_count"].fillna(0).astype(int).clip(lower=0, upper=8)

    # Time features from pickup time (available at request time)
    pickup = pd.to_datetime(df["tpep_pickup_datetime"], utc=True, errors="coerce")
    x["pickup_hour"] = pickup.dt.hour.astype(int)
    x["pickup_dow"]  = pickup.dt.dayofweek.astype(int)  # 0=Mon

    # Zone features (IDs are stable ints; we keep as is)
    x["PU_id"] = df["PULocationID"].astype(int)
    x["DO_id"] = df["DOLocationID"].astype(int)

    # Categorical encodings (one-hots with fixed vocab)
    x_vendor  = _one_hot(df["VendorID"].astype("Int64"), VENDOR_VOCAB, "vendor")
    x_rate    = _one_hot(df["RatecodeID"].astype("Int64"), RATECODE_VOCAB, "rate")
    x_payment = _one_hot(df["payment_type"].astype("Int64"), PAYMENT_VOCAB, "pay")

    x = pd.concat([x, x_vendor, x_rate, x_payment], axis=1)

    # Guardrail: no NaNs
    x = x.fillna(0)

    return x
