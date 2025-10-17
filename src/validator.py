from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .contract_spec import COLUMNS, REQUIRED, apply_rule, coerce_dtype


@dataclass
class ValidationReport:
    rows: int
    required_columns_present: bool
    missing_columns: list[str]
    anomaly_rate: float
    anomalies_by_rule: Dict[str, int]
    freshness_ok: Optional[bool] = None
    latest_dropoff: Optional[str] = None


def validate_dataframe(
    df: pd.DataFrame,
    month: Optional[str] = None,  # e.g., "2025-03"
    taxi_zone_ids: Optional[set[int]] = None,  # pass known IDs if you have them
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (validated_df_with_derivatives, report).
    Adds: duration_minutes (Int64), is_anomaly (0/1).
    Drops rows only when required columns are missing or joinability fails.
    """
    df = df.copy()
    len(df)

    # 1) Required columns present
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2) Coerce dtypes per contract
    for col in COLUMNS:
        if col.name in df.columns:
            df[col.name] = coerce_dtype(df, col)

    # 3) Per-column rules & anomaly flags
    is_anomaly = pd.Series(False, index=df.index)
    anomalies_by_rule: Dict[str, int] = {}

    for col in COLUMNS:
        if col.name not in df.columns:
            continue
        s, badmask = apply_rule(df[col.name], col)
        df[col.name] = s
        if col.required:
            # Required cannot be null after coercion
            req_bad = df[col.name].isna()
            if req_bad.any():
                # drop rows that violate required-ness
                df = df.loc[~req_bad].copy()
                badmask = badmask.loc[df.index]
        # accumulate anomaly flags (for non-dropped)
        badmask = badmask.reindex(df.index, fill_value=False)
        count_bad = int(badmask.sum())
        if count_bad > 0:
            is_anomaly = is_anomaly.reindex(df.index, fill_value=False) | badmask
            anomalies_by_rule[col.name] = count_bad

    # 4) Joinability checks (if zone set supplied)
    if taxi_zone_ids is not None:
        pu_bad = ~df["PULocationID"].isin(taxi_zone_ids)
        do_bad = ~df["DOLocationID"].isin(taxi_zone_ids)
        drop_mask = pu_bad | do_bad
        if drop_mask.any():
            df = df.loc[~drop_mask].copy()
            anomalies_by_rule["joinability_drop"] = int(drop_mask.sum())
            is_anomaly = is_anomaly.loc[df.index]

    # 5) Duration + additional rules
    dur = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df["duration_minutes"] = pd.to_numeric(dur, errors="coerce").round().astype("Int64")
    dur_bad = (
        df["duration_minutes"].isna()
        | (df["duration_minutes"] < 1)
        | (df["duration_minutes"] > 720)
    )
    if dur_bad.any():
        is_anomaly = is_anomaly | dur_bad
        anomalies_by_rule["duration_minutes"] = int(dur_bad.sum())

    df["is_anomaly"] = is_anomaly.astype(int)

    # 6) Freshness check (optional, based on month)
    freshness_ok = None
    latest_dropoff = None
    if "tpep_dropoff_datetime" in df.columns and not df.empty:
        latest_dropoff = str(df["tpep_dropoff_datetime"].max())
        if month:
            # consider fresh if any dropoff falls within that YYYY-MM month window
            y, m = month.split("-")
            start = pd.Timestamp(f"{y}-{m}-01", tz="UTC")
            end = start + pd.offsets.MonthBegin(1)
            in_window = (df["tpep_dropoff_datetime"] >= start) & (
                df["tpep_dropoff_datetime"] < end
            )
            freshness_ok = bool(in_window.any())

    report = ValidationReport(
        rows=len(df),
        required_columns_present=True,
        missing_columns=[],
        anomaly_rate=float(df["is_anomaly"].mean()) if len(df) else 0.0,
        anomalies_by_rule=anomalies_by_rule,
        freshness_ok=freshness_ok,
        latest_dropoff=latest_dropoff,
    )
    return df, report


# --- CLI entrypoint (optional) ---
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Validate NYC Yellow Taxi data against the contract."
    )
    parser.add_argument("path", help="CSV or Parquet file")
    parser.add_argument("--month", help="YYYY-MM expected month window", default=None)
    args = parser.parse_args()

    if args.path.lower().endswith(".parquet"):
        data = pd.read_parquet(args.path)
    else:
        data = pd.read_csv(args.path)

    valid_df, rep = validate_dataframe(data, month=args.month)
    print(json.dumps(rep.__dict__, indent=2))
    # Optionally save outputs:
    # valid_df.to_parquet("validated.parquet")
