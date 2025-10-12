from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any
import pandas as pd
import numpy as np

UTC = "UTC"

@dataclass(frozen=True)
class ColumnRule:
    name: str
    required: bool
    dtype: str  # "timestamp", "int", "float", "decimal", "string", "char"
    nullable: bool
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed: Optional[Sequence[Any]] = None
    default_if_null: Optional[Any] = None
    on_bad: str = "flag"  # "flag" | "drop" | "coerce_null" | "cap"
    notes: str = ""

# Contract columns (18 initial)
COLUMNS: list[ColumnRule] = [
    ColumnRule("VendorID", True, "int", False, allowed=[1,2,6,7], on_bad="flag"),
    ColumnRule("tpep_pickup_datetime", True, "timestamp", False, notes="UTC"),
    ColumnRule("tpep_dropoff_datetime", True, "timestamp", False, notes="UTC"),
    ColumnRule("PULocationID", True, "int", False),
    ColumnRule("DOLocationID", True, "int", False),
    ColumnRule("passenger_count", False, "int", True, min_val=0, max_val=8, on_bad="coerce_null"),
    ColumnRule("trip_distance", False, "float", True, min_val=0.0, max_val=200.0, on_bad="flag"),
    ColumnRule("RatecodeID", False, "int", True, allowed=[1,2,3,4,5,6,99], default_if_null=99),
    ColumnRule("store_and_fwd_flag", False, "char", True, allowed=["Y","N"], default_if_null="N"),
    ColumnRule("payment_type", False, "int", True, allowed=[0,1,2,3,4,5,6]),
    ColumnRule("fare_amount", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("extra", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("mta_tax", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("tip_amount", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("tolls_amount", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("improvement_surcharge", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("congestion_surcharge", False, "decimal", True, min_val=0.0, on_bad="coerce_null"),
    ColumnRule("total_amount", False, "decimal", True, min_val=0.0, on_bad="flag"),
]

REQUIRED = {c.name for c in COLUMNS if c.required}

def to_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s

def coerce_dtype(df: pd.DataFrame, col: ColumnRule) -> pd.Series:
    s = df[col.name]
    if col.dtype == "timestamp":
        return to_utc(s)
    if col.dtype in ("int",):
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if col.dtype in ("float","decimal"):
        return pd.to_numeric(s, errors="coerce").astype(float)
    if col.dtype == "char":
        s = s.astype("string").str.strip().str.upper()
        return s
    return s.astype("string")

def apply_rule(series: pd.Series, rule: ColumnRule) -> tuple[pd.Series, pd.Series]:
    """Returns (possibly modified series, violation_mask)"""
    s = series.copy()
    violations = pd.Series(False, index=s.index)

    # Handle null defaults
    if rule.default_if_null is not None:
        s = s.fillna(rule.default_if_null)

    # Range checks
    if rule.min_val is not None:
        bad = s.notna() & (s < rule.min_val)
        violations |= bad
        if rule.on_bad in ("coerce_null","flag"):
            s = s.mask(bad, np.nan)
        elif rule.on_bad == "cap":
            s = s.mask(bad, rule.min_val)
    if rule.max_val is not None:
        bad = s.notna() & (s > rule.max_val)
        violations |= bad
        if rule.on_bad in ("coerce_null","flag"):
            s = s.mask(bad, np.nan)
        elif rule.on_bad == "cap":
            s = s.mask(bad, rule.max_val)

    # Allowed set
    if rule.allowed is not None:
        bad = s.notna() & (~s.isin(rule.allowed))
        violations |= bad
        if rule.on_bad in ("coerce_null","flag"):
            s = s.mask(bad, np.nan)

    return s, violations
