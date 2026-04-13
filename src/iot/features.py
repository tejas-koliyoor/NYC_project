"""Temporal and rolling-window feature engineering for IoT sensor data.

Features generated (per machine, per row):
  - Rolling statistics (mean, std, min, max) over 10-min, 30-min, 1-h windows
  - Rolling gradient (linear slope) over 10-min window
  - Spike indicators (value > mean + 3*std within window)
  - Time-of-day cyclic encoding (sin/cos of hour)
  - Error code lag and rolling count

All features are built **per machine_id** to avoid leaking data across
machines.  The caller should drop rows with NaN before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Window sizes in minutes
WINDOWS = {"10min": 10, "30min": 30, "1h": 60}

SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm", "voltage"]


# -----------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute the OLS slope of a rolling window (trend indicator)."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for end in range(window - 1, len(vals)):
        y = vals[end - window + 1: end + 1]
        if np.isnan(y).any():
            continue
        x = np.arange(window, dtype=float)
        # slope via least-squares formula
        xm = x - x.mean()
        ym = y - y.mean()
        denom = (xm ** 2).sum()
        if denom == 0:
            result.iloc[end] = 0.0
        else:
            result.iloc[end] = float((xm * ym).sum() / denom)
    return result


def _build_machine_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Build features for a single machine's time series (already sorted)."""
    idx = grp.index
    col_dict: dict[str, pd.Series] = {}

    # Time features (cyclic hour encoding)
    hour = grp["timestamp"].dt.hour.astype(float)
    col_dict["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    col_dict["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Error code features
    col_dict["error_code"] = grp["error_code"].astype(float)
    col_dict["error_lag1"] = grp["error_code"].shift(1).fillna(0).astype(float)

    for win_label, win_min in WINDOWS.items():
        # Use count of rows as window size (assumes ~5-min cadence); fall back
        # to a minimum of 2 rows so stats are always defined
        n = max(2, win_min // 5)

        # Error count in rolling window
        col_dict[f"error_count_{win_label}"] = (
            (grp["error_code"] > 0).rolling(n, min_periods=1).sum().astype(float)
        )

        for col in SENSOR_COLS:
            s = grp[col]
            roll_mean = s.rolling(n, min_periods=1).mean()
            roll_std = s.rolling(n, min_periods=1).std().fillna(0.0)

            col_dict[f"{col}_mean_{win_label}"] = roll_mean
            col_dict[f"{col}_std_{win_label}"] = roll_std
            col_dict[f"{col}_min_{win_label}"] = s.rolling(n, min_periods=1).min()
            col_dict[f"{col}_max_{win_label}"] = s.rolling(n, min_periods=1).max()
            col_dict[f"{col}_var_{win_label}"] = (
                s.rolling(n, min_periods=1).var().fillna(0.0)
            )

            # Rolling gradient (trend)
            if win_label == "10min":
                col_dict[f"{col}_trend_{win_label}"] = _rolling_slope(s, n)

            # Spike indicator: current value > mean + 3*std in window
            col_dict[f"{col}_spike_{win_label}"] = (
                (s > roll_mean + 3 * roll_std).astype(float)
            )

    return pd.DataFrame(col_dict, index=idx)


# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

def build_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from validated sensor telemetry.

    Parameters
    ----------
    df:
        Validated sensor DataFrame (output of ``validate_sensor_df``).
        Must be sorted by (machine_id, timestamp).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with ``df.index``.  Contains only numeric
        columns; no NaN values (NaN rows from the start of each machine
        window are forward-filled with 0).
    """
    all_feats = []
    for _, grp in df.groupby("machine_id", sort=False):
        machine_feats = _build_machine_features(grp)
        all_feats.append(machine_feats)

    features = pd.concat(all_feats).reindex(df.index)

    # Fill any leading NaNs (start of window) with 0
    features = features.fillna(0.0)

    return features
