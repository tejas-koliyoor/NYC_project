# save_model.py — NYC Taxi baseline (trip duration regression)

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


cat_cols = ["store_and_fwd_flag", "payment_type", "RatecodeID"]
num_cols = ["trip_distance", "fare_amount", "passenger_count",
            "pickup_hour", "pickup_dow", "is_weekend",
            "PULocationID", "DOLocationID"]

cat_proc = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

num_proc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

pre = ColumnTransformer([
    ("num", num_proc, num_cols),
    ("cat", cat_proc, cat_cols)
])




# ---- 1) Load CSV (adapt path if needed) ----
CSV_PATH = os.getenv("NYC_TAXI_CSV", "data\nyc_taxi_2025-03_updated_1200_rows.csv")
df = pd.read_csv(CSV_PATH)

# ---- 2) Normalize typical TLC column names ----
# We try both yellow ("tpep_*") and green ("lpep_*") names.
pickup_col = next(
    (
        c
        for c in [
            "tpep_pickup_datetime",
            "lpep_pickup_datetime",
            "pickup_datetime",
            "pickup_time",
        ]
        if c in df.columns
    ),
    None,
)
dropoff_col = next(
    (
        c
        for c in [
            "tpep_dropoff_datetime",
            "lpep_dropoff_datetime",
            "dropoff_datetime",
            "dropoff_time",
        ]
        if c in df.columns
    ),
    None,
)

if pickup_col is None or dropoff_col is None:
    # If timestamps aren’t there, we expect duration_min already exists.
    if "duration_min" not in df.columns:
        raise ValueError(
            "Could not find pickup/dropoff timestamps and no 'duration_min' present. "
            "Add timestamp cols or a precomputed 'duration_min' column."
        )
else:
    # Parse datetimes and compute duration
    for c in [pickup_col, dropoff_col]:
        if not np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["duration_min"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0

# ---- 3) Basic cleaning / filters ----
# Keep reasonable durations/distances
if "trip_distance" in df.columns:
    df = df[df["trip_distance"].fillna(0) >= 0]
df = df[(df["duration_min"] > 0.5) & (df["duration_min"] <= 240)]  # 30s..4h

# ---- 4) Feature set (adjust to what you have) ----
num_cols = [
    c for c in ["trip_distance", "fare_amount", "passenger_count"] if c in df.columns
]

# Location IDs (categorical)
cat_cols = [
    c
    for c in [
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "RatecodeID",
        "store_and_fwd_flag",
    ]
    if c in df.columns
]

# Time features from pickup
if pickup_col in df.columns:
    df["pickup_hour"] = df[pickup_col].dt.hour
    df["pickup_dow"] = df[pickup_col].dt.dayofweek
    df["is_weekend"] = df["pickup_dow"].isin([5, 6]).astype(int)
    num_cols += ["pickup_hour", "pickup_dow", "is_weekend"]

# Ensure we have at least something
if not num_cols and not cat_cols:
    # Fallback minimal features if file is very small
    guess_num = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != "duration_min"
    ]
    if not guess_num:
        raise ValueError("No usable feature columns found.")
    num_cols = guess_num[:5]

features = num_cols + cat_cols
target = "duration_min"
X = df[features].copy()
y = df[target].copy()

# ---- 5) Preprocess + model ----
num_proc = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
cat_proc = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)

pre = ColumnTransformer([("num", num_proc, num_cols), ("cat", cat_proc, cat_cols)])

model = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1, max_depth=None
)

pipe = Pipeline([("preprocess", pre), ("model", model)])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr)
preds = pipe.predict(Xte)
mae = mean_absolute_error(yte, preds)
print(f"[info] Validation MAE (minutes): {mae:.2f} on {len(yte)} holdout trips")

# ---- 6) Save artifact + an example payload ----
Path("artifacts").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "artifacts/model.joblib")
print("[ok] Saved model → artifacts/model.joblib")

# Build an example request matching columns we trained on
example_record = {}
for c in features:
    if c in df.columns:
        example_record[c] = df[c].iloc[0]
# Cast datetimes away (we only use derived features at inference)
for c in list(example_record):
    if "datetime" in str(c).lower():
        example_record.pop(c, None)

# If pickup_hour/dow/is_weekend are in features, ensure present (already in features if computed)
example = {"records": [example_record]}
pd.Series(example).to_json("artifacts/example_payload.json")
print("[ok] Wrote artifacts/example_payload.json with one valid record")

