import argparse
import os
import json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--month", required=True)
args = ap.parse_args()

df = pd.read_csv(f"data/nyc_taxi_{args.month}.csv")

# required base columns & derived features
for c in ["trip_distance", "fare_amount", "total_amount", "payment_type"]:
    if c not in df.columns:
        df[c] = 0

dur = (
    pd.to_datetime(df["tpep_dropoff_datetime"])
    - pd.to_datetime(df["tpep_pickup_datetime"])
).dt.total_seconds() / 60.0

df["duration_min"] = dur

output_path = os.path.join("features", "offline", f"offline_{args.month}.json")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, indent=2)

print("Saved offline features to:", output_path)
