import pandas as pd

from src.validator import validate_dataframe


def test_basic_validation_pipeline():
    df = pd.DataFrame(
        {
            "VendorID": [2, 2],
            "tpep_pickup_datetime": ["2025-03-01T08:00:00Z", "2025-03-01T08:10:00Z"],
            "tpep_dropoff_datetime": ["2025-03-01T08:05:00Z", "2025-03-01T20:10:00Z"],
            "PULocationID": [142, 999999],  # one bad joinable
            "DOLocationID": [236, 236],
            "passenger_count": [1, -5],
            "trip_distance": [3.2, -1.0],
            "RatecodeID": [1, 77],
            "store_and_fwd_flag": ["N", "Z"],
            "payment_type": [1, 9],
            "fare_amount": [12.5, -2.0],
            "extra": [0.5, 0.0],
            "mta_tax": [0.5, 0.5],
            "tip_amount": [2.0, 0.0],
            "tolls_amount": [0.0, 0.0],
            "improvement_surcharge": [0.3, 0.3],
            "congestion_surcharge": [2.75, 2.75],
            "total_amount": [18.05, -1.0],
        }
    )
    valid, rep = validate_dataframe(
        df, month="2025-03", taxi_zone_ids=set(range(1, 300))
    )
    # one row should drop due to joinability
    assert len(valid) == 1
    assert "duration_minutes" in valid.columns
    assert "is_anomaly" in valid.columns
    assert rep.freshness_ok is True
