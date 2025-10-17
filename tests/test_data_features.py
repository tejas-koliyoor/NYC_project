import pandas as pd

from src.data import add_label
from src.features import build_features


def _sample_df():
    return pd.DataFrame(
        {
            "VendorID": [2, 2],
            "tpep_pickup_datetime": ["2025-03-01T08:00:00Z", "2025-03-02T09:15:00Z"],
            "tpep_dropoff_datetime": ["2025-03-01T08:20:00Z", "2025-03-02T09:45:00Z"],
            "PULocationID": [142, 100],
            "DOLocationID": [236, 236],
            "passenger_count": [1, 3],
            "trip_distance": [3.2, 10.5],
            "RatecodeID": [1, 99],
            "store_and_fwd_flag": ["N", "N"],
            "payment_type": [1, 2],
            "fare_amount": [12.5, 45.0],
            "extra": [0.5, 0.0],
            "mta_tax": [0.5, 0.5],
            "tip_amount": [2.0, 0.0],
            "tolls_amount": [0.0, 0.0],
            "improvement_surcharge": [0.3, 0.3],
            "congestion_surcharge": [2.75, 2.75],
            "total_amount": [18.05, 55.25],
            "duration_minutes": [20, 30],
            "is_anomaly": [0, 0],
        }
    )


def test_label_and_features_shapes_and_types():
    df = _sample_df()
    df = add_label(df)
    assert "HIGH_TOTAL" in df.columns
    assert set(df["HIGH_TOTAL"].unique()) <= {0, 1}

    X = build_features(df)
    # Columns we expect
    must_have = {
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dow",
        "PU_id",
        "DO_id",
        "vendor_1",
        "vendor_2",
        "vendor_6",
        "vendor_7",
        "rate_1",
        "rate_2",
        "rate_3",
        "rate_4",
        "rate_5",
        "rate_6",
        "rate_99",
        "pay_0",
        "pay_1",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
    }
    assert must_have.issubset(set(X.columns))
    # Types: numeric only
    assert all(str(t).startswith(("int", "float", "Int64")) for t in X.dtypes)
    # No NaNs
    assert X.isna().sum().sum() == 0
