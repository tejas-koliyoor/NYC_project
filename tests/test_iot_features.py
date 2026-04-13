"""Tests for IoT sensor failure prediction: data loading, feature engineering,
and model training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.iot.sensor_data import (
    add_failure_label,
    make_synthetic_sensor_data,
    validate_sensor_df,
)
from src.iot.features import build_sensor_features, SENSOR_COLS, WINDOWS
from src.iot.train import compute_metrics, train_sensor_model


# -----------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------

@pytest.fixture()
def tiny_df():
    """Minimal validated sensor DataFrame for one machine."""
    base = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i in range(30):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(minutes=5 * i),
                "machine_id": "machine_00",
                "temperature": 80.0 + i * 0.5,
                "vibration": 0.02 + i * 0.001,
                "pressure": 100.0,
                "rpm": 3500.0,
                "voltage": 220.0,
                "error_code": 1 if i >= 25 else 0,
                "will_fail": 1 if i >= 20 else 0,
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture()
def multi_machine_df():
    """Synthetic dataset with multiple machines."""
    return make_synthetic_sensor_data(n_machines=3, n_rows_per_machine=50, seed=0)


# -----------------------------------------------------------------
# sensor_data tests
# -----------------------------------------------------------------

class TestValidateSensorDf:
    def test_required_columns_raises(self):
        df = pd.DataFrame({"timestamp": ["2026-01-01"], "machine_id": ["m1"]})
        with pytest.raises(ValueError, match="Missing required sensor columns"):
            validate_sensor_df(df)

    def test_out_of_bounds_replaced_with_median(self):
        base = pd.Timestamp("2026-01-01", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": [base, base + pd.Timedelta(minutes=5)],
                "machine_id": ["m1", "m1"],
                "temperature": [80.0, 9999.0],  # 9999 > 500 → out of bounds
                "vibration": [0.02, 0.03],
                "pressure": [100.0, 100.0],
                "rpm": [3500.0, 3500.0],
                "voltage": [220.0, 220.0],
                "error_code": [0, 0],
            }
        )
        result = validate_sensor_df(df)
        # Out-of-bounds temperature replaced with median of the remaining value
        assert result["temperature"].iloc[1] == pytest.approx(80.0)

    def test_nan_timestamps_dropped(self):
        base = pd.Timestamp("2026-01-01", tz="UTC")
        df = pd.DataFrame(
            {
                "timestamp": ["not-a-date", str(base)],
                "machine_id": ["m1", "m1"],
                "temperature": [80.0, 80.0],
                "vibration": [0.02, 0.02],
                "pressure": [100.0, 100.0],
                "rpm": [3500.0, 3500.0],
                "voltage": [220.0, 220.0],
                "error_code": [0, 0],
            }
        )
        result = validate_sensor_df(df)
        assert len(result) == 1

    def test_sorted_by_machine_and_time(self, multi_machine_df):
        result = validate_sensor_df(multi_machine_df)
        # Check monotonic timestamp within each machine
        for _, grp in result.groupby("machine_id"):
            assert grp["timestamp"].is_monotonic_increasing


class TestAddFailureLabel:
    def test_label_already_present_unchanged(self, tiny_df):
        out = add_failure_label(tiny_df)
        # will_fail was in tiny_df; should be returned as-is
        pd.testing.assert_series_equal(out["will_fail"], tiny_df["will_fail"])

    def test_synthetic_data_has_label(self):
        df = make_synthetic_sensor_data(n_machines=2, n_rows_per_machine=20)
        assert "will_fail" in df.columns
        assert set(df["will_fail"].unique()).issubset({0, 1})


# -----------------------------------------------------------------
# Feature engineering tests
# -----------------------------------------------------------------

class TestBuildSensorFeatures:
    def test_output_shape(self, tiny_df):
        feats = build_sensor_features(tiny_df)
        assert len(feats) == len(tiny_df)

    def test_no_nans(self, tiny_df):
        feats = build_sensor_features(tiny_df)
        assert feats.isna().sum().sum() == 0, "Feature matrix must have no NaN"

    def test_expected_columns_present(self, tiny_df):
        feats = build_sensor_features(tiny_df)
        cols = set(feats.columns)

        # Time features
        assert "hour_sin" in cols
        assert "hour_cos" in cols

        # Rolling stats for each sensor/window combo
        for win in WINDOWS:
            for sensor in SENSOR_COLS:
                assert f"{sensor}_mean_{win}" in cols, f"Missing {sensor}_mean_{win}"
                assert f"{sensor}_std_{win}" in cols
                assert f"{sensor}_spike_{win}" in cols
                assert f"{sensor}_var_{win}" in cols

        # Trend only for 10min window
        for sensor in SENSOR_COLS:
            assert f"{sensor}_trend_10min" in cols

    def test_rolling_mean_monotonic_temperature(self, tiny_df):
        """Rising temperature → rolling mean should increase (not guaranteed
        to be strictly monotonic due to window effects, but mean over full
        history > first row mean)."""
        feats = build_sensor_features(tiny_df)
        first_mean = feats["temperature_mean_1h"].iloc[0]
        last_mean = feats["temperature_mean_1h"].iloc[-1]
        assert last_mean > first_mean

    def test_spike_indicator_binary(self, tiny_df):
        feats = build_sensor_features(tiny_df)
        spike_cols = [c for c in feats.columns if "spike" in c]
        for col in spike_cols:
            vals = feats[col].unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} contains non-binary values"

    def test_multi_machine_index_preserved(self, multi_machine_df):
        validated = validate_sensor_df(multi_machine_df)
        feats = build_sensor_features(validated)
        assert list(feats.index) == list(validated.index)

    def test_error_lag(self, tiny_df):
        feats = build_sensor_features(tiny_df)
        # First row lag should be 0
        assert feats["error_lag1"].iloc[0] == 0.0


# -----------------------------------------------------------------
# Training tests
# -----------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictor(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y, p)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.3, 0.7, 0.4, 0.6])
        m = compute_metrics(y, p)
        expected_keys = {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "threshold"}
        assert expected_keys == set(m.keys())


class TestTrainSensorModel:
    def test_rf_trains_and_returns_metrics(self, tmp_path, multi_machine_df):
        metrics = train_sensor_model(
            df=multi_machine_df,
            model_name="test_model",
            experiment="test-iot",
            algo="rf",
            test_size=0.25,
            models_dir=str(tmp_path),
        )
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert (tmp_path / "test_model.pkl").exists()
        assert (tmp_path / "test_model_feature_schema.json").exists()

    def test_gb_trains_and_returns_metrics(self, tmp_path, multi_machine_df):
        metrics = train_sensor_model(
            df=multi_machine_df,
            model_name="test_model_gb",
            experiment="test-iot",
            algo="gb",
            test_size=0.25,
            models_dir=str(tmp_path),
        )
        assert "f1" in metrics
        assert (tmp_path / "test_model_gb.pkl").exists()

    def test_invalid_algo_raises(self, tmp_path, multi_machine_df):
        with pytest.raises(ValueError, match="algo must be"):
            train_sensor_model(
                df=multi_machine_df,
                model_name="x",
                experiment="test-iot",
                algo="xgboost",
                models_dir=str(tmp_path),
            )

    def test_no_labelled_rows_raises(self, tmp_path, multi_machine_df):
        df = multi_machine_df.copy()
        df["will_fail"] = -1  # all unknown
        with pytest.raises(ValueError, match="No labelled rows"):
            train_sensor_model(
                df=df,
                model_name="x",
                experiment="test-iot",
                algo="rf",
                models_dir=str(tmp_path),
            )
