"""Training pipeline for IoT sensor failure prediction.

Supports:
  - Random Forest (``algo='rf'``)
  - Gradient Boosting (``algo='gb'``)

Uses MLflow for experiment tracking (optional – falls back gracefully if the
tracking server is unavailable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.iot.features import build_sensor_features
from src.iot.sensor_data import add_failure_label, load_sensor_csv, make_synthetic_sensor_data


# -----------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


# -----------------------------------------------------------------
# Training
# -----------------------------------------------------------------

def train_sensor_model(
    df: pd.DataFrame,
    model_name: str = "sensor_failure_model",
    experiment: str = "iot-sensor-failure",
    algo: str = "rf",
    random_state: int = 42,
    test_size: float = 0.2,
    class_weight: Optional[str] = "balanced",
    models_dir: str = "models",
) -> dict:
    """Train and evaluate a sensor failure prediction model.

    Parameters
    ----------
    df:
        Labelled sensor DataFrame.  Must contain a ``will_fail`` column with
        values 0 or 1 (rows with -1 are excluded automatically).
    model_name:
        Base name for the saved model file.
    experiment:
        MLflow experiment name.
    algo:
        ``'rf'`` for Random Forest or ``'gb'`` for Gradient Boosting.
    random_state, test_size, class_weight:
        Standard sklearn parameters.
    models_dir:
        Directory where the ``.pkl`` model file is saved.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    # Filter to known labels
    df = df[df["will_fail"].isin([0, 1])].copy()
    if len(df) == 0:
        raise ValueError("No labelled rows (will_fail in {0, 1}) found.")

    # Features + label
    X = build_sensor_features(df)
    y = df["will_fail"].astype(int).values
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"sensor-failure-{algo}"):
        if algo == "rf":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
                class_weight=class_weight,
            )
        elif algo == "gb":
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=random_state,
            )
        else:
            raise ValueError(f"algo must be 'rf' or 'gb', got '{algo}'")

        # Log params
        mlflow.log_param("algo", algo)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("class_weight", str(class_weight))
        mlflow.log_param("n_features", len(feature_names))

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        mlflow.log_metrics(metrics)

        # Save artefacts
        Path(models_dir).mkdir(exist_ok=True)
        model_path = Path(models_dir) / f"{model_name}.pkl"
        dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")

        schema = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "notes": "Sensor failure model. Keep feature order in sync with src/iot/features.py",
        }
        schema_path = Path(models_dir) / f"{model_name}_feature_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))
        mlflow.log_artifact(str(schema_path))
        mlflow.log_artifact(str(model_path))

        print(f"[MLflow] run_id={mlflow.active_run().info.run_id}")
        print(f"[Saved]  {model_path}")
        print(f"[Metrics] {json.dumps(metrics, indent=2)}")

    return metrics


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train IoT sensor failure prediction model."
    )
    parser.add_argument(
        "--data",
        help=(
            "Path to CSV or Parquet sensor telemetry file. "
            "If omitted, a synthetic dataset is used."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="iot-sensor-failure",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--algo",
        choices=["rf", "gb"],
        default="rf",
        help="Algorithm: 'rf' (Random Forest) or 'gb' (Gradient Boosting)",
    )
    parser.add_argument(
        "--model-name",
        default="sensor_failure_model",
        help="Base name for saved model file",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save model artefacts",
    )
    args = parser.parse_args()

    if args.data:
        df = load_sensor_csv(args.data)
        df = add_failure_label(df)
    else:
        print("[INFO] No --data supplied; using synthetic dataset.")
        df = make_synthetic_sensor_data(n_machines=10, n_rows_per_machine=100)

    train_sensor_model(
        df=df,
        model_name=args.model_name,
        experiment=args.experiment,
        algo=args.algo,
        models_dir=args.models_dir,
    )


if __name__ == "__main__":
    main()
