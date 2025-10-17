# TODO: wire MLflow logging and baseline training
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.data import add_label, load_month
from src.features import build_features


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


def train_once(
    df: pd.DataFrame,
    model_name: str,
    experiment: str,
    algo: str = "rf",
    random_state: int = 42,
    test_size: float = 0.2,
    class_weight: str | None = "balanced",
):
    """
    Single training run with MLflow logging and model artifact.
    """
    mlflow.set_experiment(experiment)

    # Label + features
    df = add_label(df)
    X = build_features(df)
    y = df["HIGH_TOTAL"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    feature_names = list(X.columns)

    with mlflow.start_run(run_name=f"baseline-{algo}"):

        # Choose model
        if algo == "rf":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                n_jobs=-1,
                random_state=random_state,
                class_weight=class_weight,
            )
        elif algo == "logreg":
            model = LogisticRegression(
                max_iter=200,
                n_jobs=-1,
                random_state=random_state,
                class_weight=class_weight,
            )
        else:
            raise ValueError("algo must be 'rf' or 'logreg'")

        # Log params
        mlflow.log_param("algo", algo)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("class_weight", str(class_weight))
        mlflow.log_param("n_features", len(feature_names))

        # Fit
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob, threshold=0.5)
        mlflow.log_metrics(metrics)

        # Save model locally for API
        Path("models").mkdir(exist_ok=True)
        local_model_path = f"models/{model_name}.pkl"
        dump(model, local_model_path)

        # Log model to MLflow (with feature signature as artifact)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,  # you can set a name to use the registry later
        )

        # Save & log feature schema
        schema = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "notes": "Order matters for serving; keep in sync with src/features.py",
        }
        schema_path = Path("models") / f"{model_name}_feature_schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))
        mlflow.log_artifact(str(schema_path))

        # Save & log a small evaluation report
        report = {
            "counts": {
                "train": int(len(y_train)),
                "test": int(len(y_test)),
                "positives_test": int(y_test.sum()),
                "negatives_test": int((y_test == 0).sum()),
            },
            "metrics": metrics,
        }
        report_path = Path("models") / f"{model_name}_eval.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        # Also log the local model file (handy for FastAPI)
        mlflow.log_artifact(local_model_path)

        print(f"[MLflow] run_id={mlflow.active_run().info.run_id}")
        print(f"[Saved] {local_model_path}")
        print(f"[Metrics] {json.dumps(metrics, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline model with MLflow logging."
    )
    parser.add_argument(
        "--data", required=True, help="Path to CSV or Parquet month file"
    )
    parser.add_argument(
        "--month", required=False, help="YYYY-MM expected month for freshness check"
    )
    parser.add_argument(
        "--experiment", default="nyc-taxi-high-total", help="MLflow experiment name"
    )
    parser.add_argument(
        "--algo", choices=["rf", "logreg"], default="rf", help="Algorithm"
    )
    parser.add_argument(
        "--model-name", default="model", help="Base name for saved model file"
    )
    args = parser.parse_args()

    # Load & validate month (freshness if provided)
    df = load_month(args.data, month=args.month)

    # Train + log
    train_once(
        df=df,
        model_name=args.model_name,
        experiment=args.experiment,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
