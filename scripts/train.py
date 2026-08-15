
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import build_feature_table, make_model_matrix, SELECTED_FEATURES


def train(data_dir: str, model_dir: str, test_size: float = 0.20, seed: int = 42):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_table(data_dir)
    if "TARGET" not in df.columns:
        raise ValueError("application_train.csv must contain TARGET for training.")

    X = make_model_matrix(df)
    y = df["TARGET"].astype("int8")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / max(positive, 1)

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=100, class_weight="balanced",
                solver="lbfgs", random_state=seed
            ))
        ]),
        "xgboost": XGBClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            min_child_weight=20, subsample=0.80, colsample_bytree=0.70,
            reg_alpha=0.10, reg_lambda=5.0, gamma=0.0,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", max_bin=64, device="cpu",
            n_jobs=2, scale_pos_weight=scale_pos_weight,
            random_state=seed
        )
    }

    results = []
    fitted = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype("int8")
        row = {
            "Model": name,
            "ROC_AUC": float(roc_auc_score(y_test, prob)),
            "PR_AUC": float(average_precision_score(y_test, prob)),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
        }
        results.append(row)
        fitted[name] = model
        print(row)

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]

    # Refit selected model on all labeled training data before production packaging.
    # For a simple deployment demo, this uses all available labeled examples.
    print(f"Best model: {best_name}")
    best_model.fit(X, y)

    model_path = model_dir / "home_loan_default_model.joblib"
    joblib.dump(best_model, model_path, compress=3)

    metadata = {
        "model_name": best_name,
        "target": "TARGET",
        "thresholds": {"eligible_max_exclusive": 0.20, "manual_review_max_exclusive": 0.40},
        "selected_features": SELECTED_FEATURES,
        "n_features": len(SELECTED_FEATURES),
        "random_state": seed,
        "test_size": test_size,
        "xgboost_scale_pos_weight": scale_pos_weight,
        "metrics": results_df.to_dict(orient="records"),
        "note": "This artifact scores the compact applicant-level feature matrix produced by src/features.py."
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results_df.to_csv(model_dir / "model_metrics.csv", index=False)
    print(f"Saved model to {model_path}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
