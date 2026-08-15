
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_feature_table, make_model_matrix


def risk_segment(p: float) -> str:
    if p < 0.20:
        return "Eligible / Lower Risk"
    if p < 0.40:
        return "Manual Review"
    return "High Risk"


def score(data_dir: str, model_path: str, output_path: str):
    model = joblib.load(model_path)
    features = build_feature_table(data_dir)
    X = make_model_matrix(features)
    prob = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "SK_ID_CURR": features["SK_ID_CURR"].values,
        "Predicted_Default_Probability": prob,
    })
    out["Risk_Segment"] = out["Predicted_Default_Probability"].map(risk_segment)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.head(20))
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/home_loan_default_model.joblib")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()
    score(args.data_dir, args.model_path, args.output)
