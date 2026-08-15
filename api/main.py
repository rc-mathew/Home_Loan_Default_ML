
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH", "models/home_loan_default_model.joblib"
))
METADATA_PATH = Path(os.getenv(
    "METADATA_PATH", "models/metadata.json"
))

app = FastAPI(
    title="Home Loan Default Risk API",
    version="1.0.0",
    description="Production-style API for home-loan default probability and risk segmentation."
)

model = None
metadata = {}


class LoanApplication(BaseModel):
    SK_ID_CURR: Optional[int] = None
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    CNT_CHILDREN: Optional[float] = None
    CNT_FAM_MEMBERS: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None
    REGION_RATING_CLIENT_W_CITY: Optional[float] = None
    CREDIT_TO_INCOME: Optional[float] = None
    ANNUITY_TO_INCOME: Optional[float] = None
    AGE_YEARS: Optional[float] = None
    EMPLOYMENT_YEARS: Optional[float] = None
    bureau_credit_count: Optional[float] = None
    bureau_AMT_CREDIT_SUM_mean: Optional[float] = None
    bureau_AMT_CREDIT_SUM_DEBT_mean: Optional[float] = None
    bureau_AMT_CREDIT_SUM_OVERDUE_max: Optional[float] = None
    previous_application_count: Optional[float] = None
    prev_AMT_CREDIT_mean: Optional[float] = None
    prev_AMT_APPLICATION_mean: Optional[float] = None
    pos_SK_DPD_mean: Optional[float] = None
    pos_SK_DPD_DEF_mean: Optional[float] = None
    pos_record_count: Optional[float] = None
    cc_AMT_BALANCE_mean: Optional[float] = None
    cc_AMT_CREDIT_LIMIT_ACTUAL_mean: Optional[float] = None
    cc_SK_DPD_mean: Optional[float] = None
    inst_payment_delay_mean: Optional[float] = None
    inst_payment_shortfall_mean: Optional[float] = None
    installment_record_count: Optional[float] = None


@app.on_event("startup")
def load_artifacts():
    global model, metadata
    if not MODEL_PATH.exists():
        return
    model = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text())


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model": metadata.get("model_name")
    }


@app.get("/metadata")
def get_metadata():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return metadata


def _segment(probability: float) -> str:
    if probability < 0.20:
        return "Eligible / Lower Risk"
    if probability < 0.40:
        return "Manual Review"
    return "High Risk"


@app.post("/predict")
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train the model and place the artifact in models/."
        )

    from src.features import SELECTED_FEATURES
    row = application.model_dump()
    row.pop("SK_ID_CURR", None)

    # Derive ratios if caller did not supply them.
    income = row.get("AMT_INCOME_TOTAL")
    credit = row.get("AMT_CREDIT")
    annuity = row.get("AMT_ANNUITY")

    if row.get("CREDIT_TO_INCOME") is None and income not in (None, 0) and credit is not None:
        row["CREDIT_TO_INCOME"] = credit / income
    if row.get("ANNUITY_TO_INCOME") is None and income not in (None, 0) and annuity is not None:
        row["ANNUITY_TO_INCOME"] = annuity / income
    if row.get("AGE_YEARS") is None and row.get("DAYS_BIRTH") is not None:
        row["AGE_YEARS"] = abs(row["DAYS_BIRTH"]) / 365.25
    if row.get("EMPLOYMENT_YEARS") is None and row.get("DAYS_EMPLOYED") is not None:
        row["EMPLOYMENT_YEARS"] = (
            abs(row["DAYS_EMPLOYED"]) / 365.25
            if row["DAYS_EMPLOYED"] < 0 else None
        )

    import pandas as pd
    X = pd.DataFrame([{c: row.get(c) for c in SELECTED_FEATURES}]).astype("float32")
    probability = float(model.predict_proba(X)[0, 1])
    segment = _segment(probability)

    return {
        "SK_ID_CURR": application.SK_ID_CURR,
        "default_probability": round(probability, 6),
        "risk_segment": segment,
        "decision_note": (
            "Use this score as a risk signal; final lending decisions must follow "
            "approved underwriting policy and governance."
        )
    }
