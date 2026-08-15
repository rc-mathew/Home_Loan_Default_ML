
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


SELECTED_FEATURES = [
    "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE",
    "CNT_CHILDREN","CNT_FAM_MEMBERS","DAYS_BIRTH","DAYS_EMPLOYED",
    "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
    "REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY",
    "CREDIT_TO_INCOME","ANNUITY_TO_INCOME","AGE_YEARS","EMPLOYMENT_YEARS",
    "bureau_credit_count","bureau_AMT_CREDIT_SUM_mean",
    "bureau_AMT_CREDIT_SUM_DEBT_mean","bureau_AMT_CREDIT_SUM_OVERDUE_max",
    "previous_application_count","prev_AMT_CREDIT_mean","prev_AMT_APPLICATION_mean",
    "pos_SK_DPD_mean","pos_SK_DPD_DEF_mean","pos_record_count",
    "cc_AMT_BALANCE_mean","cc_AMT_CREDIT_LIMIT_ACTUAL_mean","cc_SK_DPD_mean",
    "inst_payment_delay_mean","inst_payment_shortfall_mean",
    "installment_record_count",
]

def compact_aggregate(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != key]
    if not numeric_cols:
        return df[[key]].drop_duplicates()
    agg = {c: ["mean", "max", "min"] for c in numeric_cols}
    out = df.groupby(key, sort=False).agg(agg)
    out.columns = [f"{prefix}{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()

def _read_selected(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in columns if c in header]
    return pd.read_csv(path, usecols=usecols)

def _merge_count(features: pd.DataFrame, raw: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    counts = raw.groupby(key, sort=False).size().rename(name).reset_index()
    return features.merge(counts, on=key, how="left")

def _safe_ratio(a, b):
    return np.where(b.notna() & (b != 0), a / b, np.nan)

def build_feature_table(data_dir: str | Path) -> pd.DataFrame:
    """Build the same compact applicant-level feature table used by the 8 GB notebook."""
    data_dir = Path(data_dir)
    app_path = data_dir / "application_train.csv"
    if not app_path.exists():
        raise FileNotFoundError(f"Missing {app_path}")

    application = pd.read_csv(app_path)
    base = application.copy()

    # Bureau
    path = data_dir / "bureau.csv"
    if path.exists():
        cols = ["SK_ID_CURR","DAYS_CREDIT","CREDIT_DAY_OVERDUE",
                "AMT_CREDIT_SUM","AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE","CNT_CREDIT_PROLONG","AMT_ANNUITY"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "bureau_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "bureau_credit_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft
        gc.collect()

    # Previous applications
    path = data_dir / "previous_application.csv"
    if path.exists():
        cols = ["SK_ID_CURR","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT",
                "AMT_DOWN_PAYMENT","AMT_GOODS_PRICE","DAYS_DECISION","CNT_PAYMENT"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "prev_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "previous_application_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft
        gc.collect()

    # POS/CASH
    path = data_dir / "POS_CASH_balance.csv"
    if path.exists():
        cols = ["SK_ID_CURR","MONTHS_BALANCE","CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE","SK_DPD","SK_DPD_DEF"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "pos_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "pos_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft
        gc.collect()

    # Credit card
    path = data_dir / "credit_card_balance.csv"
    if path.exists():
        cols = ["SK_ID_CURR","MONTHS_BALANCE","AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL","AMT_DRAWINGS_ATM",
                "AMT_DRAWINGS_CURRENT","AMT_PAYMENT_TOTAL_CURRENT",
                "AMT_RECEIVABLE_PRINCIPAL","SK_DPD","SK_DPD_DEF"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "cc_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "cc_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft
        gc.collect()

    # Installments
    path = data_dir / "installments_payments.csv"
    if path.exists():
        cols = ["SK_ID_CURR","DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT","AMT_PAYMENT"]
        raw = _read_selected(path, cols)
        if {"DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT"}.issubset(raw.columns):
            raw["payment_delay"] = raw["DAYS_ENTRY_PAYMENT"] - raw["DAYS_INSTALMENT"]
        if {"AMT_INSTALMENT","AMT_PAYMENT"}.issubset(raw.columns):
            raw["payment_shortfall"] = raw["AMT_INSTALMENT"] - raw["AMT_PAYMENT"]
        ft = compact_aggregate(raw, "SK_ID_CURR", "inst_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "installment_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft
        gc.collect()

    # Application-level features
    if {"AMT_CREDIT","AMT_INCOME_TOTAL"}.issubset(base.columns):
        base["CREDIT_TO_INCOME"] = _safe_ratio(base["AMT_CREDIT"], base["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(base.columns):
        base["ANNUITY_TO_INCOME"] = _safe_ratio(base["AMT_ANNUITY"], base["AMT_INCOME_TOTAL"])
    if "DAYS_BIRTH" in base.columns:
        base["AGE_YEARS"] = base["DAYS_BIRTH"].abs() / 365.25
    if "DAYS_EMPLOYED" in base.columns:
        base["EMPLOYMENT_YEARS"] = np.where(
            base["DAYS_EMPLOYED"] < 0,
            base["DAYS_EMPLOYED"].abs() / 365.25,
            np.nan
        )

    return base

def make_model_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SELECTED_FEATURES if c not in feature_df.columns]
    X = feature_df.reindex(columns=SELECTED_FEATURES).copy()
    for c in SELECTED_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype("float32")
    return X
