from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Compact, numeric applicant-level feature set. bureau_balance-derived features
# are included to close the original 8 GB implementation gap.
SELECTED_FEATURES = [
    "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE",
    "CNT_CHILDREN","CNT_FAM_MEMBERS","DAYS_BIRTH","DAYS_EMPLOYED",
    "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
    "REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY",
    "CREDIT_TO_INCOME","ANNUITY_TO_INCOME","AGE_YEARS","EMPLOYMENT_YEARS",
    "bureau_credit_count","bureau_AMT_CREDIT_SUM_mean",
    "bureau_AMT_CREDIT_SUM_DEBT_mean","bureau_AMT_CREDIT_SUM_OVERDUE_max",
    "bureau_balance_record_count","bureau_balance_delinquency_rate",
    "bureau_balance_delinquency_max",
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


def _read_selected(path: Path, columns: Iterable[str], dtype: dict | None = None) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in columns if c in header]
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


def _merge_count(features: pd.DataFrame, raw: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    counts = raw.groupby(key, sort=False).size().rename(name).reset_index()
    return features.merge(counts, on=key, how="left")


def _safe_ratio(a, b):
    return np.where(b.notna() & (b != 0), a / b, np.nan)


def _bureau_balance_features(data_dir: Path, bureau_map: pd.DataFrame, chunksize: int = 200_000) -> pd.DataFrame:
    """Aggregate bureau_balance in chunks so it can run on an 8 GB laptop.

    The original notebook skipped this table. This version uses only SK_ID_BUREAU
    and STATUS, maps to SK_ID_CURR, and accumulates compact applicant-level sums/counts.
    """
    path = data_dir / "bureau_balance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR", "bureau_balance_record_count",
                                     "bureau_balance_delinquency_rate", "bureau_balance_delinquency_max"])

    keep = bureau_map[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates("SK_ID_BUREAU")
    totals = {}
    for chunk in pd.read_csv(path, usecols=["SK_ID_BUREAU", "STATUS"], chunksize=chunksize):
        chunk = chunk.merge(keep, on="SK_ID_BUREAU", how="inner")
        if chunk.empty:
            continue
        status = chunk["STATUS"].astype("string")
        # Home Credit statuses 1-5 are delinquency buckets; C/X/0 are treated as non-delinquent.
        delinquent = pd.to_numeric(status, errors="coerce").fillna(0).between(1, 5).astype("int8")
        chunk["_delinq"] = delinquent
        g = chunk.groupby("SK_ID_CURR", sort=False).agg(
            _records=("_delinq", "size"),
            _delinq=("_delinq", "sum"),
            _max_delinq=("_delinq", "max"),
        )
        for idx, row in g.iterrows():
            rec, de, mx = int(row["_records"]), int(row["_delinq"]), int(row["_max_delinq"])
            if idx not in totals:
                totals[idx] = [0, 0, 0]
            totals[idx][0] += rec; totals[idx][1] += de; totals[idx][2] = max(totals[idx][2], mx)
        del chunk, g
        gc.collect()
    if not totals:
        return pd.DataFrame(columns=["SK_ID_CURR", "bureau_balance_record_count",
                                     "bureau_balance_delinquency_rate", "bureau_balance_delinquency_max"])
    rows = [(k, v[0], v[1] / v[0] if v[0] else np.nan, v[2]) for k, v in totals.items()]
    return pd.DataFrame(rows, columns=["SK_ID_CURR", "bureau_balance_record_count",
                                       "bureau_balance_delinquency_rate", "bureau_balance_delinquency_max"])


def build_feature_table(data_dir: str | Path) -> pd.DataFrame:
    """Build applicant-level features using sequential, memory-conscious aggregation."""
    data_dir = Path(data_dir)
    app_path = data_dir / "application_train.csv"
    if not app_path.exists():
        raise FileNotFoundError(f"Missing {app_path}")

    application = pd.read_csv(app_path)
    base = application.copy()

    # Bureau + bureau_balance. Keep the mapping small before scanning bureau_balance.
    bureau_path = data_dir / "bureau.csv"
    if bureau_path.exists():
        map_df = _read_selected(bureau_path, ["SK_ID_BUREAU", "SK_ID_CURR"],
                                dtype={"SK_ID_BUREAU": "int64", "SK_ID_CURR": "int64"})
        cols = ["SK_ID_CURR","DAYS_CREDIT","CREDIT_DAY_OVERDUE","AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT","AMT_CREDIT_SUM_OVERDUE","CNT_CREDIT_PROLONG","AMT_ANNUITY"]
        raw = _read_selected(bureau_path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "bureau_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "bureau_credit_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        bb = _bureau_balance_features(data_dir, map_df)
        base = base.merge(bb, on="SK_ID_CURR", how="left")
        del map_df, raw, ft, bb
        gc.collect()

    # Previous applications
    path = data_dir / "previous_application.csv"
    if path.exists():
        cols = ["SK_ID_CURR","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT",
                "AMT_GOODS_PRICE","DAYS_DECISION","CNT_PAYMENT"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "prev_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "previous_application_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft; gc.collect()

    # POS/CASH
    path = data_dir / "POS_CASH_balance.csv"
    if path.exists():
        cols = ["SK_ID_CURR","MONTHS_BALANCE","CNT_INSTALMENT","CNT_INSTALMENT_FUTURE","SK_DPD","SK_DPD_DEF"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "pos_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "pos_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft; gc.collect()

    # Credit card
    path = data_dir / "credit_card_balance.csv"
    if path.exists():
        cols = ["SK_ID_CURR","MONTHS_BALANCE","AMT_BALANCE","AMT_CREDIT_LIMIT_ACTUAL","AMT_DRAWINGS_ATM",
                "AMT_DRAWINGS_CURRENT","AMT_PAYMENT_TOTAL_CURRENT","AMT_RECEIVABLE_PRINCIPAL","SK_DPD","SK_DPD_DEF"]
        raw = _read_selected(path, cols)
        ft = compact_aggregate(raw, "SK_ID_CURR", "cc_")
        ft = _merge_count(ft, raw, "SK_ID_CURR", "cc_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft; gc.collect()

    # Installments
    path = data_dir / "installments_payments.csv"
    if path.exists():
        cols = ["SK_ID_CURR","DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT","AMT_INSTALMENT","AMT_PAYMENT"]
        raw = _read_selected(path, cols)
        if {"DAYS_INSTALMENT","DAYS_ENTRY_PAYMENT"}.issubset(raw.columns):
            raw["payment_delay"] = raw["DAYS_ENTRY_PAYMENT"] - raw["DAYS_INSTALMENT"]
        if {"AMT_INSTALMENT","AMT_PAYMENT"}.issubset(raw.columns):
            raw["payment_shortfall"] = raw["AMT_INSTALMENT"] - raw["AMT_PAYMENT"]
        ft = compact_aggregate(raw, "SK_ID_CURR", "inst_")
        # Use explicit business names for the two engineered installment features.
        if "inst_payment_delay_mean" not in ft.columns and "payment_delay" in raw.columns:
            ft = ft.rename(columns={"inst_payment_delay_mean": "inst_payment_delay_mean"})
        ft = _merge_count(ft, raw, "SK_ID_CURR", "installment_record_count")
        base = base.merge(ft, on="SK_ID_CURR", how="left")
        del raw, ft; gc.collect()

    if {"AMT_CREDIT","AMT_INCOME_TOTAL"}.issubset(base.columns):
        base["CREDIT_TO_INCOME"] = _safe_ratio(base["AMT_CREDIT"], base["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(base.columns):
        base["ANNUITY_TO_INCOME"] = _safe_ratio(base["AMT_ANNUITY"], base["AMT_INCOME_TOTAL"])
    if "DAYS_BIRTH" in base.columns:
        base["AGE_YEARS"] = base["DAYS_BIRTH"].abs() / 365.25
    if "DAYS_EMPLOYED" in base.columns:
        base["EMPLOYMENT_YEARS"] = np.where(base["DAYS_EMPLOYED"] < 0,
                                             base["DAYS_EMPLOYED"].abs() / 365.25, np.nan)
    return base


def make_model_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    X = feature_df.reindex(columns=SELECTED_FEATURES).copy()
    for c in SELECTED_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype("float32")
