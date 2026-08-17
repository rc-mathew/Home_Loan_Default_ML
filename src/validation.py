from __future__ import annotations
import pandas as pd

def temporal_split(df: pd.DataFrame, date_column: str, train_fraction=0.60, validation_fraction=0.20):
    """Return chronological train/validation/test index labels.

    This must use a true application/decision timestamp. The Home Credit Kaggle
    files do not provide such a timestamp, so callers should not substitute
    SK_ID_CURR or relative feature dates and call it temporal validation.
    """
    if date_column not in df.columns:
        raise ValueError(f"True temporal validation requires '{date_column}' in the application table.")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError("Temporal validation date column contains missing/unparseable values.")
    order = dates.sort_values().index
    n = len(order); n_train = int(n*train_fraction); n_val = int(n*validation_fraction)
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]
