from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

def evaluate_binary(y_true, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype("int8")
    return {
        "ROC_AUC": float(roc_auc_score(y_true, prob)),
        "PR_AUC": float(average_precision_score(y_true, prob)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }

def find_f2_threshold(y_true, prob, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -1.0
    for t in grid:
        pred = (np.asarray(prob) >= t).astype("int8")
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f2 = (5*p*r)/(4*p+r) if (p+r) else 0.0
        if f2 > best_score:
            best_score, best_t = f2, float(t)
    return best_t, float(best_score)
