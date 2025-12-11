"""Evaluation helpers for downstream response prediction."""
from typing import Tuple
import pandas as pd
from sklearn import metrics


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict:
    """Return common classification metrics used in experiments."""
    return {
        "auc": metrics.roc_auc_score(y_true, y_prob) if y_prob is not None else None,
        "f1": metrics.f1_score(y_true, y_pred),
        "log_loss": metrics.log_loss(y_true, y_prob) if y_prob is not None else None,
    }
