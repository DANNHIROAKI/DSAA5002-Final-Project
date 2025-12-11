"""Evaluation helpers for downstream response prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


def _to_1d_array(x: Any) -> np.ndarray:
    """Safely convert common array/Series types to a 1‑D NumPy array."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        # Assume single-column DataFrame
        arr = x.to_numpy().ravel()
    else:
        arr = np.asarray(x)

    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Return common classification metrics used in experiments.

    Parameters
    ----------
    y_true :
        Ground-truth binary labels.
    y_pred :
        Predicted binary labels (after thresholding).
    y_prob :
        Optional predicted probabilities for the positive class. Required to
        compute AUC and log-loss; if omitted, those metrics are set to ``None``.

    Returns
    -------
    dict
        Dictionary with keys:
        ``"auc"``, ``"f1"``, ``"log_loss"``, ``"accuracy"``, ``"precision"``,
        ``"recall"``, ``"balanced_accuracy"``, ``"specificity"``,
        ``"support_pos"``, and ``"support_neg"``.
        Some values may be ``None`` if not computable.
    """
    y_true_arr = _to_1d_array(y_true)
    y_pred_arr = _to_1d_array(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: "
            f"{y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )

    # Optional probability array
    if y_prob is None:
        y_prob_arr: Optional[np.ndarray] = None
    else:
        y_prob_arr = _to_1d_array(y_prob)
        if y_prob_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"y_true and y_prob have different lengths: "
                f"{y_true_arr.shape[0]} vs {y_prob_arr.shape[0]}"
            )

    # Always safe to compute with sensible defaults.
    accuracy = metrics.accuracy_score(y_true_arr, y_pred_arr)
    precision = metrics.precision_score(
        y_true_arr, y_pred_arr, zero_division=0
    )
    recall = metrics.recall_score(
        y_true_arr, y_pred_arr, zero_division=0
    )
    f1 = metrics.f1_score(
        y_true_arr, y_pred_arr, zero_division=0
    )

    # Confusion-matrix-based extras (always 2x2 with labels=[0, 1])
    cm = metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_neg = tn + fp
    support_pos = tp + fn

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    # Some metrics may fail (e.g., AUC when only one class present)
    if y_prob_arr is None:
        auc: Optional[float] = None
        log_loss: Optional[float] = None
    else:
        try:
            auc = metrics.roc_auc_score(y_true_arr, y_prob_arr)
        except ValueError:
            auc = None

        try:
            log_loss = metrics.log_loss(y_true_arr, y_prob_arr)
        except ValueError:
            log_loss = None

    return {
        "auc": auc,
        "f1": f1,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "support_pos": float(support_pos),
        "support_neg": float(support_neg),
    }
