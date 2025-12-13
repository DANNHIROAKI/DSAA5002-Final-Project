"""Evaluation helpers for downstream response prediction.

Upgrades for the new methodology:
- Add choose_threshold() (validation-based threshold selection).
- Add PR-AUC (average precision) and Brier score for imbalanced response modeling.
- More robust probability handling (accept 1D probas or 2D predict_proba outputs).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Literal, Iterable

import numpy as np
import pandas as pd
from sklearn import metrics


def _to_1d_labels(x: Any) -> np.ndarray:
    """Safely convert common array/Series types to a 1‑D NumPy array (labels)."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected a single-column DataFrame for labels, got shape={x.shape}."
            )
        arr = x.iloc[:, 0].to_numpy()
    else:
        arr = np.asarray(x)

    arr = np.asarray(arr).reshape(-1)
    return arr


def _to_1d_proba(x: Any) -> np.ndarray:
    """Convert probability-like input to a 1D array of P(y=1).

    Accepts:
    - pd.Series
    - single-column DataFrame -> that column
    - two-column DataFrame/ndarray -> take column 1 as positive class prob
    - 1D ndarray/list -> as-is
    """
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            arr = x.iloc[:, 0].to_numpy()
        elif x.shape[1] == 2:
            arr = x.iloc[:, 1].to_numpy()
        else:
            raise ValueError(
                f"Expected 1 or 2 columns for probabilities, got shape={x.shape}."
            )
    else:
        arr = np.asarray(x)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                arr = arr[:, 1]
            elif arr.shape[0] == 2 and arr.shape[1] != 2:
                # ambiguous; flatten as fallback
                arr = arr.reshape(-1)
            else:
                # if model returned multi-class, user must pass the positive class prob
                raise ValueError(
                    f"Ambiguous probability array shape={arr.shape}. "
                    "Pass a 1D positive-class probability or a (n,2) predict_proba output."
                )
        else:
            arr = arr.reshape(-1)

    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr


def choose_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1",
    grid: Optional[Iterable[float]] = None,
    grid_size: int = 101,
) -> float:
    """Choose a probability threshold to optimize a metric on a validation set.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_prob:
        Predicted probabilities P(y=1).
    metric:
        - "f1": maximize F1.
        - "balanced_accuracy": maximize balanced accuracy.
        - "youden_j": maximize (TPR - FPR) = sensitivity + specificity - 1.
    grid:
        Optional iterable of thresholds. If None, use linspace(0,1,grid_size).
    grid_size:
        Number of grid points when grid is None.

    Returns
    -------
    float
        Best threshold in [0,1].
    """
    y = _to_1d_labels(y_true).astype(int)
    p = _to_1d_proba(y_prob)

    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true and y_prob have different lengths: {y.shape[0]} vs {p.shape[0]}")

    if grid is None:
        thresholds = np.linspace(0.0, 1.0, int(grid_size))
    else:
        thresholds = np.array(list(grid), dtype=float)

    best_t = 0.5
    best_score = -np.inf

    for t in thresholds:
        pred = (p >= t).astype(int)

        if metric == "f1":
            score = metrics.f1_score(y, pred, zero_division=0)
        elif metric == "balanced_accuracy":
            score = metrics.balanced_accuracy_score(y, pred)
        elif metric == "youden_j":
            cm = metrics.confusion_matrix(y, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:  # pragma: no cover
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_t = float(t)

    return float(best_t)


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
        compute AUC, PR-AUC and log-loss; if omitted, those metrics are set to ``None``.

    Returns
    -------
    dict
        Dictionary with keys:
        ``"auc"``, ``"average_precision"``, ``"f1"``, ``"log_loss"``, ``"brier"``,
        ``"accuracy"``, ``"precision"``, ``"recall"``, ``"balanced_accuracy"``,
        ``"specificity"``, ``"support_pos"``, and ``"support_neg"``.
        Some values may be ``None`` if not computable.
    """
    y_true_arr = _to_1d_labels(y_true).astype(int)
    y_pred_arr = _to_1d_labels(y_pred).astype(int)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: "
            f"{y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )

    # Always safe to compute with sensible defaults.
    accuracy = float(metrics.accuracy_score(y_true_arr, y_pred_arr))
    precision = float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # Confusion-matrix-based extras (always 2x2 with labels=[0, 1])
    cm = metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_neg = float(tn + fp)
    support_pos = float(tp + fn)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float(0.5 * (recall + specificity))

    # Optional probability metrics
    if y_prob is None:
        auc: Optional[float] = None
        average_precision: Optional[float] = None
        log_loss: Optional[float] = None
        brier: Optional[float] = None
    else:
        y_prob_arr = _to_1d_proba(y_prob)
        if y_prob_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"y_true and y_prob have different lengths: "
                f"{y_true_arr.shape[0]} vs {y_prob_arr.shape[0]}"
            )

        # clip probabilities for numeric stability (esp. log-loss)
        eps = 1e-15
        y_prob_clip = np.clip(y_prob_arr, eps, 1.0 - eps)

        try:
            auc = float(metrics.roc_auc_score(y_true_arr, y_prob_clip))
        except ValueError:
            auc = None

        try:
            average_precision = float(metrics.average_precision_score(y_true_arr, y_prob_clip))
        except ValueError:
            average_precision = None

        try:
            # labels ensures consistent behavior even if a split misses a class
            log_loss = float(metrics.log_loss(y_true_arr, y_prob_clip, labels=[0, 1]))
        except ValueError:
            log_loss = None

        try:
            brier = float(metrics.brier_score_loss(y_true_arr, y_prob_clip))
        except ValueError:
            brier = None

    return {
        "auc": auc,
        "average_precision": average_precision,
        "f1": f1,
        "log_loss": log_loss,
        "brier": brier,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "support_pos": support_pos,
        "support_neg": support_neg,
    }
