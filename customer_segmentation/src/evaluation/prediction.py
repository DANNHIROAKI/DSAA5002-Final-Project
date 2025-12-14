"""Evaluation helpers for downstream response prediction.

This module is used by both baseline and RAMoE/HyRAMoE experiment scripts.

Compared to a pure accuracy-centric evaluation, this project also cares about
**ranking quality under a marketing budget** (lift@top-q). Ranking metrics are
implemented in :mod:`customer_segmentation.src.utils.metrics_utils`; this module
provides a thin wrapper so that experiments can optionally log them together
with standard classification metrics.

Key APIs
--------
- :func:`choose_threshold`: pick a probability threshold on validation data.
- :func:`compute_classification_metrics`: compute AUC/PR-AUC/log-loss/Brier and
  thresholded metrics.
- :func:`expected_calibration_error`: simple ECE implementation.
- :func:`compute_ranking_metrics`: budget-oriented summary (precision/recall/lift).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn import metrics

from customer_segmentation.src.utils.metrics_utils import (
    compute_lift,
    precision_at_frac,
    recall_at_frac,
    positives_in_top_frac,
    expected_score_sum_in_top_frac,
)


# ---------------------------------------------------------------------------
# Array conversion helpers
# ---------------------------------------------------------------------------


def _to_1d_labels(x: Any) -> np.ndarray:
    """Convert labels to a 1D NumPy array."""
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
    return np.asarray(arr).reshape(-1)


def _to_1d_proba(x: Any) -> np.ndarray:
    """Convert probability-like input to a 1D array of P(y=1).

    Accepts:
    - pd.Series
    - single-column DataFrame -> that column
    - two-column DataFrame/ndarray -> take column 1 as positive-class probability
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
            else:
                raise ValueError(
                    f"Ambiguous probability array shape={arr.shape}. "
                    "Pass a 1D positive-class probability or a (n,2) predict_proba output."
                )
        else:
            arr = arr.reshape(-1)

    return np.asarray(arr, dtype=float).reshape(-1)


def _drop_nan_pairs(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either y or score/proba is NaN/Inf."""
    y = _to_1d_labels(y_true).astype(float)
    s = _to_1d_proba(y_score).astype(float)

    if y.shape[0] != s.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y.shape[0]} vs y_score={s.shape[0]}")

    mask = np.isfinite(y) & np.isfinite(s)
    return y[mask], s[mask]


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def choose_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    metric: Literal["f1", "balanced_accuracy", "youden_j"] = "f1",
    grid: Optional[Iterable[float]] = None,
    grid_size: int = 101,
) -> float:
    """Choose a probability threshold to optimize a metric on validation data.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_prob:
        Predicted probabilities P(y=1).
    metric:
        - "f1": maximize F1.
        - "balanced_accuracy": maximize balanced accuracy.
        - "youden_j": maximize (TPR - FPR).
    grid:
        Optional iterable of thresholds. If None, use linspace(0,1,grid_size).
    grid_size:
        Number of grid points when grid is None.

    Returns
    -------
    float
        Best threshold in [0,1].
    """
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = y.astype(int)

    if grid is None:
        thresholds = np.linspace(0.0, 1.0, int(grid_size))
    else:
        thresholds = np.array(list(grid), dtype=float)

    best_t = 0.5
    best_score = -np.inf

    for t in thresholds:
        pred = (p >= float(t)).astype(int)

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
            best_score = float(score)
            best_t = float(t)

    return float(best_t)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_b (|acc_b - conf_b| * w_b)

    where:
      - b is a probability bin,
      - acc_b is empirical accuracy in that bin,
      - conf_b is mean predicted probability in that bin,
      - w_b is the fraction of samples in that bin.

    Parameters
    ----------
    y_true:
        Binary labels (0/1).
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of bins.
    strategy:
        - "uniform": equal-width bins over [0,1].
        - "quantile": bins with equal sample counts.
    return_details:
        If True, return a dict with per-bin stats.

    Returns
    -------
    float or dict
        ECE value, optionally with details.
    """
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = (y > 0.0).astype(int)

    # Clip for numerical safety
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    n = int(y.shape[0])
    if n == 0:
        out = float("nan")
        return {"ece": out, "bins": []} if return_details else out

    n_bins = int(max(2, n_bins))

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        # Ensure unique bin edges; fall back to uniform if ties break quantiles.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(p, qs)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        if np.unique(bin_edges).size < bin_edges.size:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Assign each probability to a bin index in [0, n_bins-1]
    # Rightmost edge inclusive
    bin_ids = np.digitize(p, bin_edges[1:-1], right=False)

    ece = 0.0
    bins: list[dict[str, float]] = []
    max_gap = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        w_b = float(np.mean(mask))
        conf_b = float(np.mean(p[mask]))
        acc_b = float(np.mean(y[mask]))
        gap = abs(acc_b - conf_b)
        ece += w_b * gap
        max_gap = max(max_gap, gap)
        bins.append(
            {
                "bin": float(b),
                "count": float(np.sum(mask)),
                "weight": w_b,
                "conf": conf_b,
                "acc": acc_b,
                "gap": float(gap),
            }
        )

    if return_details:
        return {"ece": float(ece), "mce": float(max_gap), "bins": bins}
    return float(ece)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: Optional[pd.Series | np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute common classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels (after thresholding).
    y_prob:
        Optional predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys:
        - auc
        - average_precision
        - f1
        - log_loss
        - brier
        - ece
        - accuracy
        - precision
        - recall
        - balanced_accuracy
        - specificity
        - support_pos
        - support_neg

        Probability-based metrics are None when y_prob is not provided.
    """
    y_true_arr = _to_1d_labels(y_true).astype(int)
    y_pred_arr = _to_1d_labels(y_pred).astype(int)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )

    # thresholded metrics
    accuracy = float(metrics.accuracy_score(y_true_arr, y_pred_arr))
    precision = float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))

    cm = metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_neg = float(tn + fp)
    support_pos = float(tp + fn)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float(0.5 * (recall + specificity))

    # probability metrics
    if y_prob is None:
        auc: Optional[float] = None
        ap: Optional[float] = None
        log_loss: Optional[float] = None
        brier: Optional[float] = None
        ece: Optional[float] = None
    else:
        y_prob_arr = _to_1d_proba(y_prob)
        if y_prob_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"y_true and y_prob have different lengths: {y_true_arr.shape[0]} vs {y_prob_arr.shape[0]}"
            )

        eps = 1e-15
        p = np.clip(y_prob_arr, eps, 1.0 - eps)

        try:
            auc = float(metrics.roc_auc_score(y_true_arr, p))
        except ValueError:
            auc = None

        try:
            ap = float(metrics.average_precision_score(y_true_arr, p))
        except ValueError:
            ap = None

        try:
            log_loss = float(metrics.log_loss(y_true_arr, p, labels=[0, 1]))
        except ValueError:
            log_loss = None

        try:
            brier = float(metrics.brier_score_loss(y_true_arr, p))
        except ValueError:
            brier = None

        try:
            ece = float(expected_calibration_error(y_true_arr, p, n_bins=10))
        except Exception:
            ece = None

    return {
        "auc": auc,
        "average_precision": ap,
        "f1": f1,
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": balanced_accuracy,
        "specificity": specificity,
        "support_pos": support_pos,
        "support_neg": support_neg,
    }


def compute_ranking_metrics(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    *,
    top_frac: float = 0.2,
) -> Dict[str, float]:
    """Compute budget/ranking-oriented metrics at a single budget fraction.

    The definitions follow :mod:`customer_segmentation.src.utils.metrics_utils`.

    Returns a small dict with stable key names used in report tables.
    """
    y, s = _drop_nan_pairs(y_true, scores)
    y_bin = (y > 0.0).astype(int)

    base_rate = float(np.mean(y_bin)) if y_bin.size > 0 else float("nan")

    return {
        "base_rate": base_rate,
        "precision_top": float(precision_at_frac(y_bin, s, top_frac=top_frac)),
        "recall_top": float(recall_at_frac(y_bin, s, top_frac=top_frac)),
        "lift_top": float(compute_lift(y_bin, s, top_frac=top_frac)),
        "positives_in_top": float(positives_in_top_frac(y_bin, s, top_frac=top_frac)),
        "expected_score_sum_in_top": float(expected_score_sum_in_top_frac(s, top_frac=top_frac)),
    }


__all__ = [
    "choose_threshold",
    "compute_classification_metrics",
    "expected_calibration_error",
    "compute_ranking_metrics",
]
