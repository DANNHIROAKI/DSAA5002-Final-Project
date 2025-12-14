"""Generic metric helpers used across experiments.

Why ranking metrics?
--------------------
In campaign response modelling, the business objective is usually *budgeted*:
we can only target the top-q% of customers. Therefore, besides standard
classification metrics (AUC/LogLoss), we report ranking metrics such as
``lift@top-q`` and ``precision@top-q``.

Upgrades for the new methodology (RAMoE / HyRAMoE)
-------------------------------------------------
- Robust handling of NaNs and pandas index alignment.
- Provide business-oriented ranking metrics beyond lift:
  precision@q, recall@q (capture rate), positives_in_topq,
  expected_score_sum_in_topq.
- Provide diagnostics for *soft assignments* (entropy / max-prob) which are
  important for RAMoE/HyRAMoE interpretability and stability.

All helpers are dependency-light (NumPy + pandas).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, Sequence[int], Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """Convert input values to a pandas Series (preserve index if possible)."""
    if isinstance(values, pd.Series):
        s = values.copy()
        s.name = name
        return s
    return pd.Series(values, name=name)


def _coerce_numeric(series: pd.Series, *, name: str) -> pd.Series:
    """Convert to numeric with coercion (invalid -> NaN)."""
    out = pd.to_numeric(series, errors="coerce")
    out.name = name
    return out


def _aligned_nonan(y_true: ArrayLike, scores: ArrayLike) -> Tuple[pd.Series, pd.Series]:
    """Align ``y_true`` and ``scores`` (by index if possible) and drop NaNs."""
    y = _coerce_numeric(_to_series(y_true, "y_true"), name="y_true")
    s = _coerce_numeric(_to_series(scores, "score"), name="score")

    df = pd.concat([y, s], axis=1)
    df = df.dropna(subset=["y_true", "score"])
    return df["y_true"], df["score"]


def _validate_top_frac(top_frac: float) -> float:
    f = float(top_frac)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}.")
    return f


# ---------------------------------------------------------------------------
# Ranking / business metrics
# ---------------------------------------------------------------------------


def top_k_from_frac(n: int, top_frac: float) -> int:
    """Convert a budget fraction to an integer top-k (ceil)."""
    if n <= 0:
        return 0
    f = _validate_top_frac(top_frac)
    return int(np.ceil(n * f))


def precision_at_frac(y_true: ArrayLike, scores: ArrayLike, top_frac: float = 0.2) -> float:
    """Precision among top-q% ranked by ``scores`` (descending)."""
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    k = top_k_from_frac(len(y), top_frac)
    if k <= 0:
        return float("nan")

    order = s.sort_values(ascending=False).index
    y_sorted = y.loc[order].astype(float)
    return float(y_sorted.iloc[:k].mean())


def recall_at_frac(y_true: ArrayLike, scores: ArrayLike, top_frac: float = 0.2) -> float:
    """Recall (capture rate) within top-q% ranked by ``scores``.

    Defined as:
        (# positives in top-q%) / (total # positives)
    """
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    y_bin = (y.astype(float) > 0.0).astype(int)
    total_pos = int(y_bin.sum())
    if total_pos == 0:
        return float("nan")

    k = top_k_from_frac(len(y_bin), top_frac)
    order = s.sort_values(ascending=False).index
    y_sorted = y_bin.loc[order]
    pos_in_top = int(y_sorted.iloc[:k].sum())
    return float(pos_in_top / total_pos)


def positives_in_top_frac(y_true: ArrayLike, scores: ArrayLike, top_frac: float = 0.2) -> float:
    """Number of positives contained in the top-q% ranked by ``scores``."""
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    y_bin = (y.astype(float) > 0.0).astype(int)
    k = top_k_from_frac(len(y_bin), top_frac)
    order = s.sort_values(ascending=False).index
    y_sorted = y_bin.loc[order]
    return float(y_sorted.iloc[:k].sum())


def expected_score_sum_in_top_frac(scores: ArrayLike, top_frac: float = 0.2) -> float:
    """Sum of scores in the top-q%.

    When ``scores`` are calibrated probabilities, this approximates the expected
    number of responders under a budgeted targeting policy.
    """
    s = _coerce_numeric(_to_series(scores, "score"), name="score").dropna()
    if len(s) == 0:
        return float("nan")

    k = top_k_from_frac(len(s), top_frac)
    order = s.sort_values(ascending=False).index
    s_sorted = s.loc[order].astype(float)
    return float(s_sorted.iloc[:k].sum())


def compute_lift(y_true: ArrayLike, scores: ArrayLike, top_frac: float = 0.2) -> float:
    """Compute lift@top-q% based on a ranking by ``scores``.

    Lift is defined as:
        precision@top_q% / base_rate
    where base_rate is the overall positive rate.
    """
    f = _validate_top_frac(top_frac)
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    y_bin = (y.astype(float) > 0.0).astype(int)
    base_rate = float(y_bin.mean())
    if base_rate == 0.0:
        return float("nan")

    prec = precision_at_frac(y_bin, s, top_frac=f)
    if np.isnan(prec):
        return float("nan")

    return float(prec / base_rate)


def lift_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Compute lift at multiple budget fractions.

    Output schema (backward compatible):
        - frac: budget fraction
        - lift: lift@frac
    """
    rows = []
    for f in fractions:
        rows.append({"frac": float(f), "lift": compute_lift(y_true, scores, top_frac=float(f))})
    return pd.DataFrame(rows)


def ranking_summary(
    y_true: ArrayLike,
    scores: ArrayLike,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Richer ranking table for reporting.

    Columns:
      - frac, k
      - base_rate
      - precision
      - recall
      - lift
      - positives_in_top
      - expected_score_sum_in_top
    """
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "frac",
                "k",
                "base_rate",
                "precision",
                "recall",
                "lift",
                "positives_in_top",
                "expected_score_sum_in_top",
            ]
        )

    y_bin = (y.astype(float) > 0.0).astype(int)
    base_rate = float(y_bin.mean())

    rows: list[Dict[str, Any]] = []
    for f in fractions:
        f = float(f)
        k = top_k_from_frac(len(y_bin), f)
        prec = precision_at_frac(y_bin, s, top_frac=f)
        rec = recall_at_frac(y_bin, s, top_frac=f)
        lift = compute_lift(y_bin, s, top_frac=f)
        pos_top = positives_in_top_frac(y_bin, s, top_frac=f)
        exp_sum = expected_score_sum_in_top_frac(s, top_frac=f)

        rows.append(
            {
                "frac": f,
                "k": int(k),
                "base_rate": base_rate,
                "precision": prec,
                "recall": rec,
                "lift": lift,
                "positives_in_top": pos_top,
                "expected_score_sum_in_top": exp_sum,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Soft-assignment diagnostics (RAMoE / HyRAMoE)
# ---------------------------------------------------------------------------


def _to_2d_array(q: Any) -> np.ndarray:
    """Convert responsibilities / assignment probabilities to a 2D float array."""
    if isinstance(q, pd.DataFrame):
        arr = q.to_numpy(dtype=float)
    else:
        arr = np.asarray(q, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of responsibilities (n,K); got shape={arr.shape}.")
    return arr


def normalize_responsibilities(responsibilities: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize responsibilities to ensure each row sums to 1."""
    q = _to_2d_array(responsibilities)
    q = np.clip(q, eps, None)
    q_sum = np.sum(q, axis=1, keepdims=True)
    q_sum = np.clip(q_sum, eps, None)
    return q / q_sum


def assignment_entropy(responsibilities: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Per-sample assignment entropy: H(q_i) = -sum_k q_ik log q_ik."""
    q = normalize_responsibilities(responsibilities, eps=eps)
    ent = -np.sum(q * np.log(np.clip(q, eps, 1.0)), axis=1)
    return ent


def assignment_maxprob(responsibilities: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Per-sample maximum assignment probability: max_k q_ik."""
    q = normalize_responsibilities(responsibilities, eps=eps)
    return np.max(q, axis=1)


__all__ = [
    "compute_lift",
    "lift_curve",
    "ranking_summary",
    "precision_at_frac",
    "recall_at_frac",
    "positives_in_top_frac",
    "expected_score_sum_in_top_frac",
    "top_k_from_frac",
    # soft assignment
    "normalize_responsibilities",
    "assignment_entropy",
    "assignment_maxprob",
]
