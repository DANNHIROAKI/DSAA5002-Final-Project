"""Generic metric helpers used across experiments.

Upgrades for the new methodology:
- More robust handling of NaNs / index alignment.
- Provide ranking-oriented business metrics beyond lift:
  precision@q, recall@q (capture rate), positives_in_topq, expected_score_sum_in_topq.
- Keep `compute_lift` and `lift_curve` backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Optional, Iterable, Dict, Any

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, Sequence[int], Sequence[float], np.ndarray]


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """Convert input values into a pandas Series with the given name."""
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


def _aligned_nonan(y_true: ArrayLike, scores: ArrayLike) -> tuple[pd.Series, pd.Series]:
    """Align y_true and scores by index (if any) and drop rows with NaNs."""
    y = _coerce_numeric(_to_series(y_true, "y_true"), name="y_true")
    s = _coerce_numeric(_to_series(scores, "score"), name="score")

    # Align by index when possible (pandas aligns on index automatically in concat)
    df = pd.concat([y, s], axis=1)

    # Drop NaNs in either
    df = df.dropna(subset=["y_true", "score"])

    return df["y_true"], df["score"]


def top_k_from_frac(n: int, top_frac: float) -> int:
    """Convert a budget fraction to a top-k integer (ceil)."""
    if n <= 0:
        return 0
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}.")
    return int(np.ceil(n * float(top_frac)))


def precision_at_frac(
    y_true: ArrayLike,
    scores: ArrayLike,
    top_frac: float = 0.2,
) -> float:
    """Precision among top-q% ranked by scores (higher is better).

    Equivalent to mean(y_true) within the top-q% by score.
    """
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    k = top_k_from_frac(len(y), top_frac)
    if k <= 0:
        return float("nan")

    order = s.sort_values(ascending=False).index
    y_sorted = y.loc[order].astype(float)

    y_top = y_sorted.iloc[:k]
    return float(y_top.mean())


def recall_at_frac(
    y_true: ArrayLike,
    scores: ArrayLike,
    top_frac: float = 0.2,
) -> float:
    """Recall (capture rate) within top-q% ranked by scores.

    Defined as: (# positives in top-q%) / (total # positives).
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


def positives_in_top_frac(
    y_true: ArrayLike,
    scores: ArrayLike,
    top_frac: float = 0.2,
) -> float:
    """Number of positives contained in the top-q% ranked by scores."""
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    y_bin = (y.astype(float) > 0.0).astype(int)
    k = top_k_from_frac(len(y_bin), top_frac)
    order = s.sort_values(ascending=False).index
    y_sorted = y_bin.loc[order]
    return float(y_sorted.iloc[:k].sum())


def expected_score_sum_in_top_frac(
    scores: ArrayLike,
    top_frac: float = 0.2,
) -> float:
    """Sum of scores in the top-q% ranked by scores.

    When `scores` are calibrated probabilities, this approximates the expected
    number of responders under a budgeted targeting policy.
    """
    s = _coerce_numeric(_to_series(scores, "score"), name="score").dropna()
    if len(s) == 0:
        return float("nan")

    k = top_k_from_frac(len(s), top_frac)
    order = s.sort_values(ascending=False).index
    s_sorted = s.loc[order].astype(float)
    return float(s_sorted.iloc[:k].sum())


def compute_lift(
    y_true: ArrayLike,
    scores: ArrayLike,
    top_frac: float = 0.2,
) -> float:
    """Compute lift at top-q% based on a ranking by `scores`.

    Lift is defined as:

        precision@top_q%  /  overall_positive_rate

    where precision@top_q% is the fraction of positives within the top
    `top_frac` examples ranked by `scores` in descending order.

    Returns NaN if:
    - inputs are empty after dropping NaNs, or
    - the overall positive rate is 0.
    """
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}.")

    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return float("nan")

    y_bin = (y.astype(float) > 0.0).astype(int)
    base_rate = float(y_bin.mean())
    if base_rate == 0.0:
        return float("nan")

    prec = precision_at_frac(y_bin, s, top_frac=top_frac)
    if np.isnan(prec):
        return float("nan")

    return float(prec / base_rate)


def lift_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Compute lift at multiple budget fractions.

    Backward-compatible output schema:
        - frac: budget fraction
        - lift: lift at that fraction
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

    Notes
    -----
    expected_score_sum_in_top is meaningful when `scores` are probabilities.
    """
    y, s = _aligned_nonan(y_true, scores)
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "frac", "k", "base_rate", "precision", "recall", "lift",
                "positives_in_top", "expected_score_sum_in_top",
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


__all__ = [
    "compute_lift",
    "lift_curve",
    "ranking_summary",
    "precision_at_frac",
    "recall_at_frac",
    "positives_in_top_frac",
    "expected_score_sum_in_top_frac",
    "top_k_from_frac",
]
