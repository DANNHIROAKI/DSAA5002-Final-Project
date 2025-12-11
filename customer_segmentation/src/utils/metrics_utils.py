"""Generic metric helpers used across experiments.

Currently includes:

- `compute_lift`: lift of a ranking model under a top-q% budget.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, Sequence[int], Sequence[float], np.ndarray]


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """Convert input values into a pandas Series with the given name."""
    if isinstance(values, pd.Series):
        s = values.copy()
        if name is not None:
            s.name = name
        return s
    return pd.Series(values, name=name)


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

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0 / 1).
    scores
        Predicted scores or probabilities (higher = more likely to be 1).
    top_frac
        Fraction of the population to target (0 < top_frac <= 1).

    Returns
    -------
    float
        Lift value. Returns NaN if the overall positive rate is 0 or
        inputs are empty.
    """
    if not (0 < top_frac <= 1):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}.")

    y = _to_series(y_true, name="y_true").astype(float)
    s = _to_series(scores, name="score").astype(float)

    if len(y) == 0:
        return float("nan")

    if len(y) != len(s):
        raise ValueError(
            f"y_true and scores must have the same length, "
            f"got {len(y)} and {len(s)}."
        )

    # Sort by predicted score (descending)
    order = s.sort_values(ascending=False).index
    y_sorted = y.loc[order]

    k = int(np.ceil(len(y_sorted) * top_frac))
    if k <= 0:
        return float("nan")

    y_top = y_sorted.iloc[:k]
    precision_top = float(y_top.mean())
    base_rate = float(y.mean())

    if base_rate == 0.0:
        return float("nan")

    return precision_top / base_rate


def lift_curve(
    y_true: ArrayLike,
    scores: ArrayLike,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Compute lift at multiple budget fractions.

    Returns a DataFrame with columns:
        - frac: budget fraction
        - lift: lift at that fraction
    """
    rows = []
    for f in fractions:
        rows.append({"frac": f, "lift": compute_lift(y_true, scores, top_frac=f)})
    return pd.DataFrame(rows)
