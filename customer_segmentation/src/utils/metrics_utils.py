"""Utility metrics helpers used across clustering experiments.

These helpers are intentionally lightweight so they can be reused in
scripts and notebooks without pulling in the full evaluation package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


def _align_two_series(
    s1: pd.Series,
    s2: pd.Series,
    name1: str,
    name2: str,
) -> pd.DataFrame:
    """Align two series on index and drop rows with missing values."""
    df = pd.concat({name1: s1, name2: s2}, axis=1)
    df = df.dropna(subset=[name1, name2])
    return df


def compute_lift(
    responses: pd.Series,
    scores: pd.Series,
    top_frac: float = 0.2,
) -> float:
    """Compute lift by comparing top-ranked response rate to overall mean.

    Parameters
    ----------
    responses :
        Binary ground-truth responses (1 if accepted a campaign, else 0).
    scores :
        Model scores or probabilities used to rank customers.
    top_frac :
        Fraction of customers to include in the top segment (0 < top_frac <= 1).

    Returns
    -------
    float
        Lift ratio of the top segment vs. overall response rate. Returns 0 when
        the overall response rate is zero or when there are no usable samples.
    """
    if len(responses) == 0:
        return 0.0
    if not (0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1].")

    df = _align_two_series(responses, scores, "response", "score")
    if df.empty:
        return 0.0

    n = len(df)
    cutoff = max(int(n * top_frac), 1)

    ranked = df.sort_values("score", ascending=False)
    top = ranked.head(cutoff)

    top_rate = float(top["response"].mean())
    overall_rate = float(df["response"].mean())

    return float(top_rate / overall_rate) if overall_rate > 0 else 0.0


@dataclass
class ClusterResponseStats:
    """Summary statistics of per-cluster promotion responses."""

    response_rates: pd.Series
    variance: float
    range_: float


def response_rate_by_cluster(
    labels: pd.Series,
    responses: pd.Series,
) -> ClusterResponseStats:
    """Compute response rates per cluster along with variance and range.

    Parameters
    ----------
    labels :
        Cluster labels indexed in the same way as ``responses``.
    responses :
        Binary promotion responses.

    Returns
    -------
    ClusterResponseStats
        Per-cluster response rate plus variance and (max-min) range.
    """
    df = _align_two_series(labels, responses, "cluster", "response")
    if df.empty:
        rates = pd.Series(dtype=float)
    else:
        rates = df.groupby("cluster")["response"].mean().sort_index()

    if len(rates) == 0:
        variance = 0.0
        range_ = 0.0
    else:
        variance = float(np.var(rates))
        range_ = float(rates.max() - rates.min())

    return ClusterResponseStats(response_rates=rates, variance=variance, range_=range_)


def summarize_topk_lift(
    responses: pd.Series,
    scores: pd.Series,
    top_fracs: Iterable[float] = (0.1, 0.2, 0.3),
) -> pd.Series:
    """Return a series of lifts for several top-fraction cutoffs."""
    lifts = {
        f"lift_top{int(frac * 100)}": compute_lift(responses, scores, frac)
        for frac in top_fracs
    }
    return pd.Series(lifts)


def classification_summary(
    y_true: pd.Series,
    y_proba: pd.Series,
    *,
    top_fracs: Optional[Iterable[float]] = None,
) -> pd.Series:
    """Collect common classification summaries used by downstream experiments.

    Parameters
    ----------
    y_true :
        Ground-truth binary labels.
    y_proba :
        Predicted probabilities for the positive class.
    top_fracs :
        Optional iterable of fractions for lift computation. Defaults to (0.2,).

    Returns
    -------
    pd.Series
        AUC, log-loss, F1 (at 0.5 threshold), and lift-at-k values. Metrics
        that cannot be computed (e.g. AUC when only one class present) are
        returned as NaN.
    """
    top_fracs = tuple(top_fracs) if top_fracs is not None else (0.2,)

    # Ensure 1-D arrays
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_prob_arr = np.asarray(y_proba).reshape(-1)

    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(
            f"y_true and y_proba have different lengths: "
            f"{y_true_arr.shape[0]} vs {y_prob_arr.shape[0]}"
        )

    # Binary predictions at 0.5 threshold
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)

    # Robust metric computation
    try:
        auc = float(metrics.roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auc = float("nan")

    try:
        log_loss = float(metrics.log_loss(y_true_arr, y_prob_arr))
    except ValueError:
        log_loss = float("nan")

    f1 = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))

    summary = {
        "auc": auc,
        "log_loss": log_loss,
        "f1": f1,
    }
    summary.update(
        summarize_topk_lift(
            pd.Series(y_true_arr),
            pd.Series(y_prob_arr),
            top_fracs=top_fracs,
        ).to_dict()
    )
    return pd.Series(summary)


__all__ = [
    "compute_lift",
    "ClusterResponseStats",
    "response_rate_by_cluster",
    "summarize_topk_lift",
    "classification_summary",
]
