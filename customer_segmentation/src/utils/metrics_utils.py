"""Utility metrics helpers used across clustering experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def compute_lift(responses: pd.Series, scores: pd.Series, top_frac: float = 0.2) -> float:
    """Compute lift by comparing top-ranked response rate to overall mean.

    Parameters
    ----------
    responses:
        Binary ground-truth responses (1 if accepted a campaign, else 0).
    scores:
        Model scores or probabilities used to rank customers.
    top_frac:
        Fraction of customers to include in the top segment (default 20%).

    Returns
    -------
    float
        Lift ratio of the top segment vs. overall response rate. Returns 0 when
        the overall response rate is zero.
    """

    if len(responses) == 0:
        return 0.0

    cutoff = max(int(len(scores) * top_frac), 1)
    ranked = scores.sort_values(ascending=False)
    top_idx = ranked.head(cutoff).index
    top_rate = responses.loc[top_idx].mean()
    overall_rate = responses.mean()
    return float(top_rate / overall_rate) if overall_rate > 0 else 0.0


@dataclass
class ClusterResponseStats:
    """Summary statistics of per-cluster promotion responses."""

    response_rates: pd.Series
    variance: float
    range_: float


def response_rate_by_cluster(labels: pd.Series, responses: pd.Series) -> ClusterResponseStats:
    """Compute response rates per cluster along with variance and range.

    Parameters
    ----------
    labels:
        Cluster labels indexed the same as ``responses``.
    responses:
        Binary promotion responses.

    Returns
    -------
    ClusterResponseStats
        Per-cluster response rate plus variance and (max-min) range for RQ2.
    """

    grouped = responses.groupby(labels)
    rates = grouped.mean().sort_index()
    variance = float(np.var(rates)) if len(rates) > 0 else 0.0
    range_ = float(rates.max() - rates.min()) if len(rates) > 0 else 0.0
    return ClusterResponseStats(response_rates=rates, variance=variance, range_=range_)


def summarize_topk_lift(
    responses: pd.Series,
    scores: pd.Series,
    top_fracs: Iterable[float] = (0.1, 0.2, 0.3),
) -> pd.Series:
    """Return a series of lifts for several top-fraction cutoffs."""

    lifts = {f"lift_top{int(frac*100)}": compute_lift(responses, scores, frac) for frac in top_fracs}
    return pd.Series(lifts)


def classification_summary(
    y_true: pd.Series,
    y_proba: pd.Series,
    *,
    top_fracs: Optional[Iterable[float]] = None,
) -> pd.Series:
    """Collect common classification summaries used by downstream experiments."""

    from sklearn import metrics

    top_fracs = tuple(top_fracs) if top_fracs is not None else (0.2,)

    auc = metrics.roc_auc_score(y_true, y_proba)
    log_loss = metrics.log_loss(y_true, y_proba, eps=1e-15)
    f1 = metrics.f1_score(y_true, (y_proba >= 0.5).astype(int))

    summary = {
        "auc": auc,
        "log_loss": log_loss,
        "f1": f1,
    }
    summary.update(summarize_topk_lift(y_true, y_proba, top_fracs))
    return pd.Series(summary)


__all__ = [
    "compute_lift",
    "ClusterResponseStats",
    "response_rate_by_cluster",
    "summarize_topk_lift",
    "classification_summary",
]
