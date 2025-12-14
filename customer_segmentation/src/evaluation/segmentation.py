"""Segmentation-level metrics and utilities.

This module measures whether a clustering is *actionable* for marketing:
- clusters should differ in response rate (high dispersion)
- clusters should be reasonably balanced (avoid many tiny clusters)

Upgraded for RAMoE/HyRAMoE
--------------------------
- robust index alignment and NaN handling
- adds cluster_lift_table and weighted_response_rate_variance
- adds campaign_allocation_lift for budget-style analysis
"""

from __future__ import annotations

from typing import Any, Sequence, Union, Optional

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, Sequence[int], Sequence[float], np.ndarray]


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """Convert input values into a pandas Series."""
    if isinstance(values, pd.Series):
        s = values.copy()
        s.name = name
        return s
    return pd.Series(values, name=name)


def _coerce_numeric(series: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out.name = name
    return out


def _aligned_cluster_and_response(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Align cluster labels and responses by index and drop NaNs.

    This is important when callers pass pandas Series with non-trivial indices.
    """
    c = _to_series(cluster_labels, "cluster")
    y = _to_series(responses, "response")

    # concat aligns on index automatically
    df = pd.concat([c, y], axis=1)

    # response should be numeric; clusters may be numeric or categorical
    df["response"] = _coerce_numeric(df["response"], name="response")

    # drop missing
    df = df.dropna(subset=["cluster", "response"])
    return df


def cluster_response_rates(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.Series:
    """Compute mean response rate per cluster."""
    df = _aligned_cluster_and_response(cluster_labels, responses)
    if df.empty:
        return pd.Series(dtype=float, name="response_rate")

    rates = df.groupby("cluster")["response"].mean().sort_index()
    rates.name = "response_rate"
    return rates


def cluster_size_summary(cluster_labels: ArrayLike) -> pd.Series:
    """Return the size (count) of each cluster as a Series."""
    c = _to_series(cluster_labels, "cluster").dropna()
    if c.empty:
        return pd.Series(dtype=int, name="cluster_size")
    counts = c.value_counts().sort_index()
    counts.name = "cluster_size"
    return counts


def segmentation_table(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Full segmentation summary table.

    Columns:
      - size
      - positives
      - negatives
      - response_rate
    """
    df = _aligned_cluster_and_response(cluster_labels, responses)
    if df.empty:
        return pd.DataFrame(
            columns=["size", "positives", "negatives", "response_rate"]
        ).astype(
            {"size": int, "positives": float, "negatives": float, "response_rate": float}
        )

    grouped = df.groupby("cluster")
    size = grouped.size()
    positives = grouped["response"].sum()
    negatives = size - positives
    rate = positives / size.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "size": size.astype(int),
            "positives": positives.astype(float),
            "negatives": negatives.astype(float),
            "response_rate": rate.astype(float),
        }
    )
    out.index.name = "cluster"
    return out.sort_index()


def cluster_response_summary(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Alias for :func:`segmentation_table` (kept for readability)."""
    return segmentation_table(cluster_labels, responses)


def response_rate_variance(rates: ArrayLike) -> float:
    """Unweighted variance of cluster-wise response rates."""
    r = _coerce_numeric(_to_series(rates, "response_rate"), name="response_rate").to_numpy(dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanvar(r))


def response_rate_std(rates: ArrayLike, ddof: int = 0) -> float:
    """Unweighted std of cluster-wise response rates."""
    r = _coerce_numeric(_to_series(rates, "response_rate"), name="response_rate").to_numpy(dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanstd(r, ddof=int(ddof)))


def response_rate_range(rates: ArrayLike) -> float:
    """Range (max-min) of cluster-wise response rates."""
    r = _coerce_numeric(_to_series(rates, "response_rate"), name="response_rate").to_numpy(dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanmax(r) - np.nanmin(r))


def weighted_response_rate_variance(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> float:
    """Size-weighted variance of cluster response rates.

    Motivation
    ----------
    An unweighted variance can be dominated by tiny clusters. Weighting by
    cluster size makes the dispersion metric more stable and closer to an
    ANOVA-style between-cluster variance.
    """
    df = _aligned_cluster_and_response(cluster_labels, responses)
    if df.empty:
        return float("nan")

    summary = segmentation_table(df["cluster"], df["response"])
    sizes = summary["size"].to_numpy(dtype=float)
    rates = summary["response_rate"].to_numpy(dtype=float)

    if sizes.sum() <= 0:
        return float("nan")

    weights = sizes / sizes.sum()
    mean_rate = float(np.sum(weights * rates))
    var = float(np.sum(weights * (rates - mean_rate) ** 2))
    return var


def cluster_lift_table(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Per-cluster lift table relative to the global response rate.

    Returns a DataFrame with:
      - size, share
      - response_rate
      - lift_vs_global = response_rate / global_rate
    """
    df = _aligned_cluster_and_response(cluster_labels, responses)
    if df.empty:
        return pd.DataFrame(columns=["size", "share", "response_rate", "lift_vs_global"])

    summary = segmentation_table(df["cluster"], df["response"]).copy()
    n = float(summary["size"].sum())
    global_rate = float(df["response"].mean())

    summary["share"] = summary["size"].astype(float) / (n if n > 0 else np.nan)
    summary["lift_vs_global"] = summary["response_rate"].astype(float) / (global_rate if global_rate > 0 else np.nan)

    # Order by response_rate descending for readability
    return summary.sort_values("response_rate", ascending=False)


def campaign_allocation_lift(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
    budget_frac: float = 0.2,
    *,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Estimate lift under a *cluster-level* budget allocation policy.

    Policy
    ------
    - Estimate cluster response rates from data.
    - Sort clusters by response rate descending.
    - Allocate budget to customers from best clusters first until budget is used.
      (partial allocation is allowed for the last cluster).

    Lift definition
    ---------------
    expected_responders_targeted / expected_responders_random

    """
    if not (0.0 < float(budget_frac) <= 1.0):
        raise ValueError(f"budget_frac must be in (0,1], got {budget_frac}")

    df = _aligned_cluster_and_response(cluster_labels, responses)
    if df.empty:
        out = float("nan")
        return {"lift": out} if return_details else out

    n_total = int(df.shape[0])
    n_select = int(np.ceil(float(budget_frac) * n_total))
    if n_select <= 0:
        out = float("nan")
        return {"lift": out} if return_details else out

    summary = segmentation_table(df["cluster"], df["response"]).sort_values("response_rate", ascending=False)

    remaining = n_select
    expected_targeted = 0.0
    selected_from_clusters: dict[Any, int] = {}

    for cid, row in summary.iterrows():
        if remaining <= 0:
            break
        size_k = int(row["size"])
        if size_k <= 0:
            continue
        take = min(remaining, size_k)
        expected_targeted += float(take) * float(row["response_rate"])
        selected_from_clusters[cid] = int(take)
        remaining -= take

    global_rate = float(df["response"].mean())
    expected_random = global_rate * float(n_select)
    lift = float("nan") if expected_random <= 0 else float(expected_targeted / expected_random)

    if return_details:
        return {
            "budget_frac": float(budget_frac),
            "n_total": n_total,
            "n_selected": n_select,
            "global_response_rate": global_rate,
            "expected_responders_targeted": float(expected_targeted),
            "expected_responders_random": float(expected_random),
            "lift": lift,
            "selected_from_clusters": selected_from_clusters,
        }
    return lift


__all__ = [
    "cluster_response_rates",
    "cluster_size_summary",
    "segmentation_table",
    "cluster_response_summary",
    "response_rate_variance",
    "response_rate_std",
    "response_rate_range",
    "weighted_response_rate_variance",
    "cluster_lift_table",
    "campaign_allocation_lift",
]
