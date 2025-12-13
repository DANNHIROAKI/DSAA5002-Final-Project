"""Segmentation-level metrics and utilities.

This module provides helpers for analysing how well a clustering aligns
with promotion-response behaviour, e.g. cluster-wise response rates and
their dispersion.

Upgrades for the new methodology:
- Add response_rate_range / response_rate_std for richer response stratification reporting.
- Add campaign_allocation_lift for budgeted targeting analysis at the cluster level.
- Provide cluster_response_summary alias (same as segmentation_table).
"""

from __future__ import annotations

from typing import Sequence, Union, Optional, Any

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


def _coerce_binary_like(y: pd.Series) -> pd.Series:
    """Coerce a response label series into numeric values (0/1-like).

    We do not force exact {0,1} here to keep compatibility with:
    - boolean labels
    - float labels already in [0,1]
    """
    out = pd.to_numeric(y, errors="coerce")
    return out


def cluster_response_rates(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.Series:
    """Compute per-cluster promotion-response rates.

    Parameters
    ----------
    cluster_labels
        Cluster assignment for each sample (ints 0..K-1 or similar).
    responses
        Binary response label for each sample (0 / 1).

    Returns
    -------
    pd.Series
        Index is cluster id, values are mean response rate in that cluster.
    """
    clusters = _to_series(cluster_labels, name="cluster")
    y = _to_series(responses, name="response")

    if len(clusters) != len(y):
        raise ValueError(
            f"cluster_labels and responses must have the same length, "
            f"got {len(clusters)} and {len(y)}."
        )

    df = pd.DataFrame({"cluster": clusters, "response": y})
    df["response"] = _coerce_binary_like(df["response"])
    df = df.dropna(subset=["cluster", "response"])

    if df.empty:
        return pd.Series(dtype=float, name="response_rate")

    rates = df.groupby("cluster")["response"].mean().sort_index()
    rates.name = "response_rate"
    return rates


def response_rate_variance(rates: ArrayLike) -> float:
    """Variance of cluster-wise response rates.

    A larger value indicates stronger dispersion of response behaviour
    across clusters (i.e., segments are more differentiated in terms of
    campaign response).
    """
    r = np.asarray(_to_series(rates, name="response_rate"), dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanvar(r))


def response_rate_std(rates: ArrayLike, ddof: int = 0) -> float:
    """Standard deviation of cluster-wise response rates."""
    r = np.asarray(_to_series(rates, name="response_rate"), dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanstd(r, ddof=ddof))


def response_rate_range(rates: ArrayLike) -> float:
    """Range (max - min) of cluster-wise response rates."""
    r = np.asarray(_to_series(rates, name="response_rate"), dtype=float)
    if r.size == 0:
        return float("nan")
    return float(np.nanmax(r) - np.nanmin(r))


def cluster_size_summary(cluster_labels: ArrayLike) -> pd.Series:
    """Return the size (count) of each cluster as a Series."""
    clusters = _to_series(cluster_labels, name="cluster")
    clusters = clusters.dropna()
    if clusters.empty:
        return pd.Series(dtype=int, name="cluster_size")
    counts = clusters.value_counts().sort_index()
    counts.name = "cluster_size"
    return counts


def segmentation_table(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Convenience function: full segmentation summary table.

    Columns:
        - size
        - positives
        - negatives
        - response_rate
    """
    clusters = _to_series(cluster_labels, name="cluster")
    y = _to_series(responses, name="response")

    if len(clusters) != len(y):
        raise ValueError(
            f"cluster_labels and responses must have the same length, "
            f"got {len(clusters)} and {len(y)}."
        )

    df = pd.DataFrame({"cluster": clusters, "response": y})
    df["response"] = _coerce_binary_like(df["response"])
    df = df.dropna(subset=["cluster", "response"])

    if df.empty:
        return pd.DataFrame(
            columns=["size", "positives", "negatives", "response_rate"]
        ).astype({"size": int, "positives": float, "negatives": float, "response_rate": float})

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
    """Alias kept for readability in reports; identical to segmentation_table()."""
    return segmentation_table(cluster_labels, responses)


def campaign_allocation_lift(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
    budget_frac: float = 0.2,
    *,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Estimate cluster-level lift under a budgeted targeting policy.

    Policy:
    - Compute empirical response rate for each cluster.
    - Sort clusters by response rate descending.
    - Allocate a marketing budget to cover top `budget_frac` customers by
      taking customers from the best clusters first (allow partial last cluster).

    The lift is defined as:
        expected_responders_targeted / expected_responders_random

    where "random" means uniformly selecting the same number of customers
    from the whole population.

    Parameters
    ----------
    cluster_labels:
        Cluster id per customer.
    responses:
        Binary labels (0/1).
    budget_frac:
        Fraction of customers we can target (e.g., 0.2 for top-20%).
    return_details:
        If True, return a dict with intermediate values.

    Returns
    -------
    float or dict
        Lift value, or a dict with lift and intermediates.
    """
    if not (0.0 < float(budget_frac) <= 1.0):
        raise ValueError(f"budget_frac must be in (0, 1], got {budget_frac}")

    clusters = _to_series(cluster_labels, name="cluster")
    y = _to_series(responses, name="response")
    if len(clusters) != len(y):
        raise ValueError(
            f"cluster_labels and responses must have the same length, "
            f"got {len(clusters)} and {len(y)}."
        )

    df = pd.DataFrame({"cluster": clusters, "response": y})
    df["response"] = _coerce_binary_like(df["response"])
    df = df.dropna(subset=["cluster", "response"])
    if df.empty:
        out = float("nan")
        return {"lift": out} if return_details else out

    n_total = int(df.shape[0])
    n_select = int(np.ceil(float(budget_frac) * n_total))
    if n_select <= 0:
        out = float("nan")
        return {"lift": out} if return_details else out

    summary = segmentation_table(df["cluster"], df["response"])
    summary_sorted = summary.sort_values("response_rate", ascending=False)

    remaining = n_select
    expected_targeted = 0.0
    selected_from_clusters: dict[Any, int] = {}

    for cid, row in summary_sorted.iterrows():
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
