"""Segmentation-level metrics and utilities.

This module provides helpers for analysing how well a clustering aligns
with promotion-response behaviour, e.g. cluster-wise response rates and
their dispersion.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, Sequence[int], Sequence[float], np.ndarray]


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """Convert input values into a pandas Series."""
    if isinstance(values, pd.Series):
        s = values.copy()
        if name is not None:
            s.name = name
        return s
    return pd.Series(values, name=name)


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

    df = pd.DataFrame({"cluster": clusters, "response": y}).dropna(
        subset=["cluster", "response"]
    )
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


def cluster_size_summary(cluster_labels: ArrayLike) -> pd.Series:
    """Return the size (count) of each cluster as a Series."""
    clusters = _to_series(cluster_labels, name="cluster")
    counts = clusters.value_counts().sort_index()
    counts.name = "cluster_size"
    return counts


def segmentation_table(
    cluster_labels: ArrayLike,
    responses: ArrayLike,
) -> pd.DataFrame:
    """Convenience function: full segmentation summary table.

    Columns:
        - cluster
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
    grouped = df.groupby("cluster")

    size = grouped.size()
    positives = grouped["response"].sum()
    negatives = size - positives
    rate = positives / size.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "cluster": size.index,
            "size": size.values,
            "positives": positives.values,
            "negatives": negatives.values,
            "response_rate": rate.values,
        }
    ).set_index("cluster")

    return out
