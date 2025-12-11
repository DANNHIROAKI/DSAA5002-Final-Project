"""Clustering quality metrics."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from sklearn import metrics


def compute_scores(
    features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Optional[float]]:
    """Compute common clustering quality indices.

    Parameters
    ----------
    features :
        Feature matrix used for clustering (rows = samples).
    labels :
        Cluster labels for each sample.

    Returns
    -------
    dict
        Dictionary with keys:
        ``"n_clusters"``, ``"silhouette"``, ``"calinski_harabasz"``,
        and ``"davies_bouldin"``. Metrics that cannot be computed
        (e.g., only one cluster present) are set to ``None``.
    """
    n_samples = len(labels)
    n_clusters = int(labels.nunique())

    results: Dict[str, Optional[float]] = {
        "n_clusters": float(n_clusters),
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    if n_samples <= 1 or n_clusters <= 1:
        # Not enough structure for any of the indices.
        return results

    # Silhouette score
    try:
        results["silhouette"] = float(
            metrics.silhouette_score(features, labels)
        )
    except ValueError:
        results["silhouette"] = None

    # Calinski–Harabasz index
    try:
        results["calinski_harabasz"] = float(
            metrics.calinski_harabasz_score(features, labels)
        )
    except ValueError:
        results["calinski_harabasz"] = None

    # Davies–Bouldin index
    try:
        results["davies_bouldin"] = float(
            metrics.davies_bouldin_score(features, labels)
        )
    except ValueError:
        results["davies_bouldin"] = None

    return results
