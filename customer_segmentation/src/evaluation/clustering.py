"""Clustering quality metrics.

Upgrades for the new methodology:
- Accept both DataFrame and ndarray inputs.
- Add stricter guards for metric definability (e.g., silhouette requires n_samples > n_clusters >= 2).
- Keep return schema stable for experiment CSV aggregation.
"""

from __future__ import annotations

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn import metrics


def _as_array(x: Any) -> Any:
    """Return an array-like object suitable for sklearn metrics.

    We keep DataFrame as-is (sklearn accepts it), but ensure numpy for others.
    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x
    return np.asarray(x)


def compute_scores(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
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
        ``"n_samples"``, ``"n_clusters"``, ``"silhouette"``,
        ``"calinski_harabasz"``, and ``"davies_bouldin"``.
        Metrics that cannot be computed (e.g., only one cluster present)
        are set to ``None``.
    """
    labels_series = labels if isinstance(labels, pd.Series) else pd.Series(labels)
    labels_series = labels_series.dropna()

    n_samples = int(len(labels_series))
    n_clusters = int(labels_series.nunique(dropna=True))

    results: Dict[str, Optional[float]] = {
        "n_samples": float(n_samples),
        "n_clusters": float(n_clusters),
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    # Not enough structure for any indices.
    # Note: silhouette also requires n_samples > n_clusters.
    if n_samples <= 1 or n_clusters <= 1 or n_samples <= n_clusters:
        return results

    X = _as_array(features)

    # Silhouette score
    try:
        results["silhouette"] = float(metrics.silhouette_score(X, labels_series))
    except Exception:
        results["silhouette"] = None

    # Calinski–Harabasz index
    try:
        results["calinski_harabasz"] = float(
            metrics.calinski_harabasz_score(X, labels_series)
        )
    except Exception:
        results["calinski_harabasz"] = None

    # Davies–Bouldin index
    try:
        results["davies_bouldin"] = float(
            metrics.davies_bouldin_score(X, labels_series)
        )
    except Exception:
        results["davies_bouldin"] = None

    return results
