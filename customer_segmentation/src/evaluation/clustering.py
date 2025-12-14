"""Clustering quality metrics.

The baseline scripts and the proposed RAMoE/HyRAMoE method both rely on a few
standard unsupervised indices:

- Silhouette score (higher is better)
- Calinski–Harabasz index (higher is better)
- Davies–Bouldin index (lower is better)

Implementation notes
--------------------
- Accepts pandas or numpy inputs.
- Performs **index/length alignment** and drops rows with NaNs/Infs.
- Returns a stable schema so experiment runners can aggregate CSVs reliably.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn import metrics


def _to_2d_float_array(x: Any) -> np.ndarray:
    """Convert input feature matrix into a 2D float NumPy array."""
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=float)
    elif isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float).reshape(-1, 1)
    else:
        arr = np.asarray(x, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return arr


def _to_1d_labels(x: Any) -> np.ndarray:
    """Convert labels/cluster assignments to a 1D NumPy array."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected a single-column DataFrame for labels, got {x.shape}.")
        arr = x.iloc[:, 0].to_numpy()
    else:
        arr = np.asarray(x)

    return np.asarray(arr).reshape(-1)


def _valid_row_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row mask for finite features and non-missing labels."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {X.shape[0]} vs {y.shape[0]}")

    # label mask: treat NaN/None as invalid
    y_series = pd.Series(y)
    mask_y = y_series.notna().to_numpy()

    # feature mask: finite on all dims
    mask_X = np.all(np.isfinite(X), axis=1)

    return mask_y & mask_X


def compute_scores(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    *,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute standard clustering quality indices.

    Parameters
    ----------
    features:
        Feature matrix used for clustering (n_samples, n_features).
    labels:
        Cluster labels for each sample.
    sample_size:
        Optional subsample size for metric computation (useful for heavy plots).
        If provided and smaller than n_samples, a random subset is used.
    random_state:
        Random seed used for subsampling.

    Returns
    -------
    dict
        Keys:
        - n_samples
        - n_clusters
        - silhouette
        - calinski_harabasz
        - davies_bouldin

        Metrics that cannot be computed are returned as None.
    """
    X = _to_2d_float_array(features)
    y = _to_1d_labels(labels)

    mask = _valid_row_mask(X, y)
    X = X[mask]
    y = y[mask]

    n_samples = int(X.shape[0])
    n_clusters = int(pd.Series(y).nunique(dropna=True))

    results: Dict[str, Optional[float]] = {
        "n_samples": float(n_samples),
        "n_clusters": float(n_clusters),
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }

    # Guard: indices undefined for trivial partitions.
    if n_samples <= 1 or n_clusters <= 1 or n_samples <= n_clusters:
        return results

    # Optional subsampling for speed (keeps determinism by RNG seed).
    if sample_size is not None and 0 < int(sample_size) < n_samples:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(n_samples, size=int(sample_size), replace=False)
        X = X[idx]
        y = y[idx]

    # Silhouette score
    try:
        results["silhouette"] = float(metrics.silhouette_score(X, y))
    except Exception:
        results["silhouette"] = None

    # Calinski–Harabasz index
    try:
        results["calinski_harabasz"] = float(metrics.calinski_harabasz_score(X, y))
    except Exception:
        results["calinski_harabasz"] = None

    # Davies–Bouldin index
    try:
        results["davies_bouldin"] = float(metrics.davies_bouldin_score(X, y))
    except Exception:
        results["davies_bouldin"] = None

    return results


__all__ = ["compute_scores"]
