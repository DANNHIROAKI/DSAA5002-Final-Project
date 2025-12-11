"""K-Means baselines with helpers for RFM and full-feature runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class KMeansConfig:
    """Configuration for K-Means clustering runs.

    Parameters
    ----------
    n_clusters :
        Number of clusters K.
    init :
        Initialization method for centroids, passed to :class:`~sklearn.cluster.KMeans`.
    max_iter :
        Maximum number of EM iterations.
    random_state :
        Random seed for reproducibility.
    n_init :
        Number of KMeans initializations. Using an integer here keeps compatibility
        with both older and newer versions of scikit-learn.
    """

    n_clusters: int = 4
    init: str = "k-means++"
    max_iter: int = 300
    random_state: int = 42
    n_init: int = 10


class KMeansBaseline:
    """Wrapper around scikit-learn KMeans with convenience scoring."""

    def __init__(self, config: Optional[KMeansConfig] = None):
        self.config = config or KMeansConfig()
        self.model: Optional[KMeans] = None

    def fit(self, features: pd.DataFrame) -> "KMeansBaseline":
        """Fit the K-Means model on the given feature matrix."""
        self.model = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            n_init=self.config.n_init,
        )
        self.model.fit(features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign each sample to the nearest cluster center."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        return pd.Series(labels, index=features.index, name="cluster")

    def inertia(self) -> float:
        """Return the final K-Means inertia (sum of squared distances)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return float(self.model.inertia_)

    def silhouette(self, features: pd.DataFrame) -> float:
        """Compute the Silhouette score for the current clustering.

        Returns ``nan`` when Silhouette is not defined (e.g., single cluster).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        unique = np.unique(labels)
        if len(unique) < 2 or features.shape[0] <= len(unique):
            return float("nan")
        return float(silhouette_score(features, labels))


def run_kmeans(
    features: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[KMeansBaseline, pd.Series]:
    """Fit K-Means and return the model plus assignments.

    This helper is mainly used by baseline experiments.
    """
    baseline = KMeansBaseline(
        KMeansConfig(n_clusters=n_clusters, random_state=random_state)
    )
    baseline.fit(features)
    labels = baseline.predict(features)
    return baseline, labels
