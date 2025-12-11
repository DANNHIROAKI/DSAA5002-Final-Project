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
    """Configuration for K-Means clustering runs."""

    n_clusters: int = 4
    init: str = "k-means++"
    max_iter: int = 300
    random_state: int = 42


class KMeansBaseline:
    """Wrapper around scikit-learn KMeans with convenience scoring."""

    def __init__(self, config: Optional[KMeansConfig] = None):
        self.config = config or KMeansConfig()
        self.model: Optional[KMeans] = None

    def fit(self, features: pd.DataFrame) -> "KMeansBaseline":
        self.model = KMeans(
            n_clusters=self.config.n_clusters,
            init=self.config.init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            n_init="auto",
        )
        self.model.fit(features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        return pd.Series(labels, index=features.index, name="cluster")

    def inertia(self) -> float:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return float(self.model.inertia_)

    def silhouette(self, features: pd.DataFrame) -> float:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        # Silhouette requires at least 2 clusters with more than one sample each
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(silhouette_score(features, labels))


def run_kmeans(
    features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> Tuple[KMeansBaseline, pd.Series]:
    """Fit K-Means and return the model plus assignments."""

    baseline = KMeansBaseline(KMeansConfig(n_clusters=n_clusters, random_state=random_state))
    baseline.fit(features)
    labels = baseline.predict(features)
    return baseline, labels
