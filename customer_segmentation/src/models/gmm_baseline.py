"""Gaussian Mixture Model baseline for flexible clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


@dataclass
class GMMConfig:
    """Configuration for GMM clustering runs."""

    n_components: int = 4
    covariance_type: str = "full"
    max_iter: int = 200
    random_state: int = 42


class GMMBaseline:
    """Wrapper for GaussianMixture with common utilities."""

    def __init__(self, config: Optional[GMMConfig] = None):
        self.config = config or GMMConfig()
        self.model: Optional[GaussianMixture] = None

    def fit(self, features: pd.DataFrame) -> "GMMBaseline":
        """Fit a Gaussian Mixture Model on the given feature matrix."""
        self.model = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self.model.fit(features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Assign each sample to the most likely Gaussian component."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        return pd.Series(labels, index=features.index, name="cluster")

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Posterior responsibilities for each component (n_samples, n_components)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(features)

    def silhouette(self, features: pd.DataFrame) -> float:
        """Silhouette score for current GMM clustering (nan if undefined)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(features)
        unique = np.unique(labels)
        if len(unique) < 2 or features.shape[0] <= len(unique):
            return float("nan")
        return float(silhouette_score(features, labels))


def run_gmm(
    features: pd.DataFrame,
    n_components: int = 4,
    random_state: int = 42,
) -> Tuple[GMMBaseline, pd.Series]:
    """Fit a GMM and return the model plus assignments."""
    baseline = GMMBaseline(
        GMMConfig(n_components=n_components, random_state=random_state)
    )
    baseline.fit(features)
    labels = baseline.predict(features)
    return baseline, labels
