"""Gaussian Mixture Model baseline for clustering."""
import pandas as pd
from sklearn.mixture import GaussianMixture


def run_gmm(features: pd.DataFrame, n_components: int = 4, random_state: int = 42) -> GaussianMixture:
    """Fit a Gaussian Mixture Model on the provided features."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=200,
        random_state=random_state,
    )
    model.fit(features)
    return model


def assign_clusters(model: GaussianMixture, features: pd.DataFrame) -> pd.Series:
    """Return cluster assignments based on maximum posterior probability."""
    labels = model.predict(features)
    return pd.Series(labels, index=features.index, name="cluster")
