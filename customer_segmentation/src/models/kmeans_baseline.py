"""K-Means baselines using RFM or full feature sets."""
from typing import Optional
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    """Fit a K-Means model on the provided features."""
    model = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, random_state=random_state)
    model.fit(features)
    return model


def assign_clusters(model: KMeans, features: pd.DataFrame) -> pd.Series:
    """Return cluster assignments for each sample."""
    return pd.Series(model.predict(features), index=features.index, name="cluster")
