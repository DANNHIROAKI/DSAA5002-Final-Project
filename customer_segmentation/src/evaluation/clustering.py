"""Clustering quality metrics."""
import pandas as pd
from sklearn import metrics


def compute_scores(features: pd.DataFrame, labels: pd.Series) -> dict:
    """Compute silhouette, Calinski-Harabasz, and Davies-Bouldin indices."""
    scores = {
        "silhouette": metrics.silhouette_score(features, labels) if len(labels.unique()) > 1 else None,
        "calinski_harabasz": metrics.calinski_harabasz_score(features, labels) if len(labels.unique()) > 1 else None,
        "davies_bouldin": metrics.davies_bouldin_score(features, labels) if len(labels.unique()) > 1 else None,
    }
    return scores
