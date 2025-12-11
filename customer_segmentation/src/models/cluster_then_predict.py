"""Two-stage baseline: cluster first, then predict promotion response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression

from .kmeans_baseline import KMeansBaseline, run_kmeans


@dataclass
class ClusterThenPredictConfig:
    """Configuration for the two-stage baseline."""

    n_clusters: int = 4
    random_state: int = 42
    max_iter: int = 200


def fit_cluster_then_predict(
    features: pd.DataFrame,
    labels: pd.Series,
    config: ClusterThenPredictConfig = ClusterThenPredictConfig(),
) -> Tuple[KMeansBaseline, Dict[int, LogisticRegression], pd.Series]:
    """Fit K-Means clustering then per-cluster logistic regression models."""

    kmeans, cluster_assignments = run_kmeans(
        features, n_clusters=config.n_clusters, random_state=config.random_state
    )

    classifiers: Dict[int, LogisticRegression] = {}
    for cluster_id in sorted(cluster_assignments.unique()):
        idx = cluster_assignments[cluster_assignments == cluster_id].index
        if len(idx) == 0:
            continue
        clf = LogisticRegression(max_iter=config.max_iter, class_weight="balanced")
        clf.fit(features.loc[idx], labels.loc[idx])
        classifiers[cluster_id] = clf
    return kmeans, classifiers, cluster_assignments


def predict_with_clusters(
    cluster_labels: pd.Series,
    features: pd.DataFrame,
    classifiers: Dict[int, LogisticRegression],
) -> Tuple[pd.Series, pd.Series]:
    """Predict responses per cluster using pre-trained classifiers."""

    preds = pd.Series(index=features.index, dtype=float)
    for cluster_id, clf in classifiers.items():
        idx = cluster_labels[cluster_labels == cluster_id].index
        if len(idx) == 0:
            continue
        preds.loc[idx] = clf.predict_proba(features.loc[idx])[:, 1]
    binary = (preds >= 0.5).astype(int)
    return preds, binary
