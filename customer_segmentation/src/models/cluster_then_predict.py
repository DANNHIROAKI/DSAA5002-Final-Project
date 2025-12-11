"""Two-stage baseline: cluster first, then predict promotion response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .kmeans_baseline import KMeansBaseline, KMeansConfig


class _ConstantProbClassifier:
    """Simple classifier that always outputs a fixed positive probability.

    Used as a robust fallback when a cluster contains only a single class and
    scikit-learn's LogisticRegression cannot be fitted.
    """

    def __init__(self, p: float):
        # clip to avoid log-loss issues with 0 or 1
        self.p = float(np.clip(p, 1e-7, 1 - 1e-7))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = X.shape[0]
        p = np.full(n, self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class ClusterThenPredictConfig:
    """Configuration for the two-stage baseline.

    Parameters
    ----------
    n_clusters :
        Number of clusters for the initial K-Means.
    random_state :
        Random seed for reproducibility.
    max_iter :
        Maximum iterations for the per-cluster logistic regressions.
    kmeans_n_init :
        Number of random initializations for K-Means.
    """

    n_clusters: int = 4
    random_state: int = 42
    max_iter: int = 200
    kmeans_n_init: int = 10


def fit_cluster_then_predict(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Optional[ClusterThenPredictConfig] = None,
) -> Tuple[KMeansBaseline, Dict[int, object], pd.Series]:
    """Fit K-Means clustering then per-cluster logistic/constant models.

    For each cluster, a logistic regression is trained if both positive and
    negative examples are present; otherwise a constant-probability classifier
    is used instead.

    Returns
    -------
    kmeans :
        Fitted KMeansBaseline model.
    classifiers :
        Mapping from cluster_id to classifier (LogisticRegression or
        _ConstantProbClassifier).
    cluster_assignments :
        Series of cluster labels for the training data.
    """
    cfg = config or ClusterThenPredictConfig()

    # Use KMeansBaseline directly so that cfg.kmeans_n_init is honoured.
    kmeans_cfg = KMeansConfig(
        n_clusters=cfg.n_clusters,
        random_state=cfg.random_state,
        n_init=cfg.kmeans_n_init,
    )
    kmeans = KMeansBaseline(kmeans_cfg).fit(features)
    cluster_assignments = kmeans.predict(features)

    classifiers: Dict[int, object] = {}
    for cluster_id in sorted(cluster_assignments.unique()):
        idx = cluster_assignments[cluster_assignments == cluster_id].index
        if len(idx) == 0:
            continue

        y_cluster = labels.loc[idx]
        X_cluster = features.loc[idx]

        # If only one class present, fall back to constant-prob classifier
        if y_cluster.nunique() < 2:
            p = float(y_cluster.mean())
            clf: object = _ConstantProbClassifier(p)
        else:
            clf = LogisticRegression(
                max_iter=cfg.max_iter,
                class_weight="balanced",
            )
            clf.fit(X_cluster, y_cluster)

        classifiers[cluster_id] = clf

    return kmeans, classifiers, cluster_assignments


def predict_with_clusters(
    cluster_labels: pd.Series,
    features: pd.DataFrame,
    classifiers: Dict[int, object],
) -> Tuple[pd.Series, pd.Series]:
    """Predict responses per cluster using pre-trained classifiers.

    Any samples assigned to clusters without a classifier receive a neutral
    probability of 0.5.

    Returns
    -------
    probs :
        Predicted positive-class probability for each sample.
    binary :
        Binary predictions obtained by thresholding probs at 0.5.
    """
    preds = pd.Series(index=features.index, dtype=float)

    for cluster_id, clf in classifiers.items():
        idx = cluster_labels[cluster_labels == cluster_id].index
        if len(idx) == 0:
            continue
        probs = clf.predict_proba(features.loc[idx])[:, 1]
        preds.loc[idx] = probs

    # For potential unseen / empty clusters, fill with 0.5
    preds = preds.fillna(0.5)
    binary = (preds >= 0.5).astype(int)
    return preds, binary
