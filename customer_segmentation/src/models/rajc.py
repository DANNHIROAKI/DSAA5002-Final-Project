"""Response-Aware Joint Clustering (RAJC) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


class _ConstantProbClassifier:
    """Cluster-level classifier that always outputs a fixed probability."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-7, 1 - 1e-7))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = X.shape[0]
        p = np.full(n, self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class RAJCConfig:
    """Configuration for the RAJC alternating optimization algorithm."""

    n_clusters: int = 4
    lambda_: float = 1.0
    max_iter: int = 20
    tol: float = 1e-4
    random_state: int = 42
    logreg_max_iter: int = 200
    kmeans_n_init: int = 10  # for the initialization KMeans


def _initialize_assignments(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> pd.Series:
    """Initialize cluster assignments with a standard K-Means run."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(features)
    return pd.Series(labels, index=features.index, name="cluster")


def _update_centers(
    features: pd.DataFrame,
    assignments: pd.Series,
    n_clusters: int,
    random_state: int,
) -> pd.DataFrame:
    """Update cluster centers as the mean of assigned samples.

    Empty clusters are re-seeded with a random sample to avoid collapse.
    """
    centers = []
    for cluster_id in range(n_clusters):
        cluster_idx = assignments[assignments == cluster_id].index
        if len(cluster_idx) == 0:
            centers.append(features.sample(1, random_state=random_state + cluster_id).iloc[0])
        else:
            centers.append(features.loc[cluster_idx].mean(axis=0))
    return pd.DataFrame(centers, index=range(n_clusters))


def _fit_cluster_classifiers(
    features: pd.DataFrame,
    labels: pd.Series,
    assignments: pd.Series,
    config: RAJCConfig,
) -> Dict[int, object]:
    """Fit per-cluster classifiers (logistic regression or constant-prob)."""
    classifiers: Dict[int, object] = {}
    for cluster_id in range(config.n_clusters):
        idx = assignments[assignments == cluster_id].index
        if len(idx) == 0:
            continue

        X_cluster = features.loc[idx]
        y_cluster = labels.loc[idx]

        if y_cluster.nunique() < 2:
            # Only one class present -> use constant classifier
            p = float(y_cluster.mean())
            clf: object = _ConstantProbClassifier(p)
        else:
            clf = LogisticRegression(
                max_iter=config.logreg_max_iter,
                class_weight="balanced",
            )
            clf.fit(X_cluster, y_cluster)

        classifiers[cluster_id] = clf
    return classifiers


def _compute_cost_matrix(
    features: pd.DataFrame,
    labels: Optional[pd.Series],
    centers: pd.DataFrame,
    classifiers: Dict[int, object],
    config: RAJCConfig,
) -> np.ndarray:
    """Compute joint cost matrix for all samples and clusters.

    Each entry (i, k) equals:
        ||x_i - μ_k||^2 + λ * logloss_i,k
    where the second term is the logistic loss under cluster-k classifier if
    labels are provided; otherwise it is 0 (for pure distance-based assignment).
    """
    n_samples = len(features)
    n_clusters = config.n_clusters
    costs = np.zeros((n_samples, n_clusters), dtype=float)
    feature_matrix = features.to_numpy()

    for k in range(n_clusters):
        center_vec = centers.loc[k].to_numpy()
        sq_dist = np.sum((feature_matrix - center_vec) ** 2, axis=1)

        if labels is not None and k in classifiers:
            probs = classifiers[k].predict_proba(features)[:, 1]
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            y = labels.to_numpy()
            logloss = -(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        else:
            logloss = np.zeros(n_samples, dtype=float)

        costs[:, k] = sq_dist + config.lambda_ * logloss

    return costs


class RAJCModel:
    """Response-Aware Joint Clustering model with alternating optimization."""

    def __init__(self, config: Optional[RAJCConfig] = None):
        self.config = config or RAJCConfig()
        self.centers_: Optional[pd.DataFrame] = None
        self.assignments_: Optional[pd.Series] = None
        self.classifiers_: Dict[int, object] = {}
        self.global_response_rate_: float = 0.5

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "RAJCModel":
        """Run alternating optimization to jointly learn clusters and classifiers."""
        self.global_response_rate_ = float(labels.mean())

        assignments = _initialize_assignments(
            features,
            self.config.n_clusters,
            self.config.random_state,
            self.config.kmeans_n_init,
        )
        centers = _update_centers(
            features, assignments, self.config.n_clusters, self.config.random_state
        )

        for _ in range(self.config.max_iter):
            classifiers = _fit_cluster_classifiers(features, labels, assignments, self.config)
            costs = _compute_cost_matrix(features, labels, centers, classifiers, self.config)
            new_assignments = pd.Series(
                np.argmin(costs, axis=1),
                index=features.index,
                name="cluster",
            )

            shift = (assignments != new_assignments).mean()
            assignments = new_assignments
            centers = _update_centers(
                features, assignments, self.config.n_clusters, self.config.random_state
            )

            if shift < self.config.tol:
                break

        self.assignments_ = assignments
        self.centers_ = centers
        self.classifiers_ = _fit_cluster_classifiers(features, labels, assignments, self.config)
        return self

    def predict_clusters(self, features: pd.DataFrame) -> pd.Series:
        """Assign clusters for new samples using learned centers.

        Note: assignments here are based solely on distances to cluster centers,
        since ground-truth labels are not available at inference time.
        """
        if self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")
        costs = _compute_cost_matrix(
            features,
            labels=None,
            centers=self.centers_,
            classifiers=self.classifiers_,
            config=self.config,
        )
        labels = np.argmin(costs, axis=1)
        return pd.Series(labels, index=features.index, name="cluster")

    def predict_response(self, features: pd.DataFrame) -> pd.Series:
        """Predict promotion-response probabilities for given samples."""
        if self.classifiers_ is None or self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        cluster_labels = self.predict_clusters(features)
        preds = pd.Series(index=features.index, dtype=float)

        for cluster_id, clf in self.classifiers_.items():
            idx = cluster_labels[cluster_labels == cluster_id].index
            if len(idx) == 0:
                continue
            probs = clf.predict_proba(features.loc[idx])[:, 1]
            preds.loc[idx] = probs

        # Fill any remaining NaNs (e.g., unseen empty clusters) with global rate
        preds = preds.fillna(self.global_response_rate_)
        return preds


def run_rajc(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Optional[RAJCConfig] = None,
) -> Tuple[RAJCModel, pd.Series]:
    """Train RAJC and return the model plus cluster assignments."""
    model = RAJCModel(config)
    model.fit(features, labels)
    return model, model.assignments_
