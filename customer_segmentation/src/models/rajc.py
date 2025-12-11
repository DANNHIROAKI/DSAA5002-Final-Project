"""Response-Aware Joint Clustering (RAJC) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


@dataclass
class RAJCConfig:
    """Configuration for the RAJC alternating optimization algorithm."""

    n_clusters: int = 4
    lambda_: float = 1.0
    max_iter: int = 20
    tol: float = 1e-4
    random_state: int = 42
    logreg_max_iter: int = 200


def _initialize_assignments(features: pd.DataFrame, n_clusters: int, random_state: int) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(features)
    return pd.Series(labels, index=features.index, name="cluster")


def _update_centers(features: pd.DataFrame, assignments: pd.Series, n_clusters: int) -> pd.DataFrame:
    centers = []
    for cluster_id in range(n_clusters):
        cluster_idx = assignments[assignments == cluster_id].index
        if len(cluster_idx) == 0:
            # re-seed empty cluster with a random point
            centers.append(features.sample(1, random_state=cluster_id).iloc[0])
        else:
            centers.append(features.loc[cluster_idx].mean(axis=0))
    return pd.DataFrame(centers, index=range(n_clusters))


def _fit_cluster_classifiers(
    features: pd.DataFrame, labels: pd.Series, assignments: pd.Series, config: RAJCConfig
) -> Dict[int, LogisticRegression]:
    classifiers: Dict[int, LogisticRegression] = {}
    for cluster_id in range(config.n_clusters):
        idx = assignments[assignments == cluster_id].index
        if len(idx) == 0:
            continue
        clf = LogisticRegression(max_iter=config.logreg_max_iter, class_weight="balanced")
        clf.fit(features.loc[idx], labels.loc[idx])
        classifiers[cluster_id] = clf
    return classifiers


def _compute_cost_matrix(
    features: pd.DataFrame,
    labels: Optional[pd.Series],
    centers: pd.DataFrame,
    classifiers: Dict[int, LogisticRegression],
    config: RAJCConfig,
) -> np.ndarray:
    n_samples = len(features)
    n_clusters = config.n_clusters
    costs = np.zeros((n_samples, n_clusters))
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
            logloss = np.zeros(n_samples)
        costs[:, k] = sq_dist + config.lambda_ * logloss
    return costs


class RAJCModel:
    """Response-Aware Joint Clustering model with alternating optimization."""

    def __init__(self, config: RAJCConfig = RAJCConfig()):
        self.config = config
        self.centers_: pd.DataFrame | None = None
        self.assignments_: pd.Series | None = None
        self.classifiers_: Dict[int, LogisticRegression] = {}

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "RAJCModel":
        assignments = _initialize_assignments(features, self.config.n_clusters, self.config.random_state)
        centers = _update_centers(features, assignments, self.config.n_clusters)

        for _ in range(self.config.max_iter):
            classifiers = _fit_cluster_classifiers(features, labels, assignments, self.config)
            costs = _compute_cost_matrix(features, labels, centers, classifiers, self.config)
            new_assignments = pd.Series(np.argmin(costs, axis=1), index=features.index, name="cluster")

            shift = (assignments != new_assignments).mean()
            assignments = new_assignments
            centers = _update_centers(features, assignments, self.config.n_clusters)

            if shift < self.config.tol:
                break

        self.assignments_ = assignments
        self.centers_ = centers
        self.classifiers_ = _fit_cluster_classifiers(features, labels, assignments, self.config)
        return self

    def predict_clusters(self, features: pd.DataFrame) -> pd.Series:
        if self.centers_ is None or self.classifiers_ is None:
            raise ValueError("Model has not been fitted yet.")
        costs = _compute_cost_matrix(features, None, self.centers_, self.classifiers_, self.config)
        labels = np.argmin(costs, axis=1)
        return pd.Series(labels, index=features.index, name="cluster")

    def predict_response(self, features: pd.DataFrame) -> pd.Series:
        if self.assignments_ is None or not self.classifiers_:
            raise ValueError("Model has not been fitted yet.")
        cluster_labels = self.predict_clusters(features)
        preds = pd.Series(index=features.index, dtype=float)
        for cluster_id, clf in self.classifiers_.items():
            idx = cluster_labels[cluster_labels == cluster_id].index
            if len(idx) == 0:
                continue
            preds.loc[idx] = clf.predict_proba(features.loc[idx])[:, 1]
        return preds


def run_rajc(
    features: pd.DataFrame, labels: pd.Series, config: RAJCConfig = RAJCConfig()
) -> Tuple[RAJCModel, pd.Series]:
    """Train RAJC and return the model plus cluster assignments."""

    model = RAJCModel(config)
    model.fit(features, labels)
    return model, model.assignments_
