"""Response-Aware Joint Clustering (RAJC) implementations.

This module implements two closely related variants of RAJC:

1) RAJC-v2 with *cluster-wise constant probabilities* (default)
   - model_type = "constant_prob"
   - Each cluster k is characterized by a single response probability p_k.
   - The joint objective directly encourages clusters to have homogeneous
     promotion-response behaviour and large inter-cluster differences.

2) Original RAJC with *per-cluster logistic regression*
   - model_type = "logreg"
   - Each cluster k owns a local logistic regression (w_k, b_k).
   - Kept mainly for ablation or comparison.

The public API (RAJCConfig, RAJCModel, run_rajc) remains backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantProbClassifier:
    """Cluster-level classifier that always outputs a fixed probability.

    Used in the original RAJC (logreg mode) as a robust fallback when a
    cluster contains only one class and LogisticRegression cannot be fitted.
    """

    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-7, 1 - 1e-7))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = X.shape[0]
        p = np.full(n, self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class RAJCConfig:
    """Configuration for the RAJC alternating optimization algorithm.

    Parameters
    ----------
    n_clusters :
        Number of clusters K.
    lambda_ :
        Trade-off between clustering distortion and response loss.
    gamma :
        Optional polarization regularizer weight. When > 0 (in constant_prob
        mode), it encourages cluster-level probabilities p_k to be closer to
        0 or 1 by penalizing p_k(1 - p_k). The term is not used directly in
        the E/M steps here but can be leveraged in model selection.
    max_iter :
        Maximum number of outer alternating-optimization iterations.
    tol :
        Early-stopping tolerance on the fraction of samples that change
        cluster between iterations.
    random_state :
        Random seed for reproducibility.
    smoothing :
        Laplace smoothing strength when estimating cluster probabilities
        p_k = (pos_k + smoothing) / (n_k + 2 * smoothing).
    logreg_max_iter :
        Max iterations for per-cluster LogisticRegression in "logreg" mode.
    kmeans_n_init :
        Number of KMeans initializations for RAJC initialization.
    model_type :
        Either "constant_prob" (RAJC-v2, recommended) or "logreg"
        (original RAJC with per-cluster logistic regressions).
    """

    n_clusters: int = 4
    lambda_: float = 1.0
    gamma: float = 0.0
    max_iter: int = 20
    tol: float = 1e-4
    random_state: int = 42
    smoothing: float = 1.0
    logreg_max_iter: int = 200
    kmeans_n_init: int = 10
    model_type: str = "constant_prob"  # "constant_prob" or "logreg"


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
            centers.append(
                features.sample(1, random_state=random_state + cluster_id).iloc[0]
            )
        else:
            centers.append(features.loc[cluster_idx].mean(axis=0))
    return pd.DataFrame(centers, index=range(n_clusters))


# ----- Helpers for original RAJC (logreg mode) ---------------------------------


def _fit_cluster_classifiers_logreg(
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


def _compute_cost_matrix_logreg(
    features: pd.DataFrame,
    labels: Optional[pd.Series],
    centers: pd.DataFrame,
    classifiers: Dict[int, object],
    config: RAJCConfig,
) -> np.ndarray:
    """Compute cost matrix for RAJC in 'logreg' mode.

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


# ----- Helpers for RAJC-v2 (constant_prob mode) --------------------------------


def _update_cluster_probs(
    labels: pd.Series,
    assignments: pd.Series,
    n_clusters: int,
    smoothing: float,
    global_rate: float,
) -> np.ndarray:
    """Estimate cluster-wise constant probabilities with Laplace smoothing."""
    probs = np.zeros(n_clusters, dtype=float)
    y = labels.to_numpy()

    for k in range(n_clusters):
        mask = assignments == k
        n_k = int(mask.sum())
        if n_k == 0:
            # Empty cluster falls back to global rate.
            p_k = global_rate
        else:
            pos = float(y[mask.to_numpy()].sum())
            p_k = (pos + smoothing) / (n_k + 2.0 * smoothing)

        probs[k] = np.clip(p_k, 1e-6, 1 - 1e-6)

    return probs


def _compute_cost_matrix_constant(
    features: pd.DataFrame,
    labels: pd.Series,
    centers: pd.DataFrame,
    cluster_probs: np.ndarray,
    config: RAJCConfig,
) -> np.ndarray:
    """Compute cost matrix for RAJC-v2 (cluster-wise constant probabilities).

    For each sample i and cluster k:

        cost_{ik} = ||x_i - μ_k||^2
                    + λ * CE(y_i ; p_k),

    where CE is the binary cross-entropy using the cluster-wise constant
    probability p_k.
    """
    X = features.to_numpy()  # (n, d)
    centers_np = centers.to_numpy()  # (K, d)
    n, d = X.shape
    K = centers_np.shape[0]

    # Squared distances: (n, K)
    diff = X[:, None, :] - centers_np[None, :, :]
    sq_dist = np.sum(diff**2, axis=2)

    # Cluster-wise constant probabilities, broadcast to (n, K)
    y = labels.to_numpy().reshape(-1, 1)  # (n, 1)
    p = cluster_probs.reshape(1, -1)  # (1, K)

    # Binary cross-entropy CE(y, p)
    ce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    return sq_dist + config.lambda_ * ce


# ---------------------------------------------------------------------------
# Main RAJC model
# ---------------------------------------------------------------------------


class RAJCModel:
    """Response-Aware Joint Clustering model with alternating optimization.

    The behavior is controlled by :class:`RAJCConfig`:

    - model_type = "constant_prob"  -> RAJC-v2 with cluster-wise p_k (recommended).
    - model_type = "logreg"         -> original RAJC with per-cluster logistic models.
    """

    def __init__(self, config: Optional[RAJCConfig] = None):
        self.config = config or RAJCConfig()
        self.centers_: Optional[pd.DataFrame] = None
        self.assignments_: Optional[pd.Series] = None

        # Attributes for logreg mode
        self.classifiers_: Dict[int, object] = {}

        # Attributes for constant_prob mode
        self.cluster_probs_: Optional[np.ndarray] = None

        # Shared
        self.global_response_rate_: float = 0.5

    # ----- public API -----------------------------------------------------

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "RAJCModel":
        """Run alternating optimization to jointly learn clusters and response model."""
        self.global_response_rate_ = float(labels.mean())

        if self.config.model_type == "logreg":
            self._fit_logreg_mode(features, labels)
        else:
            # default and recommended
            self._fit_constant_prob_mode(features, labels)

        return self

    def predict_clusters(self, features: pd.DataFrame) -> pd.Series:
        """Assign clusters for new samples using learned centers.

        Assignments here are always based solely on distances to cluster centers,
        since ground-truth labels are not available at inference time.
        """
        if self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = features.to_numpy()
        centers_np = self.centers_.to_numpy()
        diff = X[:, None, :] - centers_np[None, :, :]
        sq_dist = np.sum(diff**2, axis=2)  # (n, K)
        labels = np.argmin(sq_dist, axis=1)
        return pd.Series(labels, index=features.index, name="cluster")

    def predict_response(self, features: pd.DataFrame) -> pd.Series:
        """Predict promotion-response probabilities for given samples."""
        if self.centers_ is None:
            raise ValueError("Model has not been fitted yet.")

        cluster_labels = self.predict_clusters(features)

        if self.config.model_type == "logreg":
            if self.classifiers_ is None:
                raise ValueError("Logistic classifiers have not been fitted.")
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

        # constant_prob mode: just map cluster -> p_k
        if self.cluster_probs_ is None:
            raise ValueError("Cluster probabilities have not been estimated.")

        probs = np.full(len(features), self.global_response_rate_, dtype=float)
        for k in range(self.config.n_clusters):
            idx = cluster_labels[cluster_labels == k].index
            if len(idx) == 0:
                continue
            probs[features.index.get_indexer(idx)] = self.cluster_probs_[k]

        return pd.Series(probs, index=features.index, name="response_prob")

    # ----- internal training routines -------------------------------------

    def _fit_logreg_mode(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Original RAJC training with per-cluster logistic regressions."""
        # Initialization
        assignments = _initialize_assignments(
            features,
            self.config.n_clusters,
            self.config.random_state,
            self.config.kmeans_n_init,
        )
        centers = _update_centers(
            features, assignments, self.config.n_clusters, self.config.random_state
        )

        # Alternating optimization
        for _ in range(self.config.max_iter):
            classifiers = _fit_cluster_classifiers_logreg(
                features, labels, assignments, self.config
            )
            costs = _compute_cost_matrix_logreg(
                features, labels, centers, classifiers, self.config
            )
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

        # Final fit
        self.assignments_ = assignments
        self.centers_ = centers
        self.classifiers_ = _fit_cluster_classifiers_logreg(
            features, labels, assignments, self.config
        )
        self.cluster_probs_ = None  # not used in this mode

    def _fit_constant_prob_mode(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """RAJC-v2 training with cluster-wise constant probabilities."""
        # Initialization via KMeans
        assignments = _initialize_assignments(
            features,
            self.config.n_clusters,
            self.config.random_state,
            self.config.kmeans_n_init,
        )
        centers = _update_centers(
            features, assignments, self.config.n_clusters, self.config.random_state
        )
        cluster_probs = _update_cluster_probs(
            labels,
            assignments,
            self.config.n_clusters,
            self.config.smoothing,
            self.global_response_rate_,
        )

        # Alternating optimization
        for _ in range(self.config.max_iter):
            costs = _compute_cost_matrix_constant(
                features, labels, centers, cluster_probs, self.config
            )
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
            cluster_probs = _update_cluster_probs(
                labels,
                assignments,
                self.config.n_clusters,
                self.config.smoothing,
                self.global_response_rate_,
            )

            if shift < self.config.tol:
                break

        self.assignments_ = assignments
        self.centers_ = centers
        self.cluster_probs_ = cluster_probs
        self.classifiers_ = {}  # not used in this mode


def run_rajc(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Optional[RAJCConfig] = None,
) -> Tuple[RAJCModel, pd.Series]:
    """Train RAJC and return the model plus cluster assignments.

    By default this uses RAJC-v2 (constant_prob). To reproduce the original
    RAJC behaviour, construct RAJCConfig(model_type="logreg").
    """
    model = RAJCModel(config)
    model.fit(features, labels)
    return model, model.assignments_
