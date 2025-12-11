"""Model implementations for clustering and response modeling.

This package centralizes the main models used in the project:

* KMeansBaseline / GMMBaseline for unsupervised clustering.
* ClusterThenPredict baseline for two-stage cluster-then-predict.
* RAJCModel for response-aware joint clustering.

The RAJC model supports two modes controlled by ``RAJCConfig.model_type``:

- ``"constant_prob"`` (default): RAJC-v2 with cluster-wise constant response
  probabilities p_k (recommended for the main experiments).
- ``"logreg"``: original RAJC with per-cluster logistic regressions, mainly
  used for ablation or comparison.
"""

from __future__ import annotations

from .kmeans_baseline import KMeansBaseline, KMeansConfig, run_kmeans
from .gmm_baseline import GMMBaseline, GMMConfig, run_gmm
from .cluster_then_predict import (
    ClusterThenPredictConfig,
    fit_cluster_then_predict,
    predict_with_clusters,
)
from .rajc import RAJCConfig, RAJCModel

__all__ = [
    "KMeansBaseline",
    "KMeansConfig",
    "run_kmeans",
    "GMMBaseline",
    "GMMConfig",
    "run_gmm",
    "ClusterThenPredictConfig",
    "fit_cluster_then_predict",
    "predict_with_clusters",
    "RAJCConfig",
    "RAJCModel",
]
