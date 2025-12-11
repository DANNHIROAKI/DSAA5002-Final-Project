"""Model implementations for clustering and response modeling.

This package centralizes the main models used in the project:

* KMeansBaseline / GMMBaseline for unsupervised clustering.
* ClusterThenPredict baseline for two-stage cluster-then-predict.
* RAJC model for response-aware joint clustering.
"""

from __future__ import annotations

from .kmeans_baseline import KMeansBaseline, KMeansConfig, run_kmeans
from .gmm_baseline import GMMBaseline, GMMConfig, run_gmm
from .cluster_then_predict import (
    ClusterThenPredictConfig,
    fit_cluster_then_predict,
    predict_with_clusters,
)
from .rajc import RAJCConfig, RAJCModel, run_rajc

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
    "run_rajc",
]
