"""Model implementations for clustering and response modeling.

This package centralizes the main models used in the project:

* KMeansBaseline / GMMBaseline for unsupervised clustering.
* ClusterThenPredict baseline for two-stage cluster-then-predict.
* RAJCModel for response-aware joint clustering.

Upgraded method (main model)
---------------------------
We upgrade the original RAJC family to support a stronger, ranking-oriented
joint model: **RAMoE (Response-Aware Mixture-of-Experts)**.

The model mode is controlled by ``RAJCConfig.model_type``:

- ``"ramoe"`` (default, recommended):
    Soft, behavior-based gating + per-segment logistic experts trained with
    an EM-like algorithm. Designed to improve hold-out AUC and lift metrics.
- ``"constant_prob"``:
    RAJC-CP++ with cluster-wise constant response probability ``p_k``.
    Useful for ablation and interpretability.
- ``"logreg"``:
    Hard-assignment joint optimization with per-cluster logistic experts.

Unified API
-----------
All joint models expose:
    - ``fit(X_beh, y, full_features=X_full)``
    - ``predict_clusters(X_beh)``
    - ``predict_response(X_beh, full_features=X_full)``
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
