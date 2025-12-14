"""customer_segmentation.src.models

This package contains all model implementations used in the project:

- **Unsupervised segmentation baselines**:
    - :class:`~customer_segmentation.src.models.kmeans_baseline.KMeansBaseline`
    - :class:`~customer_segmentation.src.models.gmm_baseline.GMMBaseline`

- **Two-stage baseline**:
    - Cluster-then-predict (K-Means on behavior features, then a classifier per cluster).

- **Main proposed model**:
    - :class:`~customer_segmentation.src.models.rajc.RAJCModel` with
      ``RAJCConfig.model_type=\"ramoe\"``.

Upgraded joint model (new method)
--------------------------------
The project upgrades the original RAJC family into a stronger joint framework:

**HyRAMoE / RAMoE (Response-Aware Mixture-of-Experts)**

Key ideas:
- **Gating/segmentation** depends only on *behavior* features ``X_beh`` (leakage-safe).
- **Experts** predict campaign response using *full* features ``X_full``.
- Training uses an EM-style loop with soft responsibilities.
- The default gating is a lightweight **diagonal-covariance GMM** (more flexible than
  soft K-Means), and the default experts can be **tree-based** (HistGradientBoosting)
  for stronger nonlinear decision boundaries.

All joint models share the same API:
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
