"""Evaluation utilities.

The repository uses a modular design: experiment runners call functions in this
package to compute:

- clustering quality metrics (Silhouette / Calinski–Harabasz / Davies–Bouldin);
- segmentation diagnostics (cluster response rates and dispersion);
- response prediction metrics (AUC / PR-AUC / log-loss / Brier, etc.).

Upgraded for the new method (RAMoE / HyRAMoE)
---------------------------------------------
- Safe index alignment and NaN filtering before metric computation.
- Adds optional calibration (ECE) and ranking/budget metrics.
"""

from __future__ import annotations

from .clustering import compute_scores
from .prediction import (
    choose_threshold,
    compute_classification_metrics,
    expected_calibration_error,
    compute_ranking_metrics,
)
from .segmentation import (
    cluster_response_rates,
    response_rate_variance,
    response_rate_std,
    response_rate_range,
    weighted_response_rate_variance,
    cluster_size_summary,
    segmentation_table,
    cluster_response_summary,
    cluster_lift_table,
    campaign_allocation_lift,
)

__all__ = [
    "compute_scores",
    "choose_threshold",
    "compute_classification_metrics",
    "expected_calibration_error",
    "compute_ranking_metrics",
    "cluster_response_rates",
    "response_rate_variance",
    "response_rate_std",
    "response_rate_range",
    "weighted_response_rate_variance",
    "cluster_size_summary",
    "segmentation_table",
    "cluster_response_summary",
    "cluster_lift_table",
    "campaign_allocation_lift",
]
