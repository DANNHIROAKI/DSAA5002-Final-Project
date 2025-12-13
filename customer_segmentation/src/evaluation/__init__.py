"""Evaluation utilities for clustering quality and downstream tasks.

This package groups together helpers for:

* Unsupervised clustering quality metrics (e.g. Silhouette, CH, DB).
* Business-focused segmentation metrics (per-cluster response rates, dispersion, budget lift).
* Downstream classification metrics for promotion-response prediction.

Upgrades for the new methodology:
- Export choose_threshold for validation-based decision threshold selection.
- Export richer segmentation helpers (range/std, campaign_allocation_lift).
"""

from __future__ import annotations

from .clustering import compute_scores
from .prediction import compute_classification_metrics, choose_threshold
from .segmentation import (
    cluster_response_rates,
    response_rate_variance,
    response_rate_std,
    response_rate_range,
    cluster_size_summary,
    segmentation_table,
    cluster_response_summary,
    campaign_allocation_lift,
)

__all__ = [
    "compute_scores",
    "compute_classification_metrics",
    "choose_threshold",
    "cluster_response_rates",
    "response_rate_variance",
    "response_rate_std",
    "response_rate_range",
    "cluster_size_summary",
    "segmentation_table",
    "cluster_response_summary",
    "campaign_allocation_lift",
]
