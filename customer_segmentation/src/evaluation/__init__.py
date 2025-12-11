"""Evaluation utilities for clustering quality and downstream tasks.

This package groups together helpers for:

* Unsupervised clustering quality metrics (e.g. Silhouette, CH, DB).
* Business-focused segmentation metrics (per-cluster response rates, lift).
* Downstream classification metrics for promotion-response prediction.

Typical usage
-------------
>>> from customer_segmentation.src.evaluation import (
...     compute_scores,
...     cluster_response_rates,
...     response_rate_variance,
...     cluster_response_summary,
...     campaign_allocation_lift,
...     compute_classification_metrics,
... )
"""

from __future__ import annotations

from .clustering import compute_scores
from .prediction import compute_classification_metrics
from .segmentation import (
    AllocationResult,
    campaign_allocation_lift,
    cluster_response_rates,
    cluster_response_summary,
    response_rate_variance,
)

__all__ = [
    "compute_scores",
    "compute_classification_metrics",
    "cluster_response_rates",
    "cluster_response_summary",
    "response_rate_variance",
    "campaign_allocation_lift",
    "AllocationResult",
]
