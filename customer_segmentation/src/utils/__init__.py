"""Utility helpers for logging, seeding, and metrics.

This package exposes a small, cohesive API that is used across the
project's experiments and scripts. Importing from here keeps the
call-sites tidy, for example::

    from customer_segmentation.src.utils import (
        configure_logging,
        set_global_seed,
        compute_lift,
    )
"""

from __future__ import annotations

from .logging_utils import DEFAULT_LOG_FORMAT, configure_logging
from .metrics_utils import (
    ClusterResponseStats,
    classification_summary,
    compute_lift,
    response_rate_by_cluster,
    summarize_topk_lift,
)
from .seed_utils import reproducible_numpy_rng, set_global_seed

__all__ = [
    "configure_logging",
    "DEFAULT_LOG_FORMAT",
    "compute_lift",
    "summarize_topk_lift",
    "ClusterResponseStats",
    "response_rate_by_cluster",
    "classification_summary",
    "set_global_seed",
    "reproducible_numpy_rng",
]
