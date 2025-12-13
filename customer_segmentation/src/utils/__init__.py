"""Utility helpers for logging, seeding, and ranking/business metrics.

This package exposes a small, cohesive API that is used across the project.

Recommended imports
-------------------
from customer_segmentation.src.utils import (
    configure_logging,
    set_global_seed,
    compute_lift,
    lift_curve,
    ranking_summary,
)

The upgraded methodology emphasizes:
- leakage-free evaluation splits (handled elsewhere),
- reproducibility (seed_utils),
- business-relevant ranking metrics (metrics_utils).
"""

from __future__ import annotations

from .logging_utils import DEFAULT_LOG_FORMAT, configure_logging
from .metrics_utils import (
    compute_lift,
    lift_curve,
    ranking_summary,
    precision_at_frac,
    recall_at_frac,
    positives_in_top_frac,
    expected_score_sum_in_top_frac,
    top_k_from_frac,
)
from .seed_utils import reproducible_numpy_rng, set_global_seed, temp_seed

__all__ = [
    "configure_logging",
    "DEFAULT_LOG_FORMAT",
    "set_global_seed",
    "temp_seed",
    "reproducible_numpy_rng",
    "compute_lift",
    "lift_curve",
    "ranking_summary",
    "precision_at_frac",
    "recall_at_frac",
    "positives_in_top_frac",
    "expected_score_sum_in_top_frac",
    "top_k_from_frac",
]
