"""Project-wide utilities (logging, seeds, ranking metrics).

This directory is deliberately lightweight. It supports the upgraded
RAMoE / HyRAMoE methodology by providing budget-oriented ranking metrics
(e.g., lift@top-q) used across experiments and plots.
"""

from __future__ import annotations

from .logging_utils import DEFAULT_LOG_FORMAT, configure_logging
from .metrics_utils import (
    assignment_entropy,
    assignment_maxprob,
    compute_lift,
    expected_score_sum_in_top_frac,
    lift_curve,
    positives_in_top_frac,
    precision_at_frac,
    recall_at_frac,
    ranking_summary,
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
    "assignment_entropy",
    "assignment_maxprob",
]
