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
from .metrics_utils import compute_lift, lift_curve
from .seed_utils import reproducible_numpy_rng, set_global_seed

__all__ = [
    "configure_logging",
    "DEFAULT_LOG_FORMAT",
    "compute_lift",
    "lift_curve",
    "set_global_seed",
    "reproducible_numpy_rng",
]
