"""Experiment entrypoints for baselines, RAJC, and ablations.

This package exposes convenience wrappers around the main experimental
scripts so they can be invoked programmatically, e.g., from notebooks.

Typical usage
-------------
>>> from customer_segmentation.src.experiments import (
...     run_baselines,
...     run_rajc,
...     run_downstream,
...     run_ablation,
... )
>>> run_baselines()   # equivalent to: python -m ...run_baselines

By default, :mod:`run_rajc` uses the RAJC-v2 configuration
(model_type="constant_prob") defined in ``configs/rajc.yaml``.
"""

from __future__ import annotations

from typing import Sequence, Optional

from .run_baselines import main as run_baselines
from .run_rajc import main as run_rajc
from .run_downstream import main as run_downstream
from .run_ablation import main as run_ablation

__all__ = [
    "run_baselines",
    "run_rajc",
    "run_downstream",
    "run_ablation",
]


def run_all(argv: Optional[Sequence[str]] = None) -> None:
    """Run baselines, RAJC, and downstream experiments in sequence.

    This is purely a convenience function for quick end-to-end experiments.
    """
    # We ignore argv here and rely on each sub-main's default CLI parameters.
    run_baselines()
    run_rajc()
    run_downstream()
    run_ablation()
