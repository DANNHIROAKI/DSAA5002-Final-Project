"""Experiment entrypoints for the project.

Each module contains a CLI-friendly ``main`` function.
This package re-exports those entrypoints so they can be called
programmatically, e.g. from ``customer_segmentation/run_all_experiments.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .run_ablation import main as run_ablation
from .run_baselines import main as run_baselines
from .run_downstream import main as run_downstream
from .run_rajc import main as run_rajc

__all__ = [
    "run_baselines",
    "run_rajc",
    "run_downstream",
    "run_ablation",
    "run_all",
]


def run_all(argv: Optional[Sequence[str]] = None) -> None:
    """Run the main experiment suite with default parameters.

    Parameters
    ----------
    argv:
        Currently unused. The wrapper is kept for backward compatibility.
    """
    _ = argv
    run_baselines()
    run_rajc()
    run_downstream()
    run_ablation()
