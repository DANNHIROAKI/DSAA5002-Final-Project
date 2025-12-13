"""Source package for the DSAA5002 customer segmentation project.

This subpackage is organized into:
- data: loading, cleaning, feature engineering
- models: baselines + RAJC family
- evaluation: clustering / segmentation / prediction metrics
- visualization: report-ready figures
- experiments: runnable scripts for baselines/RAJC/downstream/ablation
- utils: logging, seeds, and small helpers

We intentionally keep this __init__ lightweight to avoid importing heavy
dependencies (e.g., scikit-learn) on package import.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "data",
    "models",
    "evaluation",
    "visualization",
    "experiments",
    "utils",
]
