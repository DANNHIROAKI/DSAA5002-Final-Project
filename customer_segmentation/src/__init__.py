"""Source package for the DSAA5002 customer segmentation project.

Package layout
--------------
- data: loading, cleaning, feature engineering (leakage-safe transformers)
- models: baselines + RAJC family (RAMoE / HyRAMoE)
- evaluation: clustering / segmentation / prediction metrics
- visualization: report-ready figures
- experiments: runnable scripts for baselines/RAJC/downstream/ablation
- utils: logging, seeds, and small helpers

We intentionally keep this ``__init__`` lightweight to avoid importing heavy
third-party dependencies (e.g., scikit-learn) at import time.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = [
    "data",
    "models",
    "evaluation",
    "visualization",
    "experiments",
    "utils",
]
