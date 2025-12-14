"""Visualization utilities for clustering, profiling, and prediction evaluation.

This subpackage contains lightweight matplotlib-based plotting functions used
throughout the project.

Design goals
------------
- Matplotlib-only (no hard seaborn dependency).
- Robust to common data issues (index misalignment, NaNs, probability array
  shapes from sklearn's ``predict_proba``).
- Backward compatible: existing experiment/launcher scripts can continue to
  import the same function names.

Upgrades for the new method (RAMoE / HyRAMoE)
--------------------------------------------
- Adds diagnostics for *soft assignments* (entropy / max-prob distributions).
- Adds budget-oriented evaluation visualizations (lift@top-q curve).
- Adds segmentation lift and cluster-level budget allocation plots.
"""

from __future__ import annotations

from .plots_clustering import (
    # clustering visuals
    plot_cluster_centers,
    plot_elbow_curve,
    plot_pca_scatter,
    plot_silhouette_distribution,
    plot_tsne_scatter,
    # prediction-evaluation curves
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_lift_curve,
    plot_threshold_sweep,
    # RAMoE / HyRAMoE diagnostics
    plot_assignment_entropy,
    plot_assignment_maxprob,
)
from .plots_profiles import (
    # profiling visuals
    plot_age_income_kde,
    plot_channel_mix,
    plot_income_vs_spent,
    plot_response_rates,
    plot_rfm_boxplots,
    plot_cluster_size_and_response,
    # new segmentation/budget visuals
    plot_cluster_lift_bars,
    plot_cluster_budget_allocation_curve,
)

__all__ = [
    # clustering
    "plot_cluster_centers",
    "plot_elbow_curve",
    "plot_pca_scatter",
    "plot_silhouette_distribution",
    "plot_tsne_scatter",
    # prediction curves
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_lift_curve",
    "plot_threshold_sweep",
    # RAMoE diagnostics
    "plot_assignment_entropy",
    "plot_assignment_maxprob",
    # profiling
    "plot_age_income_kde",
    "plot_channel_mix",
    "plot_income_vs_spent",
    "plot_response_rates",
    "plot_rfm_boxplots",
    "plot_cluster_size_and_response",
    # segmentation/budget
    "plot_cluster_lift_bars",
    "plot_cluster_budget_allocation_curve",
]
