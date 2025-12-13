"""Plotting utilities for clustering evaluation, profiling, and new-method evaluation curves.

This subpackage groups plotting helpers used throughout the experiments and
the report.

Includes:
- Clustering evaluation plots: elbow, silhouette distribution, PCA/t-SNE.
- Cluster profiling plots: income vs spent, RFM boxplots, channel mix, response rates.
- New-method evaluation curves: ROC, PR (AP), calibration, lift@budget, threshold sweep.
"""

from __future__ import annotations

from .plots_clustering import (
    plot_cluster_centers,
    plot_elbow_curve,
    plot_pca_scatter,
    plot_silhouette_distribution,
    plot_tsne_scatter,
    # new-method curves
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_lift_curve,
    plot_threshold_sweep,
)
from .plots_profiles import (
    plot_age_income_kde,
    plot_channel_mix,
    plot_income_vs_spent,
    plot_response_rates,
    plot_rfm_boxplots,
    plot_cluster_size_and_response,
)

__all__ = [
    # clustering visuals
    "plot_cluster_centers",
    "plot_elbow_curve",
    "plot_pca_scatter",
    "plot_silhouette_distribution",
    "plot_tsne_scatter",
    # new-method evaluation curves
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_lift_curve",
    "plot_threshold_sweep",
    # profiling visuals
    "plot_age_income_kde",
    "plot_channel_mix",
    "plot_income_vs_spent",
    "plot_response_rates",
    "plot_rfm_boxplots",
    "plot_cluster_size_and_response",
]
