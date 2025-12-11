"""Plotting utilities for clustering evaluation and customer profiling.

This subpackage groups plotting helpers used throughout the experiments and
the report (e.g., elbow curves, PCA/t-SNE visualizations, and cluster
profiles such as income-vs-spending plots and channel mix charts).
"""

from __future__ import annotations

from .plots_clustering import (
    plot_cluster_centers,
    plot_elbow_curve,
    plot_pca_scatter,
    plot_silhouette_distribution,
    plot_tsne_scatter,
)
from .plots_profiles import (
    plot_age_income_kde,
    plot_channel_mix,
    plot_income_vs_spent,
    plot_response_rates,
    plot_rfm_boxplots,
)

__all__ = [
    "plot_cluster_centers",
    "plot_elbow_curve",
    "plot_pca_scatter",
    "plot_silhouette_distribution",
    "plot_tsne_scatter",
    "plot_age_income_kde",
    "plot_channel_mix",
    "plot_income_vs_spent",
    "plot_response_rates",
    "plot_rfm_boxplots",
]
