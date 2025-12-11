"""Cluster profiling visualizations aligned with the methodology section."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def _check_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in DataFrame: {missing}")


def plot_income_vs_spent(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    hue_name: str = "Cluster",
    *,
    income_col: str = "Income",
    monetary_col: str = "monetary",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of income vs spending colored by cluster.

    Parameters
    ----------
    df :
        DataFrame containing income and monetary columns.
    cluster_labels :
        Cluster labels (same index as ``df``).
    hue_name :
        Legend title for clusters.
    income_col :
        Column name for income.
    monetary_col :
        Column name for total spending.
    save_path :
        Optional path to save the figure.
    """
    _check_required_columns(df, [income_col, monetary_col])

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x=income_col,
        y=monetary_col,
        hue=cluster_labels,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("Income vs Monetary by Cluster")
    ax.set_xlabel(income_col)
    ax.set_ylabel(monetary_col)
    ax.legend(title=hue_name)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_rfm_boxplots(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    rfm_cols: Sequence[str] = ("recency", "frequency", "monetary"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Boxplots for RFM dimensions to contrast segments.

    Parameters
    ----------
    df :
        DataFrame containing RFM columns.
    cluster_labels :
        Cluster labels aligned with ``df``.
    rfm_cols :
        Names of the RFM feature columns.
    save_path :
        Optional path to save the figure.
    """
    _check_required_columns(df, list(rfm_cols))

    melted = df.assign(cluster=cluster_labels).melt(
        id_vars="cluster",
        value_vars=list(rfm_cols),
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=melted, x="metric", y="value", hue="cluster", palette="tab10", ax=ax)
    ax.set_title("RFM distributions by cluster")
    ax.set_xlabel("RFM metric")
    ax.set_ylabel("Value")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_channel_mix(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    channel_cols: Optional[List[str]] = None,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar of average channel usage per cluster (web/catalog/store).

    Parameters
    ----------
    df :
        DataFrame containing channel purchase counts.
    cluster_labels :
        Cluster labels aligned with ``df``.
    channel_cols :
        Columns representing channels. Defaults to web/catalog/store purchases.
    save_path :
        Optional path to save the figure.
    """
    if channel_cols is None:
        channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

    _check_required_columns(df, channel_cols)

    grouped = df[channel_cols].assign(cluster=cluster_labels).groupby("cluster").mean()
    proportions = grouped.div(grouped.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = None
    for col in channel_cols:
        if bottom is None:
            ax.bar(proportions.index, proportions[col], label=col)
            bottom = proportions[col]
        else:
            ax.bar(proportions.index, proportions[col], bottom=bottom, label=col)
            bottom = bottom + proportions[col]

    ax.set_title("Channel purchase mix by cluster")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cluster")
    ax.legend(title="Channel")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_response_rates(
    response_rates: pd.Series,
    title: str = "Cluster response rates",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of response rates per cluster for RQ2."""
    fig, ax = plt.subplots()
    response_rates = response_rates.sort_index()
    response_rates.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Response rate")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_age_income_kde(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    age_col: str = "Age",
    income_col: str = "Income",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Density heatmaps for age vs income per cluster to surface demographic differences.

    Parameters
    ----------
    df :
        DataFrame containing age and income columns.
    cluster_labels :
        Cluster labels aligned with ``df``.
    age_col :
        Name of the age column.
    income_col :
        Name of the income column.
    save_path :
        Optional path to save the figure.
    """
    _check_required_columns(df, [age_col, income_col])

    clusters = pd.DataFrame(
        {age_col: df[age_col], income_col: df[income_col], "cluster": cluster_labels}
    )
    unique_clusters: Iterable = sorted(clusters["cluster"].unique())
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(
        1,
        n_clusters,
        figsize=(4 * n_clusters, 3),
        sharex=True,
        sharey=True,
    )
    if n_clusters == 1:
        axes = [axes]

    for ax, cluster in zip(axes, unique_clusters):
        subset = clusters[clusters["cluster"] == cluster]
        if subset.empty:
            continue
        sns.kdeplot(
            data=subset,
            x=age_col,
            y=income_col,
            fill=True,
            cmap="Blues",
            ax=ax,
            levels=10,
            thresh=0.05,
        )
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlabel(age_col)
        ax.set_ylabel(income_col)

    fig.suptitle("Age vs Income density by cluster")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
