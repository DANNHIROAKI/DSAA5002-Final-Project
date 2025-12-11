"""Cluster profiling visualizations aligned with the methodology section."""

from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_income_vs_spent(df: pd.DataFrame, cluster_labels: pd.Series, hue_name: str = "Cluster"):
    """Create a scatter plot of income vs spending colored by cluster."""

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Income", y="monetary", hue=cluster_labels, palette="tab10", ax=ax)
    ax.set_title("Income vs Monetary by Cluster")
    ax.set_xlabel("Income")
    ax.set_ylabel("Monetary (total spend)")
    ax.legend(title=hue_name)
    return fig


def plot_rfm_boxplots(df: pd.DataFrame, cluster_labels: pd.Series):
    """Boxplots for RFM dimensions to contrast segments."""

    melted = df.assign(cluster=cluster_labels).melt(id_vars="cluster", value_vars=["recency", "frequency", "monetary"], var_name="metric")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=melted, x="metric", y="value", hue="cluster", palette="tab10", ax=ax)
    ax.set_title("RFM distributions by cluster")
    ax.set_xlabel("RFM metric")
    ax.set_ylabel("Value")
    return fig


def plot_channel_mix(df: pd.DataFrame, cluster_labels: pd.Series, channel_cols: Optional[List[str]] = None):
    """Stacked bar of average channel usage per cluster (web/catalog/store)."""

    if channel_cols is None:
        channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    grouped = df[channel_cols].assign(cluster=cluster_labels).groupby("cluster").mean()
    proportions = grouped.div(grouped.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = None
    for col in channel_cols:
        ax.bar(proportions.index, proportions[col], bottom=bottom, label=col)
        bottom = (proportions[col] if bottom is None else bottom + proportions[col])
    ax.set_title("Channel purchase mix by cluster")
    ax.set_ylabel("Proportion")
    ax.legend(title="Channel")
    return fig


def plot_response_rates(response_rates: pd.Series, title: str = "Cluster response rates"):
    """Bar chart of response rates per cluster for RQ2."""

    fig, ax = plt.subplots()
    response_rates.sort_index().plot(kind="bar", color="steelblue", ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Response rate")
    ax.set_title(title)
    return fig


def plot_age_income_kde(df: pd.DataFrame, cluster_labels: pd.Series, age_col: str = "Age", income_col: str = "Income"):
    """Density heatmaps for age vs income per cluster to surface demographic differences."""

    clusters = pd.DataFrame({age_col: df[age_col], income_col: df[income_col], "cluster": cluster_labels})
    unique_clusters: Iterable = sorted(clusters["cluster"].unique())
    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 3), sharex=True, sharey=True)
    if n_clusters == 1:
        axes = [axes]
    for ax, cluster in zip(axes, unique_clusters):
        subset = clusters[clusters["cluster"] == cluster]
        sns.kdeplot(data=subset, x=age_col, y=income_col, fill=True, cmap="Blues", ax=ax, levels=10, thresh=0.05)
        ax.set_title(f"Cluster {cluster}")
    fig.suptitle("Age vs Income density by cluster")
    fig.tight_layout()
    return fig
