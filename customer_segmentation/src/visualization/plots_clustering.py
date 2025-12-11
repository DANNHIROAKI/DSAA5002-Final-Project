"""Visualization helpers for cluster evaluation, elbow plots and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

sns.set_style("whitegrid")


def plot_elbow_curve(
    ks: Iterable[int],
    inertias: Iterable[float],
    title: str = "Elbow Curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot inertia vs. cluster number for elbow selection.

    Parameters
    ----------
    ks :
        Iterable of candidate cluster counts.
    inertias :
        Corresponding K-Means inertia values.
    title :
        Figure title.
    save_path :
        Optional path to save the figure. When ``None``, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    ks = list(ks)
    inertias = list(inertias)

    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)

    # Annotate each point with its inertia for easier reading in reports.
    for k, inertia in zip(ks, inertias):
        ax.text(k, inertia, f"{inertia:.1f}", fontsize=8, ha="center", va="bottom")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_silhouette_distribution(
    features: pd.DataFrame,
    labels: pd.Series,
    title: str = "Silhouette distribution",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw per-sample silhouette scores grouped by cluster.

    Parameters
    ----------
    features :
        Feature matrix used for clustering.
    labels :
        Cluster labels for each sample.
    title :
        Plot title.
    save_path :
        Optional path to save the figure.

    Raises
    ------
    ValueError
        If fewer than two clusters are present (silhouette undefined).
    """
    labels_series = pd.Series(labels)
    if labels_series.nunique() < 2:
        raise ValueError("Silhouette distribution requires at least two clusters.")

    scores = silhouette_samples(features, labels_series)
    df = pd.DataFrame({"score": scores, "cluster": labels_series})

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="cluster", y="score", ax=ax)
    sns.stripplot(
        data=df,
        x="cluster",
        y="score",
        ax=ax,
        color="gray",
        size=2,
        alpha=0.4,
    )
    ax.set_title(title)
    ax.axhline(scores.mean(), color="red", linestyle="--", linewidth=1, label="Mean")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_pca_scatter(
    features: pd.DataFrame,
    labels: pd.Series,
    title: str = "Cluster PCA",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of PCA-reduced features colored by cluster labels.

    Axis labels include explained variance ratio to better interpret separation. If
    the feature dimensionality is less than 2, a ValueError is raised.
    """
    if features.shape[1] < 2:
        raise ValueError("PCA scatter requires at least 2 feature dimensions.")

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_tsne_scatter(
    features: pd.DataFrame,
    labels: pd.Series,
    perplexity: float = 30.0,
    title: str = "t-SNE of clusters",
    *,
    random_state: Optional[int] = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """t-SNE embedding colored by cluster labels (useful for nonlinear structure).

    Parameters
    ----------
    features :
        Feature matrix.
    labels :
        Cluster labels.
    perplexity :
        t-SNE perplexity parameter. Should be smaller than the number of samples.
    title :
        Plot title.
    random_state :
        Random seed for t-SNE (for reproducible embeddings).
    save_path :
        Optional path to save the figure.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    reduced = tsne.fit_transform(features)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_cluster_centers(
    centers: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Cluster centers",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot of cluster centers across dimensions to compare shape differences.

    Parameters
    ----------
    centers :
        DataFrame whose rows are clusters and columns are features.
    feature_names :
        Optional sequence of feature names (defaults to ``centers.columns``).
    title :
        Plot title.
    save_path :
        Optional path to save the figure.
    """
    if feature_names is None:
        feature_names = list(centers.columns)
    else:
        feature_names = list(feature_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (_, row) in enumerate(centers.iterrows()):
        ax.plot(feature_names, row.values, marker="o", label=f"Cluster {idx}")

    ax.set_xlabel("Features")
    ax.set_ylabel("Center value")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
