"""Visualization helpers for cluster evaluation, elbow plots and embeddings."""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

sns.set_style("whitegrid")


def plot_elbow_curve(ks: Iterable[int], inertias: Iterable[float], title: str = "Elbow Curve"):
    """Plot inertia vs cluster number for elbow selection."""

    ks = list(ks)
    inertias = list(inertias)
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)
    for k, inertia in zip(ks, inertias):
        ax.text(k, inertia, f"{inertia:.1f}", fontsize=8, ha="center", va="bottom")
    return fig


def plot_silhouette_distribution(features: pd.DataFrame, labels: pd.Series, title: str = "Silhouette distribution"):
    """Draw per-sample silhouette scores grouped by cluster."""

    scores = silhouette_samples(features, labels)
    df = pd.DataFrame({"score": scores, "cluster": labels})
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="cluster", y="score", ax=ax)
    sns.stripplot(data=df, x="cluster", y="score", ax=ax, color="gray", size=2, alpha=0.4)
    ax.set_title(title)
    ax.axhline(scores.mean(), color="red", linestyle="--", linewidth=1, label="Mean")
    ax.legend()
    return fig


def plot_pca_scatter(features: pd.DataFrame, labels: pd.Series, title: str = "Cluster PCA"):
    """Scatter plot of PCA-reduced features colored by cluster labels."""

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def plot_tsne_scatter(features: pd.DataFrame, labels: pd.Series, perplexity: float = 30.0, title: str = "t-SNE of clusters"):
    """t-SNE embedding colored by cluster labels (useful for nonlinear structure)."""

    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto")
    reduced = tsne.fit_transform(features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=20)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def plot_cluster_centers(centers: pd.DataFrame, feature_names: Optional[Iterable[str]] = None, title: str = "Cluster centers"):
    """Line plot of cluster centers across dimensions to compare shape differences."""

    if feature_names is None:
        feature_names = centers.columns
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (_, row) in enumerate(centers.iterrows()):
        ax.plot(feature_names, row.values, marker="o", label=f"Cluster {idx}")
    ax.set_xlabel("Features")
    ax.set_ylabel("Center value")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig
