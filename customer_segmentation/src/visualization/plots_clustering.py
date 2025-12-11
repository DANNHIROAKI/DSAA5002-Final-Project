"""Visualization helpers for cluster evaluation and elbow plots."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
