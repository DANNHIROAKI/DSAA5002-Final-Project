"""Visualization helpers for cluster evaluation, elbow plots, embeddings, and new-method curves.

原模块主要支撑 RQ1 / Ablation：
- Elbow 曲线看选 K；
- Silhouette 分布看簇紧凑度/分离度；
- PCA / t-SNE 可视化高维特征空间中的簇结构；
- 簇中心折线图看不同特征上的“形状差异”。

Upgrades for the new methodology:
- Remove seaborn dependency (matplotlib only).
- Add prediction-evaluation curves needed by the new protocol:
  ROC / PR / Calibration / Lift / Threshold sweep.
- Add stronger guards: silhouette requires n_clusters>=2 and n_samples>n_clusters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_samples,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.calibration import calibration_curve


def _maybe_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)


def _to_1d_numpy(x: Any) -> np.ndarray:
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy().reshape(-1)
    else:
        arr = np.asarray(x).reshape(-1)
    return arr


def _drop_nan_pairs(y: Any, s: Any) -> Tuple[np.ndarray, np.ndarray]:
    yy = _to_1d_numpy(y).astype(float)
    ss = _to_1d_numpy(s).astype(float)
    if yy.shape[0] != ss.shape[0]:
        raise ValueError(f"y and score/prob length mismatch: {yy.shape[0]} vs {ss.shape[0]}")
    mask = np.isfinite(yy) & np.isfinite(ss)
    return yy[mask], ss[mask]


# -----------------------------
# Clustering plots
# -----------------------------

def plot_elbow_curve(
    ks: Iterable[int],
    inertias: Iterable[float],
    title: str = "Elbow Curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot inertia vs. cluster number for elbow selection."""
    ks = list(ks)
    inertias = list(inertias)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)

    for k, inertia in zip(ks, inertias):
        ax.text(k, inertia, f"{inertia:.1f}", fontsize=8, ha="center", va="bottom")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_silhouette_distribution(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str = "Silhouette distribution",
    *,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    show_mean: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw per-sample silhouette scores grouped by cluster (box + jitter scatter).

    Raises
    ------
    ValueError
        If fewer than two clusters are present, or n_samples <= n_clusters.
    """
    X = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    y = labels.to_numpy() if isinstance(labels, pd.Series) else np.asarray(labels)
    y = y.reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {X.shape[0]} vs {y.shape[0]}")

    # Optional subsampling for speed
    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(X.shape[0], size=int(sample_size), replace=False)
        X = X[idx]
        y = y[idx]

    n_clusters = len(np.unique(y))
    if n_clusters < 2:
        raise ValueError("Silhouette distribution requires at least two clusters.")
    if X.shape[0] <= n_clusters:
        raise ValueError("Silhouette is undefined when n_samples <= n_clusters.")

    scores = silhouette_samples(X, y)

    clusters = sorted(np.unique(y))
    data = [scores[y == c] for c in clusters]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)

    # Jitter points
    rng = np.random.default_rng(int(random_state))
    for i, c in enumerate(clusters, start=1):
        s = scores[y == c]
        if s.size == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=s.size)
        ax.scatter(np.full(s.size, i) + jitter, s, s=8, alpha=0.35)

    if show_mean:
        ax.axhline(float(scores.mean()), linestyle="--", linewidth=1, label="Mean")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_pca_scatter(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str = "Cluster PCA",
    *,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of PCA-reduced features colored by cluster labels.

    Axis labels include explained variance ratio.
    """
    X = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    y = labels.to_numpy() if isinstance(labels, pd.Series) else np.asarray(labels)
    y = y.reshape(-1)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("PCA scatter requires at least 2 feature dimensions.")

    pca = PCA(n_components=2, random_state=int(random_state))
    reduced = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    clusters = sorted(np.unique(y))
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot per cluster for a clean legend
    for cid in clusters:
        mask = y == cid
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=20, alpha=0.8, label=str(cid))

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    ax.legend(title="Cluster", fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_tsne_scatter(
    features: pd.DataFrame | np.ndarray,
    labels: pd.Series | np.ndarray,
    perplexity: float = 30.0,
    title: str = "t-SNE of clusters",
    *,
    random_state: Optional[int] = 42,
    sample_size: Optional[int] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """t-SNE embedding colored by cluster labels (useful for nonlinear structure)."""
    X = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    y = labels.to_numpy() if isinstance(labels, pd.Series) else np.asarray(labels)
    y = y.reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {X.shape[0]} vs {y.shape[0]}")

    if sample_size is not None and sample_size < X.shape[0]:
        rng = np.random.default_rng(42 if random_state is None else int(random_state))
        idx = rng.choice(X.shape[0], size=int(sample_size), replace=False)
        X = X[idx]
        y = y[idx]

    # t-SNE requires perplexity < n_samples
    if X.shape[0] <= 3:
        raise ValueError("t-SNE requires at least 4 samples.")
    if perplexity >= X.shape[0]:
        perplexity = max(5.0, float(X.shape[0] - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="random",
        learning_rate="auto",
        random_state=None if random_state is None else int(random_state),
    )
    reduced = tsne.fit_transform(X)

    clusters = sorted(np.unique(y))
    fig, ax = plt.subplots(figsize=(6, 4))
    for cid in clusters:
        mask = y == cid
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=20, alpha=0.8, label=str(cid))

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Cluster", fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_cluster_centers(
    centers: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Cluster centers",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot of cluster centers across dimensions to compare shape differences."""
    if feature_names is None:
        feature_names = list(centers.columns)
    else:
        feature_names = list(feature_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for row_idx, (cluster_id, row) in enumerate(centers.iterrows()):
        # Use the DataFrame index as the cluster id for correctness.
        ax.plot(feature_names, row.values, marker="o", label=f"Cluster {cluster_id}")

    ax.set_xlabel("Features")
    ax.set_ylabel("Center value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# -----------------------------
# New-method evaluation curves
# -----------------------------

def plot_roc_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    title: str = "ROC curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """ROC curve + AUC (needed by the new evaluation protocol)."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = y.astype(int)

    # Numerical stability
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    # AUC can fail if only one class exists
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")

    fpr, tpr, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}" if np.isfinite(auc) else "AUC=nan")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_pr_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    title: str = "Precision-Recall curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Precision-Recall curve + AP (PR-AUC)."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = y.astype(int)

    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    try:
        ap = float(average_precision_score(y, p))
    except ValueError:
        ap = float("nan")

    precision, recall, _ = precision_recall_curve(y, p)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(recall, precision, label=f"AP={ap:.3f}" if np.isfinite(ap) else "AP=nan")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_calibration_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    title: str = "Calibration curve",
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Calibration curve (reliability diagram)."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = y.astype(int)

    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=int(n_bins), strategy=str(strategy))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_lift_curve(
    y_true: Union[pd.Series, np.ndarray],
    scores: Union[pd.Series, np.ndarray],
    title: str = "Lift curve",
    *,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Lift@budget curve for business-facing evaluation.

    Lift(f) = precision@top-f / base_rate.
    """
    y, s = _drop_nan_pairs(y_true, scores)
    y = (y > 0).astype(int)

    if y.size == 0:
        raise ValueError("Empty inputs after dropping NaNs.")

    base_rate = float(y.mean())
    if base_rate <= 0:
        base_rate = float("nan")

    # Sort by score descending
    order = np.argsort(-s)
    y_sorted = y[order]

    lifts = []
    fracs = []
    n = y_sorted.size

    for f in fractions:
        f = float(f)
        if not (0.0 < f <= 1.0):
            continue
        k = int(np.ceil(n * f))
        k = max(1, min(k, n))
        prec = float(y_sorted[:k].mean())
        lift = float("nan") if not np.isfinite(base_rate) or base_rate == 0 else float(prec / base_rate)
        fracs.append(f)
        lifts.append(lift)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fracs, lifts, marker="o", label="Model lift")
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Random (lift=1)")
    ax.set_xlabel("Budget fraction (top-q)")
    ax.set_ylabel("Lift")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_threshold_sweep(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    title: str = "Threshold sweep",
    *,
    thresholds: Optional[Sequence[float]] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot F1 and Balanced Accuracy over thresholds (for validation threshold selection).

    This supports the new protocol: choose threshold on validation set, then
    report final metrics on the held-out test set.
    """
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = y.astype(int)

    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    ths = np.array(list(thresholds), dtype=float)
    f1s = np.zeros_like(ths)
    bas = np.zeros_like(ths)

    for i, t in enumerate(ths):
        pred = (p >= t).astype(int)
        f1s[i] = f1_score(y, pred, zero_division=0)
        bas[i] = balanced_accuracy_score(y, pred)

    best_f1_t = float(ths[np.argmax(f1s)])
    best_ba_t = float(ths[np.argmax(bas)])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ths, f1s, label=f"F1 (best@{best_f1_t:.2f})")
    ax.plot(ths, bas, label=f"Balanced Acc (best@{best_ba_t:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig
