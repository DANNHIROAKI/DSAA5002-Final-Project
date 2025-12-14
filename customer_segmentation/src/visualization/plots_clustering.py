"""Visualization helpers for cluster evaluation, embeddings, and prediction curves.

This module supports:
- *Clustering* plots: elbow curve, silhouette distribution, PCA/t-SNE scatter,
  and cluster-center line plots.
- *Prediction* evaluation curves: ROC, Precision-Recall, calibration, lift and
  threshold sweep (used by the leak-free evaluation protocol).
- *RAMoE / HyRAMoE* diagnostics: distributions of soft-assignment entropy and
  maximum assignment probability.

Notes
-----
- Uses matplotlib only (no seaborn hard dependency).
- Robust to sklearn ``predict_proba`` outputs of shape (n, 2).
- Drops NaNs/Infs for plotting stability.
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

try:
    # Optional: used for consistent lift computation if available
    from customer_segmentation.src.utils.metrics_utils import compute_lift

    _HAS_PROJECT_METRICS = True
except Exception:  # pragma: no cover
    compute_lift = None  # type: ignore
    _HAS_PROJECT_METRICS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)


def _to_1d_labels(x: Any) -> np.ndarray:
    """Convert labels to a 1D NumPy array."""
    if isinstance(x, (pd.Series, pd.Index)):
        arr = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            # If user passed a 2-column proba DataFrame by mistake, do not guess.
            raise ValueError(f"Expected a single-column DataFrame for labels, got shape={x.shape}.")
        arr = x.iloc[:, 0].to_numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr).reshape(-1)


def _to_1d_scores(x: Any) -> np.ndarray:
    """Convert scores/probabilities to a 1D array.

    Accepts:
    - 1D list/ndarray/Series
    - 2D ndarray of shape (n,2) from sklearn.predict_proba -> take column 1
    - DataFrame with 1 column -> that column
    - DataFrame with 2 columns -> take the second column
    """
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            arr = x.iloc[:, 0].to_numpy(dtype=float)
        elif x.shape[1] == 2:
            arr = x.iloc[:, 1].to_numpy(dtype=float)
        else:
            raise ValueError(f"Expected 1 or 2 columns for scores/proba, got shape={x.shape}.")
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                arr = arr[:, 1]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError(
                    f"Ambiguous score array shape={arr.shape}. Pass a 1D array or an (n,2) predict_proba output."
                )
    return np.asarray(arr, dtype=float).reshape(-1)


def _drop_nan_pairs(y: Any, s: Any) -> Tuple[np.ndarray, np.ndarray]:
    yy = _to_1d_labels(y).astype(float)
    ss = _to_1d_scores(s).astype(float)
    if yy.shape[0] != ss.shape[0]:
        raise ValueError(f"y and score/prob length mismatch: {yy.shape[0]} vs {ss.shape[0]}")
    mask = np.isfinite(yy) & np.isfinite(ss)
    return yy[mask], ss[mask]


def _ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Simple Expected Calibration Error (ECE).

    The implementation is intentionally lightweight and self-contained.
    """
    y_true = (y_true > 0).astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-12, 1.0 - 1e-12)

    n_bins = int(max(2, n_bins))

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, qs)
        edges[0] = 0.0
        edges[-1] = 1.0
        # handle duplicate edges due to ties
        if np.unique(edges).size < edges.size:
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unsupported ECE binning strategy: {strategy}")

    # bin index in [0, n_bins-1]
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    ece = 0.0
    n = float(len(y_true))
    if n == 0:
        return float("nan")

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        w = float(np.sum(mask)) / n
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += w * abs(acc - conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Clustering plots
# ---------------------------------------------------------------------------


def plot_elbow_curve(
    ks: Iterable[int],
    inertias: Iterable[float],
    title: str = "Elbow Curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot inertia vs. number of clusters for elbow selection."""
    ks = list(ks)
    inertias = list(inertias)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)

    for k, inertia in zip(ks, inertias):
        try:
            ax.text(k, float(inertia), f"{float(inertia):.1f}", fontsize=8, ha="center", va="bottom")
        except Exception:
            # If inertia is not numeric, skip annotation
            pass

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
    """Plot per-sample silhouette scores grouped by cluster.

    Notes
    -----
    Silhouette is undefined when:
    - fewer than 2 clusters exist, or
    - n_samples <= n_clusters.
    """
    X = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    y = _to_1d_labels(labels)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {X.shape[0]} vs {y.shape[0]}")

    # Drop rows with non-finite feature values or missing labels
    mask = np.all(np.isfinite(X), axis=1) & pd.Series(y).notna().to_numpy()
    X = X[mask]
    y = y[mask]

    # Optional subsampling for speed
    if sample_size is not None and 0 < int(sample_size) < X.shape[0]:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(X.shape[0], size=int(sample_size), replace=False)
        X = X[idx]
        y = y[idx]

    clusters = sorted(pd.unique(pd.Series(y)))
    n_clusters = len(clusters)

    if n_clusters < 2:
        raise ValueError("Silhouette distribution requires at least two clusters.")
    if X.shape[0] <= n_clusters:
        raise ValueError("Silhouette is undefined when n_samples <= n_clusters.")

    scores = silhouette_samples(X, y)
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
        ax.axhline(float(np.mean(scores)), linestyle="--", linewidth=1, label="Mean")
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
    """Scatter plot of PCA-reduced features coloured by cluster labels."""
    X = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    y = _to_1d_labels(labels)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("PCA scatter requires at least 2 feature dimensions.")

    # Drop non-finite rows
    mask = np.all(np.isfinite(X), axis=1) & pd.Series(y).notna().to_numpy()
    X = X[mask]
    y = y[mask]

    pca = PCA(n_components=2, random_state=int(random_state))
    reduced = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_

    clusters = sorted(pd.unique(pd.Series(y)))
    fig, ax = plt.subplots(figsize=(6, 4))

    for cid in clusters:
        mask_c = y == cid
        ax.scatter(reduced[mask_c, 0], reduced[mask_c, 1], s=20, alpha=0.8, label=str(cid))

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
    """t-SNE embedding coloured by cluster labels."""
    X = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    y = _to_1d_labels(labels)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {X.shape[0]} vs {y.shape[0]}")

    # Drop non-finite rows
    mask = np.all(np.isfinite(X), axis=1) & pd.Series(y).notna().to_numpy()
    X = X[mask]
    y = y[mask]

    if sample_size is not None and 0 < int(sample_size) < X.shape[0]:
        rng = np.random.default_rng(42 if random_state is None else int(random_state))
        idx = rng.choice(X.shape[0], size=int(sample_size), replace=False)
        X = X[idx]
        y = y[idx]

    if X.shape[0] <= 3:
        raise ValueError("t-SNE requires at least 4 samples.")

    # t-SNE requires perplexity < n_samples
    if float(perplexity) >= X.shape[0]:
        perplexity = max(5.0, float(X.shape[0] - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="random",
        learning_rate="auto",
        random_state=None if random_state is None else int(random_state),
    )
    reduced = tsne.fit_transform(X)

    clusters = sorted(pd.unique(pd.Series(y)))
    fig, ax = plt.subplots(figsize=(6, 4))
    for cid in clusters:
        mask_c = y == cid
        ax.scatter(reduced[mask_c, 0], reduced[mask_c, 1], s=20, alpha=0.8, label=str(cid))

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
    """Line plot of cluster centers across feature dimensions."""
    if feature_names is None:
        feature_names = list(centers.columns)
    else:
        feature_names = list(feature_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id, row in centers.iterrows():
        ax.plot(feature_names, row.values, marker="o", label=f"Cluster {cluster_id}")

    ax.set_xlabel("Features")
    ax.set_ylabel("Center value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Prediction evaluation curves
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    title: str = "ROC curve",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """ROC curve with AUC."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = (y > 0).astype(int)

    p = np.clip(p, 1e-15, 1.0 - 1e-15)

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
    """Precision-Recall curve with Average Precision (AP)."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = (y > 0).astype(int)

    p = np.clip(p, 1e-15, 1.0 - 1e-15)

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
    show_ece: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Reliability diagram (calibration curve), optionally annotated with ECE."""
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = (y > 0).astype(int)

    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=int(n_bins), strategy=str(strategy))

    ece = _ece(y, p, n_bins=int(n_bins), strategy=str(strategy)) if show_ece else None

    fig, ax = plt.subplots(figsize=(5.5, 4))
    label = "Model"
    if show_ece and ece is not None and np.isfinite(ece):
        label = f"Model (ECE={ece:.3f})"

    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
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
    """Lift@budget curve.

    Lift(f) = precision@top-f / base_rate.

    If project metric utilities are available, uses ``compute_lift`` to match
    the exact definition used in tables.
    """
    y, s = _drop_nan_pairs(y_true, scores)
    y = (y > 0).astype(int)

    if y.size == 0:
        raise ValueError("Empty inputs after dropping NaNs.")

    base_rate = float(np.mean(y))

    lifts: list[float] = []
    fracs: list[float] = []

    for f in fractions:
        f = float(f)
        if not (0.0 < f <= 1.0):
            continue
        fracs.append(f)
        if _HAS_PROJECT_METRICS and compute_lift is not None:
            lifts.append(float(compute_lift(y, s, top_frac=f)))
        else:
            # Local fallback
            order = np.argsort(-s)
            y_sorted = y[order]
            k = int(np.ceil(len(y_sorted) * f))
            k = max(1, min(k, len(y_sorted)))
            prec = float(np.mean(y_sorted[:k]))
            lifts.append(float("nan") if base_rate <= 0 else float(prec / base_rate))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fracs, lifts, marker="o", label="Model")
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
    """Plot F1 and Balanced Accuracy over thresholds.

    This supports the recommended protocol:
    - choose threshold on validation
    - report metrics on test
    """
    y, p = _drop_nan_pairs(y_true, y_prob)
    y = (y > 0).astype(int)

    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    ths = np.array(list(thresholds), dtype=float)
    f1s = np.zeros_like(ths)
    bas = np.zeros_like(ths)

    for i, t in enumerate(ths):
        pred = (p >= float(t)).astype(int)
        f1s[i] = f1_score(y, pred, zero_division=0)
        bas[i] = balanced_accuracy_score(y, pred)

    best_f1_t = float(ths[int(np.argmax(f1s))])
    best_ba_t = float(ths[int(np.argmax(bas))])

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


# ---------------------------------------------------------------------------
# RAMoE / HyRAMoE diagnostics (soft assignments)
# ---------------------------------------------------------------------------


def _to_2d_array(q: Any) -> np.ndarray:
    """Convert responsibilities / assignment probabilities to a 2D float array."""
    if isinstance(q, pd.DataFrame):
        arr = q.to_numpy(dtype=float)
    else:
        arr = np.asarray(q, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array of responsibilities (n,K); got shape={arr.shape}.")
    return arr


def plot_assignment_entropy(
    responsibilities: Any,
    title: str = "Soft assignment entropy",
    *,
    bins: int = 30,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of per-sample assignment entropy.

    Entropy is computed as:
        H(q_i) = -\sum_k q_{ik} log q_{ik}

    Lower entropy => crisper segmentation (more interpretable).
    """
    q = _to_2d_array(responsibilities)

    # Clip and normalise row-wise in case inputs are imperfect.
    q = np.clip(q, 1e-12, 1.0)
    q = q / np.clip(np.sum(q, axis=1, keepdims=True), 1e-12, None)

    ent = -np.sum(q * np.log(q), axis=1)
    ent = ent[np.isfinite(ent)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ent, bins=int(bins))
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # Annotate mean
    if ent.size > 0:
        ax.axvline(float(np.mean(ent)), linestyle="--", linewidth=1, label=f"mean={float(np.mean(ent)):.3f}")
        ax.legend()

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_assignment_maxprob(
    responsibilities: Any,
    title: str = "Max assignment probability",
    *,
    bins: int = 30,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of the maximum assignment probability per sample.

    Higher max-prob => crisper segmentation.
    """
    q = _to_2d_array(responsibilities)

    q = np.clip(q, 1e-12, 1.0)
    q = q / np.clip(np.sum(q, axis=1, keepdims=True), 1e-12, None)

    m = np.max(q, axis=1)
    m = m[np.isfinite(m)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(m, bins=int(bins), range=(0.0, 1.0))
    ax.set_xlabel("max_k q_{ik}")
    ax.set_ylabel("Count")
    ax.set_title(title)

    if m.size > 0:
        ax.axvline(float(np.mean(m)), linestyle="--", linewidth=1, label=f"mean={float(np.mean(m)):.3f}")
        ax.legend()

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


__all__ = [
    # clustering plots
    "plot_elbow_curve",
    "plot_silhouette_distribution",
    "plot_pca_scatter",
    "plot_tsne_scatter",
    "plot_cluster_centers",
    # prediction curves
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_calibration_curve",
    "plot_lift_curve",
    "plot_threshold_sweep",
    # RAMoE diagnostics
    "plot_assignment_entropy",
    "plot_assignment_maxprob",
]
