"""Cluster profiling visualizations aligned with the methodology.

这些函数主要用在「簇画像 / Profiling」部分，帮助回答：
“不同簇在收入、消费、年龄、渠道偏好和响应率上有什么差异？”.

Upgrades for the new methodology:
- Robust column-name resolution: works with both raw columns (Income/Monetary/Recency...)
  and engineered columns (income/monetary/recency...).
- Robust alignment between df index and cluster_labels index.
- Add segmentation-oriented plots: cluster size + response rate in one figure.
- Remove hard dependency on seaborn; use matplotlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _align_labels(df: pd.DataFrame, cluster_labels: Union[pd.Series, np.ndarray, List[int]]) -> pd.Series:
    """Align cluster labels with df rows.

    Rules:
    - If cluster_labels is a Series with the same index as df: use as-is.
    - If cluster_labels is a Series with different index but same length: reindex to df.index by position.
    - If cluster_labels is array-like with same length: convert to Series using df.index.
    """
    if isinstance(cluster_labels, pd.Series):
        if cluster_labels.index.equals(df.index):
            return cluster_labels
        if len(cluster_labels) == len(df):
            return pd.Series(cluster_labels.to_numpy(), index=df.index, name=cluster_labels.name or "cluster")
        # Try align by index intersection
        aligned = cluster_labels.reindex(df.index)
        if aligned.isna().any():
            raise ValueError(
                "cluster_labels index does not align with df.index, and cannot safely reindex."
            )
        return aligned.rename(cluster_labels.name or "cluster")

    arr = np.asarray(cluster_labels).reshape(-1)
    if len(arr) != len(df):
        raise ValueError(
            f"cluster_labels length must match df rows: {len(arr)} vs {len(df)}."
        )
    return pd.Series(arr, index=df.index, name="cluster")


def _resolve_col(df: pd.DataFrame, col: str, aliases: Sequence[str] = ()) -> str:
    """Resolve a column name in a case-insensitive / alias-aware way.

    If `col` does not exist, try:
    - exact match (already checked)
    - lowercase/uppercase/titlecase variants
    - case-insensitive match against df.columns
    - aliases (exact and case-insensitive)
    """
    if col in df.columns:
        return col

    candidates = [col, col.lower(), col.upper(), col.title()]
    candidates.extend(list(aliases))

    # Exact candidate match
    for c in candidates:
        if c in df.columns:
            return c

    # Case-insensitive match
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        key = str(c).lower()
        if key in lower_map:
            return lower_map[key]

    raise KeyError(
        f"Column '{col}' not found. Tried variants/aliases={list(dict.fromkeys(candidates))}. "
        f"Available columns (sample)={list(df.columns[:20])}"
    )


def _resolve_cols(df: pd.DataFrame, cols: Sequence[str], aliases_map: Optional[dict[str, Sequence[str]]] = None) -> List[str]:
    """Resolve multiple columns with an optional aliases_map."""
    out: List[str] = []
    for c in cols:
        aliases = ()
        if aliases_map and c in aliases_map:
            aliases = aliases_map[c]
        out.append(_resolve_col(df, c, aliases=aliases))
    return out


def _maybe_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)


# -----------------------------
# Profiling plots
# -----------------------------

def plot_income_vs_spent(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    hue_name: str = "Cluster",
    *,
    income_col: str = "Income",
    monetary_col: str = "monetary",
    save_path: str | Path | None = None,
    alpha: float = 0.7,
    s: float = 20.0,
) -> plt.Figure:
    """Scatter plot of income vs spending colored by cluster.

    Notes
    -----
    This function supports both:
    - Raw/cleaned data columns: Income / Monetary / Spent
    - Engineered feature columns: income / monetary / spent

    Parameters
    ----------
    df :
        DataFrame containing income and monetary/spent columns.
    cluster_labels :
        Cluster labels aligned (or alignable) with df rows.
    hue_name :
        Legend title for clusters.
    income_col :
        Preferred column name for income (default "Income").
    monetary_col :
        Preferred column name for monetary/spent (default "monetary").
    save_path :
        Optional path to save the figure.
    alpha, s :
        Scatter transparency and marker size.
    """
    labels = _align_labels(df, cluster_labels)

    xcol = _resolve_col(df, income_col, aliases=("income",))
    ycol = _resolve_col(df, monetary_col, aliases=("Monetary", "monetary", "Spent", "spent"))

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot per cluster for a clean legend
    clusters = sorted(pd.unique(labels.dropna()))
    for cid in clusters:
        mask = labels == cid
        ax.scatter(df.loc[mask, xcol], df.loc[mask, ycol], s=s, alpha=alpha, label=str(cid))

    ax.set_title("Income vs Spending by Cluster")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.legend(title=hue_name, fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_rfm_boxplots(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    rfm_cols: Sequence[str] = ("recency", "frequency", "monetary"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Boxplots for RFM dimensions to contrast segments.

    Compared with the original seaborn version, this produces a 1x3 subplot
    layout (Recency/Frequency/Monetary), which is easier to read in the report.

    Column-name robustness:
    - accepts both "Recency"/"Frequency"/"Monetary" and "recency"/"frequency"/"monetary".
    """
    labels = _align_labels(df, cluster_labels)

    aliases_map = {
        "recency": ("Recency",),
        "frequency": ("Frequency",),
        "monetary": ("Monetary", "Spent", "spent"),
    }
    resolved = _resolve_cols(df, list(rfm_cols), aliases_map=aliases_map)

    clusters = sorted(pd.unique(labels.dropna()))

    fig, axes = plt.subplots(1, len(resolved), figsize=(4 * len(resolved), 4), sharey=False)
    if len(resolved) == 1:
        axes = [axes]

    for ax, col in zip(axes, resolved):
        data = [df.loc[labels == cid, col].dropna().to_numpy() for cid in clusters]
        ax.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Value")

    fig.suptitle("RFM distributions by cluster", y=1.02)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_channel_mix(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    channel_cols: Optional[List[str]] = None,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar of average channel usage per cluster (web/catalog/store).

    We plot *proportions* within each cluster (not absolute counts) so the plot
    reflects preference rather than size.
    """
    labels = _align_labels(df, cluster_labels)

    if channel_cols is None:
        channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

    # allow engineered lower-case names too
    aliases_map = {
        "NumWebPurchases": ("numwebpurchases",),
        "NumCatalogPurchases": ("numcatalogpurchases",),
        "NumStorePurchases": ("numstorepurchases",),
    }
    resolved = _resolve_cols(df, channel_cols, aliases_map=aliases_map)

    grouped = df[resolved].assign(cluster=labels).groupby("cluster").mean(numeric_only=True)
    proportions = grouped.div(grouped.sum(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(proportions), dtype=float)

    x = np.arange(len(proportions.index))
    for col in resolved:
        ax.bar(x, proportions[col].to_numpy(), bottom=bottom, label=col)
        bottom = bottom + proportions[col].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in proportions.index])
    ax.set_title("Channel purchase mix by cluster")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cluster")
    ax.legend(title="Channel", fontsize=9)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_response_rates(
    response_rates: pd.Series,
    title: str = "Cluster response rates",
    *,
    global_rate: float | None = None,
    annotate: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of response rates per cluster.

    Parameters
    ----------
    response_rates:
        pd.Series indexed by cluster id.
    global_rate:
        Optional global response rate (draw a dashed horizontal line).
    annotate:
        Whether to annotate each bar with its value.
    """
    rr = response_rates.dropna().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(i) for i in rr.index], rr.values)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Response rate")
    ax.set_title(title)

    if global_rate is not None and np.isfinite(global_rate):
        ax.axhline(float(global_rate), linestyle="--", linewidth=1)
        ax.text(
            0.98,
            float(global_rate),
            f" global={global_rate:.3f}",
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
        )

    if annotate:
        for b, v in zip(bars, rr.values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_cluster_size_and_response(
    cluster_sizes: pd.Series,
    response_rates: pd.Series,
    title: str = "Cluster size and response rate",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cluster size (bar) and response rate (line) together.

    This is particularly useful for the new methodology because:
    - High response rate in a tiny cluster may be less useful than moderate rate in a large cluster.
    """
    sizes = cluster_sizes.dropna().sort_index()
    rates = response_rates.dropna().sort_index()

    # Align index union
    idx = sorted(set(sizes.index).union(set(rates.index)))
    sizes = sizes.reindex(idx).fillna(0).astype(float)
    rates = rates.reindex(idx).astype(float)

    x = np.arange(len(idx))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(x, sizes.values)
    ax1.set_ylabel("Cluster size")
    ax1.set_xlabel("Cluster")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in idx])
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, rates.values, marker="o")
    ax2.set_ylabel("Response rate")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_age_income_kde(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    age_col: str = "Age",
    income_col: str = "Income",
    *,
    n_cols: int = 3,
    gridsize: int = 35,
    mincnt: int = 1,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Density heatmaps for age vs income per cluster.

    Notes
    -----
    The original implementation used seaborn.kdeplot. To avoid depending on seaborn/scipy,
    we use matplotlib.hexbin as a robust density-like visualization.

    Parameters
    ----------
    age_col / income_col:
        Prefer raw names ("Age", "Income"), but engineered ("age", "income") also works.
    n_cols:
        Number of subplot columns; rows are determined automatically.
    gridsize:
        Hexbin resolution.
    """
    labels = _align_labels(df, cluster_labels)

    xcol = _resolve_col(df, age_col, aliases=("age",))
    ycol = _resolve_col(df, income_col, aliases=("income",))

    clusters = sorted(pd.unique(labels.dropna()))
    n_clusters = len(clusters)

    if n_clusters == 0:
        raise ValueError("No clusters found in cluster_labels.")

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, cid in zip(axes, clusters):
        subset = df.loc[labels == cid, [xcol, ycol]].dropna()
        if subset.empty:
            ax.set_title(f"Cluster {cid} (no data)")
            continue

        hb = ax.hexbin(
            subset[xcol].to_numpy(),
            subset[ycol].to_numpy(),
            gridsize=gridsize,
            mincnt=mincnt,
        )
        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        # Add a small colorbar for each subplot (readable for report figures)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    # Hide any unused axes
    for ax in axes[n_clusters:]:
        ax.axis("off")

    fig.suptitle("Age vs Income density by cluster", y=1.02)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig
