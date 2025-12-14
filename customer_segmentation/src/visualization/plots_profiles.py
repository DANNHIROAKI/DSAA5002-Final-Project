"""Cluster profiling and segment-level business visualizations.

These plotting functions are mainly used in the *Profiling* section of the
report, helping answer questions like:
- How do clusters differ in spending, income, age and channel preference?
- Do clusters exhibit clearly different response propensities?

Upgrades for the new method
---------------------------
- Robust column name resolution (raw vs engineered names).
- Robust alignment between a DataFrame and its cluster labels.
- Additional segment-level business plots:
    * cluster lift bars (response_rate / global_rate)
    * cluster-based budget allocation curve (lift vs budget fraction)

All functions use matplotlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_labels(df: pd.DataFrame, cluster_labels: Union[pd.Series, np.ndarray, List[int]]) -> pd.Series:
    """Align cluster labels with df rows.

    Rules
    -----
    - If cluster_labels is a Series with the same index as df: use as-is.
    - If cluster_labels is a Series with different index but same length: align by position.
    - If cluster_labels is array-like with same length: convert to Series with df.index.

    Raises
    ------
    ValueError
        If alignment is not possible.
    """
    if isinstance(cluster_labels, pd.Series):
        if cluster_labels.index.equals(df.index):
            return cluster_labels
        if len(cluster_labels) == len(df):
            return pd.Series(cluster_labels.to_numpy(), index=df.index, name=cluster_labels.name or "cluster")
        aligned = cluster_labels.reindex(df.index)
        if aligned.isna().any():
            raise ValueError("cluster_labels index does not align with df.index, and cannot safely reindex.")
        return aligned.rename(cluster_labels.name or "cluster")

    arr = np.asarray(cluster_labels).reshape(-1)
    if len(arr) != len(df):
        raise ValueError(f"cluster_labels length must match df rows: {len(arr)} vs {len(df)}.")
    return pd.Series(arr, index=df.index, name="cluster")


def _resolve_col(df: pd.DataFrame, col: str, aliases: Sequence[str] = ()) -> str:
    """Resolve a column name in a case-insensitive / alias-aware way."""
    if col in df.columns:
        return col

    candidates = [col, col.lower(), col.upper(), col.title()]
    candidates.extend(list(aliases))

    for c in candidates:
        if c in df.columns:
            return c

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


def _align_pair(y_true: Union[pd.Series, np.ndarray, List[int]],
                cluster_labels: Union[pd.Series, np.ndarray, List[int]]) -> tuple[pd.Series, pd.Series]:
    """Align y_true and cluster_labels by index when possible; otherwise by position."""
    y = pd.Series(y_true, name="y") if not isinstance(y_true, pd.Series) else y_true.rename("y")
    z = pd.Series(cluster_labels, name="cluster") if not isinstance(cluster_labels, pd.Series) else cluster_labels.rename("cluster")

    # If both have index, align on index via concat
    if isinstance(y_true, pd.Series) and isinstance(cluster_labels, pd.Series):
        df = pd.concat([y, z], axis=1).dropna()
        return df["y"], df["cluster"]

    # Fallback: align by position
    if len(y) != len(z):
        raise ValueError(f"y_true and cluster_labels length mismatch: {len(y)} vs {len(z)}")
    df = pd.DataFrame({"y": y.to_numpy(), "cluster": z.to_numpy()}).dropna()
    return df["y"], df["cluster"]


# ---------------------------------------------------------------------------
# Profiling plots
# ---------------------------------------------------------------------------


def plot_income_vs_spent(
    df: pd.DataFrame,
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    hue_name: str = "Cluster",
    *,
    income_col: str = "Income",
    monetary_col: str = "monetary",
    save_path: str | Path | None = None,
    alpha: float = 0.7,
    s: float = 20.0,
) -> plt.Figure:
    """Scatter plot of income vs spending coloured by cluster."""
    labels = _align_labels(df, cluster_labels)

    xcol = _resolve_col(df, income_col, aliases=("income",))
    ycol = _resolve_col(df, monetary_col, aliases=("Monetary", "monetary", "Spent", "spent"))

    fig, ax = plt.subplots(figsize=(6, 4))

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
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    *,
    rfm_cols: Sequence[str] = ("recency", "frequency", "monetary"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Boxplots for R/F/M distributions by cluster."""
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
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    channel_cols: Optional[List[str]] = None,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar plot of average channel usage proportions per cluster."""
    labels = _align_labels(df, cluster_labels)

    if channel_cols is None:
        channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

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
    """Bar chart of response rates per cluster."""
    rr = response_rates.dropna().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(i) for i in rr.index], rr.values)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Response rate")
    ax.set_title(title)

    if global_rate is not None and np.isfinite(global_rate):
        ax.axhline(float(global_rate), linestyle="--", linewidth=1)

    if annotate:
        for b, v in zip(bars, rr.values):
            ax.text(
                b.get_x() + b.get_width() / 2,
                float(v),
                f"{float(v):.3f}",
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
    """Plot cluster size (bar) and response rate (line) together."""
    sizes = cluster_sizes.dropna().sort_index()
    rates = response_rates.dropna().sort_index()

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
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    age_col: str = "Age",
    income_col: str = "Income",
    *,
    n_cols: int = 3,
    gridsize: int = 35,
    mincnt: int = 1,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Density-like plots for age vs income per cluster using hexbin."""
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
            gridsize=int(gridsize),
            mincnt=int(mincnt),
        )
        ax.set_title(f"Cluster {cid}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    for ax in axes[n_clusters:]:
        ax.axis("off")

    fig.suptitle("Age vs Income density by cluster", y=1.02)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# New segment-level business plots
# ---------------------------------------------------------------------------


def plot_cluster_lift_bars(
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    y_true: Union[pd.Series, np.ndarray, List[int]],
    title: str = "Cluster lift vs global",
    *,
    annotate: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of cluster lift (response_rate / global_rate).

    This is a compact way to show whether the segmentation is *actionable*:
    clusters should have clearly different lift values.
    """
    y, z = _align_pair(y_true, cluster_labels)

    y_bin = (pd.to_numeric(y, errors="coerce") > 0).astype(int)
    z = z.astype(int)

    df = pd.DataFrame({"y": y_bin, "cluster": z}).dropna()
    if df.empty:
        raise ValueError("Empty inputs after alignment / NaN dropping.")

    global_rate = float(df["y"].mean())
    rates = df.groupby("cluster")["y"].mean().sort_index()

    if global_rate <= 0:
        lifts = rates * np.nan
    else:
        lifts = rates / global_rate

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(i) for i in lifts.index], lifts.values)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Lift")
    ax.set_title(title)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Global (lift=1)")

    if annotate:
        for b, v in zip(bars, lifts.values):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    float(v),
                    f"{float(v):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_cluster_budget_allocation_curve(
    cluster_labels: Union[pd.Series, np.ndarray, List[int]],
    y_true: Union[pd.Series, np.ndarray, List[int]],
    title: str = "Cluster-based budget allocation (expected lift)",
    *,
    fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Expected lift curve of a *cluster-level* targeting policy.

    Policy
    ------
    1) Estimate response rate for each cluster.
    2) Sort clusters by rate (descending).
    3) Spend budget by targeting customers from the best clusters first.

    When the last selected cluster would exceed the budget, we assume we
    sample uniformly within that cluster (i.e., partial inclusion).

    This curve complements the *ranking-based* lift curve:
    - Ranking lift reflects how well the model orders *individuals*.
    - Cluster allocation lift reflects how useful the *segmentation* is for
      coarse-grained campaign planning.
    """
    y, z = _align_pair(y_true, cluster_labels)

    y_bin = (pd.to_numeric(y, errors="coerce") > 0).astype(int)
    z = z.astype(int)

    df = pd.DataFrame({"y": y_bin, "cluster": z}).dropna()
    if df.empty:
        raise ValueError("Empty inputs after alignment / NaN dropping.")

    global_rate = float(df["y"].mean())
    n = int(len(df))

    # cluster stats
    stats = (
        df.groupby("cluster")["y"]
        .agg([("size", "count"), ("rate", "mean")])
        .sort_values("rate", ascending=False)
    )

    fracs: list[float] = []
    lifts: list[float] = []

    for f in fractions:
        f = float(f)
        if not (0.0 < f <= 1.0):
            continue
        budget = int(np.ceil(n * f))
        budget = max(1, min(budget, n))

        remaining = budget
        expected_pos = 0.0

        for _, row in stats.iterrows():
            if remaining <= 0:
                break
            size_c = int(row["size"])
            rate_c = float(row["rate"])

            take = min(remaining, size_c)
            expected_pos += float(take) * rate_c
            remaining -= take

        expected_precision = expected_pos / float(budget)
        lift = float("nan") if global_rate <= 0 else expected_precision / global_rate
        fracs.append(f)
        lifts.append(lift)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, lifts, marker="o", label="Cluster allocation")
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Random (lift=1)")

    ax.set_xlabel("Budget fraction (top-q)")
    ax.set_ylabel("Expected lift")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


__all__ = [
    # profiling
    "plot_income_vs_spent",
    "plot_rfm_boxplots",
    "plot_channel_mix",
    "plot_response_rates",
    "plot_cluster_size_and_response",
    "plot_age_income_kde",
    # business/segmentation
    "plot_cluster_lift_bars",
    "plot_cluster_budget_allocation_curve",
]
