"""Business-focused segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def cluster_response_rates(
    cluster_labels: pd.Series,
    responses: pd.Series,
) -> pd.Series:
    """Compute promotion response rate per cluster.

    Parameters
    ----------
    cluster_labels :
        Cluster assignments for each customer.
    responses :
        Binary promotion-response label for each customer.

    Returns
    -------
    pd.Series
        Index is cluster id, value is mean response rate in that cluster.
    """
    # Align indices and drop rows with missing labels.
    df = pd.concat(
        {"cluster": cluster_labels, "response": responses}, axis=1
    ).dropna(subset=["cluster", "response"])

    # Ensure integer / categorical cluster labels behave well.
    rates = df.groupby("cluster")["response"].mean().sort_index()
    return rates


def response_rate_variance(rates: pd.Series) -> float:
    """Variance of response rates across clusters (population variance)."""
    if rates.empty:
        return float("nan")
    return float(rates.var(ddof=0))


def cluster_response_summary(
    cluster_labels: pd.Series,
    responses: pd.Series,
) -> pd.DataFrame:
    """Return a per-cluster summary table for business analysis.

    Columns
    -------
    cluster :
        Cluster id.
    size :
        Number of customers in the cluster.
    fraction :
        Proportion of all customers in the cluster.
    responders :
        Number of responding customers.
    response_rate :
        Fraction of responders in the cluster.
    """
    df = pd.concat(
        {"cluster": cluster_labels, "response": responses}, axis=1
    ).dropna(subset=["cluster", "response"])

    group = df.groupby("cluster")
    size = group.size()
    responders = group["response"].sum()
    rates = responders / size

    summary = pd.DataFrame(
        {
            "cluster": size.index,
            "size": size.values,
            "fraction": size.values / size.values.sum(),
            "responders": responders.values,
            "response_rate": rates.values,
        }
    )
    return summary.sort_values("cluster").reset_index(drop=True)


@dataclass
class AllocationResult:
    """Result of a simple cluster-level campaign allocation simulation."""

    budget_ratio: float
    selected_fraction: float
    expected_positives: int
    overall_response_rate: float
    baseline_response_rate: float
    lift: float


def campaign_allocation_lift(
    cluster_labels: pd.Series,
    responses: pd.Series,
    budget_ratio: float = 0.2,
) -> AllocationResult:
    """Simulate cluster-level targeting under a fixed user-budget constraint.

    We assume the marketer can contact at most ``budget_ratio`` fraction of all
    customers. Customers are selected cluster-by-cluster in descending order of
    cluster response rate, until the budget is exhausted.

    Parameters
    ----------
    cluster_labels :
        Cluster assignments for each customer.
    responses :
        Binary promotion-response label for each customer.
    budget_ratio :
        Fraction of customers that can be targeted (e.g., 0.2 = 20%).

    Returns
    -------
    AllocationResult
        Summary of expected positives and lift versus random targeting.
    """
    if not (0 < budget_ratio <= 1.0):
        raise ValueError("budget_ratio must be in (0, 1].")

    df = pd.concat(
        {"cluster": cluster_labels, "response": responses}, axis=1
    ).dropna(subset=["cluster", "response"])

    n_total = len(df)
    if n_total == 0:
        return AllocationResult(
            budget_ratio=budget_ratio,
            selected_fraction=0.0,
            expected_positives=0,
            overall_response_rate=float("nan"),
            baseline_response_rate=float("nan"),
            lift=float("nan"),
        )

    # Per-cluster stats
    group = df.groupby("cluster")
    size = group.size()
    responders = group["response"].sum()
    rates = responders / size

    # Order clusters by descending response rate
    ordered_clusters = rates.sort_values(ascending=False).index

    budget = int(np.ceil(budget_ratio * n_total))
    selected_mask = np.zeros(n_total, dtype=bool)

    # Greedy selection of clusters until budget is reached
    for c in ordered_clusters:
        idx = df["cluster"] == c
        idx_positions = np.where(idx.to_numpy())[0]

        if selected_mask.sum() + idx_positions.size <= budget:
            selected_mask[idx_positions] = True
        else:
            # If taking the whole cluster would exceed budget, stop.
            # (Alternative: take a random subset of this cluster.)
            break

    selected_df = df[selected_mask]
    selected_fraction = selected_df.shape[0] / n_total

    baseline_response_rate = df["response"].mean()
    overall_response_rate = (
        selected_df["response"].mean() if selected_df.shape[0] > 0 else float("nan")
    )
    expected_positives = int(selected_df["response"].sum())

    if np.isnan(overall_response_rate) or np.isnan(baseline_response_rate) or baseline_response_rate == 0:
        lift = float("nan")
    else:
        lift = overall_response_rate / baseline_response_rate

    return AllocationResult(
        budget_ratio=budget_ratio,
        selected_fraction=selected_fraction,
        expected_positives=expected_positives,
        overall_response_rate=float(overall_response_rate),
        baseline_response_rate=float(baseline_response_rate),
        lift=float(lift),
    )
