"""Business-focused segmentation metrics."""
import pandas as pd


def cluster_response_rates(cluster_labels: pd.Series, responses: pd.Series) -> pd.Series:
    """Compute promotion response rate per cluster."""
    return responses.groupby(cluster_labels).mean()


def response_rate_variance(rates: pd.Series) -> float:
    """Variance of response rates across clusters."""
    return rates.var(ddof=0)
