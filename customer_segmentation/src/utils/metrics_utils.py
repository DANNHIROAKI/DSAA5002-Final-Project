"""Utility metrics such as lift calculation."""
import pandas as pd


def compute_lift(responses: pd.Series, scores: pd.Series, top_frac: float = 0.2) -> float:
    """Compute lift by comparing top-ranked response rate to overall mean."""
    cutoff = int(len(scores) * top_frac)
    ranked = scores.sort_values(ascending=False)
    top_idx = ranked.head(max(cutoff, 1)).index
    top_rate = responses.loc[top_idx].mean()
    overall_rate = responses.mean()
    return top_rate / overall_rate if overall_rate > 0 else 0.0
