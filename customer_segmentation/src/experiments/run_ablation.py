"""Ablation study runner for lambda/gamma and cluster-count sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Optional, Any, Dict

import pandas as pd

from customer_segmentation.src.data.features import (
    assemble_feature_table,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import clustering as clustering_eval
from customer_segmentation.src.evaluation import segmentation as segmentation_eval
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"

# Default grids; can be overridden via CLI.
DEFAULT_LAMBDA_GRID: List[float] = [0.0, 0.1, 0.3, 1.0, 3.0]
DEFAULT_GAMMA_GRID: List[float] = [0.0, 0.1, 1.0]
DEFAULT_CLUSTER_GRID: List[int] = [3, 4, 5, 6]


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean and featurize the dataset.

    For ablation, we follow the main RAJC experiment and use only
    **behavioural** features as the clustering space.
    """
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, transformer = assemble_feature_table(cleaned)
    behavior_features, _ = split_behavior_and_response_features(
        features_df, transformer
    )
    return behavior_features, labels


def _run_single(
    features: pd.DataFrame,
    labels: pd.Series,
    k: int,
    lambda_: float,
    gamma: float,
) -> Dict[str, Any]:
    """Fit a RAJC model for given (K, λ, γ) and compute clustering/segmentation metrics."""
    config = RAJCConfig(
        n_clusters=k,
        lambda_=lambda_,
        gamma=gamma,
    )
    model = RAJCModel(config)
    model.fit(features, labels)
    clusters = model.assignments_

    scores = clustering_eval.compute_scores(features, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, labels)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()
    scores["n_clusters"] = k
    scores["lambda"] = lambda_
    scores["gamma"] = gamma
    scores["model_type"] = getattr(model.config, "model_type", "constant_prob")
    scores["label"] = "rajc_ablation"

    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study over lambda, gamma and number of clusters for RAJC."
    )
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="*",
        default=DEFAULT_LAMBDA_GRID,
        help=f"Grid of lambda values (default: {DEFAULT_LAMBDA_GRID}).",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=DEFAULT_GAMMA_GRID,
        help=f"Grid of gamma values (default: {DEFAULT_GAMMA_GRID}).",
    )
    parser.add_argument(
        "--cluster-grid",
        type=int,
        nargs="*",
        default=DEFAULT_CLUSTER_GRID,
        help=f"Grid of cluster counts K (default: {DEFAULT_CLUSTER_GRID}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    try:
        features, labels = _prepare_features()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` "
            "to verify dataset placement."
        )
        sys.exit(1)

    logger.info(
        "Running RAJC ablations over lambda=%s, gamma=%s and K=%s",
        args.lambda_grid,
        args.gamma_grid,
        args.cluster_grid,
    )

    ablation_results: list[Dict[str, Any]] = []

    for k in args.cluster_grid:
        for lambda_ in args.lambda_grid:
            for gamma in args.gamma_grid:
                logger.info(
                    "Fitting RAJC with K=%d, lambda=%.3f, gamma=%.3f",
                    k,
                    lambda_,
                    gamma,
                )
                try:
                    scores = _run_single(features, labels, k, lambda_, gamma)
                    ablation_results.append(scores)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception(
                        "Ablation run failed for K=%d, lambda=%.3f, gamma=%.3f: %s",
                        k,
                        lambda_,
                        gamma,
                        exc,
                    )

    results_df = pd.DataFrame(ablation_results)
    results_path = TABLE_DIR / "rajc_ablation.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved ablation results to %s", results_path)


if __name__ == "__main__":
    main()
