"""Ablation study runner for lambda and cluster-count sensitivity."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import List

import pandas as pd

from customer_segmentation.src.data.features import assemble_feature_table
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import clustering as clustering_eval
from customer_segmentation.src.evaluation import segmentation as segmentation_eval
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"


LAMBDA_GRID: List[float] = [0.0, 0.1, 1.0, 10.0]
CLUSTER_GRID: List[int] = [3, 4, 5, 6]


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> tuple[pd.DataFrame, pd.Series]:
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, _ = assemble_feature_table(cleaned)
    return features_df, labels


def _run_single(features: pd.DataFrame, labels: pd.Series, k: int, lambda_: float) -> dict:
    config = RAJCConfig(n_clusters=k, lambda_=lambda_)
    model = RAJCModel(config)
    model.fit(features, labels)
    clusters = model.assignments_

    scores = clustering_eval.compute_scores(features, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, labels)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()
    scores["n_clusters"] = k
    scores["lambda"] = lambda_
    return scores


def main() -> None:
    logger = configure_logging()
    _ensure_output_dirs()

    try:
        features, labels = _prepare_features()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` to verify dataset placement."
        )
        sys.exit(1)
    logger.info(
        "Running ablations over lambda=%s and K=%s", LAMBDA_GRID, CLUSTER_GRID
    )

    ablation_results = []
    for k in CLUSTER_GRID:
        for lambda_ in LAMBDA_GRID:
            logger.info("Fitting RAJC with K=%d, lambda=%.3f", k, lambda_)
            scores = _run_single(features, labels, k, lambda_)
            ablation_results.append(scores)

    results_df = pd.DataFrame(ablation_results)
    results_path = TABLE_DIR / "rajc_ablation.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved ablation results to %s", results_path)


if __name__ == "__main__":
    main()
