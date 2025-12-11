"""Train and evaluate the Response-Aware Joint Clustering (RAJC) model.

This script trains a RAJC model (by default the RAJC-v2 "constant_prob"
variant) on the marketing campaign dataset and evaluates:

- clustering quality (Silhouette / CH / DB),
- promotion-response segmentation (cluster-wise response rates),
- per-customer response prediction quality (AUC / F1 / etc.),
- lift@20% based on predicted response probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Any, Sequence, Optional

import pandas as pd
import yaml

from customer_segmentation.src.data.features import (
    assemble_feature_table,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import clustering as clustering_eval
from customer_segmentation.src.evaluation import prediction as prediction_eval
from customer_segmentation.src.evaluation import segmentation as segmentation_eval
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging
from customer_segmentation.src.utils.metrics_utils import compute_lift

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"

DEFAULT_CONFIG_PATH = Path("customer_segmentation/configs/rajc.yaml")


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series]:
    """Load, clean and featurize the dataset.

    RAJC is intended to cluster primarily on **behavioural** features, so we
    extract the behaviour subset from the engineered feature table and use it
    throughout this script.
    """
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, transformer = assemble_feature_table(cleaned)
    behavior_features, _ = split_behavior_and_response_features(
        features_df, transformer
    )
    return behavior_features, labels


def _evaluate(
    model: RAJCModel,
    features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    """Compute clustering, segmentation and prediction metrics for a fitted RAJC model."""
    clusters = model.assignments_
    if clusters is None:
        raise ValueError("RAJC model has not been fitted or assignments_ is None.")

    # Clustering + segmentation metrics
    scores = clustering_eval.compute_scores(features, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, labels)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    # Basic config info for traceability
    scores["n_clusters"] = model.config.n_clusters
    scores["lambda"] = model.config.lambda_
    scores["gamma"] = getattr(model.config, "gamma", 0.0)
    scores["model_type"] = getattr(model.config, "model_type", "constant_prob")

    # Promotion-response prediction metrics (using RAJC's cluster-wise model)
    probas = model.predict_response(features)
    preds = (probas >= 0.5).astype(int)
    scores.update(
        prediction_eval.compute_classification_metrics(labels, preds, probas)
    )
    scores["lift_top20"] = compute_lift(labels, probas, top_frac=0.2)
    scores["label"] = "rajc"

    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the RAJC model on the marketing dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file for RAJC (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    # Load configuration
    try:
        config_dict = yaml.safe_load(args.config.read_text()) or {}
    except FileNotFoundError:
        logger.error("RAJC config file not found at %s", args.config)
        sys.exit(1)

    rajc_cfg = config_dict.get("rajc", {})

    config = RAJCConfig(
        n_clusters=rajc_cfg.get("n_clusters", 4),
        lambda_=rajc_cfg.get("lambda", 1.0),
        gamma=rajc_cfg.get("gamma", 0.0),
        max_iter=rajc_cfg.get("max_iter", 20),
        tol=float(rajc_cfg.get("tol", 1e-4)),
        random_state=rajc_cfg.get("random_state", 42),
        smoothing=rajc_cfg.get("smoothing", 1.0),
        logreg_max_iter=rajc_cfg.get("logistic_regression", {}).get(
            "max_iter", 200
        ),
        kmeans_n_init=rajc_cfg.get("kmeans_n_init", 10),
        model_type=rajc_cfg.get("model_type", "constant_prob"),
    )

    # Prepare data
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
        "Fitting RAJC on %d samples with config: %s",
        len(features),
        config,
    )

    # Fit RAJC
    model = RAJCModel(config)
    model.fit(features, labels)

    # Evaluate
    metrics = _evaluate(model, features, labels)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = TABLE_DIR / "rajc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved RAJC metrics to %s", metrics_path)

    assignments_path = TABLE_DIR / "rajc_assignments.csv"
    assignments_series = pd.Series(
        model.assignments_, index=features.index, name="cluster"
    )
    assignments_series.to_csv(assignments_path, header=True)
    logger.info("Saved RAJC assignments to %s", assignments_path)


if __name__ == "__main__":
    main()
