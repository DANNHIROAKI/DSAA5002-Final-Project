"""Run baseline clustering pipelines and log metrics.

This script executes the baseline methods described in the methodology:
- RFM K-Means (Baseline 1)
- Full-feature K-Means (Baseline 2)
- Gaussian Mixture Model (Baseline 3)
- Cluster-then-Predict (Baseline 4)

Outputs are written under ``customer_segmentation/outputs`` for later reporting.
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
from customer_segmentation.src.models.cluster_then_predict import (
    ClusterThenPredictConfig,
    fit_cluster_then_predict,
    predict_with_clusters,
)
from customer_segmentation.src.models.gmm_baseline import GMMBaseline, GMMConfig
from customer_segmentation.src.models.kmeans_baseline import KMeansBaseline, KMeansConfig
from customer_segmentation.src.utils.logging_utils import configure_logging
from customer_segmentation.src.utils.metrics_utils import compute_lift

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

DEFAULT_CONFIG_PATH = Path("customer_segmentation/configs/baselines.yaml")


def _ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series, object]:
    """Load raw data, clean it, and build the engineered feature matrix.

    Returns
    -------
    features_df :
        Full engineered feature table (behaviour + response-related).
    labels :
        Binary promotion-response label.
    transformer :
        Fitted ColumnTransformer with attached feature-group metadata, which
        can be used to derive behaviour-only features for clustering.
    """
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, transformer = assemble_feature_table(cleaned)
    return features_df, labels, transformer


def _select_rfm_features(features: pd.DataFrame) -> pd.DataFrame:
    """Select the RFM subset of engineered features for Baseline 1."""
    cols = [col for col in ["recency", "frequency", "monetary"] if col in features.columns]
    if not cols:
        raise ValueError("RFM columns are missing from engineered features.")
    return features[cols]


def _evaluate_clusters(
    name: str,
    feature_matrix: pd.DataFrame,
    cluster_labels: pd.Series,
    responses: pd.Series,
) -> Dict[str, Any]:
    """Compute clustering and segmentation metrics for a given partition."""
    scores = clustering_eval.compute_scores(feature_matrix, cluster_labels)
    rates = segmentation_eval.cluster_response_rates(cluster_labels, responses)

    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()
    scores["label"] = name
    return scores


def _run_rfm_kmeans(
    config: dict,
    full_features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    """Run Baseline 1: RFM-only K-Means."""
    subset = _select_rfm_features(full_features)
    km_config = KMeansConfig(
        n_clusters=config.get("n_clusters", 4),
        init=config.get("init", "k-means++"),
        max_iter=config.get("max_iter", 300),
        random_state=config.get("random_state", 42),
        n_init=config.get("n_init", 10),
    )
    model = KMeansBaseline(km_config)
    model.fit(subset)
    assignments = model.predict(subset)
    return _evaluate_clusters("rfm_kmeans", subset, assignments, labels)


def _run_full_kmeans(
    config: dict,
    behavior_features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    """Run Baseline 2: full behavioural-feature K-Means."""
    km_config = KMeansConfig(
        n_clusters=config.get("n_clusters", 4),
        init=config.get("init", "k-means++"),
        max_iter=config.get("max_iter", 300),
        random_state=config.get("random_state", 42),
        n_init=config.get("n_init", 10),
    )
    model = KMeansBaseline(km_config)
    model.fit(behavior_features)
    assignments = model.predict(behavior_features)
    return _evaluate_clusters("full_kmeans", behavior_features, assignments, labels)


def _run_gmm(
    config: dict,
    behavior_features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    """Run Baseline 3: Gaussian Mixture Model on behavioral feature set."""
    gmm_config = GMMConfig(
        n_components=config.get("n_clusters", 4),
        covariance_type=config.get("covariance_type", "full"),
        max_iter=config.get("max_iter", 200),
        random_state=config.get("random_state", 42),
    )
    model = GMMBaseline(gmm_config)
    model.fit(behavior_features)
    assignments = model.predict(behavior_features)
    return _evaluate_clusters("gmm", behavior_features, assignments, labels)


def _run_cluster_then_predict(
    config: dict,
    full_features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, Any]:
    """Run Baseline 4: cluster-then-predict pipeline.

    Here we let both clustering and per-cluster classifiers see the full
    engineered feature set for a strong two-stage baseline.
    """
    ctp_config = ClusterThenPredictConfig(
        n_clusters=config.get("n_clusters", 4),
        random_state=config.get("random_state", 42),
        max_iter=config.get("max_iter", 200),
        kmeans_n_init=config.get("kmeans_n_init", 10),
    )
    kmeans, classifiers, assignments = fit_cluster_then_predict(
        full_features, labels, ctp_config
    )
    probas, preds = predict_with_clusters(assignments, full_features, classifiers)

    # Classification metrics
    clf_metrics = prediction_eval.compute_classification_metrics(labels, preds, probas)
    # Clustering + segmentation metrics
    cluster_metrics = _evaluate_clusters(
        "cluster_then_predict", full_features, assignments, labels
    )

    # Merge dictionaries (cluster metrics already contain "label").
    cluster_metrics.update(clf_metrics)
    cluster_metrics["inertia"] = kmeans.inertia()
    cluster_metrics["lift_top20"] = compute_lift(labels, probas, top_frac=0.2)
    return cluster_metrics


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for baseline experiments."""
    parser = argparse.ArgumentParser(
        description="Run baseline clustering methods for DSAA5002 final project."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config for baselines (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Skip the GMM baseline (useful when runtime is a concern).",
    )
    parser.add_argument(
        "--skip-ctp",
        action="store_true",
        help="Skip the Cluster-then-Predict baseline.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    # Read config
    try:
        configs = yaml.safe_load(args.config.read_text()) or {}
    except FileNotFoundError:
        logger.error("Baseline config file not found at %s", args.config)
        sys.exit(1)

    # Prepare features
    try:
        features_full, labels, transformer = _prepare_features()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` "
            "to verify dataset placement."
        )
        sys.exit(1)

    # Behaviour-only subset for clustering-oriented baselines
    behavior_features, _ = split_behavior_and_response_features(
        features_full, transformer
    )

    logger.info(
        "Running baseline methods on %d samples "
        "(behavioural feature dim = %d, full feature dim = %d)",
        features_full.shape[0],
        behavior_features.shape[1],
        features_full.shape[1],
    )

    results: list[Dict[str, Any]] = []

    # Baseline 1: RFM K-Means
    logger.info("Running RFM K-Means baseline")
    results.append(
        _run_rfm_kmeans(configs.get("rfm_kmeans", {}), features_full, labels)
    )

    # Baseline 2: Full-feature (behavioural) K-Means
    logger.info("Running Full-feature K-Means baseline (behavioural features)")
    results.append(
        _run_full_kmeans(configs.get("full_kmeans", {}), behavior_features, labels)
    )

    # Baseline 3: GMM
    if not args.skip_gmm:
        logger.info("Running GMM baseline")
        try:
            results.append(
                _run_gmm(configs.get("gmm", {}), behavior_features, labels)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("GMM baseline failed: %s", exc)

    # Baseline 4: Cluster-then-Predict
    if not args.skip_ctp:
        logger.info("Running Cluster-then-Predict baseline")
        try:
            results.append(
                _run_cluster_then_predict(
                    configs.get("cluster_then_predict", {}), features_full, labels
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Cluster-then-Predict baseline failed: %s", exc)

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "baseline_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved baseline metrics to %s", results_path)


if __name__ == "__main__":
    main()
