"""Run baseline clustering pipelines and log metrics.

This script executes the baseline methods described in the methodology:
- RFM K-Means (Baseline 1)
- Full-feature K-Means (Baseline 2)
- Gaussian Mixture Model (Baseline 3)
- Cluster-then-Predict (Baseline 4)

Outputs are written under ``customer_segmentation/outputs`` for later reporting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from customer_segmentation.src.data.features import assemble_feature_table
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


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series]:
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, _ = assemble_feature_table(cleaned)
    return features_df, labels


def _select_rfm_features(features: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["recency", "frequency", "monetary"] if col in features.columns]
    if not cols:
        raise ValueError("RFM columns are missing from engineered features.")
    return features[cols]


def _evaluate_clusters(
    name: str, feature_matrix: pd.DataFrame, cluster_labels: pd.Series, responses: pd.Series
) -> Dict:
    scores = clustering_eval.compute_scores(feature_matrix, cluster_labels)
    rates = segmentation_eval.cluster_response_rates(cluster_labels, responses)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()
    scores["label"] = name
    return scores


def _run_rfm_kmeans(config: dict, features: pd.DataFrame, labels: pd.Series) -> Dict:
    subset = _select_rfm_features(features)
    model = KMeansBaseline(
        KMeansConfig(
            n_clusters=config.get("n_clusters", 4),
            init=config.get("init", "k-means++"),
            max_iter=config.get("max_iter", 300),
            random_state=config.get("random_state", 42),
        )
    )
    model.fit(subset)
    assignments = model.predict(subset)
    return _evaluate_clusters("rfm_kmeans", subset, assignments, labels)


def _run_full_kmeans(config: dict, features: pd.DataFrame, labels: pd.Series) -> Dict:
    model = KMeansBaseline(
        KMeansConfig(
            n_clusters=config.get("n_clusters", 4),
            init=config.get("init", "k-means++"),
            max_iter=config.get("max_iter", 300),
            random_state=config.get("random_state", 42),
        )
    )
    model.fit(features)
    assignments = model.predict(features)
    return _evaluate_clusters("full_kmeans", features, assignments, labels)


def _run_gmm(config: dict, features: pd.DataFrame, labels: pd.Series) -> Dict:
    model = GMMBaseline(
        GMMConfig(
            n_components=config.get("n_components", 4),
            covariance_type=config.get("covariance_type", "full"),
            max_iter=config.get("max_iter", 200),
            random_state=config.get("random_state", 42),
        )
    )
    model.fit(features)
    assignments = model.predict(features)
    return _evaluate_clusters("gmm", features, assignments, labels)


def _run_cluster_then_predict(config: dict, features: pd.DataFrame, labels: pd.Series) -> Dict:
    ctp_config = ClusterThenPredictConfig(
        n_clusters=config.get("n_clusters", 4),
        random_state=config.get("random_state", 42),
        max_iter=config.get("classifier_params", {}).get("max_iter", 200),
    )
    kmeans, classifiers, assignments = fit_cluster_then_predict(features, labels, ctp_config)
    probas, preds = predict_with_clusters(assignments, features, classifiers)

    metrics = prediction_eval.compute_classification_metrics(labels, preds, probas)
    metrics.update(
        _evaluate_clusters("cluster_then_predict", features, assignments, labels)
    )
    metrics["inertia"] = kmeans.inertia()
    metrics["lift_top20"] = compute_lift(labels, probas, top_frac=0.2)
    return metrics


def main() -> None:
    logger = configure_logging()
    _ensure_output_dirs()

    configs = yaml.safe_load(Path("customer_segmentation/configs/baselines.yaml").read_text())
    features, labels = _prepare_features()

    logger.info("Running baseline methods on %d samples", len(features))

    results = []
    results.append(_run_rfm_kmeans(configs.get("rfm_kmeans", {}), features, labels))
    results.append(_run_full_kmeans(configs.get("full_kmeans", {}), features, labels))
    results.append(_run_gmm(configs.get("gmm", {}), features, labels))
    results.append(
        _run_cluster_then_predict(configs.get("cluster_then_predict", {}), features, labels)
    )

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "baseline_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved baseline metrics to %s", results_path)


if __name__ == "__main__":
    main()
