"""Run baseline clustering/prediction pipelines under a strict hold-out protocol.

Baselines implemented (aligned with the methodology):
- B1: RFM K-Means
- B2: Behavior K-Means
- B3: GMM on behavior features
- B4: Cluster-then-Predict (behavior clustering -> per-cluster supervised predictor)

Key upgrades vs the previous script:
- Uses y = Response (label_mode="recent").
- Uses train/val/test split with leakage-free preprocessing:
  fit transformer on train only, then transform val/test.
- Cluster-then-Predict is made fair: clustering uses behavior features,
  but prediction uses full features.
- Uses validation to choose a decision threshold (max F1) for supervised baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Any, Sequence, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from customer_segmentation.src.data.features import (
    add_response_label,
    assemble_feature_table,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data, train_val_test_split
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


def _choose_threshold_max_f1(y_true: pd.Series, probas: pd.Series, grid_size: int = 101) -> float:
    """Choose a probability threshold on validation set to maximize F1."""
    y = y_true.astype(int).to_numpy()
    p = probas.to_numpy()
    best_t = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.0, 1.0, grid_size):
        preds = (p >= t).astype(int)
        score = f1_score(y, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t


def _load_and_split_data(
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw dataset, clean it, create y=Response label, then split."""
    raw = load_raw_data()
    cleaned = clean_data(raw)

    # Add y = Response (recent label) for stratified splitting.
    cleaned = add_response_label(cleaned, mode="recent")
    train_df, val_df, test_df = train_val_test_split(
        cleaned,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify_col="campaign_response",
    )
    return train_df, val_df, test_df


def _build_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    object
]:
    """Fit feature transformer on train; transform val/test; split behavior subset."""
    X_train_full, y_train, transformer = assemble_feature_table(
        train_df, fit=True, label_mode="recent"
    )
    X_val_full, y_val, _ = assemble_feature_table(
        val_df, transformer=transformer, fit=False, label_mode="recent"
    )
    X_test_full, y_test, _ = assemble_feature_table(
        test_df, transformer=transformer, fit=False, label_mode="recent"
    )

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_val_beh, _ = split_behavior_and_response_features(X_val_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)

    return (
        X_train_full, y_train,
        X_val_full, y_val,
        X_test_full, y_test,
        X_train_beh, X_val_beh, X_test_beh,
        transformer,
    )


def _select_rfm_features(features: pd.DataFrame) -> pd.DataFrame:
    """Select the RFM subset (recency/frequency/monetary) from engineered features."""
    cols = [c for c in ["recency", "frequency", "monetary"] if c in features.columns]
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
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Baseline 1: RFM-only K-Means (fit on train, evaluate on test)."""
    train_rfm = _select_rfm_features(X_train_full)
    test_rfm = _select_rfm_features(X_test_full)

    km_config = KMeansConfig(
        n_clusters=config.get("n_clusters", 4),
        init=config.get("init", "k-means++"),
        max_iter=config.get("max_iter", 300),
        random_state=config.get("random_state", 42),
        n_init=config.get("n_init", 10),
    )
    model = KMeansBaseline(km_config)
    model.fit(train_rfm)
    assignments_test = model.predict(test_rfm)

    metrics = _evaluate_clusters("rfm_kmeans", test_rfm, assignments_test, y_test)
    metrics["inertia"] = model.inertia()
    metrics["n_clusters"] = km_config.n_clusters
    return metrics


def _run_full_kmeans(
    config: dict,
    X_train_beh: pd.DataFrame,
    X_test_beh: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Baseline 2: behavior-feature K-Means (fit on train, evaluate on test)."""
    km_config = KMeansConfig(
        n_clusters=config.get("n_clusters", 4),
        init=config.get("init", "k-means++"),
        max_iter=config.get("max_iter", 300),
        random_state=config.get("random_state", 42),
        n_init=config.get("n_init", 10),
    )
    model = KMeansBaseline(km_config)
    model.fit(X_train_beh)
    assignments_test = model.predict(X_test_beh)

    metrics = _evaluate_clusters("full_kmeans", X_test_beh, assignments_test, y_test)
    metrics["inertia"] = model.inertia()
    metrics["n_clusters"] = km_config.n_clusters
    return metrics


def _run_gmm(
    config: dict,
    X_train_beh: pd.DataFrame,
    X_test_beh: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Baseline 3: GMM on behavior features (fit on train, evaluate on test)."""
    gmm_config = GMMConfig(
        n_components=config.get("n_clusters", 4),
        covariance_type=config.get("covariance_type", "full"),
        max_iter=config.get("max_iter", 200),
        random_state=config.get("random_state", 42),
    )
    model = GMMBaseline(gmm_config)
    model.fit(X_train_beh)
    assignments_test = model.predict(X_test_beh)

    metrics = _evaluate_clusters("gmm", X_test_beh, assignments_test, y_test)
    metrics["n_clusters"] = gmm_config.n_components
    return metrics


def _run_cluster_then_predict(
    config: dict,
    X_train_full: pd.DataFrame,
    X_train_beh: pd.DataFrame,
    y_train: pd.Series,
    X_val_full: pd.DataFrame,
    X_val_beh: pd.DataFrame,
    y_val: pd.Series,
    X_test_full: pd.DataFrame,
    X_test_beh: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Baseline 4: Cluster-then-Predict (behavior clustering -> full-feature prediction)."""
    ctp_config = ClusterThenPredictConfig(
        n_clusters=config.get("n_clusters", 4),
        random_state=config.get("random_state", 42),
        max_iter=config.get("max_iter", 200),
        kmeans_n_init=config.get("kmeans_n_init", 10),
    )

    kmeans, classifiers, _ = fit_cluster_then_predict(
        features=X_train_full,
        labels=y_train,
        config=ctp_config,
        cluster_features=X_train_beh,   # <- IMPORTANT: cluster on behavior only
    )

    # Choose threshold on validation
    val_clusters = kmeans.predict(X_val_beh)
    probas_val, _ = predict_with_clusters(val_clusters, X_val_full, classifiers)
    threshold = _choose_threshold_max_f1(y_val, probas_val)

    # Evaluate on test
    test_clusters = kmeans.predict(X_test_beh)
    probas_test, _ = predict_with_clusters(test_clusters, X_test_full, classifiers)
    preds_test = (probas_test >= threshold).astype(int)

    clf_metrics = prediction_eval.compute_classification_metrics(
        y_test, preds_test, probas_test
    )

    cluster_metrics = _evaluate_clusters(
        "cluster_then_predict",
        X_test_beh,
        test_clusters,
        y_test,
    )

    cluster_metrics.update(clf_metrics)
    cluster_metrics["inertia"] = kmeans.inertia()
    cluster_metrics["lift_top20"] = compute_lift(y_test, probas_test, top_frac=0.2)
    cluster_metrics["threshold"] = threshold
    cluster_metrics["n_clusters"] = ctp_config.n_clusters
    return cluster_metrics


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for baseline experiments."""
    parser = argparse.ArgumentParser(
        description="Run baseline methods under strict hold-out evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config for baselines (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and models (default: 42).",
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

    # Read baseline config
    try:
        configs = yaml.safe_load(args.config.read_text()) or {}
    except FileNotFoundError:
        logger.error("Baseline config file not found at %s", args.config)
        sys.exit(1)

    # Load & split data
    try:
        train_df, val_df, test_df = _load_and_split_data(
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` "
            "to verify dataset placement."
        )
        sys.exit(1)

    # Build features (leakage-free)
    (
        X_train_full, y_train,
        X_val_full, y_val,
        X_test_full, y_test,
        X_train_beh, X_val_beh, X_test_beh,
        _,
    ) = _build_features(train_df, val_df, test_df)

    logger.info(
        "Baseline evaluation split sizes: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Feature dims: behavior=%d, full=%d",
        X_train_beh.shape[1],
        X_train_full.shape[1],
    )

    results: list[Dict[str, Any]] = []

    # B1: RFM K-Means
    logger.info("Running Baseline 1: RFM K-Means")
    results.append(
        _run_rfm_kmeans(configs.get("rfm_kmeans", {}), X_train_full, X_test_full, y_test)
    )

    # B2: behavior K-Means
    logger.info("Running Baseline 2: Behavior K-Means")
    results.append(
        _run_full_kmeans(configs.get("full_kmeans", {}), X_train_beh, X_test_beh, y_test)
    )

    # B3: GMM
    if not args.skip_gmm:
        logger.info("Running Baseline 3: GMM")
        try:
            results.append(
                _run_gmm(configs.get("gmm", {}), X_train_beh, X_test_beh, y_test)
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("GMM baseline failed: %s", exc)

    # B4: Cluster-then-Predict
    if not args.skip_ctp:
        logger.info("Running Baseline 4: Cluster-then-Predict")
        try:
            results.append(
                _run_cluster_then_predict(
                    configs.get("cluster_then_predict", {}),
                    X_train_full, X_train_beh, y_train,
                    X_val_full, X_val_beh, y_val,
                    X_test_full, X_test_beh, y_test,
                )
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Cluster-then-Predict baseline failed: %s", exc)

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "baseline_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved baseline metrics to %s", results_path)


if __name__ == "__main__":
    main()
