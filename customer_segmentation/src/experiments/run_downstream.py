"""Downstream response prediction with cluster IDs as features (leakage-free).

We compare four feature configurations:
1) Base features only.
2) Base + KMeans cluster IDs (clusters learned on behavior features).
3) Base + GMM cluster IDs (clusters learned on behavior features).
4) Base + RAJC/RAMoE cluster IDs (clusters learned on behavior features with labels).

Upgrades vs previous version:
- Uses y = Response (label_mode="recent").
- Splits BEFORE preprocessing; fits transformer on train only.
- Adds lift@20% to the result table for a marketing-budget view.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from customer_segmentation.src.data.features import (
    add_response_label,
    assemble_feature_table,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import prediction as prediction_eval
from customer_segmentation.src.models.gmm_baseline import GMMBaseline, GMMConfig
from customer_segmentation.src.models.kmeans_baseline import KMeansBaseline, KMeansConfig
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging
from customer_segmentation.src.utils.metrics_utils import compute_lift

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
DEFAULT_RAJC_CONFIG_PATH = Path("customer_segmentation/configs/rajc.yaml")


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _one_hot_clusters(
    cluster_labels: pd.Series | np.ndarray,
    index: Optional[pd.Index] = None,
    prefix: str = "cluster",
) -> pd.DataFrame:
    """Convert cluster labels into one-hot encoded indicator features."""
    if not isinstance(cluster_labels, pd.Series):
        cluster_labels = pd.Series(cluster_labels, index=index)
    return pd.get_dummies(cluster_labels, prefix=prefix, dtype=int)


def _add_cluster_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_labels: pd.Series | np.ndarray,
    test_labels: pd.Series | np.ndarray,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append one-hot cluster indicators to train/test features with aligned columns."""
    train_dummies = _one_hot_clusters(train_labels, index=train_x.index, prefix=prefix)
    test_dummies = _one_hot_clusters(test_labels, index=test_x.index, prefix=prefix)

    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    train_aug = pd.concat([train_x, train_dummies], axis=1)
    test_aug = pd.concat([test_x, test_dummies], axis=1)
    return train_aug, test_aug


def _train_classifier(train_x: pd.DataFrame, train_y: pd.Series) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(train_x, train_y)
    return clf


def _evaluate_model(
    model: LogisticRegression, test_x: pd.DataFrame, test_y: pd.Series
) -> Dict[str, float]:
    probas = pd.Series(model.predict_proba(test_x)[:, 1], index=test_x.index)
    preds = pd.Series(model.predict(test_x), index=test_x.index)
    metrics = prediction_eval.compute_classification_metrics(test_y, preds, probas)
    metrics["lift_top20"] = compute_lift(test_y, probas, top_frac=0.2)
    return metrics


def _load_rajc_config(logger) -> RAJCConfig:
    """Load RAJCConfig from YAML with safe defaults."""
    try:
        cfg_dict = yaml.safe_load(DEFAULT_RAJC_CONFIG_PATH.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {})
    except FileNotFoundError:
        logger.warning(
            "RAJC config file not found at %s, using default RAJCConfig.",
            DEFAULT_RAJC_CONFIG_PATH,
        )
        rajc_cfg = {}

    lr_cfg = rajc_cfg.get("logistic_regression", {}) or {}

    return RAJCConfig(
        n_clusters=rajc_cfg.get("n_clusters", 4),
        lambda_=float(rajc_cfg.get("lambda", 1.0)),
        gamma=float(rajc_cfg.get("gamma", 0.0)),
        max_iter=int(rajc_cfg.get("max_iter", 30)),
        tol=float(rajc_cfg.get("tol", 1e-3)),
        random_state=rajc_cfg.get("random_state", 42),
        smoothing=float(rajc_cfg.get("smoothing", 1.0)),
        kmeans_n_init=int(rajc_cfg.get("kmeans_n_init", 10)),
        model_type=rajc_cfg.get("model_type", "ramoe"),
        temperature=float(rajc_cfg.get("temperature", 1.0)),
        logreg_C=float(lr_cfg.get("C", 1.0)),
        logreg_max_iter=int(lr_cfg.get("max_iter", 200)),
        logreg_solver=str(lr_cfg.get("solver", "lbfgs")),
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream response prediction using cluster IDs as features (leakage-free)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used as test set (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42).",
    )
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Skip GMM cluster features.",
    )
    parser.add_argument(
        "--skip-rajc",
        action="store_true",
        help="Skip RAJC/RAMoE cluster features.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    # Load raw, clean, create label for stratified split
    try:
        raw = load_raw_data()
        cleaned = clean_data(raw)
        cleaned = add_response_label(cleaned, mode="recent")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` "
            "to verify dataset placement."
        )
        sys.exit(1)

    train_df, test_df = train_test_split(
        cleaned,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=cleaned["campaign_response"],
    )

    # Leakage-free feature transform: fit on train only
    X_train_full, y_train, transformer = assemble_feature_table(
        train_df, fit=True, label_mode="recent"
    )
    X_test_full, y_test, _ = assemble_feature_table(
        test_df, transformer=transformer, fit=False, label_mode="recent"
    )

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)

    logger.info(
        "Downstream split sizes: train=%d, test=%d | dims: full=%d, beh=%d",
        len(train_df), len(test_df), X_train_full.shape[1], X_train_beh.shape[1],
    )

    results = []

    # 1) Base model
    logger.info("Training base logistic regression without cluster IDs")
    base_clf = _train_classifier(X_train_full, y_train)
    base_metrics = _evaluate_model(base_clf, X_test_full, y_test)
    base_metrics["model"] = "base"
    results.append(base_metrics)

    # 2) KMeans cluster IDs (cluster on behavior features)
    logger.info("Training logistic regression with KMeans cluster IDs")
    kmeans_cfg = KMeansConfig(
        n_clusters=4,
        init="k-means++",
        max_iter=300,
        random_state=args.random_state,
        n_init=10,
    )
    kmeans_model = KMeansBaseline(kmeans_cfg)
    kmeans_model.fit(X_train_beh)
    kmeans_labels_train = kmeans_model.predict(X_train_beh)
    kmeans_labels_test = kmeans_model.predict(X_test_beh)

    kmeans_train, kmeans_test = _add_cluster_features(
        X_train_full, X_test_full,
        kmeans_labels_train, kmeans_labels_test,
        prefix="kmeans",
    )
    kmeans_clf = _train_classifier(kmeans_train, y_train)
    kmeans_metrics = _evaluate_model(kmeans_clf, kmeans_test, y_test)
    kmeans_metrics["model"] = "base+kmeansid"
    results.append(kmeans_metrics)

    # 3) GMM cluster IDs (optional)
    if not args.skip_gmm:
        logger.info("Training logistic regression with GMM cluster IDs")
        try:
            gmm_model = GMMBaseline(GMMConfig())
            gmm_model.fit(X_train_beh)
            gmm_labels_train = gmm_model.predict(X_train_beh)
            gmm_labels_test = gmm_model.predict(X_test_beh)

            gmm_train, gmm_test = _add_cluster_features(
                X_train_full, X_test_full,
                gmm_labels_train, gmm_labels_test,
                prefix="gmm",
            )
            gmm_clf = _train_classifier(gmm_train, y_train)
            gmm_metrics = _evaluate_model(gmm_clf, gmm_test, y_test)
            gmm_metrics["model"] = "base+gmmid"
            results.append(gmm_metrics)
        except Exception as exc:  # pragma: no cover
            logger.exception("GMM downstream pipeline failed: %s", exc)

    # 4) RAJC/RAMoE cluster IDs (optional)
    if not args.skip_rajc:
        logger.info("Training logistic regression with RAJC/RAMoE cluster IDs")
        try:
            rajc_config = _load_rajc_config(logger)
            rajc_model = RAJCModel(rajc_config)

            # RAMoE needs full_features to train experts; pass it for all modes.
            rajc_model.fit(X_train_beh, y_train, full_features=X_train_full)

            rajc_labels_train = pd.Series(rajc_model.assignments_, index=X_train_beh.index)
            rajc_labels_test = pd.Series(rajc_model.predict_clusters(X_test_beh), index=X_test_beh.index)

            rajc_train, rajc_test = _add_cluster_features(
                X_train_full, X_test_full,
                rajc_labels_train, rajc_labels_test,
                prefix="rajc",
            )
            rajc_clf = _train_classifier(rajc_train, y_train)
            rajc_metrics = _evaluate_model(rajc_clf, rajc_test, y_test)
            rajc_metrics["model"] = "base+rajcid"
            results.append(rajc_metrics)
        except Exception as exc:  # pragma: no cover
            logger.exception("RAJC/RAMoE downstream pipeline failed: %s", exc)

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "downstream_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved downstream prediction metrics to %s", results_path)


if __name__ == "__main__":
    main()
