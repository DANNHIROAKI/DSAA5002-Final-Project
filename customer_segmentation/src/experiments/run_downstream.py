"""Downstream promotion-response prediction with cluster IDs as features.

We compare four feature configurations:

1) Base features only.
2) Base + KMeans cluster IDs.
3) Base + GMM cluster IDs.
4) Base + RAJC cluster IDs.

For clustering-based features, we let the clustering step operate on the
behaviour-only feature subset, while the downstream classifier always
uses the full engineered feature matrix.
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

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
DEFAULT_RAJC_CONFIG_PATH = Path("customer_segmentation/configs/rajc.yaml")


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw data, clean it and build the engineered feature matrix.

    Returns
    -------
    features_full :
        Full engineered feature matrix (behaviour + response-related).
    labels :
        Binary promotion-response labels.
    behavior_features :
        Behaviour-only feature subset for clustering.
    """
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, transformer = assemble_feature_table(cleaned)
    behavior_features, _ = split_behavior_and_response_features(
        features_df, transformer
    )
    return features_df, labels, behavior_features


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
    """Append one-hot cluster indicators to train/test features.

    Ensures train and test have the **same dummy columns** even if some
    clusters are absent in the test set.
    """
    train_dummies = _one_hot_clusters(train_labels, index=train_x.index, prefix=prefix)
    test_dummies = _one_hot_clusters(test_labels, index=test_x.index, prefix=prefix)

    # Align columns to training dummies; unseen clusters in test are filled with 0.
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    train_aug = pd.concat([train_x, train_dummies], axis=1)
    test_aug = pd.concat([test_x, test_dummies], axis=1)
    return train_aug, test_aug


def _train_classifier(train_x: pd.DataFrame, train_y: pd.Series) -> LogisticRegression:
    """Fit a logistic regression classifier with balanced class weights."""
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(train_x, train_y)
    return clf


def _evaluate_model(
    model: LogisticRegression, test_x: pd.DataFrame, test_y: pd.Series
) -> Dict[str, float]:
    """Compute evaluation metrics for a trained classifier."""
    probas = pd.Series(model.predict_proba(test_x)[:, 1], index=test_x.index)
    preds = pd.Series(model.predict(test_x), index=test_x.index)
    return prediction_eval.compute_classification_metrics(test_y, preds, probas)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream response prediction using cluster IDs as features."
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
        help="Skip RAJC cluster features.",
    )
    return parser.parse_args(argv)


def _load_rajc_config(logger) -> RAJCConfig:
    """Load RAJCConfig from the shared YAML file, with safe defaults."""
    try:
        cfg_dict = yaml.safe_load(DEFAULT_RAJC_CONFIG_PATH.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {})
    except FileNotFoundError:
        logger.warning(
            "RAJC config file not found at %s, using default RAJCConfig.",
            DEFAULT_RAJC_CONFIG_PATH,
        )
        rajc_cfg = {}

    return RAJCConfig(
        n_clusters=rajc_cfg.get("n_clusters", 4),
        lambda_=rajc_cfg.get("lambda", 1.0),
        gamma=rajc_cfg.get("gamma", 0.0),
        max_iter=rajc_cfg.get("max_iter", 20),
        tol=float(rajc_cfg.get("tol", 1e-4)),
        random_state=rajc_cfg.get("random_state", 42),
        smoothing=rajc_cfg.get("smoothing", 1.0),
        logreg_max_iter=rajc_cfg.get("logistic_regression", {}).get("max_iter", 200),
        kmeans_n_init=rajc_cfg.get("kmeans_n_init", 10),
        model_type=rajc_cfg.get("model_type", "constant_prob"),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    try:
        features_full, labels, behavior_features = _prepare_features()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error(
            "Run `python -m customer_segmentation.src.data.check_data` "
            "to verify dataset placement."
        )
        sys.exit(1)

    logger.info(
        "Running downstream prediction on %d samples with %d base features.",
        features_full.shape[0],
        features_full.shape[1],
    )

    train_x_full, test_x_full, train_y, test_y = train_test_split(
        features_full,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    # Align behavioural subset with the train/test split.
    train_x_beh = behavior_features.loc[train_x_full.index]
    test_x_beh = behavior_features.loc[test_x_full.index]

    results = []

    # 1) Base model
    logger.info("Training base logistic regression without cluster IDs")
    base_clf = _train_classifier(train_x_full, train_y)
    base_metrics = _evaluate_model(base_clf, test_x_full, test_y)
    base_metrics["model"] = "base"
    results.append(base_metrics)

    # 2) KMeans clusters as features
    logger.info("Training logistic regression with KMeans cluster IDs")
    kmeans_cfg = KMeansConfig(
        n_clusters=4,
        init="k-means++",
        max_iter=300,
        random_state=args.random_state,
        n_init=10,
    )
    kmeans_model = KMeansBaseline(kmeans_cfg)
    kmeans_model.fit(train_x_beh)
    kmeans_labels_train = kmeans_model.predict(train_x_beh)
    kmeans_labels_test = kmeans_model.predict(test_x_beh)
    kmeans_train, kmeans_test = _add_cluster_features(
        train_x_full,
        test_x_full,
        kmeans_labels_train,
        kmeans_labels_test,
        prefix="kmeans",
    )
    kmeans_clf = _train_classifier(kmeans_train, train_y)
    kmeans_metrics = _evaluate_model(kmeans_clf, kmeans_test, test_y)
    kmeans_metrics["model"] = "base+kmeansid"
    results.append(kmeans_metrics)

    # 3) GMM clusters (optional; clustering on behaviour features)
    if not args.skip_gmm:
        logger.info("Training logistic regression with GMM cluster IDs")
        try:
            gmm_model = GMMBaseline(GMMConfig())
            gmm_model.fit(train_x_beh)
            gmm_labels_train = gmm_model.predict(train_x_beh)
            gmm_labels_test = gmm_model.predict(test_x_beh)
            gmm_train, gmm_test = _add_cluster_features(
                train_x_full,
                test_x_full,
                gmm_labels_train,
                gmm_labels_test,
                prefix="gmm",
            )
            gmm_clf = _train_classifier(gmm_train, train_y)
            gmm_metrics = _evaluate_model(gmm_clf, gmm_test, test_y)
            gmm_metrics["model"] = "base+gmmid"
            results.append(gmm_metrics)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("GMM downstream pipeline failed: %s", exc)

    # 4) RAJC clusters (optional; RAJC trained on behaviour features)
    if not args.skip_rajc:
        logger.info("Training logistic regression with RAJC cluster IDs")
        try:
            rajc_config = _load_rajc_config(logger)
            rajc_model = RAJCModel(rajc_config)
            rajc_model.fit(train_x_beh, train_y)
            rajc_labels_train = rajc_model.assignments_
            rajc_labels_test = rajc_model.predict_clusters(test_x_beh)
            rajc_train, rajc_test = _add_cluster_features(
                train_x_full,
                test_x_full,
                rajc_labels_train,
                rajc_labels_test,
                prefix="rajc",
            )
            rajc_clf = _train_classifier(rajc_train, train_y)
            rajc_metrics = _evaluate_model(rajc_clf, rajc_test, test_y)
            rajc_metrics["model"] = "base+rajcid"
            results.append(rajc_metrics)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("RAJC downstream pipeline failed: %s", exc)

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "downstream_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved downstream prediction metrics to %s", results_path)


if __name__ == "__main__":
    main()
