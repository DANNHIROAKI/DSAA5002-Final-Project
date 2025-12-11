"""Downstream promotion-response prediction with cluster IDs as features."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from customer_segmentation.src.data.features import assemble_feature_table
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import prediction as prediction_eval
from customer_segmentation.src.models.gmm_baseline import GMMBaseline, GMMConfig
from customer_segmentation.src.models.kmeans_baseline import KMeansBaseline, KMeansConfig
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series]:
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, _ = assemble_feature_table(cleaned)
    return features_df, labels


def _one_hot_clusters(cluster_labels: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(cluster_labels, prefix="cluster", dtype=int)


def _train_classifier(train_x: pd.DataFrame, train_y: pd.Series) -> LogisticRegression:
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(train_x, train_y)
    return clf


def _evaluate_model(model: LogisticRegression, test_x: pd.DataFrame, test_y: pd.Series) -> Dict:
    probas = pd.Series(model.predict_proba(test_x)[:, 1], index=test_x.index)
    preds = pd.Series(model.predict(test_x), index=test_x.index)
    return prediction_eval.compute_classification_metrics(test_y, preds, probas)


def main() -> None:
    logger = configure_logging()
    _ensure_output_dirs()

    features, responses = _prepare_features()
    train_x, test_x, train_y, test_y = train_test_split(
        features, responses, test_size=0.2, random_state=42, stratify=responses
    )

    logger.info("Training base logistic regression without cluster IDs")
    base_clf = _train_classifier(train_x, train_y)
    base_metrics = _evaluate_model(base_clf, test_x, test_y)
    base_metrics["model"] = "base"

    logger.info("Training with KMeans cluster IDs")
    kmeans_model = KMeansBaseline(KMeansConfig())
    kmeans_model.fit(train_x)
    kmeans_labels_train = kmeans_model.predict(train_x)
    kmeans_labels_test = kmeans_model.predict(test_x)
    kmeans_train = pd.concat([train_x, _one_hot_clusters(kmeans_labels_train)], axis=1)
    kmeans_test = pd.concat([test_x, _one_hot_clusters(kmeans_labels_test)], axis=1)
    kmeans_clf = _train_classifier(kmeans_train, train_y)
    kmeans_metrics = _evaluate_model(kmeans_clf, kmeans_test, test_y)
    kmeans_metrics["model"] = "base+kmeansid"

    logger.info("Training with GMM cluster IDs")
    gmm_model = GMMBaseline(GMMConfig())
    gmm_model.fit(train_x)
    gmm_labels_train = gmm_model.predict(train_x)
    gmm_labels_test = gmm_model.predict(test_x)
    gmm_train = pd.concat([train_x, _one_hot_clusters(gmm_labels_train)], axis=1)
    gmm_test = pd.concat([test_x, _one_hot_clusters(gmm_labels_test)], axis=1)
    gmm_clf = _train_classifier(gmm_train, train_y)
    gmm_metrics = _evaluate_model(gmm_clf, gmm_test, test_y)
    gmm_metrics["model"] = "base+gmmid"

    logger.info("Training with RAJC cluster IDs")
    rajc_model = RAJCModel(RAJCConfig())
    rajc_model.fit(train_x, train_y)
    rajc_labels_train = rajc_model.assignments_
    rajc_labels_test = rajc_model.predict_clusters(test_x)
    rajc_train = pd.concat([train_x, _one_hot_clusters(rajc_labels_train)], axis=1)
    rajc_test = pd.concat([test_x, _one_hot_clusters(rajc_labels_test)], axis=1)
    rajc_clf = _train_classifier(rajc_train, train_y)
    rajc_metrics = _evaluate_model(rajc_clf, rajc_test, test_y)
    rajc_metrics["model"] = "base+rajcid"

    results_df = pd.DataFrame([base_metrics, kmeans_metrics, gmm_metrics, rajc_metrics])
    results_path = TABLE_DIR / "downstream_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved downstream prediction metrics to %s", results_path)


if __name__ == "__main__":
    main()
