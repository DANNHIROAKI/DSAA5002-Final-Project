"""Train and evaluate the Response-Aware Joint Clustering (RAJC) model."""
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
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging
from customer_segmentation.src.utils.metrics_utils import compute_lift

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_features() -> Tuple[pd.DataFrame, pd.Series]:
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features_df, labels, _ = assemble_feature_table(cleaned)
    return features_df, labels


def _evaluate(model: RAJCModel, features: pd.DataFrame, labels: pd.Series) -> Dict:
    clusters = model.assignments_
    if clusters is None:
        raise ValueError("RAJC model has not been fitted.")

    scores = clustering_eval.compute_scores(features, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, labels)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    probas = model.predict_response(features)
    preds = (probas >= 0.5).astype(int)
    scores.update(prediction_eval.compute_classification_metrics(labels, preds, probas))
    scores["lift_top20"] = compute_lift(labels, probas, top_frac=0.2)
    return scores


def main() -> None:
    logger = configure_logging()
    _ensure_output_dirs()

    config_dict = yaml.safe_load(Path("customer_segmentation/configs/rajc.yaml").read_text())
    rajc_cfg = config_dict.get("rajc", {})
    config = RAJCConfig(
        n_clusters=rajc_cfg.get("n_clusters", 4),
        lambda_=rajc_cfg.get("lambda", 1.0),
        max_iter=rajc_cfg.get("max_iter", 20),
        tol=float(rajc_cfg.get("tol", 1e-4)),
        random_state=rajc_cfg.get("random_state", 42),
        logreg_max_iter=rajc_cfg.get("logistic_regression", {}).get("max_iter", 200),
    )

    features, labels = _prepare_features()
    logger.info("Fitting RAJC on %d samples", len(features))

    model = RAJCModel(config)
    model.fit(features, labels)
    metrics = _evaluate(model, features, labels)

    metrics_df = pd.DataFrame([metrics])
    metrics_path = TABLE_DIR / "rajc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved RAJC metrics to %s", metrics_path)

    assignments_path = TABLE_DIR / "rajc_assignments.csv"
    model.assignments_.to_csv(assignments_path, header=["cluster"])
    logger.info("Saved RAJC assignments to %s", assignments_path)


if __name__ == "__main__":
    main()
