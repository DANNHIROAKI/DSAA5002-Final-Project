"""Train and evaluate the Response-Aware Joint model under a strict hold-out protocol.

Upgraded behavior vs the previous version:
- Uses y = Response (label_mode="recent").
- Uses leakage-free preprocessing (fit on train; transform val/test).
- Supports RAMoE as the default RAJC model_type ("ramoe").
- Trains RAJC/RAMoE on behavior features (X_beh) but experts use full features (X_full).
- Selects a probability threshold on validation (max F1), then reports test metrics.
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
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils.logging_utils import configure_logging
from customer_segmentation.src.utils.metrics_utils import compute_lift

OUTPUT_DIR = Path("customer_segmentation/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"

DEFAULT_CONFIG_PATH = Path("customer_segmentation/configs/rajc.yaml")


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _choose_threshold_max_f1(y_true: pd.Series, probas: pd.Series, grid_size: int = 101) -> float:
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


def _load_rajc_config(config_path: Path, logger) -> RAJCConfig:
    """Load RAJCConfig from YAML with safe defaults."""
    try:
        cfg_dict = yaml.safe_load(config_path.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {})
    except FileNotFoundError:
        logger.warning("RAJC config file not found at %s; using defaults.", config_path)
        rajc_cfg = {}

    lr_cfg = rajc_cfg.get("logistic_regression", {}) or {}

    return RAJCConfig(
        n_clusters=rajc_cfg.get("n_clusters", 4),
        lambda_=float(rajc_cfg.get("lambda", 1.0)),
        gamma=float(rajc_cfg.get("gamma", 0.0)),
        smoothing=float(rajc_cfg.get("smoothing", 1.0)),
        max_iter=int(rajc_cfg.get("max_iter", 30)),
        tol=float(rajc_cfg.get("tol", 1e-3)),
        random_state=rajc_cfg.get("random_state", 42),
        kmeans_n_init=int(rajc_cfg.get("kmeans_n_init", 10)),
        model_type=rajc_cfg.get("model_type", "ramoe"),
        # RAMoE-specific
        temperature=float(rajc_cfg.get("temperature", 1.0)),
        # Expert LR settings
        logreg_C=float(lr_cfg.get("C", 1.0)),
        logreg_max_iter=int(lr_cfg.get("max_iter", 200)),
        logreg_solver=str(lr_cfg.get("solver", "lbfgs")),
    )


def _prepare_splits(
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_raw_data()
    cleaned = clean_data(raw)
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
    object,
]:
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


def _evaluate_on_split(
    model: RAJCModel,
    X_beh: pd.DataFrame,
    X_full: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> Dict[str, Any]:
    """Compute clustering + segmentation + prediction metrics on a given split."""
    clusters = pd.Series(model.predict_clusters(X_beh), index=X_beh.index, name="cluster")
    probas = pd.Series(
        model.predict_response(X_beh, full_features=X_full),
        index=X_beh.index,
        name="proba",
    )
    preds = (probas >= threshold).astype(int)

    scores = clustering_eval.compute_scores(X_beh, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, y)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    scores.update(prediction_eval.compute_classification_metrics(y, preds, probas))
    scores["lift_top20"] = compute_lift(y, probas, top_frac=0.2)

    scores["split"] = split_name
    scores["threshold"] = float(threshold)
    scores["n_clusters"] = model.config.n_clusters
    scores["lambda"] = model.config.lambda_
    scores["gamma"] = getattr(model.config, "gamma", 0.0)
    scores["temperature"] = getattr(model.config, "temperature", None)
    scores["model_type"] = getattr(model.config, "model_type", "ramoe")
    scores["label"] = "rajc"
    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RAJC/RAMoE with strict hold-out evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file for RAJC (default: {DEFAULT_CONFIG_PATH})",
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
        help="Random seed for split and models (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    # Load configuration
    config = _load_rajc_config(args.config, logger)

    # Prepare data splits
    try:
        train_df, val_df, test_df = _prepare_splits(
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

    # Features (fit on train only)
    (
        X_train_full, y_train,
        X_val_full, y_val,
        X_test_full, y_test,
        X_train_beh, X_val_beh, X_test_beh,
        _,
    ) = _build_features(train_df, val_df, test_df)

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Fitting RAJC/RAMoE with config: %s",
        config,
    )

    # Fit model (RAMoE requires full_features)
    model = RAJCModel(config)
    model.fit(X_train_beh, y_train, full_features=X_train_full)

    # Choose threshold on validation
    probas_val = pd.Series(
        model.predict_response(X_val_beh, full_features=X_val_full),
        index=X_val_beh.index,
    )
    threshold = _choose_threshold_max_f1(y_val, probas_val)

    # Evaluate
    val_metrics = _evaluate_on_split(
        model, X_val_beh, X_val_full, y_val, threshold, split_name="val"
    )
    test_metrics = _evaluate_on_split(
        model, X_test_beh, X_test_full, y_test, threshold, split_name="test"
    )

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_path = TABLE_DIR / "rajc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved RAJC metrics to %s", metrics_path)

    # Save train assignments and test predicted clusters for reporting
    train_assignments_path = TABLE_DIR / "rajc_assignments_train.csv"
    pd.Series(model.assignments_, index=X_train_beh.index, name="cluster").to_csv(
        train_assignments_path, header=True
    )
    logger.info("Saved train cluster assignments to %s", train_assignments_path)

    test_clusters_path = TABLE_DIR / "rajc_clusters_test.csv"
    pd.Series(model.predict_clusters(X_test_beh), index=X_test_beh.index, name="cluster").to_csv(
        test_clusters_path, header=True
    )
    logger.info("Saved test cluster predictions to %s", test_clusters_path)


if __name__ == "__main__":
    main()
