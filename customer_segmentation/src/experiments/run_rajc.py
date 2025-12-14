"""Train and evaluate the proposed RAJC family under a strict hold-out protocol.

This script is the main entrypoint for the upgraded method:
**RAMoE / HyRAMoE (Response-Aware Mixture-of-Experts)**.

Key properties
--------------
- Leak-safe evaluation: split raw rows first; fit preprocessing on train only.
- Fair feature usage:
    - Segmentation/gating uses behaviour features X_beh only.
    - Response prediction uses full features X_full.
- Validation-based threshold selection: choose threshold on val (max F1),
  report test metrics.

Outputs
-------
Writes:
- ``customer_segmentation/outputs/tables/rajc_metrics.csv``
- ``customer_segmentation/outputs/tables/rajc_assignments_train.csv``
- ``customer_segmentation/outputs/tables/rajc_clusters_test.csv``
- ``customer_segmentation/outputs/tables/rajc_assignments.csv``
  (cluster assignment for the *full* cleaned dataset, produced using the
   train-fitted transformer; useful for profiling/plots).
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

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


def _as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return default


def _load_rajc_config(
    config_path: Path,
    logger,
    *,
    override_model_type: Optional[str] = None,
) -> RAJCConfig:
    """Load RAJCConfig from YAML with backward/forward compatibility.

    The repository's older ``rajc.yaml`` may contain only a small subset of
    parameters (e.g., n_clusters/lambda/gamma). This loader maps those keys into
    the newer :class:`~customer_segmentation.src.models.rajc.RAJCConfig`.

    Unknown keys are ignored.
    """

    if not config_path.is_file():
        logger.warning("RAJC config file not found at %s; using RAJCConfig defaults.", config_path)
        cfg = RAJCConfig()
        if override_model_type is not None:
            cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg

    try:
        cfg_dict = yaml.safe_load(config_path.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {}) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read RAJC config (%s); using defaults.", exc)
        cfg = RAJCConfig()
        if override_model_type is not None:
            cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg

    # Nested sub-configs (optional)
    lr_cfg = rajc_cfg.get("logistic_regression", {}) or {}
    hgbdt_cfg = rajc_cfg.get("hgbdt", {}) or {}

    valid_fields = {f.name for f in fields(RAJCConfig)}

    # Start with direct pass-through for keys that match RAJCConfig fields.
    kwargs: Dict[str, Any] = {k: v for k, v in rajc_cfg.items() if k in valid_fields}

    # Backward-compatible key mapping
    if "lambda" in rajc_cfg and "lambda_" not in kwargs:
        kwargs["lambda_"] = float(rajc_cfg.get("lambda"))
    if "n_clusters" in rajc_cfg:
        kwargs["n_clusters"] = int(rajc_cfg.get("n_clusters"))
    if "kmeans_n_init" in rajc_cfg:
        kwargs["kmeans_n_init"] = int(rajc_cfg.get("kmeans_n_init"))

    # Common optimisation keys
    if "max_iter" in rajc_cfg:
        kwargs["max_iter"] = int(rajc_cfg.get("max_iter"))
    if "tol" in rajc_cfg:
        kwargs["tol"] = float(rajc_cfg.get("tol"))
    if "random_state" in rajc_cfg:
        kwargs["random_state"] = int(rajc_cfg.get("random_state"))

    # Constant-prob (RAJC-CP++) keys
    if "gamma" in rajc_cfg:
        kwargs["gamma"] = float(rajc_cfg.get("gamma"))
    if "smoothing" in rajc_cfg:
        kwargs["smoothing"] = float(rajc_cfg.get("smoothing"))

    # RAMoE keys (some may exist in newer configs)
    if "temperature" in rajc_cfg:
        kwargs["temperature"] = float(rajc_cfg.get("temperature"))
    if "gating_temperature" in rajc_cfg:
        kwargs["gating_temperature"] = float(rajc_cfg.get("gating_temperature"))

    # Expert blocks
    if "C" in lr_cfg and "logreg_C" not in kwargs:
        kwargs["logreg_C"] = float(lr_cfg.get("C"))
    if "max_iter" in lr_cfg and "logreg_max_iter" not in kwargs:
        kwargs["logreg_max_iter"] = int(lr_cfg.get("max_iter"))
    if "solver" in lr_cfg and "logreg_solver" not in kwargs:
        kwargs["logreg_solver"] = str(lr_cfg.get("solver"))

    # HGBDT block (optional)
    if "max_depth" in hgbdt_cfg and "hgbdt_max_depth" not in kwargs:
        kwargs["hgbdt_max_depth"] = int(hgbdt_cfg.get("max_depth"))
    if "learning_rate" in hgbdt_cfg and "hgbdt_learning_rate" not in kwargs:
        kwargs["hgbdt_learning_rate"] = float(hgbdt_cfg.get("learning_rate"))
    if "max_iter" in hgbdt_cfg and "hgbdt_max_iter" not in kwargs:
        kwargs["hgbdt_max_iter"] = int(hgbdt_cfg.get("max_iter"))

    # Booleans (yaml already parses but keep robust)
    if "use_global_expert" in rajc_cfg:
        kwargs["use_global_expert"] = _as_bool(rajc_cfg.get("use_global_expert"), default=True)

    # Model type default: prefer new method unless explicitly overridden
    model_type = str(rajc_cfg.get("model_type", "ramoe")).strip().lower()
    if override_model_type is not None:
        model_type = str(override_model_type).strip().lower()
    kwargs["model_type"] = model_type

    # Filter to dataclass fields
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    try:
        return RAJCConfig(**kwargs)
    except TypeError as exc:
        logger.warning("Failed to instantiate RAJCConfig from YAML (%s). Using defaults.", exc)
        cfg = RAJCConfig()
        if override_model_type is not None:
            cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg


def _prepare_splits(
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load -> clean -> label -> train/val/test split.

    Returns
    -------
    cleaned_all, train_df, val_df, test_df
    """
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
    return cleaned, train_df, val_df, test_df


def _build_features(
    *,
    cleaned_all: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    object,
]:
    """Fit transformer on train; transform val/test and the full dataset."""
    X_train_full, y_train, transformer = assemble_feature_table(train_df, fit=True, label_mode="recent")
    X_val_full, y_val, _ = assemble_feature_table(val_df, transformer=transformer, fit=False, label_mode="recent")
    X_test_full, y_test, _ = assemble_feature_table(test_df, transformer=transformer, fit=False, label_mode="recent")

    # For profiling plots we also want all-sample cluster assignments with the same transformer.
    X_all_full, y_all, _ = assemble_feature_table(
        cleaned_all,
        transformer=transformer,
        fit=False,
        label_mode="recent",
    )

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_val_beh, _ = split_behavior_and_response_features(X_val_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)
    X_all_beh, _ = split_behavior_and_response_features(X_all_full, transformer)

    return (
        X_train_full,
        y_train,
        X_val_full,
        y_val,
        X_test_full,
        y_test,
        X_train_beh,
        X_val_beh,
        X_test_beh,
        X_all_full,
        X_all_beh,
        transformer,
    )


def _evaluate_on_split(
    *,
    model: RAJCModel,
    X_beh: pd.DataFrame,
    X_full: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str,
) -> Dict[str, Any]:
    """Compute clustering + segmentation + prediction metrics."""
    clusters = pd.Series(model.predict_clusters(X_beh), index=X_beh.index, name="cluster")
    probas = pd.Series(model.predict_response(X_beh, full_features=X_full), index=X_beh.index, name="proba")
    preds = (probas >= threshold).astype(int)

    scores: Dict[str, Any] = clustering_eval.compute_scores(X_beh, clusters)
    rates = segmentation_eval.cluster_response_rates(clusters, y)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    scores.update(prediction_eval.compute_classification_metrics(y, preds, probas))
    scores["lift_top20"] = compute_lift(y, probas, top_frac=0.2)

    # Trace config fields used in the report table
    cfg = model.config
    scores.update(
        {
            "split": split_name,
            "threshold": float(threshold),
            "lambda": float(getattr(cfg, "lambda_", np.nan)),
            "gamma": float(getattr(cfg, "gamma", np.nan)),
            "temperature": float(getattr(cfg, "temperature", np.nan)),
            "model_type": str(getattr(cfg, "model_type", "ramoe")),
            "expert_type": str(getattr(cfg, "expert_type", "")),
            "gating_type": str(getattr(cfg, "gating_type", "")),
            "use_global_expert": bool(getattr(cfg, "use_global_expert", False)),
            "hybrid_alpha": float(getattr(cfg, "hybrid_alpha", np.nan)),
            "budget_reweight_alpha": float(getattr(cfg, "budget_reweight_alpha", 0.0)),
            "label": "rajc",
        }
    )

    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RAJC/RAMoE/HyRAMoE with strict hold-out evaluation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration file for RAJC (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ramoe",
        choices=["ramoe", "logreg", "constant_prob"],
        help=(
            "Override model_type from YAML (default: ramoe). "
            "Set to constant_prob to reproduce the older RAJC-CP++ baseline."
        ),
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
        help="Random seed for split and model initialisation (default: 42).",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy", "youden_j"],
        help="Metric used to choose threshold on validation (default: f1).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    # Load configuration (override model_type to the new method by default)
    config = _load_rajc_config(args.config, logger, override_model_type=args.model_type)

    # Prepare data splits
    try:
        cleaned_all, train_df, val_df, test_df = _prepare_splits(
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
        X_train_full,
        y_train,
        X_val_full,
        y_val,
        X_test_full,
        y_test,
        X_train_beh,
        X_val_beh,
        X_test_beh,
        X_all_full,
        X_all_beh,
        _,
    ) = _build_features(
        cleaned_all=cleaned_all,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Feature dims: behaviour=%d, full=%d",
        X_train_beh.shape[1],
        X_train_full.shape[1],
    )
    logger.info("Fitting RAJC model with config: %s", config)

    # Fit model
    model = RAJCModel(config)
    model.fit(X_train_beh, y_train, full_features=X_train_full)

    # Choose threshold on validation (only affects binary metrics; AUC/lift remain threshold-free)
    probas_val = pd.Series(
        model.predict_response(X_val_beh, full_features=X_val_full),
        index=X_val_beh.index,
    )
    threshold = prediction_eval.choose_threshold(y_val, probas_val, metric=args.threshold_metric)

    # Evaluate
    val_metrics = _evaluate_on_split(
        model=model,
        X_beh=X_val_beh,
        X_full=X_val_full,
        y=y_val,
        threshold=threshold,
        split_name="val",
    )
    test_metrics = _evaluate_on_split(
        model=model,
        X_beh=X_test_beh,
        X_full=X_test_full,
        y=y_test,
        threshold=threshold,
        split_name="test",
    )

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_path = TABLE_DIR / "rajc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved RAJC metrics to %s", metrics_path)

    # Save train assignments and test predicted clusters
    train_assignments_path = TABLE_DIR / "rajc_assignments_train.csv"
    pd.Series(model.assignments_, index=X_train_beh.index, name="cluster").to_csv(
        train_assignments_path,
        header=True,
    )
    logger.info("Saved train cluster assignments to %s", train_assignments_path)

    test_clusters_path = TABLE_DIR / "rajc_clusters_test.csv"
    pd.Series(model.predict_clusters(X_test_beh), index=X_test_beh.index, name="cluster").to_csv(
        test_clusters_path,
        header=True,
    )
    logger.info("Saved test cluster predictions to %s", test_clusters_path)

    # Save *all-sample* assignments (useful for profiling/plots)
    all_assignments_path = TABLE_DIR / "rajc_assignments.csv"
    pd.Series(model.predict_clusters(X_all_beh), index=X_all_beh.index, name="cluster").to_csv(
        all_assignments_path,
        header=True,
    )
    logger.info("Saved full-dataset cluster assignments to %s", all_assignments_path)


if __name__ == "__main__":
    main()
