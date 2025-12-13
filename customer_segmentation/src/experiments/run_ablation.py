"""Ablation study runner under a strict hold-out protocol.

Upgraded behavior:
- Uses y = Response (label_mode="recent").
- Splits data into train/val/test BEFORE preprocessing.
- Fits transformers on train only; evaluates on validation (ablation) split.
- Supports model_type in {"ramoe", "logreg", "constant_prob"}.
- For RAMoE, we ablate K / lambda / temperature (and optionally gamma is ignored).
- For constant_prob, we ablate K / lambda / gamma.

Outputs are saved under customer_segmentation/outputs/tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Optional, Any, Dict

import numpy as np
import pandas as pd

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

# Defaults can be overridden via CLI.
DEFAULT_MODEL_TYPE = "ramoe"
DEFAULT_LAMBDA_GRID: List[float] = [0.0, 0.1, 0.3, 1.0, 3.0]
DEFAULT_TEMP_GRID: List[float] = [0.5, 1.0, 2.0]
DEFAULT_GAMMA_GRID: List[float] = [0.0, 0.1, 0.3]
DEFAULT_CLUSTER_GRID: List[int] = [3, 4, 5, 6]


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_val_split(
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load data, split, fit on train, transform val/test, return feature blocks."""
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
        train_df, val_df, test_df,
        X_train_full, y_train,
        X_val_full, y_val,
        X_test_full, y_test,
        X_train_beh, X_val_beh, X_test_beh,
    )


def _run_single(
    model_type: str,
    X_train_beh: pd.DataFrame,
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    X_val_beh: pd.DataFrame,
    X_val_full: pd.DataFrame,
    y_val: pd.Series,
    k: int,
    lambda_: float,
    temperature: float,
    gamma: float,
    max_iter: int,
    tol: float,
    random_state: int,
) -> Dict[str, Any]:
    """Fit a RAJC/RAMoE model and evaluate on validation split."""
    cfg = RAJCConfig(
        n_clusters=k,
        lambda_=lambda_,
        gamma=gamma,
        temperature=temperature,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        model_type=model_type,
    )
    model = RAJCModel(cfg)
    model.fit(X_train_beh, y_train, full_features=X_train_full)

    clusters_val = pd.Series(model.predict_clusters(X_val_beh), index=X_val_beh.index, name="cluster")
    probas_val = pd.Series(model.predict_response(X_val_beh, full_features=X_val_full), index=X_val_beh.index)

    # clustering/segmentation
    scores: Dict[str, Any] = clustering_eval.compute_scores(X_val_beh, clusters_val)
    rates = segmentation_eval.cluster_response_rates(clusters_val, y_val)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    # prediction
    preds_val = (probas_val >= 0.5).astype(int)
    scores.update(prediction_eval.compute_classification_metrics(y_val, preds_val, probas_val))
    scores["lift_top20"] = compute_lift(y_val, probas_val, top_frac=0.2)

    # config trace
    scores["n_clusters"] = k
    scores["lambda"] = lambda_
    scores["temperature"] = temperature
    scores["gamma"] = gamma
    scores["model_type"] = model_type
    scores["split"] = "val"
    scores["label"] = "rajc_ablation"

    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study over K/lambda/temperature (RAMoE) or K/lambda/gamma (constant_prob)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["ramoe", "logreg", "constant_prob"],
        help=f"Model type for ablation (default: {DEFAULT_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="*",
        default=DEFAULT_LAMBDA_GRID,
        help=f"Grid of lambda values (default: {DEFAULT_LAMBDA_GRID}).",
    )
    parser.add_argument(
        "--temperature-grid",
        type=float,
        nargs="*",
        default=DEFAULT_TEMP_GRID,
        help=f"Grid of temperature values for RAMoE (default: {DEFAULT_TEMP_GRID}).",
    )
    parser.add_argument(
        "--gamma-grid",
        type=float,
        nargs="*",
        default=DEFAULT_GAMMA_GRID,
        help=f"Grid of gamma values for constant_prob (default: {DEFAULT_GAMMA_GRID}).",
    )
    parser.add_argument(
        "--cluster-grid",
        type=int,
        nargs="*",
        default=DEFAULT_CLUSTER_GRID,
        help=f"Grid of cluster counts K (default: {DEFAULT_CLUSTER_GRID}).",
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
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Max iterations for RAJC/RAMoE training (default: 30).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance on cluster-change ratio (default: 1e-3).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logger = configure_logging()
    _ensure_output_dirs()
    args = _parse_args(argv)

    try:
        (
            _train_df, _val_df, _test_df,
            X_train_full, y_train,
            X_val_full, y_val,
            _X_test_full, _y_test,
            X_train_beh, X_val_beh, _X_test_beh,
        ) = _prepare_val_split(
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

    logger.info(
        "Running ablation for model_type=%s over K=%s, lambda=%s",
        args.model_type,
        args.cluster_grid,
        args.lambda_grid,
    )
    if args.model_type == "ramoe":
        logger.info("RAMoE temperature grid: %s", args.temperature_grid)
    else:
        logger.info("Gamma grid (used only for constant_prob): %s", args.gamma_grid)

    ablation_results: list[Dict[str, Any]] = []

    for k in args.cluster_grid:
        for lambda_ in args.lambda_grid:
            if args.model_type == "ramoe":
                for temp in args.temperature_grid:
                    logger.info("Fitting %s with K=%d, lambda=%.3f, temperature=%.3f", args.model_type, k, lambda_, temp)
                    try:
                        scores = _run_single(
                            model_type=args.model_type,
                            X_train_beh=X_train_beh,
                            X_train_full=X_train_full,
                            y_train=y_train,
                            X_val_beh=X_val_beh,
                            X_val_full=X_val_full,
                            y_val=y_val,
                            k=k,
                            lambda_=lambda_,
                            temperature=float(temp),
                            gamma=0.0,
                            max_iter=args.max_iter,
                            tol=args.tol,
                            random_state=args.random_state,
                        )
                        ablation_results.append(scores)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Ablation failed: %s", exc)
            else:
                for gamma in args.gamma_grid:
                    logger.info("Fitting %s with K=%d, lambda=%.3f, gamma=%.3f", args.model_type, k, lambda_, gamma)
                    try:
                        scores = _run_single(
                            model_type=args.model_type,
                            X_train_beh=X_train_beh,
                            X_train_full=X_train_full,
                            y_train=y_train,
                            X_val_beh=X_val_beh,
                            X_val_full=X_val_full,
                            y_val=y_val,
                            k=k,
                            lambda_=lambda_,
                            temperature=1.0,
                            gamma=float(gamma),
                            max_iter=args.max_iter,
                            tol=args.tol,
                            random_state=args.random_state,
                        )
                        ablation_results.append(scores)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Ablation failed: %s", exc)

    results_df = pd.DataFrame(ablation_results)
    results_path = TABLE_DIR / "rajc_ablation.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved ablation results to %s", results_path)


if __name__ == "__main__":
    main()
