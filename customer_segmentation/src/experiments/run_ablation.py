"""Ablation study runner under a strict hold-out protocol.

This script is designed to quickly stress-test the proposed method and
understand which components matter most.

Protocol
--------
- Use y = Response (label_mode='recent').
- Split raw rows into train/val/test BEFORE preprocessing.
- Fit transformers on train only.
- Evaluate each setting on the validation split (val).

What can be ablated
-------------------
The ablation grid is configurable via CLI.
The default grid is intentionally small so the script remains runnable on a laptop.

Outputs
-------
Writes ``customer_segmentation/outputs/tables/rajc_ablation.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

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

# ---------------------------------------------------------------------------
# Default grids (small by design)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_TYPE = "ramoe"
DEFAULT_EXPERT_TYPE_GRID: List[str] = ["hgbdt"]
DEFAULT_LAMBDA_GRID: List[float] = [0.0, 0.1, 0.3, 1.0, 3.0]
DEFAULT_TEMP_GRID: List[float] = [0.5, 1.0, 2.0]
DEFAULT_GAMMA_GRID: List[float] = [0.0, 0.1, 0.3]
DEFAULT_CLUSTER_GRID: List[int] = [3, 4, 5, 6]
DEFAULT_HYBRID_ALPHA_GRID: List[float] = [0.0, 0.2]
DEFAULT_BUDGET_REWEIGHT_ALPHA_GRID: List[float] = [0.0]


def _ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_val_split(
    *,
    test_size: float,
    val_size: float,
    random_state: int,
):
    """Load data, split, fit transformer on train, transform val/test."""
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

    X_train_full, y_train, transformer = assemble_feature_table(train_df, fit=True, label_mode="recent")
    X_val_full, y_val, _ = assemble_feature_table(val_df, transformer=transformer, fit=False, label_mode="recent")
    X_test_full, y_test, _ = assemble_feature_table(test_df, transformer=transformer, fit=False, label_mode="recent")

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_val_beh, _ = split_behavior_and_response_features(X_val_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)

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
    )


def _run_single(
    *,
    cfg: RAJCConfig,
    X_train_beh: pd.DataFrame,
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    X_val_beh: pd.DataFrame,
    X_val_full: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """Fit a RAJC/RAMoE model and evaluate on validation split."""
    model = RAJCModel(cfg)
    model.fit(X_train_beh, y_train, full_features=X_train_full)

    clusters_val = pd.Series(model.predict_clusters(X_val_beh), index=X_val_beh.index, name="cluster")
    probas_val = pd.Series(model.predict_response(X_val_beh, full_features=X_val_full), index=X_val_beh.index)

    # clustering/segmentation
    scores: Dict[str, Any] = clustering_eval.compute_scores(X_val_beh, clusters_val)
    rates = segmentation_eval.cluster_response_rates(clusters_val, y_val)
    scores["response_rate_variance"] = segmentation_eval.response_rate_variance(rates)
    scores["response_rates"] = rates.to_dict()

    # prediction (threshold fixed at 0.5 for ablation comparability)
    preds_val = (probas_val >= 0.5).astype(int)
    scores.update(prediction_eval.compute_classification_metrics(y_val, preds_val, probas_val))
    scores["lift_top20"] = compute_lift(y_val, probas_val, top_frac=0.2)

    # config trace
    scores.update(
        {
            "split": "val",
            "label": "rajc_ablation",
            "model_type": cfg.model_type,
            "expert_type": cfg.expert_type,
            "gating_type": cfg.gating_type,
            "n_clusters": cfg.n_clusters,
            "lambda": cfg.lambda_,
            "gamma": cfg.gamma,
            "temperature": cfg.temperature,
            "use_global_expert": cfg.use_global_expert,
            "hybrid_alpha": cfg.hybrid_alpha,
            "budget_reweight_alpha": cfg.budget_reweight_alpha,
        }
    )

    return scores


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study over K/lambda/temperature/expert_type (RAMoE) or K/lambda/gamma (constant_prob)."
        )
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["ramoe", "logreg", "constant_prob"],
        help=f"Model type for ablation (default: {DEFAULT_MODEL_TYPE}).",
    )
    parser.add_argument(
        "--expert-type-grid",
        type=str,
        nargs="*",
        default=DEFAULT_EXPERT_TYPE_GRID,
        choices=["logreg", "hgbdt"],
        help=f"Expert type grid (default: {DEFAULT_EXPERT_TYPE_GRID}).",
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
        "--hybrid-alpha-grid",
        type=float,
        nargs="*",
        default=DEFAULT_HYBRID_ALPHA_GRID,
        help=f"Grid of hybrid_alpha when use_global_expert=True (default: {DEFAULT_HYBRID_ALPHA_GRID}).",
    )
    parser.add_argument(
        "--budget-reweight-alpha-grid",
        type=float,
        nargs="*",
        default=DEFAULT_BUDGET_REWEIGHT_ALPHA_GRID,
        help=(
            f"Grid of budget_reweight_alpha (default: {DEFAULT_BUDGET_REWEIGHT_ALPHA_GRID}). "
            "0.0 disables budget-aware training."
        ),
    )
    parser.add_argument(
        "--use-global-expert",
        action="store_true",
        default=False,
        help="If set, include a global expert and ablate hybrid_alpha.",
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
        help="Max iterations for RAJC/RAMoE alternating optimisation (default: 30).",
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
            X_train_full,
            y_train,
            X_val_full,
            y_val,
            _X_test_full,
            _y_test,
            X_train_beh,
            X_val_beh,
            _X_test_beh,
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
        "Running ablation: model_type=%s | K=%s | lambda=%s",
        args.model_type,
        args.cluster_grid,
        args.lambda_grid,
    )

    results: list[Dict[str, Any]] = []

    for k in args.cluster_grid:
        for lambda_ in args.lambda_grid:
            # constant_prob does not use experts/temperature in a meaningful way
            if args.model_type == "constant_prob":
                for gamma in args.gamma_grid:
                    cfg = RAJCConfig(
                        model_type="constant_prob",
                        n_clusters=int(k),
                        lambda_=float(lambda_),
                        gamma=float(gamma),
                        max_iter=int(args.max_iter),
                        tol=float(args.tol),
                        random_state=int(args.random_state),
                    )
                    logger.info(
                        "[CP++] K=%d lambda=%.3f gamma=%.3f",
                        cfg.n_clusters,
                        cfg.lambda_,
                        cfg.gamma,
                    )
                    try:
                        results.append(
                            _run_single(
                                cfg=cfg,
                                X_train_beh=X_train_beh,
                                X_train_full=X_train_full,
                                y_train=y_train,
                                X_val_beh=X_val_beh,
                                X_val_full=X_val_full,
                                y_val=y_val,
                            )
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Ablation failed: %s", exc)
                continue

            # RAMoE / logreg modes
            for expert_type in args.expert_type_grid:
                for temp in args.temperature_grid:
                    hybrid_grid = args.hybrid_alpha_grid if args.use_global_expert else [0.0]
                    for ha in hybrid_grid:
                        for bra in args.budget_reweight_alpha_grid:
                            cfg = RAJCConfig(
                                model_type=str(args.model_type),
                                expert_type=str(expert_type),
                                n_clusters=int(k),
                                lambda_=float(lambda_),
                                temperature=float(temp),
                                max_iter=int(args.max_iter),
                                tol=float(args.tol),
                                random_state=int(args.random_state),
                                use_global_expert=bool(args.use_global_expert),
                                hybrid_alpha=float(ha),
                                budget_reweight_alpha=float(bra),
                            )
                            logger.info(
                                "[%s] expert=%s K=%d lambda=%.3f temp=%.3f global=%s ha=%.2f bra=%.2f",
                                cfg.model_type,
                                cfg.expert_type,
                                cfg.n_clusters,
                                cfg.lambda_,
                                cfg.temperature,
                                cfg.use_global_expert,
                                cfg.hybrid_alpha,
                                cfg.budget_reweight_alpha,
                            )
                            try:
                                results.append(
                                    _run_single(
                                        cfg=cfg,
                                        X_train_beh=X_train_beh,
                                        X_train_full=X_train_full,
                                        y_train=y_train,
                                        X_val_beh=X_val_beh,
                                        X_val_full=X_val_full,
                                        y_val=y_val,
                                    )
                                )
                            except Exception as exc:  # pragma: no cover
                                logger.exception("Ablation failed: %s", exc)

    df = pd.DataFrame(results)
    out_path = TABLE_DIR / "rajc_ablation.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved ablation results to %s", out_path)


if __name__ == "__main__":
    main()
