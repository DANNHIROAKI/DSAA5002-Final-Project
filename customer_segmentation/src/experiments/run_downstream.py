"""Downstream response prediction with cluster IDs as additional features (leakage-free).

We compare several feature configurations for a supervised response model:

1) Base features only (X_full).
2) Base + KMeans cluster IDs (clusters learned on X_beh).
3) Base + GMM cluster IDs (clusters learned on X_beh).
4) Base + RAJC/RAMoE cluster IDs (clusters learned on X_beh with labels via the proposed method).

Optionally, we also report the *direct* RAJC/RAMoE probability predictions as a stand-alone predictor (no extra classifier).

Leakage-free protocol
---------------------
- Split raw rows into train/val/test.
- Fit preprocessing transformer on train only.
- Derive clusters on train; apply to val/test using behaviour features only.
- For models that output probabilities, select a decision threshold on val (max F1), then report test metrics.

Outputs
-------
Writes ``customer_segmentation/outputs/tables/downstream_metrics.csv``.
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
from sklearn.linear_model import LogisticRegression

from customer_segmentation.src.data.features import (
    add_response_label,
    assemble_feature_table,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data, train_val_test_split
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
    val_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_labels: pd.Series | np.ndarray,
    val_labels: pd.Series | np.ndarray,
    test_labels: pd.Series | np.ndarray,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append one-hot cluster indicators with aligned columns."""
    tr = _one_hot_clusters(train_labels, index=train_x.index, prefix=prefix)
    va = _one_hot_clusters(val_labels, index=val_x.index, prefix=prefix)
    te = _one_hot_clusters(test_labels, index=test_x.index, prefix=prefix)

    # Align val/test columns to train dummies
    va = va.reindex(columns=tr.columns, fill_value=0)
    te = te.reindex(columns=tr.columns, fill_value=0)

    return (
        pd.concat([train_x, tr], axis=1),
        pd.concat([val_x, va], axis=1),
        pd.concat([test_x, te], axis=1),
    )


def _train_logreg(train_x: pd.DataFrame, train_y: pd.Series) -> LogisticRegression:
    clf = LogisticRegression(max_iter=800, class_weight="balanced")
    clf.fit(train_x, train_y)
    return clf


def _evaluate_probs(y_true: pd.Series, probas: pd.Series, threshold: float) -> Dict[str, float]:
    preds = (probas >= float(threshold)).astype(int)
    metrics: Dict[str, float] = {
        **prediction_eval.compute_classification_metrics(y_true, preds, probas),
        "lift_top20": compute_lift(y_true, probas, top_frac=0.2),
        "threshold": float(threshold),
    }
    return metrics


def _load_rajc_config(logger, *, override_model_type: str = "ramoe") -> RAJCConfig:
    """Load RAJCConfig from YAML with safe defaults.

    This is shared with `run_rajc`, but duplicated here to keep the script standalone.
    """
    if not DEFAULT_RAJC_CONFIG_PATH.is_file():
        logger.warning(
            "RAJC config file not found at %s, using default RAJCConfig().",
            DEFAULT_RAJC_CONFIG_PATH,
        )
        cfg = RAJCConfig()
        cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg

    try:
        cfg_dict = yaml.safe_load(DEFAULT_RAJC_CONFIG_PATH.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {}) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to parse RAJC YAML (%s). Using defaults.", exc)
        cfg = RAJCConfig()
        cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg

    valid_fields = {f.name for f in fields(RAJCConfig)}
    kwargs: Dict[str, Any] = {k: v for k, v in rajc_cfg.items() if k in valid_fields}

    # Backward-compatible mapping
    if "lambda" in rajc_cfg and "lambda_" not in kwargs:
        kwargs["lambda_"] = float(rajc_cfg.get("lambda"))

    # Always use new method for downstream cluster features unless user edits YAML
    kwargs["model_type"] = str(rajc_cfg.get("model_type", override_model_type)).strip().lower()
    if override_model_type is not None:
        kwargs["model_type"] = override_model_type

    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    try:
        return RAJCConfig(**kwargs)
    except TypeError:
        cfg = RAJCConfig()
        cfg.model_type = override_model_type  # type: ignore[assignment]
        return cfg


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream response prediction using cluster IDs as features (leakage-free).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used as test set (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data used as validation set (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split (default: 42).",
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
    parser.add_argument(
        "--skip-direct-rajc",
        action="store_true",
        help="Skip direct RAJC/RAMoE probability predictions.",
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
            "to verify dataset placement.",
        )
        sys.exit(1)

    train_df, val_df, test_df = train_val_test_split(
        cleaned,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        stratify_col="campaign_response",
    )

    # Leakage-free feature transform: fit on train only
    X_train_full, y_train, transformer = assemble_feature_table(train_df, fit=True, label_mode="recent")
    X_val_full, y_val, _ = assemble_feature_table(val_df, transformer=transformer, fit=False, label_mode="recent")
    X_test_full, y_test, _ = assemble_feature_table(test_df, transformer=transformer, fit=False, label_mode="recent")

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_val_beh, _ = split_behavior_and_response_features(X_val_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)

    logger.info(
        "Downstream split sizes: train=%d, val=%d, test=%d | dims: full=%d, beh=%d",
        len(train_df),
        len(val_df),
        len(test_df),
        X_train_full.shape[1],
        X_train_beh.shape[1],
    )

    results: list[Dict[str, Any]] = []

    # 1) Base model
    logger.info("Training base logistic regression without cluster IDs")
    base_clf = _train_logreg(X_train_full, y_train)
    prob_val = pd.Series(base_clf.predict_proba(X_val_full)[:, 1], index=X_val_full.index)
    thr = prediction_eval.choose_threshold(y_val, prob_val, metric="f1")
    prob_test = pd.Series(base_clf.predict_proba(X_test_full)[:, 1], index=X_test_full.index)
    base_metrics = _evaluate_probs(y_test, prob_test, thr)
    base_metrics["model"] = "base"
    results.append(base_metrics)

    # 2) KMeans cluster IDs (cluster on behaviour features)
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
    k_tr = kmeans_model.predict(X_train_beh)
    k_va = kmeans_model.predict(X_val_beh)
    k_te = kmeans_model.predict(X_test_beh)

    Xtr, Xva, Xte = _add_cluster_features(X_train_full, X_val_full, X_test_full, k_tr, k_va, k_te, prefix="kmeans")
    clf = _train_logreg(Xtr, y_train)
    prob_val = pd.Series(clf.predict_proba(Xva)[:, 1], index=Xva.index)
    thr = prediction_eval.choose_threshold(y_val, prob_val, metric="f1")
    prob_test = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)
    met = _evaluate_probs(y_test, prob_test, thr)
    met["model"] = "base+kmeansid"
    results.append(met)

    # 3) GMM cluster IDs (optional)
    if not args.skip_gmm:
        logger.info("Training logistic regression with GMM cluster IDs")
        try:
            gmm_model = GMMBaseline(GMMConfig())
            gmm_model.fit(X_train_beh)
            g_tr = gmm_model.predict(X_train_beh)
            g_va = gmm_model.predict(X_val_beh)
            g_te = gmm_model.predict(X_test_beh)

            Xtr, Xva, Xte = _add_cluster_features(X_train_full, X_val_full, X_test_full, g_tr, g_va, g_te, prefix="gmm")
            clf = _train_logreg(Xtr, y_train)
            prob_val = pd.Series(clf.predict_proba(Xva)[:, 1], index=Xva.index)
            thr = prediction_eval.choose_threshold(y_val, prob_val, metric="f1")
            prob_test = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)
            met = _evaluate_probs(y_test, prob_test, thr)
            met["model"] = "base+gmmid"
            results.append(met)
        except Exception as exc:  # pragma: no cover
            logger.exception("GMM downstream pipeline failed: %s", exc)

    # 4) RAJC/RAMoE cluster IDs (optional)
    if not args.skip_rajc:
        logger.info("Training logistic regression with RAJC/RAMoE cluster IDs")
        try:
            rajc_config = _load_rajc_config(logger, override_model_type="ramoe")
            rajc_model = RAJCModel(rajc_config)

            # RAMoE uses full_features to train experts; pass it for all modes.
            rajc_model.fit(X_train_beh, y_train, full_features=X_train_full)

            r_tr = pd.Series(rajc_model.predict_clusters(X_train_beh), index=X_train_beh.index)
            r_va = pd.Series(rajc_model.predict_clusters(X_val_beh), index=X_val_beh.index)
            r_te = pd.Series(rajc_model.predict_clusters(X_test_beh), index=X_test_beh.index)

            Xtr, Xva, Xte = _add_cluster_features(X_train_full, X_val_full, X_test_full, r_tr, r_va, r_te, prefix="rajc")
            clf = _train_logreg(Xtr, y_train)
            prob_val = pd.Series(clf.predict_proba(Xva)[:, 1], index=Xva.index)
            thr = prediction_eval.choose_threshold(y_val, prob_val, metric="f1")
            prob_test = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)
            met = _evaluate_probs(y_test, prob_test, thr)
            met["model"] = "base+rajcid"
            results.append(met)
        except Exception as exc:  # pragma: no cover
            logger.exception("RAJC/RAMoE downstream pipeline failed: %s", exc)

    # 5) Direct RAJC/RAMoE probability predictions (optional)
    if not args.skip_direct_rajc and not args.skip_rajc:
        logger.info("Evaluating direct RAJC/RAMoE probability predictions")
        try:
            # Reuse fitted rajc_model if available above; otherwise fit here.
            # If the above branch failed, this will refit.
            if "rajc_model" not in locals():
                rajc_config = _load_rajc_config(logger, override_model_type="ramoe")
                rajc_model = RAJCModel(rajc_config)
                rajc_model.fit(X_train_beh, y_train, full_features=X_train_full)

            prob_val = pd.Series(
                rajc_model.predict_response(X_val_beh, full_features=X_val_full),
                index=X_val_beh.index,
            )
            thr = prediction_eval.choose_threshold(y_val, prob_val, metric="f1")
            prob_test = pd.Series(
                rajc_model.predict_response(X_test_beh, full_features=X_test_full),
                index=X_test_beh.index,
            )
            met = _evaluate_probs(y_test, prob_test, thr)
            met["model"] = "rajc_direct"
            results.append(met)
        except Exception as exc:  # pragma: no cover
            logger.exception("Direct RAJC/RAMoE evaluation failed: %s", exc)

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "downstream_metrics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved downstream prediction metrics to %s", results_path)


if __name__ == "__main__":
    main()
