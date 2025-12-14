"""Run the full DSAA5002 customer segmentation experiment suite.

This launcher orchestrates:
1) Baseline segmentations (KMeans/GMM/Cluster-then-Predict)
2) Proposed RAJC family (default: RAMoE / HyRAMoE)
3) Downstream response prediction with cluster IDs
4) Ablation study (small default grid)
5) Report-ready figures (clustering / profiling / prediction curves)

Design goals
------------
- Robust to current working directory: you can run from repo root or from inside
  ``customer_segmentation``.
- Leakage-safe evaluation: the underlying experiment scripts split rows first and
  fit preprocessing on train only.
- Reproducible: global seeding and a single consolidated log file.

Usage
-----
From the repository root:

    python customer_segmentation/run_all_experiments.py

Or from inside the folder:

    cd customer_segmentation
    python run_all_experiments.py

The outputs are written to:
- ``customer_segmentation/outputs/tables``
- ``customer_segmentation/outputs/figures``
- ``customer_segmentation/outputs/logs/run_all.log``
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make imports & paths robust to the current working directory.
# ---------------------------------------------------------------------------

import os
import sys
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

# Ensure repo root is importable (needed when running from inside customer_segmentation/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from customer_segmentation.src.data.check_data import dataset_status
from customer_segmentation.src.data.features import (
    DEFAULT_LABEL_COL,
    add_response_label,
    assemble_feature_table,
    build_rfm_features,
    split_behavior_and_response_features,
)
from customer_segmentation.src.data.load import load_raw_data
from customer_segmentation.src.data.preprocess import clean_data
from customer_segmentation.src.evaluation import segmentation as segmentation_eval
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel
from customer_segmentation.src.utils import configure_logging, set_global_seed
from customer_segmentation.src.experiments import (
    run_ablation,
    run_baselines,
    run_downstream,
    run_rajc,
)
from customer_segmentation.src.visualization import (
    # baselines
    plot_elbow_curve,
    # clustering
    plot_cluster_centers,
    plot_pca_scatter,
    plot_silhouette_distribution,
    # prediction curves
    plot_calibration_curve,
    plot_lift_curve,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_sweep,
    # RAMoE diagnostics
    plot_assignment_entropy,
    plot_assignment_maxprob,
    # profiling
    plot_age_income_kde,
    plot_channel_mix,
    plot_cluster_budget_allocation_curve,
    plot_cluster_lift_bars,
    plot_cluster_size_and_response,
    plot_income_vs_spent,
    plot_response_rates,
    plot_rfm_boxplots,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
LOG_DIR = OUTPUT_DIR / "logs"

FIG_BASELINES_DIR = FIG_DIR / "baselines"
FIG_RAJC_DIR = FIG_DIR / "rajc"
FIG_PROFILE_DIR = FIG_DIR / "profiles"
FIG_PRED_DIR = FIG_DIR / "prediction"

DEFAULT_BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baselines.yaml"
DEFAULT_RAJC_CONFIG = PROJECT_ROOT / "configs" / "rajc.yaml"


def _ensure_dirs() -> None:
    for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, LOG_DIR, FIG_BASELINES_DIR, FIG_RAJC_DIR, FIG_PROFILE_DIR, FIG_PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


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


def _load_rajc_config(config_path: Path, logger) -> RAJCConfig:
    """Load RAJCConfig from YAML (backward/forward compatible)."""
    if not config_path.is_file():
        logger.warning("RAJC config not found at %s; using defaults.", config_path)
        return RAJCConfig()

    try:
        cfg_dict = yaml.safe_load(config_path.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {}) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read RAJC YAML (%s); using defaults.", exc)
        return RAJCConfig()

    # Allow both legacy (lambda) and new (lambda_) key.
    valid_fields = {f.name for f in fields(RAJCConfig)}
    kwargs: Dict[str, Any] = {k: v for k, v in rajc_cfg.items() if k in valid_fields}

    if "lambda" in rajc_cfg and "lambda_" not in kwargs:
        kwargs["lambda_"] = float(rajc_cfg.get("lambda"))

    # Nested blocks (optional): keep compatibility with older configs.
    lr_cfg = rajc_cfg.get("logistic_regression", {}) or {}
    if "C" in lr_cfg and "logreg_C" not in kwargs:
        kwargs["logreg_C"] = float(lr_cfg.get("C"))
    if "max_iter" in lr_cfg and "logreg_max_iter" not in kwargs:
        kwargs["logreg_max_iter"] = int(lr_cfg.get("max_iter"))
    if "solver" in lr_cfg and "logreg_solver" not in kwargs:
        kwargs["logreg_solver"] = str(lr_cfg.get("solver"))

    hgbdt_cfg = rajc_cfg.get("hgbdt", {}) or {}
    if "max_depth" in hgbdt_cfg and "hgbdt_max_depth" not in kwargs:
        kwargs["hgbdt_max_depth"] = int(hgbdt_cfg.get("max_depth"))
    if "learning_rate" in hgbdt_cfg and "hgbdt_learning_rate" not in kwargs:
        kwargs["hgbdt_learning_rate"] = float(hgbdt_cfg.get("learning_rate"))
    if "max_iter" in hgbdt_cfg and "hgbdt_max_iter" not in kwargs:
        kwargs["hgbdt_max_iter"] = int(hgbdt_cfg.get("max_iter"))

    # Booleans
    if "use_global_expert" in rajc_cfg:
        kwargs["use_global_expert"] = _as_bool(rajc_cfg.get("use_global_expert"), default=True)
    if "hgbdt_early_stopping" in rajc_cfg:
        kwargs["hgbdt_early_stopping"] = _as_bool(rajc_cfg.get("hgbdt_early_stopping"), default=True)

    # Filter again to dataclass fields
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    try:
        return RAJCConfig(**kwargs)
    except TypeError as exc:
        logger.warning("Invalid RAJC YAML fields (%s); falling back to defaults.", exc)
        return RAJCConfig()


# ---------------------------------------------------------------------------
# Data helpers for plots
# ---------------------------------------------------------------------------


def _load_core_data_for_plots(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw data and build full + behaviour features for plotting.

    Notes
    -----
    For *plots only* we fit a transformer on the full dataset for convenience.
    All *reported metrics* are produced by the experiment scripts which use
    leakage-free splitting.
    """
    raw = load_raw_data(data_dir=data_dir)
    cleaned = clean_data(raw)
    cleaned = add_response_label(cleaned, label_col=DEFAULT_LABEL_COL, mode="recent")

    X_full, y, transformer = assemble_feature_table(cleaned, label_col=DEFAULT_LABEL_COL, label_mode="recent", fit=True)
    X_beh, _ = split_behavior_and_response_features(X_full, transformer)

    return cleaned, X_full, X_beh, y


def _align_rajc_assignments(logger, idx: pd.Index) -> Optional[pd.Series]:
    """Load RAJC/RAMoE full-dataset assignments and align to a given index."""
    path = TABLE_DIR / "rajc_assignments.csv"
    if not path.is_file():
        logger.warning("Missing %s; run the RAJC step first.", path)
        return None

    df = pd.read_csv(path, index_col=0)
    if "cluster" not in df.columns:
        # In case it was saved as a Series with name "cluster"
        if df.shape[1] == 1:
            df.columns = ["cluster"]
        else:
            logger.warning("Unexpected columns in %s: %s", path, df.columns.tolist())
            return None

    clusters = df["cluster"]
    # Reindex to the feature table index (fills missing with NaN)
    clusters = clusters.reindex(idx)

    missing = clusters.isna().sum()
    if missing > 0:
        logger.warning("%d indices missing in rajc_assignments.csv after alignment.", int(missing))

    return clusters.dropna().astype(int)


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------


def _generate_baseline_elbow_plot(cleaned: pd.DataFrame) -> None:
    """RFM elbow curve for KMeans (diagnostic)."""
    rfm_df = build_rfm_features(cleaned).rename(columns=str.lower)
    rfm = rfm_df[["recency", "frequency", "monetary"]].copy().fillna(0.0)

    ks = list(range(2, 11))
    inertias: list[float] = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm)
        inertias.append(float(km.inertia_))

    plot_elbow_curve(ks, inertias, save_path=FIG_BASELINES_DIR / "elbow_rfm_kmeans.png")


def _generate_rajc_clustering_plots(X_beh: pd.DataFrame, clusters: pd.Series) -> None:
    """Cluster-center + PCA + silhouette for RAMoE segmentation."""
    # centers in behaviour-feature space
    centers = X_beh.groupby(clusters).mean()
    plot_cluster_centers(centers, title="RAMoE cluster centers (behaviour features)", save_path=FIG_RAJC_DIR / "rajc_cluster_centers.png")

    # PCA scatter
    plot_pca_scatter(X_beh, clusters, title="RAMoE clusters (PCA)", save_path=FIG_RAJC_DIR / "rajc_pca_clusters.png")

    # silhouette
    plot_silhouette_distribution(X_beh, clusters, title="RAMoE silhouette distribution", save_path=FIG_RAJC_DIR / "rajc_silhouette_distribution.png")


def _generate_profile_plots(cleaned: pd.DataFrame, clusters: pd.Series, y: pd.Series) -> None:
    """Business-oriented profiling for clusters."""
    # Derived RFM/spending for plotting convenience
    prof = build_rfm_features(cleaned)

    # Response rate summary
    response_rates = segmentation_eval.cluster_response_rates(clusters, y)
    global_rate = float(pd.to_numeric(y, errors="coerce").mean())
    cluster_sizes = segmentation_eval.cluster_size_summary(clusters)

    plot_response_rates(
        response_rates,
        title="Cluster response rates (y=Response)",
        global_rate=global_rate,
        save_path=FIG_PROFILE_DIR / "rajc_response_rates.png",
    )

    plot_cluster_size_and_response(
        cluster_sizes,
        response_rates,
        title="Cluster size and response rate",
        save_path=FIG_PROFILE_DIR / "rajc_size_and_response.png",
    )

    # Segment-level lift and budget allocation curve (coarse campaign planning)
    plot_cluster_lift_bars(
        clusters,
        y,
        title="Cluster lift vs global (y=Response)",
        save_path=FIG_PROFILE_DIR / "rajc_cluster_lift.png",
    )

    plot_cluster_budget_allocation_curve(
        clusters,
        y,
        title="Cluster-based budget allocation (expected lift)",
        fractions=(0.05, 0.1, 0.2, 0.3, 0.5),
        save_path=FIG_PROFILE_DIR / "rajc_cluster_budget_allocation.png",
    )

    # RFM distribution by cluster
    plot_rfm_boxplots(
        prof,
        clusters,
        title="RFM distribution by cluster",
        save_path=FIG_PROFILE_DIR / "rajc_rfm_boxplots.png",
    )

    # Channel mix
    plot_channel_mix(
        prof,
        clusters,
        save_path=FIG_PROFILE_DIR / "rajc_channel_mix.png",
    )

    # Income vs spent
    plot_income_vs_spent(
        prof,
        clusters,
        save_path=FIG_PROFILE_DIR / "rajc_income_vs_monetary.png",
    )

    # Age vs income KDE + density
    plot_age_income_kde(
        prof,
        clusters,
        save_path=FIG_PROFILE_DIR / "rajc_age_income_kde.png",
    )

    # density plot (hexbin)
    plot_age_income_kde(
        prof,
        clusters,
        kind="hexbin",
        save_path=FIG_PROFILE_DIR / "rajc_age_income_density.png",
    )


def _one_hot(labels: np.ndarray, prefix: str = "cluster") -> pd.DataFrame:
    s = pd.Series(labels, name="cluster")
    return pd.get_dummies(s, prefix=prefix, dtype=int)


def _prepare_leak_free_train_test(
    *,
    data_dir: Path,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simple leakage-free train/test split for curve generation."""
    raw = load_raw_data(data_dir=data_dir)
    cleaned = clean_data(raw)
    cleaned = add_response_label(cleaned, label_col=DEFAULT_LABEL_COL, mode="recent")

    train_df, test_df = train_test_split(
        cleaned,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=cleaned[DEFAULT_LABEL_COL],
    )

    X_train_full, y_train, transformer = assemble_feature_table(train_df, label_col=DEFAULT_LABEL_COL, label_mode="recent", fit=True)
    X_test_full, y_test, _ = assemble_feature_table(test_df, label_col=DEFAULT_LABEL_COL, label_mode="recent", transformer=transformer, fit=False)

    X_train_beh, _ = split_behavior_and_response_features(X_train_full, transformer)
    X_test_beh, _ = split_behavior_and_response_features(X_test_full, transformer)

    return X_train_full, X_test_full, y_train, y_test, X_train_beh, X_test_beh


def _generate_downstream_curves(logger, data_dir: Path, *, test_size: float = 0.2) -> None:
    """Generate ROC/PR/Calibration/Lift/Threshold plots for key predictors."""
    X_train_full, X_test_full, y_train, y_test, X_train_beh, X_test_beh = _prepare_leak_free_train_test(
        data_dir=data_dir,
        test_size=test_size,
        random_state=42,
    )

    # Base logistic regression
    base_clf = LogisticRegression(max_iter=800, class_weight="balanced")
    base_clf.fit(X_train_full, y_train)
    base_prob = base_clf.predict_proba(X_test_full)[:, 1]

    # KMeans cluster IDs + logistic regression
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_train_beh)
    tr_k = km.predict(X_train_beh)
    te_k = km.predict(X_test_beh)

    tr_k_oh = _one_hot(tr_k, prefix="kmeans")
    te_k_oh = _one_hot(te_k, prefix="kmeans").reindex(columns=tr_k_oh.columns, fill_value=0)

    Xtr_k = pd.concat([X_train_full.reset_index(drop=True), tr_k_oh.reset_index(drop=True)], axis=1)
    Xte_k = pd.concat([X_test_full.reset_index(drop=True), te_k_oh.reset_index(drop=True)], axis=1)

    kmeans_clf = LogisticRegression(max_iter=800, class_weight="balanced")
    kmeans_clf.fit(Xtr_k, y_train.reset_index(drop=True))
    kmeans_prob = kmeans_clf.predict_proba(Xte_k)[:, 1]

    # RAMoE cluster IDs + logistic regression, and RAMoE direct probabilities
    rajc_cfg = _load_rajc_config(DEFAULT_RAJC_CONFIG, logger)
    rajc_model = RAJCModel(rajc_cfg)
    rajc_model.fit(X_train_beh, y_train, full_features=X_train_full)

    tr_r = rajc_model.predict_clusters(X_train_beh)
    te_r = rajc_model.predict_clusters(X_test_beh)

    tr_r_oh = _one_hot(tr_r, prefix="rajc")
    te_r_oh = _one_hot(te_r, prefix="rajc").reindex(columns=tr_r_oh.columns, fill_value=0)

    Xtr_r = pd.concat([X_train_full.reset_index(drop=True), tr_r_oh.reset_index(drop=True)], axis=1)
    Xte_r = pd.concat([X_test_full.reset_index(drop=True), te_r_oh.reset_index(drop=True)], axis=1)

    rajcid_clf = LogisticRegression(max_iter=800, class_weight="balanced")
    rajcid_clf.fit(Xtr_r, y_train.reset_index(drop=True))
    rajcid_prob = rajcid_clf.predict_proba(Xte_r)[:, 1]

    rajc_direct_prob = rajc_model.predict_response(X_test_beh, full_features=X_test_full)

    models = {
        "base": base_prob,
        "base+kmeansid": kmeans_prob,
        "base+rajcid": rajcid_prob,
        "rajc_direct": rajc_direct_prob,
    }

    # Save curves
    for name, prob in models.items():
        plot_roc_curve(y_test, prob, title=f"ROC – {name}", save_path=FIG_PRED_DIR / f"roc_{name}.png")
        plot_pr_curve(y_test, prob, title=f"PR – {name}", save_path=FIG_PRED_DIR / f"pr_{name}.png")
        plot_calibration_curve(
            y_test,
            prob,
            title=f"Calibration – {name}",
            save_path=FIG_PRED_DIR / f"calibration_{name}.png",
        )
        plot_lift_curve(y_test, prob, title=f"Lift – {name}", save_path=FIG_PRED_DIR / f"lift_{name}.png")
        plot_threshold_sweep(
            y_test,
            prob,
            title=f"Threshold sweep – {name}",
            save_path=FIG_PRED_DIR / f"threshold_{name}.png",
        )

    # RAMoE soft-assignment diagnostics (train responsibilities)
    if rajc_model.responsibilities_ is not None:
        plot_assignment_entropy(
            rajc_model.responsibilities_,
            title="RAMoE assignment entropy (train)",
            save_path=FIG_RAJC_DIR / "ramoe_assignment_entropy.png",
        )
        plot_assignment_maxprob(
            rajc_model.responsibilities_,
            title="RAMoE max assignment prob (train)",
            save_path=FIG_RAJC_DIR / "ramoe_assignment_maxprob.png",
        )


def _generate_all_plots(logger, *, data_dir: Path, downstream_test_size: float) -> None:
    """Generate the full figure set."""
    cleaned, X_full, X_beh, y = _load_core_data_for_plots(data_dir)

    # Baseline elbow
    try:
        _generate_baseline_elbow_plot(cleaned)
        logger.info("Saved baseline elbow plot.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to generate elbow plot: %s", exc)

    # RAJC cluster visuals + profiles (requires assignments file)
    clusters = _align_rajc_assignments(logger, idx=X_beh.index)
    if clusters is not None and not clusters.empty:
        try:
            _generate_rajc_clustering_plots(X_beh.loc[clusters.index], clusters)
            logger.info("Saved RAMoE clustering plots.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to generate RAMoE clustering plots: %s", exc)

        try:
            _generate_profile_plots(cleaned.loc[clusters.index], clusters, y.loc[clusters.index])
            logger.info("Saved cluster profile plots.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to generate profile plots: %s", exc)
    else:
        logger.warning("Skipping cluster profile plots (no RAJC assignments available).")

    # Downstream prediction curves
    try:
        _generate_downstream_curves(logger, data_dir=data_dir, test_size=float(downstream_test_size))
        logger.info("Saved downstream prediction curves.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to generate downstream curves: %s", exc)

    plt.close("all")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments (baselines + RAMoE + downstream + ablation).")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing marketing_campaign.csv (default: customer_segmentation/data/raw)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")

    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline experiments")
    parser.add_argument("--skip-rajc", action="store_true", help="Skip RAMoE/RAJC experiment")
    parser.add_argument("--skip-downstream", action="store_true", help="Skip downstream prediction experiment")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")

    parser.add_argument(
        "--downstream-test-size",
        type=float,
        default=0.2,
        help="Test fraction used for downstream curve plots only (default: 0.2)",
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue even if a step fails (default: stop on first failure)",
    )

    return parser.parse_args(argv)


def _check_dataset(logger, data_dir: Path) -> bool:
    # 1) Verify the user-provided location
    exists, csv_path = dataset_status(data_dir=data_dir)
    if not exists:
        logger.error("Dataset not found at %s", csv_path)
        logger.error(
            "Please place marketing_campaign.csv under customer_segmentation/data/raw/ (recommended) "
            "or pass --data-dir to point to the folder containing it."
        )
        return False
    logger.info("Found dataset at %s", csv_path)

    # 2) Ensure the *default* path exists too, because the experiment scripts
    #    load the dataset from customer_segmentation/data/raw by default.
    default_dir = PROJECT_ROOT / "data" / "raw"
    default_path = default_dir / csv_path.name

    try:
        if default_path.resolve() != csv_path.resolve():
            if not default_path.is_file():
                default_dir.mkdir(parents=True, exist_ok=True)
                # Try symlink first (fast), fallback to copy.
                try:
                    os.symlink(csv_path, default_path)
                    logger.info("Created symlink to dataset at %s", default_path)
                except Exception:
                    shutil.copy2(csv_path, default_path)
                    logger.info("Copied dataset to default location at %s", default_path)
            else:
                logger.info("Default dataset already exists at %s", default_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not mirror dataset into default location: %s", exc)
    return True


def _run_step(logger, name: str, fn, *, keep_going: bool) -> bool:
    logger.info("\n===== Running: %s =====", name)
    try:
        fn([])
        logger.info("✅ Completed: %s", name)
        return True
    except SystemExit as exc:
        logger.error("❌ %s exited with code %s", name, getattr(exc, "code", exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("❌ %s failed: %s", name, exc)

    if keep_going:
        logger.warning("Continuing because --keep-going is set.")
        return False

    raise RuntimeError(f"Step '{name}' failed")


def main(argv: Optional[Sequence[str]] = None) -> None:
    _ensure_dirs()

    # Make sure relative paths in the experiment scripts resolve correctly.
    os.chdir(REPO_ROOT)

    log_file = LOG_DIR / "run_all.log"
    logger = configure_logging(log_file=log_file, logger_name="run_all")

    args = _parse_args(argv)

    set_global_seed(args.seed)

    if not _check_dataset(logger, data_dir=args.data_dir):
        sys.exit(1)

    # ---------------------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------------------

    if not args.skip_baselines:
        _run_step(logger, "Baselines", run_baselines, keep_going=args.keep_going)

    if not args.skip_rajc:
        _run_step(logger, "RAJC / RAMoE", run_rajc, keep_going=args.keep_going)

    if not args.skip_downstream:
        _run_step(logger, "Downstream", run_downstream, keep_going=args.keep_going)

    if not args.skip_ablation:
        _run_step(logger, "Ablation", run_ablation, keep_going=args.keep_going)

    # ---------------------------------------------------------------------
    # Generate figures
    # ---------------------------------------------------------------------

    if not args.skip_plots:
        logger.info("\n===== Generating figures =====")
        try:
            _generate_all_plots(logger, data_dir=args.data_dir, downstream_test_size=args.downstream_test_size)
            logger.info("✅ Figures generated under %s", FIG_DIR)
        except Exception as exc:  # pragma: no cover
            logger.exception("Figure generation failed: %s", exc)
            if not args.keep_going:
                raise

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
