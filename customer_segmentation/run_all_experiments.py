"""One-stop launcher to run all experiments end-to-end.

This script will:

1. Check whether the marketing_campaign.csv dataset is present.
2. Set a global random seed for reproducibility.
3. Run all experimental pipelines in sequence:
   - Baseline clustering (4 baselines)
   - RAJC main experiment
   - Downstream response prediction with cluster IDs
   - RAJC ablation over lambda and K
4. Optionally generate key figures for the report:
   - elbow curve (baseline)
   - RAJC PCA / silhouette / centers
   - segment profiling plots (income-vs-spent, RFM boxplots, channel mix, response rate)
   - NEW (for the upgraded methodology):
       ROC / PR / Calibration / Lift / Threshold sweep for downstream prediction
     with leak-free preprocessing (fit transforms only on train split).

Results are written under ``customer_segmentation/outputs`` as defined
in the individual experiment scripts. Figures are additionally written
under ``customer_segmentation/outputs/figures`` by this launcher.

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence, Callable, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from customer_segmentation.src.data.check_data import dataset_status  # type: ignore[import]
from customer_segmentation.src.data.load import load_raw_data  # type: ignore[import]
from customer_segmentation.src.data.preprocess import clean_data  # type: ignore[import]
from customer_segmentation.src.data.features import (  # type: ignore[import]
    add_response_label,
    DEFAULT_LABEL_COL,
    assemble_feature_table,
    build_rfm_features,
    split_behavior_and_response_features,
)
from customer_segmentation.src.evaluation.segmentation import (  # type: ignore[import]
    cluster_response_rates,
)
from customer_segmentation.src.models.rajc import RAJCConfig, RAJCModel  # type: ignore[import]
from customer_segmentation.src.utils import configure_logging, set_global_seed  # type: ignore[import]
from customer_segmentation.src.experiments import (  # type: ignore[import]
    run_ablation,
    run_baselines,
    run_downstream,
    run_rajc,
)
from customer_segmentation.src.visualization import (  # type: ignore[import]
    # clustering visuals
    plot_elbow_curve,
    plot_pca_scatter,
    plot_silhouette_distribution,
    plot_cluster_centers,
    # profiling visuals
    plot_income_vs_spent,
    plot_rfm_boxplots,
    plot_channel_mix,
    plot_response_rates,
    plot_age_income_kde,
    plot_cluster_size_and_response,
    # NEW: prediction-evaluation curves
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_lift_curve,
    plot_threshold_sweep,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
GLOBAL_LOG = LOG_DIR / "run_all.log"

TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_BASELINES_DIR = FIG_DIR / "baselines"
FIG_RAJC_DIR = FIG_DIR / "rajc"
FIG_PROFILES_DIR = FIG_DIR / "profiles"
FIG_PRED_DIR = FIG_DIR / "prediction"  # NEW

DEFAULT_RAJC_CONFIG_PATH = PROJECT_ROOT / "configs" / "rajc.yaml"


# ---------------------------------------------------------------------------
# CLI parsing & logging
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the end-to-end launcher."""
    parser = argparse.ArgumentParser(
        description="Run all DSAA5002 final project experiments end-to-end.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help=(
            "Directory where marketing_campaign.csv is expected "
            "(default: customer_segmentation/data/raw)."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        default=False,
        help="Skip running baseline clustering experiments.",
    )
    parser.add_argument(
        "--skip-rajc",
        action="store_true",
        default=False,
        help="Skip running the main RAJC experiment.",
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        default=False,
        help="Skip running downstream prediction experiments.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        default=False,
        help="Skip RAJC ablation study over lambda and K.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        default=False,
        help="Skip figure generation step.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=False,
        help=(
            "If set, continue running remaining steps even if one step fails. "
            "By default, the script stops on the first failure."
        ),
    )
    # NEW: used for leak-free downstream curves
    parser.add_argument(
        "--downstream-test-size",
        type=float,
        default=0.2,
        help="Test fraction for downstream curves (default: 0.2).",
    )
    return parser.parse_args(argv)


def _configure_global_logging() -> logging.Logger:
    """Configure a global logger for the entire run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        level=logging.INFO,
        log_file=GLOBAL_LOG,
        logger_name="run_all",
    )
    logger.info("Global log file: %s", GLOBAL_LOG)
    return logger


def _check_dataset(data_dir: Path, logger: logging.Logger) -> None:
    """Verify that the dataset is present; exit with error if not."""
    exists, csv_path = dataset_status(data_dir)
    if not exists:
        logger.error("Dataset not found at %s", csv_path)
        logger.error(
            "Please place `marketing_campaign.csv` from the Kaggle "
            "Customer Personality Analysis dataset in this location "
            "and rerun the script."
        )
        sys.exit(1)
    logger.info("✅ Found dataset at %s", csv_path)


def _run_step(
    name: str,
    func: Callable[[], None],
    logger: logging.Logger,
    stop_on_error: bool,
) -> None:
    """Run a single experiment step with logging and error handling."""
    logger.info("=" * 80)
    logger.info("Starting step: %s", name)
    logger.info("=" * 80)
    try:
        func()
        logger.info("Finished step: %s", name)
    except SystemExit:
        logger.exception("Step '%s' aborted via SystemExit.", name)
        if stop_on_error:
            raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Step '%s' failed with error: %s", name, exc)
        if stop_on_error:
            raise


# ---------------------------------------------------------------------------
# Plot generation helpers
# ---------------------------------------------------------------------------


def _ensure_figure_dirs() -> None:
    """Ensure figure output directories exist."""
    for d in (FIG_DIR, FIG_BASELINES_DIR, FIG_RAJC_DIR, FIG_PROFILES_DIR, FIG_PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _load_core_data_for_plots(
    data_dir: Path, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load raw/cleaned data and assemble (leaky) feature/label table for clustering/profile plots.

    Notes
    -----
    This is used for cluster/profile visualization only.
    For downstream prediction curves, we use a leak-free split where
    preprocessing is fitted only on the training subset (see _generate_downstream_curves()).
    """
    logger.info("Loading data for clustering/profile plotting.")
    raw = load_raw_data(data_dir=data_dir, parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features, responses, _ = assemble_feature_table(cleaned)
    return cleaned, features, responses


def _align_rajc_assignments(
    features: pd.DataFrame,
    logger: logging.Logger,
) -> pd.Series:
    """Load RAJC assignments and align index with the feature table.

    If index mismatch happens, fill missing with the most frequent cluster
    to avoid crashing figures (plotting-only fix).
    """
    assignments_path = TABLE_DIR / "rajc_assignments.csv"
    if not assignments_path.is_file():
        raise FileNotFoundError(assignments_path)

    raw_assignments = pd.read_csv(assignments_path, index_col=0)["cluster"]

    rajc_assignments = raw_assignments.reindex(features.index)

    if rajc_assignments.isna().any():
        n_missing = int(rajc_assignments.isna().sum())
        logger.warning(
            "RAJC assignments missing for %d samples; filling them with the "
            "most frequent cluster for plotting only.",
            n_missing,
        )
        if rajc_assignments.notna().any():
            most_freq = int(rajc_assignments.dropna().mode().iloc[0])
        else:
            most_freq = 0
        rajc_assignments = rajc_assignments.fillna(most_freq)

    return rajc_assignments.astype(int)


def _load_rajc_config(logger: logging.Logger) -> RAJCConfig:
    """Load RAJCConfig from YAML for consistency with main experiment."""
    if not DEFAULT_RAJC_CONFIG_PATH.is_file():
        logger.warning(
            "RAJC config file not found at %s; using default RAJCConfig().",
            DEFAULT_RAJC_CONFIG_PATH,
        )
        return RAJCConfig()

    try:
        cfg_dict = yaml.safe_load(DEFAULT_RAJC_CONFIG_PATH.read_text()) or {}
        rajc_cfg = cfg_dict.get("rajc", {}) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read RAJC config (%s); using defaults.", exc)
        return RAJCConfig()

    return RAJCConfig(
        n_clusters=int(rajc_cfg.get("n_clusters", 4)),
        lambda_=float(rajc_cfg.get("lambda", 1.0)),
        gamma=float(rajc_cfg.get("gamma", 0.0)),
        max_iter=int(rajc_cfg.get("max_iter", 20)),
        tol=float(rajc_cfg.get("tol", 1e-4)),
        random_state=int(rajc_cfg.get("random_state", 42)),
        smoothing=float(rajc_cfg.get("smoothing", 1.0)),
        logreg_max_iter=int(rajc_cfg.get("logistic_regression", {}).get("max_iter", 200)),
        kmeans_n_init=int(rajc_cfg.get("kmeans_n_init", 10)),
        model_type=str(rajc_cfg.get("model_type", "constant_prob")),
    )


def _one_hot_clusters(
    cluster_labels: pd.Series | np.ndarray,
    *,
    index: pd.Index,
    prefix: str,
) -> pd.DataFrame:
    """Convert cluster labels into one-hot encoded indicator features."""
    if not isinstance(cluster_labels, pd.Series):
        cluster_labels = pd.Series(cluster_labels, index=index)
    return pd.get_dummies(cluster_labels, prefix=prefix, dtype=int)


def _add_cluster_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_clusters: pd.Series | np.ndarray,
    test_clusters: pd.Series | np.ndarray,
    *,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append aligned one-hot cluster indicators to train/test features."""
    train_d = _one_hot_clusters(train_clusters, index=train_x.index, prefix=prefix)
    test_d = _one_hot_clusters(test_clusters, index=test_x.index, prefix=prefix)
    test_d = test_d.reindex(columns=train_d.columns, fill_value=0)

    train_aug = pd.concat([train_x, train_d], axis=1)
    test_aug = pd.concat([test_x, test_d], axis=1)
    return train_aug, test_aug


def _train_lr(train_x: pd.DataFrame, train_y: pd.Series, *, max_iter: int = 500) -> LogisticRegression:
    """Fit a balanced logistic regression for downstream prediction curves."""
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(train_x, train_y)
    return clf


def _predict_lr(model: LogisticRegression, x: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (proba, pred) for positive class."""
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return prob, pred


def _prepare_leak_free_train_test(
    cleaned_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prepare leak-free train/test features.

    Key point:
    - Fit assemble_feature_table() ONLY on train_df.
    - Use the fitted transformer to transform test_df.
    """
    labeled = add_response_label(cleaned_df, label_col=DEFAULT_LABEL_COL)

    y_all = labeled[DEFAULT_LABEL_COL].astype(int)
    train_df, test_df = train_test_split(
        labeled,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y_all,
    )

    # Fit transforms on train only
    x_train_full, y_train, transformer = assemble_feature_table(train_df, label_col=DEFAULT_LABEL_COL)

    # Transform test using fitted transformer and the SAME feature names
    x_test_full, y_test, _ = assemble_feature_table(
        test_df, label_col=DEFAULT_LABEL_COL, transformer=transformer, fit=False
    )

    # Behaviour-only subset (for clustering/RAJC)
    x_train_beh, _ = split_behavior_and_response_features(x_train_full, transformer)
    x_test_beh, _ = split_behavior_and_response_features(x_test_full, transformer)

    logger.info(
        "Leak-free split: train=%d, test=%d, full_dim=%d, beh_dim=%d",
        x_train_full.shape[0],
        x_test_full.shape[0],
        x_train_full.shape[1],
        x_train_beh.shape[1],
    )

    return x_train_full, x_test_full, y_train, y_test, x_train_beh, x_test_beh


def _generate_downstream_curves(
    cleaned_df: pd.DataFrame,
    *,
    logger: logging.Logger,
    seed: int,
    test_size: float,
) -> None:
    """Generate ROC/PR/Calibration/Lift/Threshold sweep curves for downstream prediction.

    Models compared (for curves):
    - base logistic regression (full engineered features)
    - base + KMeans cluster IDs
    - base + RAJC cluster IDs  (our key improvement)
    """
    logger.info("Generating downstream prediction curves (ROC/PR/Calibration/Lift/Threshold).")
    FIG_PRED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Leak-free train/test features
    x_train_full, x_test_full, y_train, y_test, x_train_beh, x_test_beh = _prepare_leak_free_train_test(
        cleaned_df,
        test_size=test_size,
        random_state=seed,
        logger=logger,
    )

    # 2) Base LR
    base_lr = _train_lr(x_train_full, y_train, max_iter=800)
    base_prob, _ = _predict_lr(base_lr, x_test_full)

    # 3) KMeans cluster IDs
    km = KMeans(n_clusters=4, random_state=int(seed), n_init=10)
    km.fit(x_train_beh)
    km_train = pd.Series(km.predict(x_train_beh), index=x_train_beh.index)
    km_test = pd.Series(km.predict(x_test_beh), index=x_test_beh.index)
    x_train_km, x_test_km = _add_cluster_features(
        x_train_full, x_test_full, km_train, km_test, prefix="kmeans"
    )
    km_lr = _train_lr(x_train_km, y_train, max_iter=800)
    km_prob, _ = _predict_lr(km_lr, x_test_km)

    # 4) RAJC cluster IDs
    rajc_cfg = _load_rajc_config(logger)
    rajc = RAJCModel(rajc_cfg)
    rajc.fit(x_train_beh, y_train)
    rajc_train = pd.Series(rajc.assignments_, index=x_train_beh.index)
    rajc_test = pd.Series(rajc.predict_clusters(x_test_beh), index=x_test_beh.index)

    x_train_r, x_test_r = _add_cluster_features(
        x_train_full, x_test_full, rajc_train, rajc_test, prefix="rajc"
    )
    rajc_lr = _train_lr(x_train_r, y_train, max_iter=800)
    rajc_prob, _ = _predict_lr(rajc_lr, x_test_r)

    # 5) Plot + save
    curves: Dict[str, np.ndarray] = {
        "base": base_prob,
        "base+kmeansid": km_prob,
        "base+rajcid": rajc_prob,
    }

    for name, prob in curves.items():
        try:
            fig = plot_roc_curve(y_test, prob, title=f"ROC ({name})")
            fig.savefig(FIG_PRED_DIR / f"roc_{name}.png", bbox_inches="tight")
            plt.close(fig)

            fig = plot_pr_curve(y_test, prob, title=f"PR curve ({name})")
            fig.savefig(FIG_PRED_DIR / f"pr_{name}.png", bbox_inches="tight")
            plt.close(fig)

            fig = plot_calibration_curve(y_test, prob, title=f"Calibration ({name})", n_bins=10)
            fig.savefig(FIG_PRED_DIR / f"calibration_{name}.png", bbox_inches="tight")
            plt.close(fig)

            fig = plot_lift_curve(y_test, prob, title=f"Lift curve ({name})")
            fig.savefig(FIG_PRED_DIR / f"lift_{name}.png", bbox_inches="tight")
            plt.close(fig)

            fig = plot_threshold_sweep(y_test, prob, title=f"Threshold sweep ({name})")
            fig.savefig(FIG_PRED_DIR / f"threshold_{name}.png", bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to generate prediction curves for %s: %s", name, exc)

    logger.info("Saved downstream prediction curves under %s", FIG_PRED_DIR)


def _generate_plots(
    logger: logging.Logger,
    *,
    data_dir: Path,
    seed: int,
    downstream_test_size: float,
) -> None:
    """Generate key figures for baselines, RAJC clusters, profiling, and downstream curves."""
    _ensure_figure_dirs()

    try:
        cleaned, features, responses = _load_core_data_for_plots(data_dir, logger)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to prepare data for plotting: %s", exc)
        return

    # -------------------- 1) Baseline: RFM KMeans Elbow --------------------
    try:
        logger.info("Generating elbow curve for RFM K-Means baseline.")
        rfm_df = build_rfm_features(cleaned).rename(columns=str.lower)
        rfm = rfm_df[["recency", "frequency", "monetary"]]

        ks: List[int] = list(range(2, 9))
        inertias: List[float] = []
        for k in ks:
            km = KMeans(
                n_clusters=k,
                random_state=int(seed),
                n_init=10,
            )
            km.fit(rfm)
            inertias.append(float(km.inertia_))

        fig = plot_elbow_curve(ks, inertias, title="Elbow curve (RFM K-Means)")
        fig.savefig(FIG_BASELINES_DIR / "elbow_rfm_kmeans.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to generate elbow curve plot: %s", exc)

    # -------------------- 2) RAJC cluster-level plots ----------------------
    try:
        rajc_assignments = _align_rajc_assignments(features, logger)
    except Exception as exc:
        logger.exception(
            "Failed to load or align RAJC assignments for plotting: %s. "
            "Skipping RAJC plots.",
            exc,
        )
        return

    # 2.1 PCA visualization
    try:
        logger.info("Generating PCA scatter for RAJC clusters.")
        fig = plot_pca_scatter(features, rajc_assignments, title="RAJC clusters (PCA)")
        fig.savefig(FIG_RAJC_DIR / "rajc_pca_clusters.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.exception("Failed to generate RAJC PCA plot: %s", exc)

    # 2.2 Silhouette distribution
    try:
        logger.info("Generating silhouette distribution for RAJC clusters.")
        fig = plot_silhouette_distribution(
            features,
            rajc_assignments,
            title="RAJC silhouette distribution",
        )
        fig.savefig(FIG_RAJC_DIR / "rajc_silhouette_distribution.png", bbox_inches="tight")
        plt.close(fig)
    except ValueError as exc:
        logger.warning("Cannot compute RAJC silhouette distribution: %s", exc)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to generate RAJC silhouette plot: %s", exc)

    # 2.3 Feature-space cluster centers
    try:
        logger.info("Generating RAJC cluster centers plot.")
        centers = features.groupby(rajc_assignments).mean()
        fig = plot_cluster_centers(
            centers,
            title="RAJC cluster centers (feature means)",
        )
        fig.savefig(FIG_RAJC_DIR / "rajc_cluster_centers.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.exception("Failed to generate RAJC cluster centers plot: %s", exc)

    # -------------------- 3) Customer profile plots ------------------------
    try:
        logger.info("Generating RAJC profile plots (income, RFM, channels, etc.).")
        rfm_df = build_rfm_features(cleaned).rename(columns=str.lower)
        clusters = rajc_assignments

        # 3.1 income vs spending
        fig = plot_income_vs_spent(rfm_df, clusters, income_col="income", monetary_col="monetary")
        fig.savefig(FIG_PROFILES_DIR / "rajc_income_vs_monetary.png", bbox_inches="tight")
        plt.close(fig)

        # 3.2 RFM boxplots
        fig = plot_rfm_boxplots(rfm_df, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_rfm_boxplots.png", bbox_inches="tight")
        plt.close(fig)

        # 3.3 channel mix
        fig = plot_channel_mix(cleaned, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_channel_mix.png", bbox_inches="tight")
        plt.close(fig)

        # 3.4 response rates (bar)
        rates = cluster_response_rates(clusters, responses)
        global_rate = float(pd.Series(responses).mean())
        fig = plot_response_rates(rates, title="RAJC cluster response rates", global_rate=global_rate)
        fig.savefig(FIG_PROFILES_DIR / "rajc_response_rates.png", bbox_inches="tight")
        plt.close(fig)

        # 3.4b NEW: size + response rate in one figure
        sizes = pd.Series(clusters).value_counts().sort_index()
        fig = plot_cluster_size_and_response(sizes, rates, title="RAJC cluster size & response rate")
        fig.savefig(FIG_PROFILES_DIR / "rajc_size_and_response.png", bbox_inches="tight")
        plt.close(fig)

        # 3.5 age vs income density (hexbin)
        fig = plot_age_income_kde(cleaned, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_age_income_density.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.exception("Failed to generate RAJC profile plots: %s", exc)

    # -------------------- 4) NEW: downstream curves for report -------------
    try:
        _generate_downstream_curves(
            cleaned_df=cleaned,
            logger=logger,
            seed=seed,
            test_size=downstream_test_size,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to generate downstream prediction curves: %s", exc)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entrypoint for running all experiments and plots."""
    args = _parse_args(argv)
    logger = _configure_global_logging()

    # 1) Check dataset
    _check_dataset(args.data_dir, logger)

    # 2) Global seed
    set_global_seed(args.seed)
    logger.info("Global random seed set to %d", args.seed)

    # 3) Build experiment plan
    steps: List[Tuple[str, Callable[[], None]]] = []
    if not args.skip_baselines:
        steps.append(("Baseline clustering", lambda: run_baselines([])))
    if not args.skip_rajc:
        steps.append(("RAJC main experiment", lambda: run_rajc([])))
    if not args.skip_downstream:
        steps.append(("Downstream prediction", lambda: run_downstream([])))
    if not args.skip_ablation:
        steps.append(("RAJC ablation", lambda: run_ablation([])))
    if not args.skip_plots:
        steps.append(
            (
                "Figure generation",
                lambda: _generate_plots(
                    logger,
                    data_dir=args.data_dir,
                    seed=args.seed,
                    downstream_test_size=args.downstream_test_size,
                ),
            )
        )

    if not steps:
        logger.warning("No steps selected to run (all were skipped). Nothing to do.")
        return

    logger.info("Planned steps (in order): %s", " -> ".join(name for name, _ in steps))

    stop_on_error = not args.keep_going

    # 4) Run all steps
    for name, func in steps:
        try:
            _run_step(name, func, logger, stop_on_error=stop_on_error)
        except SystemExit:
            logger.error("Stopping run_all due to SystemExit from step '%s'.", name)
            sys.exit(1)
        except Exception:
            logger.error("Stopping run_all due to failure in step '%s'.", name)
            sys.exit(1)

    logger.info("✅ All selected experiments and plots completed successfully.")


if __name__ == "__main__":
    main()
