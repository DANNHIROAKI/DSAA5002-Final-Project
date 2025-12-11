"""One-stop launcher to run all experiments end-to-end.

This script will:

1. Check whether the marketing_campaign.csv dataset is present.
2. Set a global random seed for reproducibility.
3. Run all experimental pipelines in sequence:
   - Baseline clustering (4 baselines)
   - RAJC main experiment
   - Downstream response prediction with cluster IDs
   - RAJC ablation over lambda and K
4. Generate key figures for the report (elbow curve, RAJC cluster plots, profiles).

Results are written under ``customer_segmentation/outputs`` as defined
in the individual experiment scripts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence, Callable, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from customer_segmentation.src.data.check_data import dataset_status  # type: ignore[import]
from customer_segmentation.src.data.load import load_raw_data  # type: ignore[import]
from customer_segmentation.src.data.preprocess import clean_data  # type: ignore[import]
from customer_segmentation.src.data.features import (  # type: ignore[import]
    assemble_feature_table,
    build_rfm_features,
)
from customer_segmentation.src.evaluation.segmentation import (  # type: ignore[import]
    cluster_response_rates,
)
from customer_segmentation.src.utils import configure_logging, set_global_seed  # type: ignore[import]
from customer_segmentation.src.experiments import (  # type: ignore[import]
    run_ablation,
    run_baselines,
    run_downstream,
    run_rajc,
)
from customer_segmentation.src.visualization import (  # type: ignore[import]
    plot_elbow_curve,
    plot_pca_scatter,
    plot_silhouette_distribution,
    plot_income_vs_spent,
    plot_rfm_boxplots,
    plot_channel_mix,
    plot_response_rates,
    plot_age_income_kde,
    plot_cluster_centers,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Where we put the global log file for this launcher
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
GLOBAL_LOG = LOG_DIR / "run_all.log"

TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_BASELINES_DIR = FIG_DIR / "baselines"
FIG_RAJC_DIR = FIG_DIR / "rajc"
FIG_PROFILES_DIR = FIG_DIR / "profiles"


# ---------------------------------------------------------------------------
# CLI parsing & logging
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop immediately if any sub-experiment fails (default: True).",
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
        # Allow sub-scripts to signal critical failure
        logger.exception("Step '%s' aborted via SystemExit.", name)
        if stop_on_error:
            raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Step '%s' failed with error: %s", name, exc)
        if stop_on_error:
            raise


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _ensure_figure_dirs() -> None:
    """Ensure figure output directories exist."""
    for d in (FIG_DIR, FIG_BASELINES_DIR, FIG_RAJC_DIR, FIG_PROFILES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _generate_plots(logger: logging.Logger) -> None:
    """Generate key figures for baselines and RAJC clusters."""

    _ensure_figure_dirs()

    logger.info("Preparing data for plotting.")
    # 重复一遍数据加载/特征构造，方便画图时使用
    raw = load_raw_data(parse_dates=["Dt_Customer"])
    cleaned = clean_data(raw)
    features, responses, _ = assemble_feature_table(cleaned)

    # -------------------- 1) Baseline: RFM KMeans Elbow --------------------
    try:
        logger.info("Generating elbow curve for RFM K-Means baseline.")
        rfm_df = build_rfm_features(cleaned)
        rfm = rfm_df[["recency", "frequency", "monetary"]]

        ks: List[int] = list(range(2, 9))
        inertias: List[float] = []
        for k in ks:
            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init="auto",
            )
            km.fit(rfm)
            inertias.append(float(km.inertia_))

        fig = plot_elbow_curve(ks, inertias, title="Elbow curve (RFM K-Means)")
        fig.savefig(FIG_BASELINES_DIR / "elbow_rfm_kmeans.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - 防御性
        logger.exception("Failed to generate elbow curve plot: %s", exc)

    # -------------------- 2) RAJC cluster-level plots ----------------------
    try:
        assignments_path = TABLE_DIR / "rajc_assignments.csv"
        if not assignments_path.is_file():
            raise FileNotFoundError(assignments_path)
        rajc_assignments = pd.read_csv(assignments_path, index_col=0)["cluster"]
        # 对齐索引
        rajc_assignments = rajc_assignments.loc[features.index]
    except Exception as exc:
        logger.exception(
            "Failed to load RAJC assignments for plotting: %s. "
            "Skipping RAJC plots.",
            exc,
        )
        return

    # 2.1 PCA 可视化
    try:
        logger.info("Generating PCA scatter for RAJC clusters.")
        fig = plot_pca_scatter(features, rajc_assignments, title="RAJC clusters (PCA)")
        fig.savefig(FIG_RAJC_DIR / "rajc_pca_clusters.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.exception("Failed to generate RAJC PCA plot: %s", exc)

    # 2.2 Silhouette 分布
    try:
        logger.info("Generating silhouette distribution for RAJC clusters.")
        fig = plot_silhouette_distribution(
            features,
            rajc_assignments,
            title="RAJC silhouette distribution",
        )
        fig.savefig(FIG_RAJC_DIR / "rajc_silhouette_distribution.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
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
        rfm_df = build_rfm_features(cleaned)
        clusters = rajc_assignments

        # 3.1 收入 vs 总消费
        fig = plot_income_vs_spent(rfm_df, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_income_vs_monetary.png", bbox_inches="tight")
        plt.close(fig)

        # 3.2 RFM 箱线图
        fig = plot_rfm_boxplots(rfm_df, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_rfm_boxplots.png", bbox_inches="tight")
        plt.close(fig)

        # 3.3 渠道组合
        fig = plot_channel_mix(cleaned, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_channel_mix.png", bbox_inches="tight")
        plt.close(fig)

        # 3.4 各簇响应率条形图
        rates = cluster_response_rates(clusters, responses)
        fig = plot_response_rates(rates, title="RAJC cluster response rates")
        fig.savefig(FIG_PROFILES_DIR / "rajc_response_rates.png", bbox_inches="tight")
        plt.close(fig)

        # 3.5 年龄 vs 收入 KDE
        fig = plot_age_income_kde(cleaned, clusters)
        fig.savefig(FIG_PROFILES_DIR / "rajc_age_income_kde.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.exception("Failed to generate RAJC profile plots: %s", exc)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logger = _configure_global_logging()

    # 1) Check dataset
    _check_dataset(args.data_dir, logger)

    # 2) Global seed
    set_global_seed(args.seed)
    logger.info("Global random seed set to %d", args.seed)

    # 3) Build experiment plan
    steps: list[tuple[str, Callable[[], None]]] = []
    if not args.skip_baselines:
        # run_baselines([]) runs all 4 clustering baselines.
        steps.append(("Baseline clustering", lambda: run_baselines([])))
    if not args.skip_rajc:
        # run_rajc([]) fits and evaluates RAJC.
        steps.append(("RAJC main experiment", lambda: run_rajc([])))
    if not args.skip_downstream:
        # run_downstream([]) runs the downstream response prediction comparison.
        steps.append(("Downstream prediction", lambda: run_downstream([])))
    if not args.skip_ablation:
        # run_ablation([]) sweeps over lambda and K.
        steps.append(("RAJC ablation", lambda: run_ablation([])))
    if not args.skip_plots:
        # 最后统一画图
        steps.append(("Figure generation", lambda: _generate_plots(logger)))

    if not steps:
        logger.warning("No steps selected to run (all were skipped). Nothing to do.")
        return

    logger.info(
        "Planned steps (in order): %s",
        " -> ".join(name for name, _ in steps),
    )

    # 4) Run all steps
    for name, func in steps:
        try:
            _run_step(name, func, logger, args.stop_on_error)
        except SystemExit:
            logger.error("Stopping run_all due to SystemExit from step '%s'.", name)
            sys.exit(1)
        except Exception:
            logger.error("Stopping run_all due to failure in step '%s'.", name)
            sys.exit(1)

    logger.info("✅ All selected experiments and plots completed successfully.")


if __name__ == "__main__":
    main()
