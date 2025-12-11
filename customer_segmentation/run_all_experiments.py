"""One-stop launcher to run all experiments end-to-end.

This script will:

1. Check whether the marketing_campaign.csv dataset is present.
2. Set a global random seed for reproducibility.
3. Run all experimental pipelines in sequence:
   - Baseline clustering (4 baselines)
   - RAJC main experiment
   - Downstream response prediction with cluster IDs
   - RAJC ablation over lambda and K

Results are written under ``customer_segmentation/outputs`` as defined
in the individual experiment scripts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence

from customer_segmentation.src.data.check_data import dataset_status
from customer_segmentation.src.utils import configure_logging, set_global_seed  # type: ignore[import]  # noqa: E501
from customer_segmentation.src.experiments import (  # type: ignore[import]
    run_ablation,
    run_baselines,
    run_downstream,
    run_rajc,
)


# Where we put the global log file for this launcher
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
GLOBAL_LOG = LOG_DIR / "run_all.log"


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
        action="bool",
        default=False,
        help="Skip running baseline clustering experiments.",
    )
    parser.add_argument(
        "--skip-rajc",
        action="bool",
        default=False,
        help="Skip running the main RAJC experiment.",
    )
    parser.add_argument(
        "--skip-downstream",
        action="bool",
        default=False,
        help="Skip running downstream prediction experiments.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="bool",
        default=False,
        help="Skip RAJC ablation study over lambda and K.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="bool",
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
    func,
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logger = _configure_global_logging()

    # 1) Check dataset
    _check_dataset(args.data_dir, logger)

    # 2) Global seed
    set_global_seed(args.seed)
    logger.info("Global random seed set to %d", args.seed)

    # 3) Build experiment plan
    steps: list[tuple[str, callable]] = []
    if not args.skip_baselines:
        # run_baselines.main() runs all 4 clustering baselines. :contentReference[oaicite:5]{index=5}
        steps.append(("Baseline clustering", run_baselines.main))
    if not args.skip_rajc:
        # run_rajc.main() fits and evaluates RAJC. :contentReference[oaicite:6]{index=6}
        steps.append(("RAJC main experiment", run_rajc.main))
    if not args.skip_downstream:
        # run_downstream.main() runs the downstream response prediction comparison. :contentReference[oaicite:7]{index=7}
        steps.append(("Downstream prediction", run_downstream.main))
    if not args.skip_ablation:
        # run_ablation.main() sweeps over lambda and K. :contentReference[oaicite:8]{index=8}
        steps.append(("RAJC ablation", run_ablation.main))

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

    logger.info("✅ All selected experiments completed successfully.")


if __name__ == "__main__":
    main()
