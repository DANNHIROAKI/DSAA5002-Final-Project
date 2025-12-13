"""Utility to verify the presence of the marketing campaign dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


DEFAULT_DATA_DIR = Path("customer_segmentation/data/raw")
DEFAULT_FILENAME = "marketing_campaign.csv"


def dataset_status(
    data_dir: Path = DEFAULT_DATA_DIR,
    filename: str = DEFAULT_FILENAME,
) -> Tuple[bool, Path]:
    """Return whether the marketing campaign CSV file exists."""
    csv_path = data_dir / filename
    return csv_path.is_file(), csv_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the marketing campaign dataset is available."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the CSV (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_FILENAME,
        help=f"CSV filename (default: {DEFAULT_FILENAME})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Examples
    --------
    From the project root:

    .. code-block:: bash

        python -m customer_segmentation.src.data.check_data
    """
    args = _parse_args(argv)

    # Make sure the directory exists to give a clear hint to the user.
    if not args.data_dir.exists():
        args.data_dir.mkdir(parents=True, exist_ok=True)

    exists, csv_path = dataset_status(args.data_dir, args.filename)
    if exists:
        print(f"✅ Found dataset at: {csv_path}")
    else:
        print(
            "❌ Dataset is missing.\n"
            "   Expected Kaggle 'Customer Personality Analysis' CSV at:\n"
            f"   {csv_path}"
        )


if __name__ == "__main__":
    main()
