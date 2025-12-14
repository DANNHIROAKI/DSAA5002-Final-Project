"""CLI utility to verify dataset availability and basic schema.

Run from the project root:

.. code-block:: bash

    python -m customer_segmentation.src.data.check_data

This script is intentionally lightweight and does not modify the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from .load import load_raw_data
from .preprocess import DEFAULT_DATE_FORMAT


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
    parser = argparse.ArgumentParser(description="Check dataset placement and schema.")
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
    args = _parse_args(argv)

    # Ensure directory exists so users immediately see where to put the file.
    args.data_dir.mkdir(parents=True, exist_ok=True)

    exists, csv_path = dataset_status(args.data_dir, args.filename)
    if not exists:
        print(
            "❌ Dataset is missing.\n"
            "   Expected Kaggle 'Customer Personality Analysis' CSV at:\n"
            f"   {csv_path}\n"
        )
        return

    print(f"✅ Found dataset at: {csv_path}")

    # Try loading and print quick sanity stats.
    try:
        df = load_raw_data(data_dir=args.data_dir, filename=args.filename)
    except Exception as exc:  # pragma: no cover
        print(f"❌ Failed to load dataset: {exc}")
        return

    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("Columns:")
    print(", ".join(df.columns.tolist()))

    # Date sanity check
    if "Dt_Customer" in df.columns:
        s = pd.to_datetime(df["Dt_Customer"], format=DEFAULT_DATE_FORMAT, errors="coerce")
        if s.notna().any():
            print(f"Dt_Customer range (parsed with {DEFAULT_DATE_FORMAT}): {s.min().date()} → {s.max().date()}")
        else:
            print("Warning: Dt_Customer could not be parsed with the expected date format.")


if __name__ == "__main__":
    main()
