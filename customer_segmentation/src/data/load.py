"""Data loading helpers for the marketing campaign dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from pandas.errors import ParserError


def load_raw_data(
    data_dir: Path = Path("customer_segmentation/data/raw"),
    filename: str = "marketing_campaign.csv",
    parse_dates: Optional[Sequence[str]] = None,
    min_expected_columns: int = 10,
) -> pd.DataFrame:
    """Load the marketing campaign CSV file with robust delimiter handling.

    The Kaggle *Customer Personality Analysis* dataset is often stored as a
    semicolon-separated file. This helper first attempts to read the file with
    pandas' default settings, and if the result appears to have too few columns
    (e.g., everything in one column) or parsing fails, it falls back to
    ``sep=None, engine='python'`` to auto-detect the delimiter.
    """
    csv_path = data_dir / filename
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. Please place the Kaggle CSV in this location."
        )

    # First attempt: default pandas CSV parsing.
    try:
        df = pd.read_csv(csv_path, parse_dates=parse_dates)
    except (ParserError, ValueError):
        # ValueError can be raised when ``parse_dates`` columns are not found due to
        # an unexpected delimiter collapsing the header into a single column.
        df = None

    # If parsing failed or produced suspiciously few columns, try auto-detecting the separator.
    if df is None or df.shape[1] < min_expected_columns:
        try:
            df = pd.read_csv(
                csv_path,
                parse_dates=parse_dates,
                sep=None,  # infer delimiters such as ';', '\t', etc.
                engine="python",
            )
        except ParserError as exc:
            raise ValueError(
                f"Failed to parse dataset at {csv_path} with automatic delimiter detection."
            ) from exc

    # Basic sanity check.
    if df.shape[1] < min_expected_columns:
        raise ValueError(
            f"Parsed dataset from {csv_path} appears to have only {df.shape[1]} columns; "
            "please verify that the file is the correct marketing_campaign CSV."
        )

    return df
