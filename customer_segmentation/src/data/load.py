"""Data loading helpers for the marketing campaign dataset.

The Kaggle *Customer Personality Analysis* dataset is often saved with a
non-comma delimiter (tab or semicolon). This module provides a small,
robust loader that:

1) tries standard :func:`pandas.read_csv` parsing;
2) falls back to delimiter auto-detection if parsing looks suspicious.

Important
---------
The ``Dt_Customer`` column contains ambiguous strings such as "04-09-2012".
To avoid day/month confusion, we **do not recommend** parsing it via
``read_csv(parse_dates=...)``. Instead, parse it explicitly inside
``customer_segmentation.src.data.preprocess.clean_data`` using the dataset's
fixed format ``%d-%m-%Y``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import warnings

import pandas as pd
from pandas.errors import ParserError


def load_raw_data(
    data_dir: Path = Path("customer_segmentation/data/raw"),
    filename: str = "marketing_campaign.csv",
    parse_dates: Optional[Sequence[str]] = None,
    min_expected_columns: int = 10,
) -> pd.DataFrame:
    """Load the marketing campaign CSV file with robust delimiter handling.

    Parameters
    ----------
    data_dir:
        Directory containing ``marketing_campaign.csv``.
    filename:
        CSV filename.
    parse_dates:
        Optional list of columns to parse as dates.

        **Recommendation:** do not include ``Dt_Customer`` here. If it is
        included, we silently remove it and issue a warning, because parsing it
        without an explicit format can swap day/month.
    min_expected_columns:
        Sanity threshold: if fewer columns are parsed, we assume delimiter
        parsing failed and fall back to auto-detection.
    """

    csv_path = data_dir / filename
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. Please place the Kaggle CSV in this location."
        )

    # Avoid ambiguous parsing for Dt_Customer.
    if parse_dates is not None and "Dt_Customer" in parse_dates:
        parse_dates = [c for c in parse_dates if c != "Dt_Customer"]
        warnings.warn(
            "Ignoring parse_dates for 'Dt_Customer'. Parse it with an explicit format "
            "in preprocess.clean_data(date_format='%d-%m-%Y') to avoid ambiguity.",
            RuntimeWarning,
        )
        if len(parse_dates) == 0:
            parse_dates = None

    # First attempt: default pandas CSV parsing.
    try:
        df = pd.read_csv(csv_path, parse_dates=parse_dates)
    except (ParserError, ValueError):
        # ValueError may occur when ``parse_dates`` columns are not found due to an
        # unexpected delimiter collapsing the header into a single column.
        df = None

    # If parsing failed or produced too few columns, try auto-detecting the separator.
    if df is None or df.shape[1] < min_expected_columns:
        try:
            df = pd.read_csv(
                csv_path,
                parse_dates=parse_dates,
                sep=None,  # infer delimiters such as '\t', ';', etc.
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
