"""Data loading helpers for the marketing campaign dataset."""
from pathlib import Path
from typing import Optional

import pandas as pd


def load_raw_data(
    data_dir: Path = Path("customer_segmentation/data/raw"),
    filename: str = "marketing_campaign.csv",
    parse_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load the marketing campaign CSV file.

    Args:
        data_dir: Directory containing the raw CSV file.
        filename: Dataset filename (defaults to ``marketing_campaign.csv``).
        parse_dates: Optional list of columns to parse as datetimes.

    Returns:
        DataFrame with the raw campaign data.

    Raises:
        FileNotFoundError: If the expected CSV is absent.
    """

    csv_path = data_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. Please place the Kaggle CSV in this location."
        )

    return pd.read_csv(csv_path, parse_dates=parse_dates)
