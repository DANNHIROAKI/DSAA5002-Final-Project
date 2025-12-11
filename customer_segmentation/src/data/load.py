"""Data loading helpers for the marketing campaign dataset."""
from pathlib import Path
import pandas as pd


def load_raw_data(data_dir: Path = Path("customer_segmentation/data/raw")) -> pd.DataFrame:
    """Load the marketing campaign CSV file.

    Args:
        data_dir: Directory containing `marketing_campaign.csv`.

    Returns:
        DataFrame with the raw campaign data.
    """
    csv_path = data_dir / "marketing_campaign.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset at {csv_path}")
    return pd.read_csv(csv_path)
