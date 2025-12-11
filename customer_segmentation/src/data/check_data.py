"""Utility to verify the presence of the marketing campaign dataset."""
from pathlib import Path


def dataset_status(data_dir: Path = Path("customer_segmentation/data/raw")) -> tuple[bool, Path]:
    """Return whether the marketing_campaign.csv file exists.

    Args:
        data_dir: Directory where the raw CSV should live.

    Returns:
        Tuple of (exists flag, expected file path).
    """
    csv_path = data_dir / "marketing_campaign.csv"
    return csv_path.exists(), csv_path


def main() -> None:
    exists, csv_path = dataset_status()
    if exists:
        print(f"✅ Found dataset at: {csv_path}")
    else:
        print(
            "❌ marketing_campaign.csv is missing. "
            f"Place the Kaggle Customer Personality Analysis CSV at: {csv_path}"
        )


if __name__ == "__main__":
    main()
