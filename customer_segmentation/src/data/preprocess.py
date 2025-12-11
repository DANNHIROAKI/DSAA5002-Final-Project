"""Data cleaning and preprocessing routines."""
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REFERENCE_YEAR = 2024
INCOME_CLIP_QUANTILES = (0.01, 0.99)
AGE_CLIP_QUANTILES = (0.01, 0.99)


def _clip_outliers(series: pd.Series, quantiles: Tuple[float, float]) -> pd.Series:
    lower, upper = series.quantile(quantiles[0]), series.quantile(quantiles[1])
    return series.clip(lower=lower, upper=upper)


def _fill_missing(df: pd.DataFrame, categorical_cols: Iterable[str], numeric_cols: Iterable[str]) -> pd.DataFrame:
    filled = df.copy()
    for col in numeric_cols:
        if col in filled:
            filled[col] = filled[col].fillna(filled[col].median())
    for col in categorical_cols:
        if col in filled:
            filled[col] = filled[col].fillna(filled[col].mode().iloc[0])
    return filled


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, derive age, and mitigate outliers.

    Steps
    -----
    1. Drop duplicate rows.
    2. Median/mode imputation for numeric/categorical columns.
    3. Convert ``Dt_Customer`` to datetime when present.
    4. Derive ``Age`` from ``Year_Birth`` using ``REFERENCE_YEAR``.
    5. Winsorize ``Income`` and ``Age`` using quantile clipping to reduce the impact of extreme outliers.
    """

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Parse join date if available
    if "Dt_Customer" in cleaned:
        cleaned["Dt_Customer"] = pd.to_datetime(cleaned["Dt_Customer"], errors="coerce")

    categorical_cols = [
        col
        for col in cleaned.columns
        if cleaned[col].dtype == "object" or str(cleaned[col].dtype).startswith("category")
    ]
    numeric_cols = [col for col in cleaned.columns if col not in categorical_cols]
    cleaned = _fill_missing(cleaned, categorical_cols, numeric_cols)

    # Age derived from Year_Birth
    if "Year_Birth" in cleaned:
        cleaned["Age"] = REFERENCE_YEAR - cleaned["Year_Birth"]

    if "Income" in cleaned:
        cleaned["Income"] = _clip_outliers(cleaned["Income"], INCOME_CLIP_QUANTILES)

    if "Age" in cleaned:
        cleaned["Age"] = _clip_outliers(cleaned["Age"], AGE_CLIP_QUANTILES)

    return cleaned


def train_val_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, label_col: str = "response"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned dataset into training and validation subsets with stratification."""

    stratify_target = df[label_col] if label_col in df.columns else None
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_target)
