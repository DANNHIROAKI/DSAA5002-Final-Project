"""Data cleaning and preprocessing routines."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reference year used to derive age from Year_Birth.
REFERENCE_YEAR = 2024

# Quantiles used to mitigate extreme outliers (Winsorization),
# consistent with the Tutorial's suggestion to handle Income and Age outliers. :contentReference[oaicite:5]{index=5}
INCOME_CLIP_QUANTILES = (0.01, 0.99)
AGE_CLIP_QUANTILES = (0.01, 0.99)


def _clip_outliers(series: pd.Series, quantiles: Tuple[float, float]) -> pd.Series:
    """Clip a numeric series to the given lower/upper quantiles.

    Non-numeric series are returned unchanged.
    """
    if not np.issubdtype(series.dtype, np.number):
        return series

    lower, upper = series.quantile(quantiles[0]), series.quantile(quantiles[1])
    return series.clip(lower=lower, upper=upper)


def _fill_missing(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
) -> pd.DataFrame:
    """Fill missing values using median for numerics and mode for categoricals."""
    filled = df.copy()

    for col in numeric_cols:
        if col in filled:
            median = filled[col].median()
            filled[col] = filled[col].fillna(median)

    for col in categorical_cols:
        if col in filled and not filled[col].empty:
            mode = filled[col].mode(dropna=True)
            if not mode.empty:
                filled[col] = filled[col].fillna(mode.iloc[0])

    return filled


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, derive age, and mitigate outliers.

    Steps
    -----
    1. Drop duplicate rows.
    2. Parse ``Dt_Customer`` as datetime when present.
    3. Separate categorical vs numeric columns.
    4. Median/mode imputation for numeric/categorical columns.
    5. Derive ``Age`` from ``Year_Birth`` using :data:`REFERENCE_YEAR`.
    6. Winsorize ``Income`` and ``Age`` using quantile clipping to reduce
       the impact of extreme outliers, reflecting the instructor's recommendation
       to handle obvious outliers in these fields. :contentReference[oaicite:6]{index=6}

    Parameters
    ----------
    df :
        Raw marketing campaign DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame suitable for downstream feature engineering.
    """
    cleaned = df.copy()

    # Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Parse join date if available
    if "Dt_Customer" in cleaned.columns:
        cleaned["Dt_Customer"] = pd.to_datetime(cleaned["Dt_Customer"], errors="coerce")

    # Identify categorical and numeric columns
    categorical_cols = [
        col
        for col in cleaned.columns
        if cleaned[col].dtype == "object"
        or str(cleaned[col].dtype).startswith("category")
    ]
    numeric_cols = [col for col in cleaned.columns if col not in categorical_cols]

    cleaned = _fill_missing(cleaned, categorical_cols, numeric_cols)

    # Age derived from Year_Birth
    if "Year_Birth" in cleaned.columns:
        cleaned["Age"] = REFERENCE_YEAR - cleaned["Year_Birth"]

    # Winsorize Income and Age to soften the effect of extreme outliers
    if "Income" in cleaned.columns:
        cleaned["Income"] = _clip_outliers(cleaned["Income"], INCOME_CLIP_QUANTILES)

    if "Age" in cleaned.columns:
        cleaned["Age"] = _clip_outliers(cleaned["Age"], AGE_CLIP_QUANTILES)

    return cleaned


def train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "response",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned dataset into training and validation subsets with stratification.

    Parameters
    ----------
    df :
        Cleaned DataFrame containing (optionally) the label column.
    test_size :
        Fraction of the data to use as the validation set.
    random_state :
        Random seed for reproducibility.
    label_col :
        Name of the label column used for stratification. If absent, no
        stratification is applied.

    Returns
    -------
    train_df, val_df :
        Two DataFrames with disjoint indices.
    """
    stratify_target = df[label_col] if label_col in df.columns else None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
