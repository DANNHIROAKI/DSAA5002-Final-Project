# src/data/preprocess.py
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATE_FORMAT = "%d-%m-%Y"  # Kaggle 原始数据常见格式：dd-mm-YYYY
DEFAULT_CLIP_QUANTILES: Tuple[float, float] = (0.01, 0.99)
DEFAULT_CLIP_COLUMNS: Tuple[str, ...] = ("Age", "Income")


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Numerical features -> median; categorical features -> mode."""
    out = df.copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    categorical_cols = out.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        median = out[col].median()
        out[col] = out[col].fillna(median)

    for col in categorical_cols:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            out[col] = out[col].fillna(mode.iloc[0])
        else:
            out[col] = out[col].fillna("Unknown")

    return out


def _clip_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorize a numerical series by quantiles to reduce extreme values."""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def clean_data(
    df: pd.DataFrame,
    date_format: str = DEFAULT_DATE_FORMAT,
    reference_year: int | None = None,
    clip_quantiles: Tuple[float, float] = DEFAULT_CLIP_QUANTILES,
    clip_columns: Iterable[str] = DEFAULT_CLIP_COLUMNS,
) -> pd.DataFrame:
    """Clean the raw marketing_campaign dataset.

    Steps
    -----
    1) Drop duplicates
    2) Parse Dt_Customer using a fixed date format and compute customer_tenure_days
    3) Compute Age from Year_Birth (reference_year inferred from data if not provided)
    4) Impute missing values
    5) Winsorize selected long-tailed continuous columns (default: Age & Income)
    """
    out = df.copy()

    # 1) Deduplicate
    out = out.drop_duplicates().reset_index(drop=True)

    # 2) Parse date and compute tenure
    inferred_year: int | None = None
    if "Dt_Customer" in out.columns:
        out["Dt_Customer"] = pd.to_datetime(out["Dt_Customer"], format=date_format, errors="coerce")
        max_date = out["Dt_Customer"].max()
        if pd.notna(max_date):
            inferred_year = int(max_date.year)
            out["customer_tenure_days"] = (max_date - out["Dt_Customer"]).dt.days
            out["customer_tenure_days"] = out["customer_tenure_days"].fillna(0).clip(lower=0)
        else:
            out["customer_tenure_days"] = 0
    else:
        out["customer_tenure_days"] = 0

    # 3) Compute Age
    if reference_year is None:
        # Prefer a dataset-consistent year; fall back to current year if Dt_Customer is missing/broken.
        reference_year = inferred_year if inferred_year is not None else pd.Timestamp.today().year

    if "Year_Birth" in out.columns:
        out["Age"] = reference_year - out["Year_Birth"]
    else:
        out["Age"] = np.nan

    # 4) Missing value imputation
    out = _fill_missing(out)

    # 5) Winsorization for selected continuous columns
    lower_q, upper_q = clip_quantiles
    for col in clip_columns:
        if col in out.columns:
            # Only apply to numeric dtypes.
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = _clip_outliers(out[col], lower_q, upper_q)

    return out


def train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple train/validation split (kept for backward compatibility)."""
    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise KeyError(f"Column '{stratify_col}' not found for stratification.")
        stratify_vals = df[stratify_col]
    else:
        stratify_vals = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_vals,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-friendly train/val/test split.

    Notes
    -----
    - The split is performed on the *row-level dataset*.
    - Feature transformers must be fit on train only (see assemble_feature_table).
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.")

    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise KeyError(f"Column '{stratify_col}' not found for stratification.")
        stratify_vals = df[stratify_col]
    else:
        stratify_vals = None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vals,
    )

    # Split validation from the remaining portion
    rel_val_size = val_size / (1.0 - test_size)

    if stratify_col is not None:
        stratify_vals_tv = train_val_df[stratify_col]
    else:
        stratify_vals_tv = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=rel_val_size,
        random_state=random_state,
        stratify=stratify_vals_tv,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
