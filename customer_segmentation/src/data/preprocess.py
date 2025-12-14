"""Deterministic dataset cleaning and splitting utilities.

This module purposely keeps *data-dependent* preprocessing (imputation, outlier
clipping, scaling, encoding) out of :func:`clean_data` to support a strict
leakage-free workflow:

1) split raw rows into train/val/test;
2) fit any preprocessing parameters on *train only*;
3) reuse the fitted transformers to transform val/test.

In this repository, the train-only preprocessing lives in
``customer_segmentation.src.data.features.assemble_feature_table``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Kaggle dataset commonly stores dates as dd-mm-YYYY.
DEFAULT_DATE_FORMAT = "%d-%m-%Y"

# The dataset's observation window ends at 2014-06-29 (verified on the provided CSV).
# We keep these fixed to avoid any split-dependent leakage.
DEFAULT_REFERENCE_DATE = pd.Timestamp("2014-06-29")
DEFAULT_REFERENCE_YEAR = 2014

# Hard, domain-reasonable bounds (deterministic, not data-driven)
AGE_BOUNDS = (0, 120)


def _parse_customer_date(
    s: pd.Series,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> pd.Series:
    """Parse Dt_Customer robustly.

    We intentionally *do not* rely on pandas' generic parser because strings like
    "04-09-2012" are ambiguous (day-first vs month-first). We instead use the
    dataset's documented format (dd-mm-YYYY).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, format=date_format, errors="coerce")


def clean_data(
    df: pd.DataFrame,
    date_format: str = DEFAULT_DATE_FORMAT,
    reference_date: pd.Timestamp | str | None = DEFAULT_REFERENCE_DATE,
    reference_year: int | None = DEFAULT_REFERENCE_YEAR,
    drop_identifier_columns: bool = True,
    drop_raw_date_columns: bool = True,
) -> pd.DataFrame:
    """Apply *deterministic* cleaning and feature derivation.

    This function performs only operations that do not estimate statistics from
    data distributions (so it is safe to call before splitting).

    Steps
    -----
    1) Drop duplicate rows.
    2) Parse ``Dt_Customer`` using a fixed format and derive ``customer_tenure_days``
       relative to a fixed ``reference_date``.
    3) Derive ``Age`` from ``Year_Birth`` relative to a fixed ``reference_year``.
    4) Apply simple deterministic sanity rules (e.g., invalid ages -> NaN).
    5) Optionally drop identifier and raw date columns.

    Notes
    -----
    - Missing value imputation and outlier clipping are intentionally deferred
      to the train-fitted transformer in ``assemble_feature_table``.
    """
    out = df.copy()

    # 1) Deduplicate
    out = out.drop_duplicates().reset_index(drop=True)

    # 2) Parse date and compute tenure
    if "Dt_Customer" in out.columns:
        out["Dt_Customer"] = _parse_customer_date(out["Dt_Customer"], date_format)
        ref = pd.Timestamp(DEFAULT_REFERENCE_DATE if reference_date is None else reference_date)
        tenure = (ref - out["Dt_Customer"]).dt.days
        out["customer_tenure_days"] = tenure.fillna(0).clip(lower=0)
    else:
        out["customer_tenure_days"] = 0

    # 3) Compute Age
    ref_year = DEFAULT_REFERENCE_YEAR if reference_year is None else int(reference_year)

    if "Year_Birth" in out.columns:
        out["Age"] = ref_year - out["Year_Birth"]
    else:
        out["Age"] = np.nan

    # 4) Deterministic sanity checks
    lo, hi = AGE_BOUNDS
    out.loc[(out["Age"] < lo) | (out["Age"] > hi), "Age"] = np.nan

    if "Income" in out.columns:
        # Negative income is invalid; mark as missing.
        out.loc[out["Income"] < 0, "Income"] = np.nan

    # These are constant columns in this dataset and should never be used.
    for col in ("Z_CostContact", "Z_Revenue"):
        if col in out.columns:
            out = out.drop(columns=[col])

    # 5) Drop identifier / raw-date columns to avoid accidental usage.
    if drop_identifier_columns:
        for col in ("ID",):
            if col in out.columns:
                out = out.drop(columns=[col])

    if drop_raw_date_columns:
        for col in ("Dt_Customer", "Year_Birth"):
            if col in out.columns:
                out = out.drop(columns=[col])

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
    """Train/validation/test split helper.

    Notes
    -----
    This split is performed at the *row level*. Downstream feature transformers
    must still be fit on train only.
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

    rel_val_size = val_size / (1.0 - test_size)
    stratify_vals_tv = train_val_df[stratify_col] if stratify_col is not None else None

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
