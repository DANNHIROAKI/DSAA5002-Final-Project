"""Data loading, deterministic cleaning, and leakage-free feature building.

This package implements a **two-stage** data workflow:

1) **Deterministic row cleaning** (:func:`clean_data`)
   - safe to call *before* train/val/test splitting
   - derives only stable columns (Age, customer_tenure_days)
   - does **not** impute or winsorize

2) **Train-fitted preprocessing + feature table** (:func:`assemble_feature_table`)
   - imputation, quantile clipping (winsorization), scaling, one-hot encoding
   - **fit on train only**, then reused for val/test

The design is tailored for the course default customer segmentation setting
and the upgraded response-aware methods.
"""

from __future__ import annotations

from .load import load_raw_data
from .preprocess import (
    AGE_BOUNDS,
    DEFAULT_DATE_FORMAT,
    DEFAULT_REFERENCE_DATE,
    DEFAULT_REFERENCE_YEAR,
    clean_data,
    train_val_split,
    train_val_test_split,
)
from .features import (
    FeatureConfig,
    DEFAULT_LABEL_COL,
    add_response_history_features,
    add_response_label,
    assemble_feature_table,
    build_rfm_features,
    add_structural_features,
    split_behavior_and_response_features,
    BEHAVIOR_NUMERIC_FEATURES,
    RESPONSE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

__all__ = [
    # loading
    "load_raw_data",
    # deterministic cleaning
    "clean_data",
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_REFERENCE_DATE",
    "DEFAULT_REFERENCE_YEAR",
    "AGE_BOUNDS",
    # splitting
    "train_val_split",
    "train_val_test_split",
    # labels & feature engineering
    "FeatureConfig",
    "DEFAULT_LABEL_COL",
    "add_response_history_features",
    "add_response_label",
    "build_rfm_features",
    "add_structural_features",
    # leakage-free feature tables
    "assemble_feature_table",
    "split_behavior_and_response_features",
    # feature lists
    "BEHAVIOR_NUMERIC_FEATURES",
    "RESPONSE_NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
]
