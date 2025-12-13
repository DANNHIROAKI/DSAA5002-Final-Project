"""Data loading, cleaning, and feature engineering utilities.

This package provides a small, cohesive API for working with the Kaggle
*Customer Personality Analysis* ("marketing_campaign.csv") dataset used in
the DSAA 5002 final project.

Key design choices (for the upgraded method)
--------------------------------------------
- Default target label is the most recent campaign response, i.e. raw column
  ``Response`` (see ``add_response_label(..., mode="recent")``).
- Historical campaign acceptances (``AcceptedCmp1``-``AcceptedCmp5``) are
  transformed into features (``PastAcceptCnt`` / ``EverAccepted``) and are NOT
  merged into the label by default.
- Feature transformation (scaling / one-hot encoding) supports a leakage-free
  protocol: fit on train only, then transform on validation/test via
  ``assemble_feature_table(..., fit=False, transformer=preprocessor)``.

Typical usage
-------------
>>> from customer_segmentation.src.data import (
...     load_raw_data, clean_data,
...     build_rfm_features, add_structural_features,
...     add_response_history_features, add_response_label,
...     train_val_test_split, assemble_feature_table,
... )
>>> df_raw = load_raw_data()
>>> df = clean_data(df_raw)
>>> df = build_rfm_features(df)
>>> df = add_structural_features(df)
>>> df = add_response_history_features(df)
>>> df = add_response_label(df, mode="recent")  # y = Response
>>> train_df, val_df, test_df = train_val_test_split(df, stratify_col="campaign_response")
>>> X_train, y_train, pre = assemble_feature_table(train_df, fit=True)
>>> X_val, y_val, _ = assemble_feature_table(val_df, transformer=pre, fit=False)
"""

from __future__ import annotations

from .load import load_raw_data
from .preprocess import clean_data, train_val_split, train_val_test_split
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
    "load_raw_data",
    "clean_data",
    "train_val_split",
    "train_val_test_split",
    "FeatureConfig",
    "DEFAULT_LABEL_COL",
    "add_response_history_features",
    "add_response_label",
    "assemble_feature_table",
    "build_rfm_features",
    "add_structural_features",
    "split_behavior_and_response_features",
    "BEHAVIOR_NUMERIC_FEATURES",
    "RESPONSE_NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
]
