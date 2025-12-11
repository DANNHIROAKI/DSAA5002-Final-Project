"""Data loading, cleaning, and feature engineering utilities.

This package provides a small, cohesive API for working with the
marketing campaign dataset used in the DSAA 5002 final project. :contentReference[oaicite:3]{index=3}

Typical usage
-------------
>>> from customer_segmentation.src.data import (
...     load_raw_data,
...     clean_data,
...     assemble_feature_table,
... )
>>>
>>> df_raw = load_raw_data(parse_dates=["Dt_Customer"])
>>> df_clean = clean_data(df_raw)
>>> X, y, transformer = assemble_feature_table(df_clean)
>>> from customer_segmentation.src.data import split_behavior_and_response_features
>>> X_beh, X_resp = split_behavior_and_response_features(X, transformer)
"""

from __future__ import annotations

from .load import load_raw_data
from .preprocess import clean_data, train_val_split
from .features import (
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
    "add_response_label",
    "assemble_feature_table",
    "build_rfm_features",
    "add_structural_features",
    "split_behavior_and_response_features",
    "BEHAVIOR_NUMERIC_FEATURES",
    "RESPONSE_NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
]
