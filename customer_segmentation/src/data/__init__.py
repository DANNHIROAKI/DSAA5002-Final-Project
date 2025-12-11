"""Data loading, cleaning, and feature engineering utilities.

This package provides a small, cohesive API for working with the
marketing campaign dataset used in the DSAA 5002 final project. :contentReference[oaicite:8]{index=8}

Typical usage
-------------
>>> from customer_segmentation.src.data import (
...     load_raw_data,
...     clean_data,
...     assemble_feature_table,
... )

>>> df_raw = load_raw_data(parse_dates=["Dt_Customer"])
>>> df_clean = clean_data(df_raw)
>>> X, y, transformer = assemble_feature_table(df_clean)
"""

from __future__ import annotations

from .load import load_raw_data
from .preprocess import clean_data, train_val_split
from .features import (
    add_response_label,
    assemble_feature_table,
    build_rfm_features,
    add_structural_features,
)

__all__ = [
    "load_raw_data",
    "clean_data",
    "train_val_split",
    "add_response_label",
    "assemble_feature_table",
    "build_rfm_features",
    "add_structural_features",
]
