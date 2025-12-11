"""Feature engineering helpers for RFM, demographic, and channel features.

This module now explicitly distinguishes between:

- *behavior features*: used for all clustering models (incl. RAJC-v2),
- *response-related features*: only used in downstream response prediction.

`assemble_feature_table` remains backward compatible: it still returns a single
dense feature matrix, the binary label, and a fitted ColumnTransformer.
In addition, the returned transformer is enriched with metadata attributes:

- `behavior_numeric_features_`
- `response_numeric_features_`
- `categorical_features_`
- `categorical_feature_names_`
- `behavior_feature_names_`   (numeric-behavior + all categorical)
- `response_feature_names_`   (numeric-response only)

These attributes can be used by models such as RAJC-v2 to obtain separate
behavior and response feature matrices from the same encoded DataFrame.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

# Campaign-related columns used to construct the binary response label.
CAMPAIGN_COLUMNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]


def add_response_label(df: pd.DataFrame, label_col: str = "response") -> pd.DataFrame:
    """Construct a binary promotion-response label from campaign fields.

    A customer is considered a responder (label 1) if they have accepted at least
    one of the campaigns or the final offer, otherwise 0.

    Parameters
    ----------
    df :
        Input DataFrame containing the campaign columns.
    label_col :
        Name of the label column to be created.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an additional binary ``label_col`` column.

    Raises
    ------
    KeyError
        If any of the required campaign columns is missing.
    """
    missing_cols = [col for col in CAMPAIGN_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing campaign columns required for label construction: {missing_cols}"
        )

    labeled = df.copy()
    labeled[label_col] = (labeled[CAMPAIGN_COLUMNS].sum(axis=1) > 0).astype(int)
    return labeled


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value features.

    Frequency is defined as the sum of web, catalog, and store purchases.
    Monetary is the sum of all monetary (``Mnt*``) fields.
    Recency uses the existing ``Recency`` column if present.

    Parameters
    ----------
    df :
        Input DataFrame containing purchase information.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with added ``recency``, ``frequency``, and ``monetary`` columns.
    """
    engineered = df.copy()

    # Purchase frequency across all channels.
    engineered["frequency"] = (
        engineered.get("NumWebPurchases", 0)
        + engineered.get("NumCatalogPurchases", 0)
        + engineered.get("NumStorePurchases", 0)
    )

    # Total monetary spending across all Mnt* columns.
    mnt_cols = [col for col in engineered.columns if col.startswith("Mnt")]
    if mnt_cols:
        engineered["monetary"] = engineered[mnt_cols].sum(axis=1)
    else:
        engineered["monetary"] = np.nan

    # Recency: smaller value means more recent purchase, consistent with the dataset.
    engineered["recency"] = engineered.get("Recency", np.nan)

    return engineered


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add family composition, tenure, and channel-mix helper features.

    Added fields
    ------------
    family_size :
        1 (customer) + Kidhome + Teenhome.
    has_child :
        1 if Kidhome + Teenhome > 0, else 0.
    customer_tenure_days :
        (max(Dt_Customer) - Dt_Customer) in days, if join dates are available.
    *_purchase_ratio :
        Ratio of purchases from each channel among total purchases.
    """
    enriched = df.copy()

    # Family structure
    enriched["family_size"] = 1 + enriched.get("Kidhome", 0) + enriched.get("Teenhome", 0)
    enriched["has_child"] = (
        (enriched.get("Kidhome", 0) + enriched.get("Teenhome", 0)) > 0
    ).astype(int)

    # Ensure Dt_Customer is datetime if present.
    if "Dt_Customer" in enriched.columns and not np.issubdtype(
        enriched["Dt_Customer"].dtype, np.datetime64
    ):
        enriched["Dt_Customer"] = pd.to_datetime(
            enriched["Dt_Customer"], errors="coerce"
        )

    if "Dt_Customer" in enriched.columns:
        latest_date = enriched["Dt_Customer"].max()
        enriched["customer_tenure_days"] = (latest_date - enriched["Dt_Customer"]).dt.days
    else:
        enriched["customer_tenure_days"] = np.nan

    # Channel mix ratios
    total_purchases = enriched.get("frequency", 0).astype(float)
    web = enriched.get("NumWebPurchases", 0).astype(float)
    catalog = enriched.get("NumCatalogPurchases", 0).astype(float)
    store = enriched.get("NumStorePurchases", 0).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["web_purchase_ratio"] = np.where(
            total_purchases > 0, web / total_purchases, 0.0
        )
        enriched["catalog_purchase_ratio"] = np.where(
            total_purchases > 0, catalog / total_purchases, 0.0
        )
        enriched["store_purchase_ratio"] = np.where(
            total_purchases > 0, store / total_purchases, 0.0
        )

    return enriched


# ---------------------------------------------------------------------------
# Feature group definitions (behavior vs response-related)
# ---------------------------------------------------------------------------

# Numeric behavior features used for clustering (RFM + channels + demographics).
BEHAVIOR_NUMERIC_FEATURES: List[str] = [
    "recency",
    "frequency",
    "monetary",
    "Income",
    "Age",
    "Kidhome",
    "Teenhome",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "customer_tenure_days",
    "web_purchase_ratio",
    "catalog_purchase_ratio",
    "store_purchase_ratio",
]

# Numeric response-related features used only in downstream prediction
# (they tend to be strongly correlated with the label but are conceptually
# not part of "core behavior clustering space").
RESPONSE_NUMERIC_FEATURES: List[str] = [
    "NumDealsPurchases",
    "NumWebVisitsMonth",
]

# Categorical features (all treated as behavior/demographics).
CATEGORICAL_FEATURES: List[str] = ["Education", "Marital_Status"]


# ---------------------------------------------------------------------------
# Main entry point(s)
# ---------------------------------------------------------------------------


def assemble_feature_table(
    df: pd.DataFrame,
    label_col: str = "response",
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Combine engineered features, scale numerics, and one-hot encode categoricals.

    This is the main entry point for building the modeling-ready feature matrix.
    It is **backward compatible** with the original implementation, but additionally
    annotates the returned ColumnTransformer with feature-group metadata so that
    callers can easily split behavior vs response-related features.

    Parameters
    ----------
    df :
        Raw or cleaned marketing campaign data.
    label_col :
        Name of the binary label column (constructed via :func:`add_response_label`).

    Returns
    -------
    features_df :
        Dense feature matrix ready for modeling (rows aligned with ``df``).
    labels :
        Binary promotion-response label.
    transformer :
        Fitted :class:`~sklearn.compose.ColumnTransformer` that encapsulates numeric
        scaling and categorical one-hot encoding, with extra attributes:

        - ``behavior_numeric_features_``
        - ``response_numeric_features_``
        - ``categorical_features_``
        - ``categorical_feature_names_``
        - ``behavior_feature_names_``
        - ``response_feature_names_``
    """
    # 1) RFM + structural features
    engineered = build_rfm_features(df)
    engineered = add_structural_features(engineered)

    # 2) Overall promotion-response label
    engineered = add_response_label(engineered, label_col=label_col)

    # Categorical columns to one-hot encode (intersection with available cols).
    categorical_cols: List[str] = [
        col for col in CATEGORICAL_FEATURES if col in engineered.columns
    ]

    # Numeric behavior vs response-related columns (also intersect with df).
    behavior_numeric_cols: List[str] = [
        col for col in BEHAVIOR_NUMERIC_FEATURES if col in engineered.columns
    ]
    response_numeric_cols: List[str] = [
        col for col in RESPONSE_NUMERIC_FEATURES if col in engineered.columns
    ]

    # Combined numeric columns for the ColumnTransformer.
    numeric_cols: List[str] = behavior_numeric_cols + response_numeric_cols

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # always return dense arrays for easier downstream use
    )

    transformed = transformer.fit_transform(engineered)

    # Build human-readable column names for the encoded feature matrix.
    cat_feature_names: List[str] = []
    if categorical_cols:
        cat_encoder = transformer.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))

    feature_names = numeric_cols + cat_feature_names

    features_df = pd.DataFrame(
        transformed, columns=feature_names, index=engineered.index
    )
    labels = engineered[label_col].astype(int)

    # ------------------------------------------------------------------
    # Enrich the transformer with metadata about feature groups.
    # ------------------------------------------------------------------
    behavior_feature_names: List[str] = behavior_numeric_cols + cat_feature_names
    response_feature_names: List[str] = response_numeric_cols

    transformer.behavior_numeric_features_ = behavior_numeric_cols
    transformer.response_numeric_features_ = response_numeric_cols
    transformer.categorical_features_ = categorical_cols
    transformer.categorical_feature_names_ = cat_feature_names
    transformer.behavior_feature_names_ = behavior_feature_names
    transformer.response_feature_names_ = response_feature_names

    return features_df, labels, transformer


def split_behavior_and_response_features(
    features_df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a full feature matrix into behavior vs response-related sub-matrices.

    This utility is mainly intended for RAJC-v2 and downstream experiments that
    want to treat behavior features (for clustering) and response-related features
    (for prediction) differently, while still reusing the same encoded matrix.

    Parameters
    ----------
    features_df :
        Full encoded feature matrix returned by :func:`assemble_feature_table`.
    transformer :
        The fitted ColumnTransformer returned by :func:`assemble_feature_table`.

    Returns
    -------
    X_behavior, X_response :
        Two DataFrames containing behavior features (including all categorical
        demographics) and response-related features, respectively.
    """
    # Prefer using metadata attached by assemble_feature_table; if unavailable,
    # fall back to a simple name-based split.
    behavior_cols = getattr(transformer, "behavior_feature_names_", None)
    response_cols = getattr(transformer, "response_feature_names_", None)

    if behavior_cols is None or response_cols is None:
        # Fallback: treat RESPONSE_NUMERIC_FEATURES as response, others as behavior.
        response_cols = [c for c in features_df.columns if c in RESPONSE_NUMERIC_FEATURES]
        behavior_cols = [c for c in features_df.columns if c not in response_cols]

    X_behavior = features_df[behavior_cols].copy()
    X_response = features_df[response_cols].copy()

    return X_behavior, X_response
