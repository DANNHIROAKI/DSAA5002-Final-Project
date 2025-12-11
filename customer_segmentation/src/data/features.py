"""Feature engineering helpers for RFM and demographic features."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def assemble_feature_table(
    df: pd.DataFrame,
    label_col: str = "response",
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Combine engineered features, scale numerics, and one-hot encode categoricals.

    This is the main entry point for building the modeling-ready feature matrix.

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
        scaling and categorical one-hot encoding, useful for inference or other models.
    """
    # 1) RFM + structural features
    engineered = build_rfm_features(df)
    engineered = add_structural_features(engineered)

    # 2) Overall promotion-response label
    engineered = add_response_label(engineered, label_col=label_col)

    # Categorical columns to one-hot encode
    categorical_cols: List[str] = [
        col for col in ["Education", "Marital_Status"] if col in engineered.columns
    ]

    # Numeric columns to scale
    numeric_cols: List[str] = [
        col
        for col in [
            "recency",
            "frequency",
            "monetary",
            "Income",
            "Age",
            "Kidhome",
            "Teenhome",
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
            "NumWebVisitsMonth",
            "customer_tenure_days",
            "web_purchase_ratio",
            "catalog_purchase_ratio",
            "store_purchase_ratio",
        ]
        if col in engineered.columns
    ]

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

    return features_df, labels, transformer
