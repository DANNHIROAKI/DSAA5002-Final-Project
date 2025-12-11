"""Feature engineering helpers for RFM and demographic features."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAMPAIGN_COLUMNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]


def add_response_label(df: pd.DataFrame, label_col: str = "response") -> pd.DataFrame:
    """Construct the binary promotion-response label ``response`` from campaign fields."""

    missing_cols = [col for col in CAMPAIGN_COLUMNS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing campaign columns required for label construction: {missing_cols}")

    labeled = df.copy()
    labeled[label_col] = labeled[CAMPAIGN_COLUMNS].sum(axis=1) > 0
    labeled[label_col] = labeled[label_col].astype(int)
    return labeled


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value features."""

    engineered = df.copy()
    engineered["frequency"] = (
        engineered.get("NumWebPurchases", 0)
        + engineered.get("NumCatalogPurchases", 0)
        + engineered.get("NumStorePurchases", 0)
    )
    mnt_cols = [col for col in engineered.columns if col.startswith("Mnt")]
    engineered["monetary"] = engineered[mnt_cols].sum(axis=1)
    engineered["recency"] = engineered.get("Recency", np.nan)
    return engineered


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add family composition, tenure, and channel mix helpers."""

    enriched = df.copy()
    enriched["family_size"] = 1 + enriched.get("Kidhome", 0) + enriched.get("Teenhome", 0)
    enriched["has_child"] = ((enriched.get("Kidhome", 0) + enriched.get("Teenhome", 0)) > 0).astype(int)

    if "Dt_Customer" in enriched and not np.issubdtype(enriched["Dt_Customer"].dtype, np.datetime64):
        enriched["Dt_Customer"] = pd.to_datetime(enriched["Dt_Customer"], errors="coerce")

    if "Dt_Customer" in enriched:
        latest_date = enriched["Dt_Customer"].max()
        enriched["customer_tenure_days"] = (latest_date - enriched["Dt_Customer"]).dt.days

    total_purchases = enriched.get("frequency", 0)
    web = enriched.get("NumWebPurchases", 0)
    catalog = enriched.get("NumCatalogPurchases", 0)
    store = enriched.get("NumStorePurchases", 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["web_purchase_ratio"] = np.where(total_purchases > 0, web / total_purchases, 0)
        enriched["catalog_purchase_ratio"] = np.where(total_purchases > 0, catalog / total_purchases, 0)
        enriched["store_purchase_ratio"] = np.where(total_purchases > 0, store / total_purchases, 0)

    return enriched


def assemble_feature_table(
    df: pd.DataFrame, label_col: str = "response"
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Combine engineered features, scale numeric fields, and one-hot encode categoricals.

    Returns
    -------
    features_df : pd.DataFrame
        Dense feature matrix ready for modeling.
    labels : pd.Series
        Binary promotion-response label.
    transformer : ColumnTransformer
        Fitted transformer for consistent preprocessing at inference time.
    """

    engineered = build_rfm_features(df)
    engineered = add_structural_features(engineered)
    engineered = add_response_label(engineered, label_col=label_col)

    categorical_cols: List[str] = [
        col
        for col in ["Education", "Marital_Status"]
        if col in engineered.columns
    ]

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
        sparse_threshold=0.0,
    )

    transformed = transformer.fit_transform(engineered)
    cat_feature_names = transformer.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_feature_names)

    features_df = pd.DataFrame(transformed, columns=feature_names, index=engineered.index)
    labels = engineered[label_col]

    return features_df, labels, transformer
