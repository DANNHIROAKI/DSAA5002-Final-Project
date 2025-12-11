"""Feature engineering helpers for RFM and demographic features."""
import pandas as pd


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value features."""
    engineered = df.copy()
    engineered["frequency"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )
    mnt_cols = [col for col in df.columns if col.startswith("Mnt")]
    engineered["monetary"] = df[mnt_cols].sum(axis=1)
    engineered["recency"] = df["Recency"]
    return engineered


def assemble_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Combine baseline and engineered features into a single table."""
    # TODO: add scaling, encoding, and additional derived features
    return build_rfm_features(df)
