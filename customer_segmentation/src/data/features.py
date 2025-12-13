# src/data/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LabelMode = Literal["recent", "any"]

# Historical campaign columns (safe to use as "past behavior" features)
CAMPAIGN_HISTORY_COLUMNS: List[str] = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]

# Most recent campaign response column (used as the default label in the upgraded method)
CAMPAIGN_RESPONSE_COLUMN = "Response"

# For backward compatibility: y = 1 if any campaign accepted in history + recent
CAMPAIGN_COLUMNS_ANY: List[str] = CAMPAIGN_HISTORY_COLUMNS + [CAMPAIGN_RESPONSE_COLUMN]

DEFAULT_LABEL_COL = "campaign_response"


def add_response_label(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    mode: LabelMode = "recent",
    campaign_cols: List[str] | None = None,
    response_col: str = CAMPAIGN_RESPONSE_COLUMN,
) -> pd.DataFrame:
    """Create a binary response label.

    Parameters
    ----------
    mode:
        - "recent" (default): y = Response (the most recent campaign response).
          This is the recommended target for the upgraded method.
        - "any": y = 1 if accepted any of {AcceptedCmp1-5, Response}.
          This keeps compatibility with the earlier definition.
    """
    labeled = df.copy()

    if mode == "recent":
        if response_col not in labeled.columns:
            raise KeyError(f"Missing response column '{response_col}' in DataFrame.")
        labeled[label_col] = labeled[response_col].astype(int)
        return labeled

    if mode == "any":
        if campaign_cols is None:
            campaign_cols = CAMPAIGN_COLUMNS_ANY
        missing = [c for c in campaign_cols if c not in labeled.columns]
        if missing:
            raise KeyError(f"Missing campaign columns in DataFrame: {missing}")
        labeled[label_col] = (labeled[campaign_cols].sum(axis=1) > 0).astype(int)
        return labeled

    raise ValueError(f"Unknown label mode: {mode}. Expected 'recent' or 'any'.")


def add_response_history_features(
    df: pd.DataFrame,
    history_cols: List[str] | None = None,
    past_accept_cnt_col: str = "PastAcceptCnt",
    ever_accepted_col: str = "EverAccepted",
) -> pd.DataFrame:
    """Derive historical campaign engagement features from AcceptedCmp1-5.

    These are safe "past behavior" features and should NOT be merged into the label
    when the target is the most recent response (Response).
    """
    if history_cols is None:
        history_cols = CAMPAIGN_HISTORY_COLUMNS

    out = df.copy()
    missing = [c for c in history_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing history campaign columns in DataFrame: {missing}")

    out[past_accept_cnt_col] = out[history_cols].sum(axis=1).astype(int)
    out[ever_accepted_col] = (out[past_accept_cnt_col] > 0).astype(int)
    return out


# --------- Feature name constants (for external modules) ---------

BEHAVIOR_NUMERIC_FEATURES: List[str] = [
    # RFM + spending
    "Recency",
    "Frequency",
    "Monetary",
    "Spent",
    # Demographics / structure
    "Income",
    "Age",
    "Kidhome",
    "Teenhome",
    "family_size",
    "has_child",
    "customer_tenure_days",
    # Basket / preference
    "Avg_Basket",
    "Deal_Sensitivity",
    "Online_Intensity",
    "Variety_Index",
    # Raw channel counts
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    # Channel ratios (new: ensure they are actually part of clustering space)
    "Web_Ratio",
    "Catalog_Ratio",
    "Store_Ratio",
    # Complaint flag
    "Complain",
]

RESPONSE_NUMERIC_FEATURES: List[str] = [
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    # Historical acceptance features (new)
    "PastAcceptCnt",
    "EverAccepted",
]

CATEGORICAL_FEATURES: List[str] = [
    "Education",
    "Marital_Status",
]


# --------- Feature engineering helpers ---------


def _log1p_safe(series: pd.Series) -> pd.Series:
    """Apply log1p to non-negative long-tailed amounts safely."""
    s = series.clip(lower=0)
    return np.log1p(s)


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Frequency / Monetary / Spent from raw dataset columns."""
    out = df.copy()

    # Frequency
    freq_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    for col in freq_cols:
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' not found in DataFrame.")
    out["Frequency"] = out[freq_cols].sum(axis=1)

    # Monetary / Spent
    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    for col in mnt_cols:
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' not found in DataFrame.")
    out["Monetary"] = out[mnt_cols].sum(axis=1)
    out["Spent"] = out["Monetary"]

    # Recency must exist
    if "Recency" not in out.columns:
        raise KeyError("Expected column 'Recency' not found in DataFrame.")

    return out


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add household / basket / preference / channel ratio features.

    Requires build_rfm_features() to have been called (needs Frequency / Spent).
    """
    out = df.copy()

    # Household structure
    kid = out.get("Kidhome", 0)
    teen = out.get("Teenhome", 0)
    out["family_size"] = 1 + kid + teen
    out["has_child"] = ((kid + teen) > 0).astype(int)

    # Ensure Frequency exists
    if "Frequency" not in out.columns:
        raise KeyError("Please call build_rfm_features() before add_structural_features().")

    freq = out["Frequency"].replace(0, np.nan)

    # Avg basket
    out["Avg_Basket"] = out["Spent"] / freq

    # Deal sensitivity
    out["Deal_Sensitivity"] = out["NumDealsPurchases"] / freq if "NumDealsPurchases" in out.columns else 0.0

    # Online intensity (purchase ratio proxy)
    out["Online_Intensity"] = out["NumWebPurchases"] / freq if "NumWebPurchases" in out.columns else 0.0

    # Channel ratios
    for col in ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]:
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' for channel features.")
    out["Web_Ratio"] = out["NumWebPurchases"] / freq
    out["Catalog_Ratio"] = out["NumCatalogPurchases"] / freq
    out["Store_Ratio"] = out["NumStorePurchases"] / freq

    # Variety index
    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    out["Variety_Index"] = (out[mnt_cols] > 0).sum(axis=1)

    # Fill NaNs from division by zero
    ratio_cols = [
        "Avg_Basket",
        "Deal_Sensitivity",
        "Online_Intensity",
        "Web_Ratio",
        "Catalog_Ratio",
        "Store_Ratio",
    ]
    out[ratio_cols] = out[ratio_cols].fillna(0.0)

    return out


@dataclass
class FeatureConfig:
    """Configure which raw columns belong to behavior / response / categorical groups."""
    behavior_numeric: List[str]
    response_numeric: List[str]
    categorical: List[str]


def _default_feature_config(df: pd.DataFrame) -> FeatureConfig:
    """Default feature config based on the upgraded method."""
    behavior_numeric = list(BEHAVIOR_NUMERIC_FEATURES)
    response_numeric = list(RESPONSE_NUMERIC_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)

    # Keep only columns that exist in df
    behavior_numeric = [c for c in behavior_numeric if c in df.columns]
    response_numeric = [c for c in response_numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]

    return FeatureConfig(
        behavior_numeric=behavior_numeric,
        response_numeric=response_numeric,
        categorical=categorical,
    )


def _make_onehot_encoder() -> OneHotEncoder:
    """Make OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # sklearn<1.2 uses "sparse"
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def assemble_feature_table(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    label_mode: LabelMode = "recent",
    feature_config: FeatureConfig | None = None,
    transformer: ColumnTransformer | None = None,
    fit: bool = True,
    auto_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Build the model feature table with a leakage-free fit/transform option.

    Parameters
    ----------
    transformer:
        - If None: a new ColumnTransformer is created and fit=True is enforced.
        - If provided: set fit=False to transform validation/test without refitting.
    fit:
        Whether to fit the transformer on this df. Use fit=True for train, fit=False for val/test.
    auto_feature_engineering:
        If True, automatically generate required engineered columns (RFM/structure/history)
        when they are missing. This prevents silent feature-drop bugs.

    Returns
    -------
    features_df:
        Transformed feature matrix as a pandas DataFrame.
    labels:
        Binary target label series.
    transformer:
        The fitted (or reused) transformer with stored feature-name metadata.
    """
    work = df.copy()

    # 0) Auto-generate required engineered columns to avoid silent feature drops.
    if auto_feature_engineering:
        # RFM
        if any(col not in work.columns for col in ["Frequency", "Monetary", "Spent"]):
            work = build_rfm_features(work)
        # structure & ratios
        if any(col not in work.columns for col in ["family_size", "Avg_Basket", "Variety_Index", "Web_Ratio", "Catalog_Ratio", "Store_Ratio"]):
            work = add_structural_features(work)
        # history response features
        if any(col not in work.columns for col in ["PastAcceptCnt", "EverAccepted"]):
            work = add_response_history_features(work)

    # 1) Ensure label exists (default: y = Response)
    if label_col not in work.columns:
        work = add_response_label(work, label_col=label_col, mode=label_mode)

    # 2) Log transform for long-tailed amount-like columns
    amount_cols = [
        "Income",
        "Monetary",
        "Spent",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    for col in amount_cols:
        if col in work.columns:
            work[col] = _log1p_safe(work[col])

    # 3) Feature config
    if feature_config is None:
        feature_config = _default_feature_config(work)

    numeric_features_all = feature_config.behavior_numeric + feature_config.response_numeric
    categorical_features = feature_config.categorical

    # 4) Build / reuse transformer
    if transformer is None:
        numeric_transformer = StandardScaler()
        transformers = [("numeric", numeric_transformer, numeric_features_all)]
        if categorical_features:
            transformers.append(("categorical", _make_onehot_encoder(), categorical_features))

        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        fit = True  # must fit a fresh transformer

    # 5) Fit/transform
    if fit:
        transformed = transformer.fit_transform(work)

        # numeric names -> lower-case
        numeric_feature_names = [col.lower() for col in numeric_features_all]

        # categorical names -> lower-case (if any)
        cat_feature_names: list[str] = []
        if categorical_features:
            cat_ohe = transformer.named_transformers_["categorical"]
            try:
                cat_feature_names = list(cat_ohe.get_feature_names_out(categorical_features))
            except AttributeError:
                # very old sklearn
                cat_feature_names = list(cat_ohe.get_feature_names(categorical_features))
            cat_feature_names = [c.lower() for c in cat_feature_names]

        all_feature_names = numeric_feature_names + cat_feature_names

        # record split meta
        n_beh = len(feature_config.behavior_numeric)
        behavior_numeric_names = numeric_feature_names[:n_beh]
        response_numeric_names = numeric_feature_names[n_beh:]

        # By default: categorical features are treated as part of behavior space.
        behavior_feature_names = behavior_numeric_names + cat_feature_names
        response_feature_names = response_numeric_names

        # attach metadata to transformer for later val/test transform & splitting
        transformer.feature_config_ = feature_config
        transformer.numeric_features_all_ = numeric_features_all
        transformer.categorical_features_ = categorical_features

        transformer.all_feature_names_ = all_feature_names
        transformer.behavior_feature_names_ = behavior_feature_names
        transformer.response_feature_names_ = response_feature_names

    else:
        # transform only: require feature-name metadata from train-fit transformer
        if not hasattr(transformer, "all_feature_names_"):
            raise AttributeError(
                "Transformer has no stored feature names. "
                "Fit it on the training set first (assemble_feature_table(..., fit=True))."
            )
        transformed = transformer.transform(work)
        all_feature_names = list(transformer.all_feature_names_)

    features_df = pd.DataFrame(transformed, columns=all_feature_names, index=df.index)
    labels = work[label_col].astype(int)

    return features_df, labels, transformer


def split_behavior_and_response_features(
    features_df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split transformed features into behavior part and response-related part.

    - behavior_features: used for clustering / gating (x_beh)
    - response_features: used for response prediction only (x_resp)
    """
    if not hasattr(transformer, "behavior_feature_names_"):
        raise AttributeError(
            "transformer.behavior_feature_names_ not found. "
            "Please fit transformer via assemble_feature_table(..., fit=True) first."
        )

    behavior_cols = [c for c in transformer.behavior_feature_names_ if c in features_df.columns]
    response_cols = [c for c in transformer.response_feature_names_ if c in features_df.columns]

    behavior_features = features_df[behavior_cols].copy()
    response_features = features_df[response_cols].copy()

    return behavior_features, response_features
