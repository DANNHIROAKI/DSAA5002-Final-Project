"""Feature engineering and leakage-free preprocessing.

This module builds two families of features:

* **Behavioral features** ``x_beh`` used for segmentation / gating.
* **Response-related features** ``x_resp`` used only for response prediction.

The key design goal is *strict* leakage-free evaluation:

1) raw rows are split into train/val/test;
2) preprocessing parameters (imputation, winsorization, scaling, encoding) are
   **fit on train only**;
3) the fitted transformer is reused to transform val/test.

The main entry point is :func:`assemble_feature_table`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import DEFAULT_DATE_FORMAT, DEFAULT_REFERENCE_DATE, DEFAULT_REFERENCE_YEAR


LabelMode = Literal["recent", "any"]


# -----------------------------------------------------------------------------
# Label + history feature helpers
# -----------------------------------------------------------------------------

# Historical campaign columns (safe to use as "past behavior" features)
CAMPAIGN_HISTORY_COLUMNS: List[str] = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]

# Most recent campaign response column (used as the default label)
CAMPAIGN_RESPONSE_COLUMN = "Response"

# Backward-compatible label definition: any accepted across history + recent
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
        - ``"recent"`` (default): :math:`y = \mathrm{Response}`.
        - ``"any"``: :math:`y = 1` if accepted any of
          {AcceptedCmp1-5, Response}. This keeps compatibility with an older
          (less strict) label definition.
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
    """Derive historical engagement features from AcceptedCmp1-5.

    These are *safe* predictors when the target is the most recent campaign
    response (``Response``), because they only summarize *past* campaigns.
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


# -----------------------------------------------------------------------------
# Feature name constants (used by external modules)
# -----------------------------------------------------------------------------

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
    "Children",
    "family_size",
    "has_child",
    "customer_tenure_days",
    # Basket / preference
    "Avg_Basket",
    "Deal_Sensitivity",
    "Online_Intensity",
    "WebVisits_per_purchase",
    "Variety_Index",
    # Raw channel counts
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    # Channel ratios
    "Web_Ratio",
    "Catalog_Ratio",
    "Store_Ratio",
    # Economic ratios
    "Spent_to_income",
    "Income_per_member",
    "Spent_per_member",
    # Complaint flag
    "Complain",
]

RESPONSE_NUMERIC_FEATURES: List[str] = [
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "PastAcceptCnt",
    "EverAccepted",
]

CATEGORICAL_FEATURES: List[str] = [
    "Education",
    "Marital_Status",
]


# -----------------------------------------------------------------------------
# Feature engineering helpers
# -----------------------------------------------------------------------------


def _log1p_safe(series: pd.Series) -> pd.Series:
    """Apply log1p to non-negative long-tailed amounts safely."""
    s = series.copy()
    s = s.clip(lower=0)
    return np.log1p(s)


def _ensure_age_and_tenure(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = DEFAULT_REFERENCE_DATE,
    reference_year: int = DEFAULT_REFERENCE_YEAR,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> pd.DataFrame:
    """Ensure Age and customer_tenure_days exist (deterministic).

    This makes the pipeline robust even if callers skip ``clean_data``.
    """
    out = df.copy()

    if "Age" not in out.columns:
        if "Year_Birth" in out.columns:
            out["Age"] = reference_year - out["Year_Birth"]
        else:
            out["Age"] = np.nan

    if "customer_tenure_days" not in out.columns:
        if "Dt_Customer" in out.columns:
            dt = out["Dt_Customer"]
            if not pd.api.types.is_datetime64_any_dtype(dt):
                dt = pd.to_datetime(dt, format=date_format, errors="coerce")
            out["customer_tenure_days"] = (pd.Timestamp(reference_date) - dt).dt.days
            out["customer_tenure_days"] = out["customer_tenure_days"].fillna(0).clip(lower=0)
        else:
            out["customer_tenure_days"] = 0

    return out


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Frequency / Monetary / Spent from raw dataset columns."""
    out = df.copy()

    # Frequency
    freq_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    missing = [c for c in freq_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns required for Frequency: {missing}")
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
    missing = [c for c in mnt_cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns required for Monetary: {missing}")
    out["Monetary"] = out[mnt_cols].sum(axis=1)
    out["Spent"] = out["Monetary"]

    if "Recency" not in out.columns:
        raise KeyError("Missing column 'Recency' required for RFM.")

    return out


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add household / basket / preference / ratio features.

    Requires :func:`build_rfm_features` to have been called (needs Frequency/Spent).
    """
    out = df.copy()

    if "Frequency" not in out.columns or "Spent" not in out.columns:
        raise KeyError("Please call build_rfm_features() before add_structural_features().")

    # Household structure
    kid = out.get("Kidhome", 0)
    teen = out.get("Teenhome", 0)
    out["Children"] = kid + teen
    out["family_size"] = 1 + out["Children"]
    out["has_child"] = (out["Children"] > 0).astype(int)

    # Avoid division-by-zero for ratios
    freq = out["Frequency"].replace(0, np.nan)

    out["Avg_Basket"] = out["Spent"] / freq

    # Deal sensitivity (deals per purchase)
    if "NumDealsPurchases" in out.columns:
        out["Deal_Sensitivity"] = out["NumDealsPurchases"] / freq
    else:
        out["Deal_Sensitivity"] = np.nan

    # Online intensity: share of web purchases
    if "NumWebPurchases" in out.columns:
        out["Online_Intensity"] = out["NumWebPurchases"] / freq
    else:
        out["Online_Intensity"] = np.nan

    # Web visits per purchase (interest vs purchase)
    if "NumWebVisitsMonth" in out.columns:
        out["WebVisits_per_purchase"] = out["NumWebVisitsMonth"] / freq
    else:
        out["WebVisits_per_purchase"] = np.nan

    # Channel ratios
    for col in ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]:
        if col not in out.columns:
            raise KeyError(f"Missing '{col}' required for channel features.")
    out["Web_Ratio"] = out["NumWebPurchases"] / freq
    out["Catalog_Ratio"] = out["NumCatalogPurchases"] / freq
    out["Store_Ratio"] = out["NumStorePurchases"] / freq

    # Variety index: number of non-zero product categories
    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    out["Variety_Index"] = (out[mnt_cols] > 0).sum(axis=1)

    # Income-based ratios: keep NaNs when income is missing/non-positive (imputer handles)
    if "Income" in out.columns:
        income_pos = out["Income"].where(out["Income"] > 0)
        out["Spent_to_income"] = out["Spent"] / income_pos
        out["Income_per_member"] = out["Income"] / out["family_size"]
        out["Spent_per_member"] = out["Spent"] / out["family_size"]
    else:
        out["Spent_to_income"] = np.nan
        out["Income_per_member"] = np.nan
        out["Spent_per_member"] = np.nan

    # Fill only the NaNs caused by zero-frequency divisions with 0.
    # (Derived from freq=0 customers; operationally they have no purchases.)
    freq_ratio_cols = [
        "Avg_Basket",
        "Deal_Sensitivity",
        "Online_Intensity",
        "WebVisits_per_purchase",
        "Web_Ratio",
        "Catalog_Ratio",
        "Store_Ratio",
    ]
    out[freq_ratio_cols] = out[freq_ratio_cols].fillna(0.0)

    return out


# -----------------------------------------------------------------------------
# Transformer utilities (train-only fitting)
# -----------------------------------------------------------------------------


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Column-wise winsorization by quantiles.

    This transformer learns per-feature lower/upper quantile thresholds on the
    training data and applies clipping during transform.
    """

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99, copy: bool = True):
        if not (0.0 <= lower_q < upper_q <= 1.0):
            raise ValueError("Require 0 <= lower_q < upper_q <= 1")
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.copy = bool(copy)

        # fitted
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=float)
        # If X has shape (n,), treat as single feature.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.lower_ = np.quantile(X, self.lower_q, axis=0)
        self.upper_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("QuantileClipper must be fitted before transform.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        out = X.copy() if self.copy else X
        np.clip(out, self.lower_, self.upper_, out=out)
        return out


def _make_onehot_encoder() -> OneHotEncoder:
    """Make OneHotEncoder compatible across scikit-learn versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # sklearn<1.2 uses "sparse"
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


@dataclass
class FeatureConfig:
    """Configure which columns belong to behavior/response/categorical groups."""

    behavior_numeric: List[str]
    response_numeric: List[str]
    categorical: List[str]


def _default_feature_config(df: pd.DataFrame) -> FeatureConfig:
    """Default feature config used by the upgraded method."""
    behavior_numeric = [c for c in BEHAVIOR_NUMERIC_FEATURES if c in df.columns]
    response_numeric = [c for c in RESPONSE_NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return FeatureConfig(
        behavior_numeric=behavior_numeric,
        response_numeric=response_numeric,
        categorical=categorical,
    )


def assemble_feature_table(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    label_mode: LabelMode = "recent",
    feature_config: FeatureConfig | None = None,
    transformer: ColumnTransformer | None = None,
    fit: bool = True,
    auto_feature_engineering: bool = True,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
    winsorize: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Build a leakage-free feature table.

    Parameters
    ----------
    transformer:
        - If None: a new :class:`~sklearn.compose.ColumnTransformer` is created and
          ``fit=True`` is enforced.
        - If provided: set ``fit=False`` to transform validation/test without refitting.
    fit:
        Whether to fit the transformer on this dataframe. Use ``fit=True`` for train,
        ``fit=False`` for val/test.
    auto_feature_engineering:
        If True, automatically generate required engineered columns when missing.
    clip_quantiles:
        (lower, upper) winsorization quantiles learned on train only.
    winsorize:
        Whether to enable train-fitted quantile clipping.

    Returns
    -------
    features_df:
        Transformed feature matrix.
    labels:
        Target label series.
    transformer:
        The fitted (or reused) transformer.
    """
    work = df.copy()

    # 0) Deterministic temporal features (robust even if clean_data is skipped)
    if auto_feature_engineering:
        work = _ensure_age_and_tenure(work)

        # RFM
        if any(col not in work.columns for col in ["Frequency", "Monetary", "Spent"]):
            work = build_rfm_features(work)

        # Structural features
        required_struct = [
            "family_size",
            "Avg_Basket",
            "Variety_Index",
            "Web_Ratio",
            "Catalog_Ratio",
            "Store_Ratio",
        ]
        if any(col not in work.columns for col in required_struct):
            work = add_structural_features(work)

        # History response features
        if any(col not in work.columns for col in ["PastAcceptCnt", "EverAccepted"]):
            work = add_response_history_features(work)

    # 1) Ensure label exists
    if label_col not in work.columns:
        work = add_response_label(work, label_col=label_col, mode=label_mode)

    # 2) Log-transform long-tailed amount columns (deterministic)
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
        lower_q, upper_q = clip_quantiles

        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if winsorize:
            numeric_steps.append(("winsor", QuantileClipper(lower_q=lower_q, upper_q=upper_q)))
        numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(steps=numeric_steps)

        transformers = [("numeric", numeric_transformer, numeric_features_all)]
        if categorical_features:
            cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", _make_onehot_encoder()),
                ]
            )
            transformers.append(("categorical", cat_transformer, categorical_features))

        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        fit = True

    # 5) Fit/transform
    if fit:
        transformed = transformer.fit_transform(work)

        # numeric names -> lower-case
        numeric_feature_names = [col.lower() for col in numeric_features_all]

        # categorical one-hot names -> lower-case
        cat_feature_names: list[str] = []
        if categorical_features:
            cat_pipe: Pipeline = transformer.named_transformers_["categorical"]
            ohe: OneHotEncoder = cat_pipe.named_steps["onehot"]
            try:
                cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
            except AttributeError:
                # old sklearn
                cat_feature_names = list(ohe.get_feature_names(categorical_features))
            cat_feature_names = [c.lower() for c in cat_feature_names]

        all_feature_names = numeric_feature_names + cat_feature_names

        # record split meta
        n_beh = len(feature_config.behavior_numeric)
        behavior_numeric_names = numeric_feature_names[:n_beh]
        response_numeric_names = numeric_feature_names[n_beh:]

        # By default: categorical features are treated as part of behavior space.
        behavior_feature_names = behavior_numeric_names + cat_feature_names
        response_feature_names = response_numeric_names

        # attach metadata for later val/test transform & splitting
        transformer.feature_config_ = feature_config
        transformer.numeric_features_all_ = numeric_features_all
        transformer.categorical_features_ = categorical_features
        transformer.all_feature_names_ = all_feature_names
        transformer.behavior_feature_names_ = behavior_feature_names
        transformer.response_feature_names_ = response_feature_names

    else:
        if not hasattr(transformer, "all_feature_names_"):
            raise AttributeError(
                "Transformer has no stored feature names. Fit it on the training set first "
                "(assemble_feature_table(..., fit=True))."
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
    """Split transformed features into behavior part and response-related part."""
    if not hasattr(transformer, "behavior_feature_names_"):
        raise AttributeError(
            "transformer.behavior_feature_names_ not found. "
            "Please fit transformer via assemble_feature_table(..., fit=True) first."
        )

    behavior_cols = [c for c in transformer.behavior_feature_names_ if c in features_df.columns]
    response_cols = [c for c in transformer.response_feature_names_ if c in features_df.columns]

    return features_df[behavior_cols].copy(), features_df[response_cols].copy()
