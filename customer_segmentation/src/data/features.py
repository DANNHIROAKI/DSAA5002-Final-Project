# src/data/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 原始促销字段（来自数据集）
CAMPAIGN_COLUMNS: List[str] = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]

DEFAULT_LABEL_COL = "campaign_response"


def add_response_label(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    campaign_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    根据 AcceptedCmp1-5 + Response 生成二元促销响应标签。

    y = 1 表示至少在 6 次活动中任意一次接受。
    """
    if campaign_cols is None:
        campaign_cols = CAMPAIGN_COLUMNS

    missing = [c for c in campaign_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing campaign columns in DataFrame: {missing}")

    labeled = df.copy()
    labeled[label_col] = (labeled[campaign_cols].sum(axis=1) > 0).astype(int)
    return labeled


# --------- 特征配置（用于 __all__ 暴露） ---------

# 行为 / 响应 / 类别特征常量，供外部模块引用。
# 这些名称与 `_default_feature_config` 中的默认设置保持一致。
BEHAVIOR_NUMERIC_FEATURES: List[str] = [
    "Recency",
    "Frequency",
    "Monetary",
    "Spent",
    "Income",
    "Age",
    "Kidhome",
    "Teenhome",
    "family_size",
    "has_child",
    "customer_tenure_days",
    "Avg_Basket",
    "Deal_Sensitivity",
    "Online_Intensity",
    "Variety_Index",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "Complain",
]

RESPONSE_NUMERIC_FEATURES: List[str] = [
    "NumDealsPurchases",
    "NumWebVisitsMonth",
]

CATEGORICAL_FEATURES: List[str] = [
    "Education",
    "Marital_Status",
]


# --------- 特征工程辅助函数 ---------


def _log1p_safe(series: pd.Series) -> pd.Series:
    """对严格非负的长尾金额特征做 log1p 变换。"""
    # clip 防止极少数负值（如果有的话）
    s = series.clip(lower=0)
    return np.log1p(s)


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造 RFM 三要素 + 总消费 Spent。
    - Recency: 原始 Recency
    - Frequency: 各渠道购买次数之和
    - Monetary / Spent: 所有品类消费金额之和
    """
    out = df.copy()

    # 频次
    freq_cols = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    for col in freq_cols:
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' not found in DataFrame.")

    out["Frequency"] = out[freq_cols].sum(axis=1)

    # 金额 / Spent
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

    # Recency 直接沿用
    if "Recency" not in out.columns:
        raise KeyError("Expected column 'Recency' not found in DataFrame.")

    return out


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增加家庭结构 / 渠道占比 / 业务相关行为特征：
    - family_size, has_child
    - 平均客单价 Avg_Basket
    - 折扣敏感度 Deal_Sensitivity
    - 线上强度 Online_Intensity
    - 品类多样性 Variety_Index
    - 渠道占比（Web / Catalog / Store）
    """
    out = df.copy()

    # 家庭结构
    kid = out.get("Kidhome", 0)
    teen = out.get("Teenhome", 0)
    out["family_size"] = 1 + kid + teen
    out["has_child"] = ((kid + teen) > 0).astype(int)

    # 购买频次确保存在
    if "Frequency" not in out.columns:
        raise KeyError("Please call build_rfm_features() before add_structural_features().")

    freq = out["Frequency"].replace(0, np.nan)

    # 平均客单价
    out["Avg_Basket"] = out["Spent"] / freq

    # 折扣敏感度
    if "NumDealsPurchases" in out.columns:
        out["Deal_Sensitivity"] = out["NumDealsPurchases"] / freq
    else:
        out["Deal_Sensitivity"] = 0.0

    # 线上强度
    if "NumWebPurchases" in out.columns:
        out["Online_Intensity"] = out["NumWebPurchases"] / freq
    else:
        out["Online_Intensity"] = 0.0

    # 渠道占比
    for col in ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]:
        if col not in out.columns:
            raise KeyError(f"Expected column '{col}' for channel features.")

    out["Web_Ratio"] = out["NumWebPurchases"] / freq
    out["Catalog_Ratio"] = out["NumCatalogPurchases"] / freq
    out["Store_Ratio"] = out["NumStorePurchases"] / freq

    # 品类多样性
    mnt_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    out["Variety_Index"] = (out[mnt_cols] > 0).sum(axis=1)

    # 缺失填 0（主要是除以 0 的 NaN）
    out[["Avg_Basket", "Deal_Sensitivity", "Online_Intensity",
         "Web_Ratio", "Catalog_Ratio", "Store_Ratio"]] = out[
        ["Avg_Basket", "Deal_Sensitivity", "Online_Intensity",
         "Web_Ratio", "Catalog_Ratio", "Store_Ratio"]
    ].fillna(0.0)

    return out


@dataclass
class FeatureConfig:
    """
    配置哪些特征属于行为 / 响应 / 类别。

    这里的列名都是“原始表中的列名”，在 transform 后会统一转成小写。
    """

    behavior_numeric: List[str]
    response_numeric: List[str]
    categorical: List[str]


def _default_feature_config(df: pd.DataFrame) -> FeatureConfig:
    """根据数据集列名，构造默认的特征配置。"""

    # 行为数值特征：RFM + 金额 & 结构 & 渠道 + 人口属性
    behavior_numeric = [
        "Recency",
        "Frequency",
        "Monetary",
        "Spent",
        "Income",
        "Age",
        "Kidhome",
        "Teenhome",
        "family_size",
        "has_child",
        "customer_tenure_days",
        "Avg_Basket",
        "Deal_Sensitivity",
        "Online_Intensity",
        "Variety_Index",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "Complain",
    ]

    # 响应相关数值特征（不参与聚类，只在下游预测中用）
    response_numeric = [
        "NumDealsPurchases",
        "NumWebVisitsMonth",
    ]

    # 类别特征
    categorical = [
        "Education",
        "Marital_Status",
    ]

    # 过滤掉当前 df 里不存在的列（容错）
    behavior_numeric = [c for c in behavior_numeric if c in df.columns]
    response_numeric = [c for c in response_numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]

    return FeatureConfig(
        behavior_numeric=behavior_numeric,
        response_numeric=response_numeric,
        categorical=categorical,
    )


def assemble_feature_table(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    feature_config: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    构造用于聚类 / 下游预测的特征表：
    - 对金额类特征做 log1p 变换
    - 数值特征标准化
    - 类别特征 one-hot
    返回：
    - features_df: 所有特征的 DataFrame
    - labels: 二元响应标签
    - transformer: sklearn ColumnTransformer，用于后续 split_behavior_and_response_features
    """

    work = df.copy()

    # If the label column is missing, derive it from the campaign response columns.
    # This makes the function robust when callers forget to explicitly add the label
    # via add_response_label beforehand.
    if label_col not in work.columns:
        try:
            work = add_response_label(work, label_col=label_col)
        except KeyError as exc:
            raise KeyError(
                f"Label column '{label_col}' not found and campaign columns are missing. "
                "Please ensure the raw data includes AcceptedCmp1-5 and Response, or "
                "call add_response_label() prior to assemble_feature_table()."
            ) from exc

    # log1p 变换：Income + 所有 Mnt* + Spent/Monetary
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

    # 默认特征配置
    if feature_config is None:
        feature_config = _default_feature_config(work)

    numeric_features_all = feature_config.behavior_numeric + feature_config.response_numeric
    categorical_features = feature_config.categorical

    # 构造 ColumnTransformer
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features_all),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    transformed = preprocessor.fit_transform(work)

    # 生成列名：数值特征 -> 小写；类别特征 -> one-hot 名
    numeric_feature_names = [col.lower() for col in numeric_features_all]

    if categorical_features:
        cat_ohe = preprocessor.named_transformers_["categorical"]
        cat_feature_names = list(
            cat_ohe.get_feature_names_out(categorical_features)
        )
    else:
        cat_feature_names = []

    all_feature_names = numeric_feature_names + cat_feature_names
    features_df = pd.DataFrame(transformed, columns=all_feature_names, index=df.index)

    # 记录行为 / 响应 / 类别特征在变换后对应的列名，方便后续 split
    n_beh = len(feature_config.behavior_numeric)
    behavior_numeric_names = numeric_feature_names[:n_beh]
    response_numeric_names = numeric_feature_names[n_beh:]

    behavior_feature_names = behavior_numeric_names + cat_feature_names
    response_feature_names = response_numeric_names

    # 将这些信息挂在 transformer 上，方便 split_behavior_and_response_features 使用
    preprocessor.behavior_numeric_features_ = feature_config.behavior_numeric
    preprocessor.response_numeric_features_ = feature_config.response_numeric
    preprocessor.categorical_features_ = categorical_features

    preprocessor.behavior_feature_names_ = behavior_feature_names
    preprocessor.response_feature_names_ = response_feature_names

    # 输出标签
    labels = work[label_col].astype(int)

    return features_df, labels, preprocessor


def split_behavior_and_response_features(
    features_df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根据 assemble_feature_table 中挂在 transformer 上的元数据，
    将特征表拆成：
    - 行为特征（用于所有聚类 + RAJC）
    - 响应相关特征（仅用于下游预测）
    """
    if not hasattr(transformer, "behavior_feature_names_"):
        raise AttributeError(
            "transformer.behavior_feature_names_ not found. "
            "Please create transformer via assemble_feature_table()."
        )

    behavior_cols = list(transformer.behavior_feature_names_)
    response_cols = list(transformer.response_feature_names_)

    # 只选取当前 DataFrame 中存在的列，避免因为外部修改导致的 KeyError
    behavior_cols = [c for c in behavior_cols if c in features_df.columns]
    response_cols = [c for c in response_cols if c in features_df.columns]

    behavior_features = features_df[behavior_cols].copy()
    response_features = features_df[response_cols].copy()

    return behavior_features, response_features
