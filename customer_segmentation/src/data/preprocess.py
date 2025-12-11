# src/data/preprocess.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REFERENCE_YEAR = 2024  # 用于计算 Age 的参考年份，可按需要调整


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """数值特征用中位数，类别特征用众数填补缺失。"""
    out = df.copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    categorical_cols = out.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        median = out[col].median()
        out[col] = out[col].fillna(median)

    for col in categorical_cols:
        mode = out[col].mode(dropna=True)
        if not mode.empty:
            out[col] = out[col].fillna(mode.iloc[0])
        else:
            out[col] = out[col].fillna("Unknown")

    return out


def _clip_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """按给定分位数做 winsorization，减少极端值影响。"""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def clean_data(
    df: pd.DataFrame,
    reference_year: int = REFERENCE_YEAR,
    clip_quantiles: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    清洗原始 marketing_campaign 数据：
    1. 去重
    2. 解析 Dt_Customer，计算 customer_tenure_days
    3. 由 Year_Birth 计算 Age
    4. 填补缺失
    5. 对 Age 与 Income 做分位数裁剪（winsorization）
    """

    out = df.copy()

    # 去重
    out = out.drop_duplicates().reset_index(drop=True)

    # 解析日期、计算关系长度（天）
    if "Dt_Customer" in out.columns:
        out["Dt_Customer"] = pd.to_datetime(out["Dt_Customer"], errors="coerce")
        max_date = out["Dt_Customer"].max()
        out["customer_tenure_days"] = (max_date - out["Dt_Customer"]).dt.days
    else:
        out["customer_tenure_days"] = 0

    # 计算 Age
    if "Year_Birth" in out.columns:
        out["Age"] = reference_year - out["Year_Birth"]
    else:
        out["Age"] = np.nan

    # 缺失值填补
    out = _fill_missing(out)

    # Age & Income 的 winsorization，呼应课程 PPT 中对 outlier 的处理
    lower_q, upper_q = clip_quantiles

    if "Age" in out.columns:
        out["Age"] = _clip_outliers(out["Age"], lower_q, upper_q)

    if "Income" in out.columns:
        out["Income"] = _clip_outliers(out["Income"], lower_q, upper_q)

    return out


def train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    可选：将数据简单划分为 train / validation。
    对无监督聚类不是必须，但在需要做弱监督调参时很有用。
    """
    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise KeyError(f"Column '{stratify_col}' not found for stratification.")
        stratify_vals = df[stratify_col]
    else:
        stratify_vals = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_vals,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
