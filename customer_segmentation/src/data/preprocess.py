"""Data cleaning and preprocessing routines."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for missing value handling and outlier mitigation."""
    # TODO: implement domain-specific cleaning (imputation, winsorization, etc.)
    return df.copy()


def train_val_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned dataset into training and validation subsets."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df.get("Response"))
