"""
data_preprocessing.py
----------------------
Handles missing values, encoding, and normalization for the loan dataset.
Designed to be used both in training and inference pipelines.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Categorical and numeric feature definitions
CATEGORICAL_FEATURES = ["Employment_Status"]
NUMERIC_FEATURES = [
    "Age", "Income", "Years_Employed", "Credit_History_Length",
    "Number_of_Credit_Cards", "Outstanding_Loan_Amount", "Debt_to_Income_Ratio",
    "Monthly_Expenses", "Previous_Defaults", "Loan_Amount", "Loan_Term",
    "Interest_Rate", "Savings_Balance", "Transaction_Volume", "Late_Payments",
]
TARGET = "Loan_Default"
DROP_COLS = ["Applicant_ID"]


class DataPreprocessor:
    """
    Full preprocessing pipeline: imputation → encoding → scaling.

    Attributes:
        label_encoders: Fitted LabelEncoder instances keyed by column name.
        scaler: Fitted StandardScaler for numeric features.
        feature_columns: Ordered list of feature column names after preprocessing.
    """

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor on training data and return transformed X, y.

        Args:
            df: Raw dataset including the target column.

        Returns:
            Tuple of (feature DataFrame, target Series).
        """
        df = df.copy()
        df = self._drop_unused(df)
        df = self._impute(df)
        df = self._encode_categoricals(df, fit=True)
        X, y = self._split_target(df)
        X = self._scale_numerics(X, fit=True)
        self.feature_columns = list(X.columns)
        logger.info(f"fit_transform complete | shape: {X.shape}")
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new / inference data.

        Args:
            df: Raw feature data (target column optional / ignored).

        Returns:
            Transformed feature DataFrame.
        """
        df = df.copy()
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])
        df = self._drop_unused(df)
        df = self._impute(df)
        df = self._encode_categoricals(df, fit=False)

        # Ensure column alignment with training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        df = self._scale_numerics(df, fit=False)
        return df

    # ── Private helpers ─────────────────────────────────────────────────────

    def _drop_unused(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric NaNs with median; categorical with mode."""
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    raise RuntimeError(f"LabelEncoder for '{col}' not fitted.")
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def _split_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        y = df[TARGET]
        X = df.drop(columns=[TARGET])
        return X, y

    def _scale_numerics(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and basic-validate the raw CSV dataset."""
    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df
