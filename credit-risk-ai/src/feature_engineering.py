"""
feature_engineering.py
-----------------------
Derives high-signal features from raw borrower data before model training.
Engineered features capture credit utilisation, spending patterns, and
repayment stability — all critical signals in credit risk modelling.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Raw numeric columns used in derived features
_EPS = 1e-6  # avoid division-by-zero


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a copy of the raw DataFrame.

    Args:
        df: Raw loan dataset (before preprocessing / scaling).

    Returns:
        pd.DataFrame with additional engineered columns appended.
    """
    df = df.copy()

    # ── 1. Credit Utilisation Ratio ─────────────────────────────────────────
    # Outstanding debt relative to declared income × loan term proxy
    df["Credit_Utilisation_Ratio"] = df["Outstanding_Loan_Amount"] / (
        df["Income"] / 12 * df.get("Loan_Term", pd.Series(np.full(len(df), 36))) + _EPS
    )
    df["Credit_Utilisation_Ratio"] = df["Credit_Utilisation_Ratio"].clip(0, 10).round(4)

    # ── 2. Spending Ratio ───────────────────────────────────────────────────
    # Monthly expenses as a fraction of monthly income
    df["Spending_Ratio"] = df["Monthly_Expenses"] / (df["Income"] / 12 + _EPS)
    df["Spending_Ratio"] = df["Spending_Ratio"].clip(0, 5).round(4)

    # ── 3. Repayment Stability Score ────────────────────────────────────────
    # Inverse-weighted combination of late payments and previous defaults
    df["Repayment_Stability"] = 1 / (
        1 + df["Late_Payments"] * 0.5 + df["Previous_Defaults"] * 1.5
    )
    df["Repayment_Stability"] = df["Repayment_Stability"].clip(0, 1).round(4)

    # ── 4. Savings Coverage Months ──────────────────────────────────────────
    # How many months of expenses savings can cover
    df["Savings_Coverage_Months"] = df["Savings_Balance"] / (
        df["Monthly_Expenses"] + _EPS
    )
    df["Savings_Coverage_Months"] = df["Savings_Coverage_Months"].clip(0, 120).round(2)

    # ── 5. Loan Burden Ratio ────────────────────────────────────────────────
    # New loan monthly instalment estimate vs monthly income
    monthly_rate = df.get("Interest_Rate", pd.Series(np.full(len(df), 10.0))) / 1200
    n_payments = df.get("Loan_Term", pd.Series(np.full(len(df), 36)))
    monthly_instalment = df["Loan_Amount"] * (
        monthly_rate * (1 + monthly_rate) ** n_payments
    ) / ((1 + monthly_rate) ** n_payments - 1 + _EPS)
    df["Loan_Burden_Ratio"] = monthly_instalment / (df["Income"] / 12 + _EPS)
    df["Loan_Burden_Ratio"] = df["Loan_Burden_Ratio"].clip(0, 5).round(4)

    # ── 6. Financial Stress Index ───────────────────────────────────────────
    # Composite: high DTI + high spending + high burden = stress
    df["Financial_Stress_Index"] = (
        df["Debt_to_Income_Ratio"] * 0.4
        + df["Spending_Ratio"] * 0.3
        + df["Loan_Burden_Ratio"] * 0.3
    ).clip(0, 10).round(4)

    # ── 7. Credit Experience Score ──────────────────────────────────────────
    # Longer history + more cards = more experience (normalised)
    df["Credit_Experience_Score"] = (
        df["Credit_History_Length"] * 0.7
        + df["Number_of_Credit_Cards"] * 0.3
    ).clip(0, 50).round(2)

    # ── 8. Income-to-Loan Ratio ─────────────────────────────────────────────
    df["Income_to_Loan_Ratio"] = df["Income"] / (df["Loan_Amount"] + _EPS)
    df["Income_to_Loan_Ratio"] = df["Income_to_Loan_Ratio"].clip(0, 100).round(4)

    n_new = 8
    logger.info(f"Feature engineering complete — added {n_new} derived features.")
    return df


ENGINEERED_FEATURES = [
    "Credit_Utilisation_Ratio",
    "Spending_Ratio",
    "Repayment_Stability",
    "Savings_Coverage_Months",
    "Loan_Burden_Ratio",
    "Financial_Stress_Index",
    "Credit_Experience_Score",
    "Income_to_Loan_Ratio",
]
