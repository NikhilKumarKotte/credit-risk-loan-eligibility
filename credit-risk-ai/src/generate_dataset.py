"""
generate_dataset.py
-------------------
Synthetic loan dataset generator for the AI Credit Risk Platform.
Produces a realistic, statistically consistent dataset of 10,000+ borrowers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def generate_loan_dataset(n_samples: int = 12000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic but statistically realistic loan dataset.

    Args:
        n_samples: Number of borrower records to generate.
        random_seed: Seed for reproducibility.

    Returns:
        pd.DataFrame: Full dataset with features and target column.
    """
    rng = np.random.default_rng(random_seed)

    # ── Applicant demographics ──────────────────────────────────────────────
    applicant_ids = [f"APP{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    age = rng.integers(21, 70, size=n_samples)
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"],
        size=n_samples,
        p=[0.60, 0.20, 0.12, 0.08],
    )

    # Income correlated with age + employment status
    base_income = (
        rng.normal(loc=60_000, scale=25_000, size=n_samples)
        + (age - 21) * 500
    )
    income_multiplier = np.where(
        employment_status == "Self-Employed", rng.uniform(0.8, 1.4, n_samples),
        np.where(employment_status == "Unemployed", rng.uniform(0.1, 0.3, n_samples),
        np.where(employment_status == "Retired", rng.uniform(0.4, 0.7, n_samples), 1.0))
    )
    income = np.clip(base_income * income_multiplier, 8_000, 500_000).astype(int)

    # ── Credit history ───────────────────────────────────────────────────────
    years_employed = np.clip(
        rng.integers(0, min(40, 1), size=n_samples) + (age - 21) * rng.uniform(0, 0.6, n_samples),
        0, 40
    ).astype(int)
    credit_history_length = np.clip(
        years_employed + rng.integers(0, 10, size=n_samples), 0, 45
    ).astype(int)
    number_of_credit_cards = rng.integers(0, 12, size=n_samples)

    # ── Financial position ───────────────────────────────────────────────────
    savings_balance = np.clip(
        income * rng.uniform(0.01, 2.0, n_samples) + rng.normal(0, 5_000, n_samples),
        0, 1_000_000
    ).astype(int)
    outstanding_loan_amount = np.clip(
        rng.exponential(scale=25_000, size=n_samples), 0, 400_000
    ).astype(int)
    monthly_expenses = np.clip(
        income / 12 * rng.uniform(0.3, 0.95, n_samples) + rng.normal(0, 500, n_samples),
        500, 30_000
    ).astype(int)
    transaction_volume = rng.integers(5, 200, size=n_samples)

    # ── Loan application ─────────────────────────────────────────────────────
    loan_amount = np.clip(
        rng.normal(loc=35_000, scale=20_000, size=n_samples), 1_000, 200_000
    ).astype(int)
    loan_term = rng.choice([12, 24, 36, 48, 60, 84, 120], size=n_samples)
    interest_rate = np.clip(
        6.0 + rng.exponential(scale=4.0, size=n_samples), 3.0, 36.0
    ).round(2)

    # ── Risk indicators ──────────────────────────────────────────────────────
    debt_to_income_ratio = np.clip(
        (outstanding_loan_amount + loan_amount) / np.maximum(income, 1), 0.0, 5.0
    ).round(3)
    previous_defaults = rng.choice(
        [0, 1, 2, 3, 4, 5],
        size=n_samples,
        p=[0.65, 0.18, 0.09, 0.05, 0.02, 0.01],
    )
    late_payments = rng.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        size=n_samples,
        p=[0.45, 0.20, 0.13, 0.09, 0.06, 0.04, 0.02, 0.007, 0.003],
    )

    # ── Target: Loan Default ─────────────────────────────────────────────────
    # Logistic model for realistic default probability
    log_odds = (
        -4.0
        + 0.03 * debt_to_income_ratio * 10
        + 0.40 * previous_defaults
        + 0.25 * late_payments
        - 0.02 * (credit_history_length)
        - 0.000008 * income
        + 0.000005 * outstanding_loan_amount
        - 0.000003 * savings_balance
        + 0.5 * (employment_status == "Unemployed").astype(int)
        + 0.2 * (employment_status == "Self-Employed").astype(int)
        + rng.normal(0, 0.5, n_samples)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    loan_default = rng.binomial(1, default_prob)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "Applicant_ID": applicant_ids,
        "Age": age,
        "Income": income,
        "Employment_Status": employment_status,
        "Years_Employed": years_employed,
        "Credit_History_Length": credit_history_length,
        "Number_of_Credit_Cards": number_of_credit_cards,
        "Outstanding_Loan_Amount": outstanding_loan_amount,
        "Debt_to_Income_Ratio": debt_to_income_ratio,
        "Monthly_Expenses": monthly_expenses,
        "Previous_Defaults": previous_defaults,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Interest_Rate": interest_rate,
        "Savings_Balance": savings_balance,
        "Transaction_Volume": transaction_volume,
        "Late_Payments": late_payments,
        "Loan_Default": loan_default,
    })

    default_rate = loan_default.mean()
    logger.info(f"Generated {n_samples:,} records | Default rate: {default_rate:.2%}")
    return df


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "data" / "loan_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_loan_dataset()
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved → {output_path}")
    print(df.describe())
