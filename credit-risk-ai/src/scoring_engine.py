"""
scoring_engine.py
-----------------
Converts raw default probability into a CIBIL-style credit score (300-900)
and assigns a risk category tier.
"""

from dataclasses import dataclass
from typing import Literal

RiskCategory = Literal["LOW RISK", "MEDIUM RISK", "HIGH RISK"]

# Score band boundaries
SCORE_MIN = 300
SCORE_MAX = 900
HIGH_RISK_UPPER = 500
MEDIUM_RISK_UPPER = 700

# Colour codes for UI
RISK_COLOURS = {
    "LOW RISK": "#00C851",
    "MEDIUM RISK": "#FF8800",
    "HIGH RISK": "#FF4444",
}

RISK_ICONS = {
    "LOW RISK": "✅",
    "MEDIUM RISK": "⚠️",
    "HIGH RISK": "🚨",
}


@dataclass
class CreditAssessment:
    """Complete credit risk assessment result."""
    default_probability: float
    credit_score: int
    risk_category: RiskCategory
    score_colour: str
    risk_icon: str
    recommendation: str
    score_percentile: float  # 0–100 where you stand vs population


def compute_credit_score(default_probability: float) -> int:
    """
    Map default probability [0, 1] to a credit score [300, 900].

    Formula mirrors CIBIL-style scoring:
        score = 300 + (1 - p_default) * 600

    Args:
        default_probability: Predicted probability of loan default.

    Returns:
        Integer credit score in [300, 900].
    """
    if not 0.0 <= default_probability <= 1.0:
        raise ValueError(f"default_probability must be in [0,1], got {default_probability}")
    score = SCORE_MIN + (1.0 - default_probability) * (SCORE_MAX - SCORE_MIN)
    return round(int(score))


def classify_risk(credit_score: int) -> RiskCategory:
    """
    Assign a risk tier based on credit score.

    Bands:
        300–500 → HIGH RISK
        500–700 → MEDIUM RISK
        700–900 → LOW RISK

    Args:
        credit_score: Integer score in [300, 900].

    Returns:
        RiskCategory string.
    """
    if credit_score < HIGH_RISK_UPPER:
        return "HIGH RISK"
    elif credit_score < MEDIUM_RISK_UPPER:
        return "MEDIUM RISK"
    return "LOW RISK"


def build_recommendation(risk_category: RiskCategory, default_prob: float) -> str:
    """Generate a short banker-facing recommendation."""
    if risk_category == "LOW RISK":
        return (
            f"Applicant demonstrates strong creditworthiness (default prob: {default_prob:.1%}). "
            "Recommend APPROVAL with standard terms."
        )
    elif risk_category == "MEDIUM RISK":
        return (
            f"Applicant presents moderate risk (default prob: {default_prob:.1%}). "
            "Consider CONDITIONAL APPROVAL with higher interest rate or collateral requirement."
        )
    else:
        return (
            f"Applicant is high risk (default prob: {default_prob:.1%}). "
            "Recommend REJECTION or request significant collateral and co-signer."
        )


def score_percentile(credit_score: int) -> float:
    """
    Approximate population percentile for a given credit score.
    Based on typical CIBIL distribution (skewed toward higher scores).
    """
    # Simplified piecewise linear approximation
    bands = [
        (300, 0.0), (500, 12.0), (600, 28.0), (650, 40.0),
        (700, 58.0), (750, 72.0), (800, 85.0), (850, 94.0), (900, 100.0),
    ]
    for i in range(len(bands) - 1):
        lo_score, lo_pct = bands[i]
        hi_score, hi_pct = bands[i + 1]
        if lo_score <= credit_score <= hi_score:
            t = (credit_score - lo_score) / (hi_score - lo_score)
            return round(lo_pct + t * (hi_pct - lo_pct), 1)
    return 100.0


def assess(default_probability: float) -> CreditAssessment:
    """
    Full credit assessment from a single default probability.

    Args:
        default_probability: Float in [0, 1].

    Returns:
        CreditAssessment dataclass with all derived fields.
    """
    score = compute_credit_score(default_probability)
    risk = classify_risk(score)
    return CreditAssessment(
        default_probability=round(default_probability, 4),
        credit_score=score,
        risk_category=risk,
        score_colour=RISK_COLOURS[risk],
        risk_icon=RISK_ICONS[risk],
        recommendation=build_recommendation(risk, default_probability),
        score_percentile=score_percentile(score),
    )
