"""
predict.py
----------
Inference API: loads the saved model bundle and generates predictions
for new applicants. Designed to be called from the Streamlit app or
a future FastAPI endpoint.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import engineer_features
from scoring_engine import CreditAssessment, assess

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "credit_model.pkl"

# Module-level cache to avoid re-loading on every call
_bundle_cache: dict[str, Any] = {}


def load_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load (and cache) the model bundle from disk."""
    key = str(model_path)
    if key not in _bundle_cache:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )
        _bundle_cache[key] = joblib.load(model_path)
        logger.info(f"Model bundle loaded from {model_path}")
    return _bundle_cache[key]


def predict_single(applicant: dict, model_path: Path = DEFAULT_MODEL_PATH) -> CreditAssessment:
    """
    Predict default probability and credit score for a single applicant.

    Args:
        applicant: Dictionary of raw feature values (un-scaled, un-encoded).
        model_path: Path to the saved model bundle.

    Returns:
        CreditAssessment dataclass with score, risk, and explanation metadata.
    """
    bundle = load_bundle(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    # Convert to DataFrame
    df = pd.DataFrame([applicant])

    # Engineer features
    df = engineer_features(df)

    # Preprocess (encode + scale)
    X = preprocessor.transform(df)

    # Predict
    proba = model.predict_proba(X)[0, 1]
    return assess(float(proba))


def predict_batch(df_raw: pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """
    Batch prediction for a DataFrame of applicants.

    Args:
        df_raw: Raw applicant records (may include Applicant_ID and Loan_Default).
        model_path: Path to saved model bundle.

    Returns:
        Original DataFrame augmented with Default_Probability, Credit_Score, Risk_Category.
    """
    bundle = load_bundle(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    df = df_raw.copy()
    df_eng = engineer_features(df)
    X = preprocessor.transform(df_eng)

    probas = model.predict_proba(X)[:, 1]
    assessments = [assess(float(p)) for p in probas]

    df_raw = df_raw.copy()
    df_raw["Default_Probability"] = [a.default_probability for a in assessments]
    df_raw["Credit_Score"] = [a.credit_score for a in assessments]
    df_raw["Risk_Category"] = [a.risk_category for a in assessments]
    return df_raw


def get_feature_names(model_path: Path = DEFAULT_MODEL_PATH) -> list[str]:
    """Return the ordered feature column names used by the model."""
    bundle = load_bundle(model_path)
    return bundle["feature_columns"]
