"""
train_model.py
--------------
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Evaluates all three and saves the best-performing model pipeline.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# ── Local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import DataPreprocessor, load_raw_data
from feature_engineering import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parents[1] / "logs" / "training.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_model.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
"""
train_model.py
--------------
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Evaluates all three and saves the best-performing model pipeline.
"""

import json
import logging
import time
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_model.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training.log"

# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Local imports
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import DataPreprocessor, load_raw_data
from feature_engineering import engineer_features


# ─────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────

def build_models():
    """Return dictionary of candidate models"""

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }

    return models


# ─────────────────────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics"""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy": round(accuracy_score(y_test, y_pred), 4),

        "precision": round(
            precision_score(y_test, y_pred, zero_division=0), 4
        ),

        "recall": round(
            recall_score(y_test, y_pred, zero_division=0), 4
        ),

        "roc_auc": round(
            roc_auc_score(y_test, y_proba), 4
        ),

        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ).tolist(),

        "classification_report": classification_report(
            y_test, y_pred
        )
    }

    return metrics


# ─────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────

def train_and_select(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    test_size=0.20,
    random_state=42
):

    logger.info("Loading dataset...")
    df_raw = load_raw_data(data_path)

    logger.info("Performing feature engineering...")
    df_eng = engineer_features(df_raw)

    logger.info("Preprocessing dataset...")
    preprocessor = DataPreprocessor()

    X, y = preprocessor.fit_transform(df_eng)

    logger.info(
        f"Feature Matrix Shape: {X.shape} | Default Rate: {y.mean():.2%}"
    )

    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    models = build_models()

    all_metrics = {}

    # Train each model
    for name, model in models.items():

        logger.info(f"Training {name}...")

        start_time = time.time()

        model.fit(X_train, y_train)

        elapsed = round(time.time() - start_time, 2)

        metrics = evaluate_model(model, X_test, y_test)

        metrics["training_time_sec"] = elapsed

        all_metrics[name] = metrics

        logger.info(
            f"{name} | AUC={metrics['roc_auc']} | "
            f"Accuracy={metrics['accuracy']} | "
            f"Recall={metrics['recall']} | "
            f"Time={elapsed}s"
        )

    # Select best model by ROC-AUC
    best_model_name = max(
        all_metrics,
        key=lambda k: all_metrics[k]["roc_auc"]
    )

    best_model = models[best_model_name]

    logger.info(
        f"Best Model: {best_model_name} "
        f"(AUC={all_metrics[best_model_name]['roc_auc']})"
    )

    # Save model bundle
    bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "model_name": best_model_name,
        "feature_columns": preprocessor.feature_columns
    }

    joblib.dump(bundle, model_path)

    logger.info(f"Model saved to {model_path}")

    # Save metrics
    serializable_metrics = {
        name: {
            k: v for k, v in metrics.items()
            if k != "classification_report"
        }
        for name, metrics in all_metrics.items()
    }

    serializable_metrics["best_model"] = best_model_name

    with open(METRICS_PATH, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Metrics saved to {METRICS_PATH}")

    return {
        "best_model": best_model_name,
        "metrics": all_metrics
    }


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    result = train_and_select()

    best = result["best_model"]

    metrics = result["metrics"][best]

    print("\n" + "=" * 60)
    print(f"Best Model : {best}")
    print(f"ROC-AUC    : {metrics['roc_auc']}")
    print(f"Accuracy   : {metrics['accuracy']}")
    print(f"Precision  : {metrics['precision']}")
    print(f"Recall     : {metrics['recall']}")
    print("=" * 60 + "\n")

    print(metrics["classification_report"])

def build_models() -> dict:
    """Return a dictionary of candidate model instances."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }


def train_and_select(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Full training pipeline:
    1. Load & engineer features
    2. Preprocess
    3. Train all candidates
    4. Select best by ROC-AUC
    5. Save model + preprocessor + metrics

    Returns:
        dict with best model name and metrics for all models.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    df_raw = load_raw_data(data_path)

    # ── 2. Feature engineering ───────────────────────────────────────────────
    logger.info("Engineering features …")
    df_eng = engineer_features(df_raw)

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    logger.info("Preprocessing …")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df_eng)

    logger.info(f"Final feature matrix: {X.shape} | Default rate: {y.mean():.2%}")

    # ── 4. Train / Test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ── 5. Train candidates ──────────────────────────────────────────────────
    models = build_models()
    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        logger.info(f"Training {name} …")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_time_sec"] = elapsed
        all_metrics[name] = metrics
        logger.info(
            f"  {name} | AUC={metrics['roc_auc']} | "
            f"Acc={metrics['accuracy']} | Recall={metrics['recall']} | "
            f"Time={elapsed}s"
        )

    # ── 6. Select best by ROC-AUC ────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    best_model = models[best_name]
    logger.info(f"Best model: {best_name} (AUC={all_metrics[best_name]['roc_auc']})")

    # ── 7. Save artefacts ────────────────────────────────────────────────────
    bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "model_name": best_name,
        "feature_columns": preprocessor.feature_columns,
    }
    joblib.dump(bundle, model_path)
    logger.info(f"Model bundle saved → {model_path}")

    # Save human-readable metrics
    serialisable = {
        name: {k: v for k, v in m.items() if k != "classification_report"}
        for name, m in all_metrics.items()
    }
    serialisable["best_model"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Metrics saved → {METRICS_PATH}")

    return {"best_model": best_name, "metrics": all_metrics}


if __name__ == "__main__":
    result = train_and_select()
    best = result["best_model"]
    m = result["metrics"][best]
    print(f"\n{'='*55}")
    print(f"  Best Model : {best}")
    print(f"  ROC-AUC    : {m['roc_auc']}")
    print(f"  Accuracy   : {m['accuracy']}")
    print(f"  Precision  : {m['precision']}")
    print(f"  Recall     : {m['recall']}")
    print(f"{'='*55}\n")
    print(m["classification_report"])
