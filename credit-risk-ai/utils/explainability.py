"""
explainability.py
-----------------
SHAP-based local and global model explanations.
Provides feature importance charts and per-prediction explanation data.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# ── SHAP Explainer factory ───────────────────────────────────────────────────

def build_explainer(model, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Build an appropriate SHAP explainer for the model type.

    Supports XGBoost (TreeExplainer), RandomForest (TreeExplainer),
    and Logistic Regression (LinearExplainer).
    """
    model_type = type(model).__name__
    logger.info(f"Building SHAP explainer for {model_type} …")

    if model_type in ("XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model, data=X_background, model_output="probability")
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_background)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba, shap.sample(X_background, 100)
        )
    return explainer


def explain_single(
    explainer: shap.Explainer,
    X_instance: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """
    Generate a local explanation for a single prediction.

    Args:
        explainer: Fitted SHAP explainer.
        X_instance: Single-row feature DataFrame.
        top_n: Number of top features to return.

    Returns:
        dict with keys 'shap_values', 'feature_names', 'feature_values',
        'top_features' (sorted by abs impact).
    """
    try:
        shap_values = explainer(X_instance)
        # Handle multi-output (binary classification → take class-1 column)
        if hasattr(shap_values, "values"):
            vals = shap_values.values[0]
            if vals.ndim == 2:          # shape (n_features, n_classes)
                vals = vals[:, 1]
        else:
            vals = np.array(shap_values[0])

        feature_names = list(X_instance.columns)
        feature_values = X_instance.iloc[0].tolist()

        # Build ranked list
        importance = sorted(
            zip(feature_names, vals, feature_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        return {
            "shap_values": vals,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "top_features": [
                {"feature": f, "shap_value": round(float(s), 4), "value": float(v)}
                for f, s, v in importance
            ],
        }
    except Exception as exc:
        logger.warning(f"SHAP explanation failed: {exc}")
        return {"top_features": [], "shap_values": [], "feature_names": [], "feature_values": []}


def explain_global(
    explainer: shap.Explainer,
    X_sample: pd.DataFrame,
    top_n: int = 15,
) -> dict:
    """
    Compute global feature importance from mean(|SHAP values|).

    Args:
        explainer: Fitted SHAP explainer.
        X_sample: Sample of the training/test set.
        top_n: Number of top features to return.

    Returns:
        dict with 'feature_names' and 'mean_abs_shap' arrays.
    """
    try:
        shap_values = explainer(X_sample)
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
        else:
            vals = np.array(shap_values)

        mean_abs = np.abs(vals).mean(axis=0)
        feature_names = list(X_sample.columns)

        ranked = sorted(
            zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True
        )[:top_n]

        return {
            "feature_names": [r[0] for r in ranked],
            "mean_abs_shap": [round(float(r[1]), 5) for r in ranked],
        }
    except Exception as exc:
        logger.warning(f"Global SHAP failed: {exc}")
        return {"feature_names": [], "mean_abs_shap": []}


def shap_waterfall_figure(explanation: dict, title: str = "SHAP Explanation") -> plt.Figure:
    """
    Build a horizontal waterfall bar chart for local explanation.

    Args:
        explanation: Output of explain_single().
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    top = explanation.get("top_features", [])
    if not top:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No explanation available", ha="center", va="center")
        return fig

    features = [t["feature"].replace("_", " ") for t in top]
    shap_vals = [t["shap_value"] for t in top]

    colours = ["#FF4444" if v > 0 else "#00C851" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.55)))
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")

    y_pos = range(len(features))
    bars = ax.barh(list(y_pos), shap_vals, color=colours, height=0.6, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(features, fontsize=10, color="white")
    ax.set_xlabel("SHAP Value (impact on default probability)", color="#AAAAAA", fontsize=9)
    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    for bar, val in zip(bars, shap_vals):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right",
            color="white", fontsize=8,
        )

    plt.tight_layout()
    return fig
