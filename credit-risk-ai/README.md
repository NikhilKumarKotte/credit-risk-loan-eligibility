# 🏦 AI Credit Risk & Loan Default Prediction Platform

> Enterprise-grade machine learning system for credit risk assessment and loan default prediction, built with XGBoost, SHAP, and Streamlit.

---

## 📋 Overview

This platform simulates how modern banks evaluate borrower risk and creditworthiness. It combines:

- **Loan Default Prediction** — XGBoost classifier predicting P(default)
- **Credit Scoring Engine** — CIBIL-style score (300–900)
- **Explainable AI** — SHAP-powered per-applicant explanations
- **Analytics Dashboard** — ROC-AUC, confusion matrix, feature importance

## 🏗️ Project Structure

```
credit-risk-ai/
├── data/
│   └── loan_data.csv              # 12,000-row synthetic dataset
├── models/
│   ├── credit_model.pkl           # Best trained model bundle
│   └── training_metrics.json      # Performance metrics for all models
├── src/
│   ├── generate_dataset.py        # Synthetic data generator
│   ├── data_preprocessing.py      # Imputation, encoding, scaling
│   ├── feature_engineering.py     # 8 derived features
│   ├── train_model.py             # Train + select best model
│   ├── predict.py                 # Inference API
│   └── scoring_engine.py          # Credit score computation
├── app/
│   └── streamlit_app.py           # 4-page banking dashboard
├── utils/
│   └── explainability.py          # SHAP explainer utilities
├── config/
│   └── config.yaml                # Central configuration
├── logs/
│   └── training.log               # Training logs
├── requirements.txt
├── run.sh                         # One-command launcher
└── README.md
```

## 🚀 Quick Start

### Option 1: One command
```bash
chmod +x run.sh
./run.sh
```

### Option 2: Step by step
```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset
python src/generate_dataset.py

# Train models
python src/train_model.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Machine Learning Pipeline

| Model | AUC | Notes |
|---|---|---|
| XGBoost | ~0.92+ | Best performer, selected automatically |
| Random Forest | ~0.89+ | Strong ensemble baseline |
| Logistic Regression | ~0.82+ | Interpretable baseline |

### Features Used (25 total)
**Raw (17):** Age, Income, Employment Status, Years Employed, Credit History Length, Number of Credit Cards, Outstanding Loan Amount, Debt-to-Income Ratio, Monthly Expenses, Previous Defaults, Loan Amount, Loan Term, Interest Rate, Savings Balance, Transaction Volume, Late Payments

**Engineered (8):** Credit Utilisation Ratio, Spending Ratio, Repayment Stability, Savings Coverage Months, Loan Burden Ratio, Financial Stress Index, Credit Experience Score, Income-to-Loan Ratio

---

## 💯 Credit Scoring Formula

```
Credit Score = 300 + (1 − P_default) × 600

300–499  →  HIGH RISK   (Reject / Heavy collateral)
500–699  →  MEDIUM RISK (Conditional approval)
700–900  →  LOW RISK    (Approve with standard terms)
```

---

## ☁️ Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set Main file: `app/streamlit_app.py`

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python src/generate_dataset.py && python src/train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
```

### HuggingFace Spaces
Set `app_file: app/streamlit_app.py` in your Space settings, SDK: `streamlit`.

---

## ⚠️ Disclaimer

This system uses **synthetic data** for demonstration purposes. It is not intended for real-world credit decisioning without proper regulatory compliance, bias auditing, and model validation.
