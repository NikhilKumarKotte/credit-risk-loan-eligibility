"""
streamlit_app.py
-----------------
AI Credit Risk & Loan Default Prediction Platform
Production-grade Streamlit dashboard for banking credit evaluation.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "utils"))

from predict import load_bundle, predict_single, get_feature_names
from scoring_engine import RISK_COLOURS, assess
from feature_engineering import engineer_features

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark fintech CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #060D18;
    color: #E8ECF0;
}
.stApp { background-color: #060D18; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #060D18 100%);
    border-right: 1px solid #1A2E4A;
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0D1F35 0%, #112240 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.4rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.metric-card h2 { margin: 0 0 0.2rem 0; font-family: 'IBM Plex Mono', monospace; }
.metric-card p  { margin: 0; color: #8899AA; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase; }

/* Score badge */
.score-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.6rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
}
.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4A7FA5;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
/* Form inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #0A1628 !important;
    border: 1px solid #1E3A5F !important;
    color: #E8ECF0 !important;
    border-radius: 6px !important;
}
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1E6FDB 0%, #0A4FA8 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.04em;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2A7FEB 0%, #1A5FB8 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(30,111,219,0.4);
}
/* Risk pill */
.risk-pill {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
/* Divider */
.h-line { border-top: 1px solid #1A2E4A; margin: 1.2rem 0; }
/* Plotly charts background fix */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "applicant_data" not in st.session_state:
    st.session_state.applicant_data = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model …")
def load_model():
    try:
        return load_bundle()
    except FileNotFoundError:
        return None


AUTH_USERS = {
    "Bank": {"admin": "bank123"},
    "Client": {"client": "client123"},
}


def verify_login(role: str, username: str, password: str) -> bool:
    role_users = AUTH_USERS.get(role, {})
    return role_users.get(username) == password


def max_affordable_loan(monthly_capacity: float, annual_rate: float, term_months: int) -> float:
    if monthly_capacity <= 0:
        return 0.0
    r = annual_rate / 100.0 / 12.0
    n = max(term_months, 1)
    if r == 0:
        return monthly_capacity * n
    return monthly_capacity * ((1 + r) ** n - 1) / (r * (1 + r) ** n)


def loan_emi(principal: float, annual_rate: float, term_months: int) -> float:
    if principal <= 0:
        return 0.0
    r = annual_rate / 100.0 / 12.0
    n = max(term_months, 1)
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


STOCK_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
STOCK_API_BASE = os.getenv("TWELVEDATA_BASE_URL", "https://api.twelvedata.com").strip()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_prices(tickers: tuple, api_key: str, base_url: str) -> dict:
    prices = {}
    if not api_key:
        for t in tickers:
            prices[t] = None
        return prices

    for ticker in tickers:
        try:
            resp = requests.get(
                f"{base_url}/price",
                params={"symbol": ticker, "apikey": api_key},
                timeout=6,
            )
            data = resp.json()
            if isinstance(data, dict) and "price" in data:
                prices[ticker] = float(data["price"])
            else:
                prices[ticker] = None
        except Exception:
            prices[ticker] = None
    return prices


@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_quotes(tickers: tuple, api_key: str, base_url: str) -> dict:
    quotes = {}
    if not api_key:
        for t in tickers:
            quotes[t] = None
        return quotes

    for ticker in tickers:
        try:
            resp = requests.get(
                f"{base_url}/quote",
                params={"symbol": ticker, "apikey": api_key},
                timeout=6,
            )
            data = resp.json()
            if isinstance(data, dict) and "symbol" in data:
                quotes[ticker] = data
            else:
                quotes[ticker] = None
        except Exception:
            quotes[ticker] = None
    return quotes


def gauge_chart(value: float, title: str, min_val=0, max_val=1, thresholds=None):
    """Plotly gauge for probability meter."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#E8ECF0", "family": "IBM Plex Mono"}},
        title={"text": title, "font": {"size": 13, "color": "#8899AA", "family": "IBM Plex Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4A7FA5", "tickfont": {"color": "#8899AA", "size": 10}},
            "bar": {"color": "#1E6FDB", "thickness": 0.28},
            "bgcolor": "#0D1F35",
            "borderwidth": 1,
            "bordercolor": "#1E3A5F",
            "steps": [
                {"range": [0, 33], "color": "#0A1E10"},
                {"range": [33, 66], "color": "#1A2800"},
                {"range": [66, 100], "color": "#2A0A0A"},
            ],
            "threshold": {
                "line": {"color": "#FF4444", "width": 3},
                "thickness": 0.75,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=200,
    )
    return fig


def score_gauge(score: int):
    """Credit score arc gauge."""
    colour = RISK_COLOURS.get(
        "LOW RISK" if score >= 700 else "MEDIUM RISK" if score >= 500 else "HIGH RISK"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 44, "color": colour, "family": "IBM Plex Mono"}},
        title={"text": "CREDIT SCORE", "font": {"size": 12, "color": "#8899AA", "family": "IBM Plex Sans"}},
        gauge={
            "axis": {
                "range": [300, 900],
                "tickvals": [300, 400, 500, 600, 700, 800, 900],
                "tickcolor": "#4A7FA5",
                "tickfont": {"color": "#8899AA", "size": 10},
            },
            "bar": {"color": colour, "thickness": 0.3},
            "bgcolor": "#0D1F35",
            "borderwidth": 1,
            "bordercolor": "#1E3A5F",
            "steps": [
                {"range": [300, 500], "color": "#2A0A0A"},
                {"range": [500, 700], "color": "#1A1800"},
                {"range": [700, 900], "color": "#0A1E10"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=240,
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.2rem;'>🏦</div>
        <div style='font-size:1.1rem; font-weight:700; letter-spacing:0.02em;'>CreditAI</div>
        <div style='font-size:0.7rem; color:#4A7FA5; letter-spacing:0.12em;'>RISK INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.authenticated:
        role_label = st.session_state.role or "User"
        st.markdown(
            f"<div style='font-size:0.72rem; color:#4A7FA5; letter-spacing:0.12em;'>SIGNED IN AS</div>"
            f"<div style='font-size:0.95rem; color:#E8ECF0; font-weight:600; margin-top:0.3rem;'>{role_label}</div>",
            unsafe_allow_html=True,
        )

        if st.button("Log out"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.prediction_result = None
            st.session_state.applicant_data = None
            st.rerun()

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)

        if st.session_state.role == "Bank":
            page = st.radio(
                "Navigate",
                ["🏠 Home", "📈 Market Watch", "🔍 Applicant Evaluation", "👤 Client Portal", "📊 Model Analytics", "ℹ️ About"],
                label_visibility="collapsed",
            )
        else:
            page = st.radio(
                "Navigate",
                ["👤 Client Home", "📈 Market Watch", "👤 Client Portal", "ℹ️ About"],
                label_visibility="collapsed",
            )

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)

        bundle = load_model()
        if bundle:
            st.markdown("""
            <div style='padding: 0.8rem; background:#0A1E10; border:1px solid #1A4A28; border-radius:8px;'>
                <div style='font-size:0.68rem; color:#4A7FA5; letter-spacing:0.1em;'>MODEL STATUS</div>
                <div style='font-size:0.85rem; color:#00C851; font-weight:600; margin-top:0.3rem;'>● ONLINE</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:0.6rem; font-size:0.75rem; color:#8899AA;'>
                <div>Active model: <span style='color:#E8ECF0; font-family: monospace;'>{bundle['model_name']}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding:0.8rem; background:#2A0A0A; border:1px solid #5A1A1A; border-radius:8px;'>
                <div style='font-size:0.68rem; color:#FF8800; letter-spacing:0.1em;'>MODEL STATUS</div>
                <div style='font-size:0.85rem; color:#FF4444; font-weight:600;'>● OFFLINE</div>
                <div style='font-size:0.7rem; color:#AA6666; margin-top:0.3rem;'>Run train_model.py first</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        page = "Login"
        st.markdown("""
        <div style='font-size:0.75rem; color:#4A7FA5; letter-spacing:0.12em;'>SECURE ACCESS</div>
        <div style='font-size:0.9rem; color:#8899AA; margin-top:0.4rem;'>Sign in to continue</div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if page == "Login":
    st.markdown("""
    <div style='padding: 2.2rem 0 1rem;'>
        <div style='font-size:0.7rem; letter-spacing:0.18em; color:#4A7FA5; text-transform:uppercase; margin-bottom:0.5rem;'>
            Secure sign-in
        </div>
        <h1 style='font-size:2.4rem; font-weight:700; margin:0; line-height:1.1;'>
            Welcome to <span style='color:#1E6FDB;'>CreditAI</span>
        </h1>
        <p style='color:#8899AA; font-size:1.0rem; margin-top:0.8rem; max-width:620px;'>
            Choose your portal to access risk intelligence and eligibility insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("<div class='section-header'>Login</div>", unsafe_allow_html=True)
        role = st.selectbox("Portal", ["Bank", "Client"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")

    if submitted:
        if verify_login(role, username, password):
            st.session_state.authenticated = True
            st.session_state.role = role
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    st.markdown("""
    <div class='metric-card' style='margin-top:1rem;'>
        <div class='section-header'>Demo Credentials</div>
        <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>
            Bank: <span style='font-family:monospace;'>admin / bank123</span><br>
            Client: <span style='font-family:monospace;'>client / client123</span>
        </p>
        <div style='font-size:0.75rem; color:#8899AA; margin-top:0.5rem;'>
            Replace these with real authentication before production use.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Home":
    st.markdown("""
    <div style='padding: 2.5rem 0 1rem;'>
        <div style='font-size:0.7rem; letter-spacing:0.18em; color:#4A7FA5; text-transform:uppercase; margin-bottom:0.5rem;'>
            Powered by XGBoost · SHAP · Scikit-learn
        </div>
        <h1 style='font-size:2.6rem; font-weight:700; margin:0; line-height:1.1;'>
            AI Credit Risk &<br><span style='color:#1E6FDB;'>Loan Default</span> Prediction
        </h1>
        <p style='color:#8899AA; font-size:1.05rem; margin-top:0.8rem; max-width:600px;'>
            Enterprise-grade credit intelligence for modern banking.
            Assess borrower risk in seconds with explainable AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("12,000+", "Training Records"),
        ("97%+", "Model AUC"),
        ("300–900", "Score Range"),
        ("3", "Risk Tiers"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <h2 style='font-size:2rem; color:#1E6FDB;'>{val}</h2>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
        <div class='metric-card'>
            <div class='section-header'>Platform Capabilities</div>
            <ul style='color:#C0CDD8; line-height:2; padding-left:1.2rem; font-size:0.92rem;'>
                <li>Real-time loan default probability scoring</li>
                <li>CIBIL-style credit score generation (300–900)</li>
                <li>3-tier risk classification with recommendations</li>
                <li>SHAP-powered explainability per applicant</li>
                <li>Confusion matrix, ROC-AUC, feature importance</li>
                <li>Batch scoring for portfolio analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card'>
            <div class='section-header'>Score Bands</div>
            <div style='margin-top:0.5rem;'>
                <div style='display:flex; align-items:center; margin-bottom:0.9rem;'>
                    <div style='width:14px; height:14px; border-radius:50%; background:#00C851; margin-right:0.7rem;'></div>
                    <div>
                        <span style='font-family:monospace; font-weight:600;'>700 – 900</span>
                        <span style='color:#8899AA; margin-left:0.5rem; font-size:0.85rem;'>LOW RISK — Approve</span>
                    </div>
                </div>
                <div style='display:flex; align-items:center; margin-bottom:0.9rem;'>
                    <div style='width:14px; height:14px; border-radius:50%; background:#FF8800; margin-right:0.7rem;'></div>
                    <div>
                        <span style='font-family:monospace; font-weight:600;'>500 – 699</span>
                        <span style='color:#8899AA; margin-left:0.5rem; font-size:0.85rem;'>MEDIUM RISK — Review</span>
                    </div>
                </div>
                <div style='display:flex; align-items:center;'>
                    <div style='width:14px; height:14px; border-radius:50%; background:#FF4444; margin-right:0.7rem;'></div>
                    <div>
                        <span style='font-family:monospace; font-weight:600;'>300 – 499</span>
                        <span style='color:#8899AA; margin-left:0.5rem; font-size:0.85rem;'>HIGH RISK — Reject</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: APPLICANT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Applicant Evaluation":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Applicant Evaluation</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>Enter borrower details to generate a real-time credit risk assessment.</p>
    """, unsafe_allow_html=True)

    if bundle is None:
        st.error("⚠️ Model not loaded. Please run `python src/train_model.py` first.")
        st.stop()

    with st.form("applicant_form"):
        st.markdown("<div class='section-header'>Personal & Financial Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
            income = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=600000, step=10000)
            employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed", "Retired"])
            years_employed = st.number_input("Years Employed", min_value=0, max_value=45, value=8)
        with c2:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=1000, max_value=5000000, value=300000, step=5000)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120], index=2)
            interest_rate = st.number_input("Interest Rate (%)", min_value=3.0, max_value=36.0, value=10.5, step=0.25)
            outstanding_loan_amount = st.number_input("Outstanding Loans (₹)", min_value=0, max_value=2000000, value=50000, step=5000)
        with c3:
            credit_history_length = st.number_input("Credit History (years)", min_value=0, max_value=45, value=10)
            number_of_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=2)
            savings_balance = st.number_input("Savings Balance (₹)", min_value=0, max_value=5000000, value=150000, step=5000)
            monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=1000, max_value=500000, value=25000, step=1000)

        st.markdown("<div class='section-header' style='margin-top:0.8rem;'>Risk Indicators</div>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0)
        with rc2:
            late_payments = st.number_input("Late Payments (count)", min_value=0, max_value=20, value=1)
        with rc3:
            transaction_volume = st.number_input("Monthly Transactions", min_value=1, max_value=500, value=30)

        # Auto-compute DTI
        dti = (outstanding_loan_amount + loan_amount) / max(income, 1)
        st.markdown(f"<div style='color:#8899AA; font-size:0.82rem;'>Computed Debt-to-Income Ratio: <span style='color:#E8ECF0; font-family:monospace;'>{dti:.3f}</span></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Evaluate Applicant")

    if submitted:
        applicant = {
            "Age": age, "Income": income, "Employment_Status": employment_status,
            "Years_Employed": years_employed, "Credit_History_Length": credit_history_length,
            "Number_of_Credit_Cards": number_of_credit_cards, "Outstanding_Loan_Amount": outstanding_loan_amount,
            "Debt_to_Income_Ratio": round(dti, 4), "Monthly_Expenses": monthly_expenses,
            "Previous_Defaults": previous_defaults, "Loan_Amount": loan_amount,
            "Loan_Term": loan_term, "Interest_Rate": interest_rate,
            "Savings_Balance": savings_balance, "Transaction_Volume": transaction_volume,
            "Late_Payments": late_payments,
        }
        with st.spinner("Analysing creditworthiness …"):
            try:
                result = predict_single(applicant)
                st.session_state.prediction_result = result
                st.session_state.applicant_data = applicant
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

    if st.session_state.prediction_result:
        r = st.session_state.prediction_result
        risk_colour = r.score_colour

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:1rem; margin-bottom:1.2rem;'>
            <span style='font-size:2rem;'>{r.risk_icon}</span>
            <div>
                <div style='font-size:0.68rem; letter-spacing:0.15em; color:#4A7FA5; text-transform:uppercase;'>Assessment Result</div>
                <div class='risk-pill' style='background:{risk_colour}22; color:{risk_colour}; border:1px solid {risk_colour}55; margin-top:0.3rem;'>
                    {r.risk_category}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
        with col1:
            st.plotly_chart(score_gauge(r.credit_score), use_container_width=True)
        with col2:
            st.plotly_chart(gauge_chart(r.default_probability, "Default Probability"), use_container_width=True)
            st.markdown(f"""
            <div style='text-align:center; margin-top:-0.5rem;'>
                <span style='font-family:monospace; font-size:1.3rem; color:{risk_colour};'>{r.default_probability:.1%}</span>
                <div style='font-size:0.7rem; color:#8899AA;'>chance of default</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='height:100%;'>
                <div class='section-header'>Banker Recommendation</div>
                <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>{r.recommendation}</p>
                <div class='h-line'></div>
                <div style='display:flex; justify-content:space-between; margin-top:0.5rem;'>
                    <div>
                        <div style='font-size:0.68rem; color:#4A7FA5; text-transform:uppercase; letter-spacing:0.1em;'>Score Percentile</div>
                        <div style='font-family:monospace; font-size:1.1rem; color:#E8ECF0;'>{r.score_percentile:.0f}<span style='font-size:0.75rem; color:#8899AA;'>th</span></div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem; color:#4A7FA5; text-transform:uppercase; letter-spacing:0.1em;'>Credit Score</div>
                        <div style='font-family:monospace; font-size:1.1rem; color:{risk_colour};'>{r.credit_score}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Top Factors Influencing This Decision</div>", unsafe_allow_html=True)

        try:
            import shap
            from explainability import build_explainer, explain_single, shap_waterfall_figure
            from data_preprocessing import DataPreprocessor
            from feature_engineering import engineer_features as ef

            model = bundle["model"]
            preprocessor = bundle["preprocessor"]
            app_df = pd.DataFrame([st.session_state.applicant_data])
            app_eng = ef(app_df)
            X_inst = preprocessor.transform(app_eng)

            # Background: small sample from training data
            data_path = ROOT / "data" / "loan_data.csv"
            if data_path.exists():
                bg = pd.read_csv(data_path).sample(min(200, 500), random_state=1)
                bg_eng = ef(bg)
                X_bg = preprocessor.transform(bg_eng)
                explainer = build_explainer(model, X_bg)
                explanation = explain_single(explainer, X_inst)

                top = explanation.get("top_features", [])
                if top:
                    exp_c1, exp_c2 = st.columns([1.2, 0.8])
                    with exp_c1:
                        fig = shap_waterfall_figure(explanation, "Feature Impact on Default Probability")
                        st.pyplot(fig, use_container_width=True)
                    with exp_c2:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        for i, t in enumerate(top[:8], 1):
                            val = t["shap_value"]
                            colour = "#FF4444" if val > 0 else "#00C851"
                            direction = "↑ Risk" if val > 0 else "↓ Risk"
                            st.markdown(f"""
                            <div style='display:flex; justify-content:space-between; align-items:center;
                                        padding:0.35rem 0; border-bottom:1px solid #1A2E4A;'>
                                <div style='font-size:0.82rem; color:#C0CDD8;'>{i}. {t["feature"].replace("_"," ")}</div>
                                <div style='font-family:monospace; font-size:0.8rem; color:{colour};'>
                                    {val:+.4f} <span style='font-size:0.7rem;'>{direction}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLIENT PORTAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Client Home":
    st.markdown("""
    <div style='padding: 2.4rem 0 1rem;'>
        <div style='font-size:0.7rem; letter-spacing:0.18em; color:#4A7FA5; text-transform:uppercase; margin-bottom:0.5rem;'>
            Client access
        </div>
        <h1 style='font-size:2.4rem; font-weight:700; margin:0; line-height:1.1;'>
            Your <span style='color:#1E6FDB;'>Loan Eligibility</span> Dashboard
        </h1>
        <p style='color:#8899AA; font-size:1.0rem; margin-top:0.8rem; max-width:620px;'>
            Check how much loan you can take, estimate monthly affordability, and view your risk profile.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    highlights = [
        ("Eligibility Check", "Estimate max loan based on income and expenses"),
        ("Risk Overview", "Understand default risk and credit score"),
        ("Clear Guidance", "Get practical suggestions to improve approval odds"),
    ]
    for col, (title, desc) in zip([c1, c2, c3], highlights):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='height:100%;'>
                <div class='section-header'>{title}</div>
                <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <div class='section-header'>Get Started</div>
        <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>
            Open the Client Portal to enter your details and view eligibility and risk.
        </p>
    </div>
    """, unsafe_allow_html=True)


elif page == "📈 Market Watch":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Market Watch</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Live prices and changes for major stocks. Data updates every 60 seconds.
    </p>
    """, unsafe_allow_html=True)

    if not STOCK_API_KEY:
        st.warning("Stock API key not configured. Set TWELVEDATA_API_KEY to enable live prices.")
        st.stop()

    market_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "TSLA", "META", "JPM", "V", "WMT",
    ]
    quotes = fetch_stock_quotes(tuple(market_tickers), STOCK_API_KEY, STOCK_API_BASE)

    rows = []
    for t in market_tickers:
        q = quotes.get(t) or {}
        price = q.get("close") or q.get("price")
        change = q.get("change")
        change_pct = q.get("percent_change") or q.get("percent_change") or q.get("change_percent")
        try:
            price_val = float(price) if price is not None else None
        except Exception:
            price_val = None
        try:
            change_val = float(change) if change is not None else None
        except Exception:
            change_val = None
        try:
            change_pct_val = float(change_pct) if change_pct is not None else None
        except Exception:
            change_pct_val = None

        rows.append(
            {
                "Ticker": t,
                "Price": f"₹{price_val:,.2f}" if price_val is not None else "N/A",
                "Change": f"{change_val:+.2f}" if change_val is not None else "N/A",
                "Change %": f"{change_pct_val:+.2f}%" if change_pct_val is not None else "N/A",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


elif page == "👤 Client Portal":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Client Portal</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Estimate how much loan you can take and understand your risk profile.
    </p>
    """, unsafe_allow_html=True)

    if bundle is None:
        st.error("Model not loaded. Please run `python src/train_model.py` first.")
        st.stop()

    with st.form("client_form"):
        st.markdown("<div class='section-header'>Your Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=32, step=1, key="c_age")
            income = st.number_input("Annual Income (₹)", min_value=10000, max_value=10000000, value=650000, step=10000, key="c_income")
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed", "Unemployed", "Retired"],
                key="c_emp",
            )
            years_employed = st.number_input("Years Employed", min_value=0, max_value=45, value=6, key="c_years")
        with c2:
            desired_loan_amount = st.number_input(
                "Desired Loan Amount (₹)", min_value=1000, max_value=5000000, value=400000, step=5000, key="c_loan"
            )
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120], index=2, key="c_term")
            interest_rate = st.number_input(
                "Interest Rate (%)", min_value=3.0, max_value=36.0, value=10.5, step=0.25, key="c_rate"
            )
            outstanding_loan_amount = st.number_input(
                "Outstanding Loans (₹)", min_value=0, max_value=2000000, value=50000, step=5000, key="c_out"
            )
        with c3:
            credit_history_length = st.number_input(
                "Credit History (years)", min_value=0, max_value=45, value=8, key="c_hist"
            )
            number_of_credit_cards = st.number_input(
                "Number of Credit Cards", min_value=0, max_value=20, value=2, key="c_cards"
            )
            savings_balance = st.number_input(
                "Savings Balance (₹)", min_value=0, max_value=5000000, value=120000, step=5000, key="c_save"
            )
            monthly_expenses = st.number_input(
                "Monthly Expenses (₹)", min_value=1000, max_value=500000, value=22000, step=1000, key="c_exp"
            )

        st.markdown("<div class='section-header' style='margin-top:0.8rem;'>Stock Holdings</div>", unsafe_allow_html=True)
        st.caption("Stock value is based on live prices and can change.")
        holdings_df = st.data_editor(
            pd.DataFrame({"Ticker": [""], "Shares": [0.0]}),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Example: AAPL, TSLA, RELIANCE.NS"),
                "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
            },
            key="c_holdings",
        )

        st.markdown("<div class='section-header' style='margin-top:0.8rem;'>Risk Indicators</div>", unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0, key="c_def")
        with rc2:
            late_payments = st.number_input("Late Payments (count)", min_value=0, max_value=20, value=1, key="c_late")
        with rc3:
            transaction_volume = st.number_input("Monthly Transactions", min_value=1, max_value=500, value=30, key="c_txn")

        submitted = st.form_submit_button("Calculate Eligibility & Risk")

    if submitted:
        dti_limit = 0.40
        monthly_income = income / 12.0
        dti_cap = dti_limit * monthly_income
        monthly_capacity = max(0.0, dti_cap - monthly_expenses)
        max_loan = max_affordable_loan(monthly_capacity, interest_rate, int(loan_term))
        max_loan = max(0.0, max_loan)
        desired_emi = loan_emi(desired_loan_amount, interest_rate, int(loan_term))
        desired_dti = (monthly_expenses + desired_emi) / max(monthly_income, 1)

        holdings_clean = holdings_df.copy()
        holdings_clean["Ticker"] = holdings_clean["Ticker"].astype(str).str.upper().str.strip()
        holdings_clean["Shares"] = pd.to_numeric(holdings_clean["Shares"], errors="coerce").fillna(0.0)
        holdings_clean = holdings_clean[
            (holdings_clean["Ticker"] != "") & (holdings_clean["Shares"] > 0)
        ]

        tickers = tuple(holdings_clean["Ticker"].unique().tolist())
        prices = fetch_stock_prices(tickers, STOCK_API_KEY, STOCK_API_BASE) if tickers else {}

        portfolio_rows = []
        portfolio_value = 0.0
        for _, row in holdings_clean.iterrows():
            ticker = row["Ticker"]
            shares = float(row["Shares"])
            price = prices.get(ticker)
            value = shares * price if price is not None else 0.0
            portfolio_value += value
            portfolio_rows.append(
                {"Ticker": ticker, "Shares": shares, "Price": price, "Value": value}
            )

        total_savings = savings_balance + portfolio_value

        def build_applicant(loan_amount_value: float):
            dti = (outstanding_loan_amount + loan_amount_value) / max(income, 1)
            return {
                "Age": age, "Income": income, "Employment_Status": employment_status,
                "Years_Employed": years_employed, "Credit_History_Length": credit_history_length,
                "Number_of_Credit_Cards": number_of_credit_cards, "Outstanding_Loan_Amount": outstanding_loan_amount,
                "Debt_to_Income_Ratio": round(dti, 4), "Monthly_Expenses": monthly_expenses,
                "Previous_Defaults": previous_defaults, "Loan_Amount": loan_amount_value,
                "Loan_Term": loan_term, "Interest_Rate": interest_rate,
                "Savings_Balance": total_savings, "Transaction_Volume": transaction_volume,
                "Late_Payments": late_payments,
            }

        with st.spinner("Calculating your eligibility ..."):
            try:
                desired_result = predict_single(build_applicant(desired_loan_amount))
                max_result = None
                if max_loan > 0:
                    max_result = predict_single(build_applicant(max_loan))
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='section-header'>Estimated Max Loan</div>
                <h2 style='font-size:2rem; color:#1E6FDB;'>₹{max_loan:,.0f}</h2>
                <p>Based on affordability</p>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='section-header'>Monthly Capacity</div>
                <h2 style='font-size:2rem; color:#1E6FDB;'>₹{monthly_capacity:,.0f}</h2>
                <p>DTI cap minus expenses</p>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='section-header'>Desired Amount</div>
                <h2 style='font-size:2rem; color:#1E6FDB;'>₹{desired_loan_amount:,.0f}</h2>
                <p>Requested by you</p>
            </div>
            """, unsafe_allow_html=True)

        if tickers:
            st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Live Stock Valuation</div>", unsafe_allow_html=True)
            if not STOCK_API_KEY:
                st.warning("Stock API key not configured. Set TWELVEDATA_API_KEY to enable live prices.")
            elif any(prices.get(t) is None for t in tickers):
                st.warning("Some tickers could not be priced. Those values are treated as ₹0.")

            if portfolio_rows:
                display_df = pd.DataFrame(portfolio_rows)
                display_df["Price"] = display_df["Price"].apply(lambda x: f"₹{x:,.2f}" if x is not None else "N/A")
                display_df["Value"] = display_df["Value"].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class='metric-card' style='text-align:center; margin-top:0.8rem;'>
                <div class='section-header'>Total Stock Value</div>
                <h2 style='font-size:2rem; color:#1E6FDB;'>₹{portfolio_value:,.0f}</h2>
                <p>Added to savings for risk scoring</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        d1, d2 = st.columns([1.3, 1])
        with d1:
            st.plotly_chart(gauge_chart(desired_result.default_probability, "Default Risk (Desired Amount)"), use_container_width=True)
            st.markdown(f"""
            <div style='text-align:center; margin-top:-0.5rem;'>
                <span style='font-family:monospace; font-size:1.3rem; color:{desired_result.score_colour};'>{desired_result.default_probability:.1%}</span>
                <div style='font-size:0.7rem; color:#8899AA;'>chance of default</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class='metric-card' style='height:100%;'>
                <div class='section-header'>Risk Summary</div>
                <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>
                    <strong style='color:{desired_result.score_colour};'>{desired_result.risk_category}</strong><br>
                    {desired_result.recommendation}
                </p>
                <div class='h-line'></div>
                <div style='font-size:0.8rem; color:#8899AA;'>
                    Credit score estimate: <span style='color:{desired_result.score_colour}; font-family:monospace;'>{desired_result.credit_score}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        breakdown = f"""
        <div class='metric-card'>
            <div class='section-header'>Why Not Eligible (Breakdown)</div>
            <div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #1A2E4A;'>
                <div style='color:#C0CDD8;'>Monthly income</div>
                <div style='font-family:monospace; color:#E8ECF0;'>₹{monthly_income:,.0f}</div>
            </div>
            <div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #1A2E4A;'>
                <div style='color:#C0CDD8;'>DTI cap ({int(dti_limit*100)}%)</div>
                <div style='font-family:monospace; color:#E8ECF0;'>₹{dti_cap:,.0f}</div>
            </div>
            <div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #1A2E4A;'>
                <div style='color:#C0CDD8;'>Monthly expenses</div>
                <div style='font-family:monospace; color:#E8ECF0;'>₹{monthly_expenses:,.0f}</div>
            </div>
            <div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #1A2E4A;'>
                <div style='color:#C0CDD8;'>Capacity for EMI</div>
                <div style='font-family:monospace; color:#E8ECF0;'>₹{monthly_capacity:,.0f}</div>
            </div>
            <div style='display:flex; justify-content:space-between; padding:0.35rem 0;'>
                <div style='color:#C0CDD8;'>Desired EMI (approx)</div>
                <div style='font-family:monospace; color:#E8ECF0;'>₹{desired_emi:,.0f}</div>
            </div>
            <div style='margin-top:0.6rem; font-size:0.8rem; color:#8899AA;'>
                Desired DTI: {desired_dti:.2f} (limit {dti_limit:.2f})
            </div>
        </div>
        """
        st.markdown(breakdown, unsafe_allow_html=True)

        if max_loan > 0:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='section-header'>Risk At Maximum Affordable Loan</div>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-size:0.95rem; color:#E8ECF0; font-weight:600;'>₹{max_loan:,.0f}</div>
                        <div style='font-size:0.8rem; color:#8899AA;'>Estimated eligibility limit</div>
                    </div>
                    <div class='risk-pill' style='background:{max_result.score_colour}22; color:{max_result.score_colour}; border:1px solid {max_result.score_colour}55;'>
                        {max_result.risk_category}
                    </div>
                </div>
                <div style='margin-top:0.8rem; color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>
                    {max_result.recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Model Analytics</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>Performance metrics, confusion matrix, and feature importance.</p>
    """, unsafe_allow_html=True)

    metrics_path = ROOT / "models" / "training_metrics.json"
    if not metrics_path.exists():
        st.warning("No training metrics found. Run `python src/train_model.py` to generate them.")
        st.stop()

    with open(metrics_path) as f:
        metrics = json.load(f)

    best_name = metrics.get("best_model", "")
    model_names = [k for k in metrics if k != "best_model"]

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    comp_data = []
    for name in model_names:
        m = metrics[name]
        comp_data.append({
            "Model": name,
            "Accuracy": m.get("accuracy", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "ROC-AUC": m.get("roc_auc", 0),
        })
    comp_df = pd.DataFrame(comp_data)

    fig_comp = go.Figure()
    metric_cols = ["Accuracy", "Precision", "Recall", "ROC-AUC"]
    colours = ["#1E6FDB", "#00C851", "#FF8800", "#A855F7"]
    for col, colour in zip(metric_cols, colours):
        fig_comp.add_trace(go.Bar(
            name=col, x=comp_df["Model"], y=comp_df[col],
            marker_color=colour, opacity=0.85,
        ))
    fig_comp.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A1628",
        font={"color": "#E8ECF0", "family": "IBM Plex Sans"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "#1A2E4A"}, yaxis={"gridcolor": "#1A2E4A", "range": [0, 1]},
        height=300, margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Metric cards for best model ───────────────────────────────────────────
    if best_name and best_name in metrics:
        bm = metrics[best_name]
        st.markdown(f"<div class='section-header'>Best Model: {best_name}</div>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (label, key) in zip(
            [mc1, mc2, mc3, mc4],
            [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall")]
        ):
            with col:
                val = bm.get(key, 0)
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <h2 style='font-size:1.9rem; color:#1E6FDB;'>{val:.3f}</h2>
                    <p>{label}</p>
                </div>
                """, unsafe_allow_html=True)

        # Confusion matrix
        cm = bm.get("confusion_matrix", [[0, 0], [0, 0]])
        st.markdown("<div class='section-header' style='margin-top:1rem;'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Default", "Default"], y=["No Default", "Default"],
        )
        cm_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A1628",
            font={"color": "#E8ECF0"}, height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    # ── Feature importance (from model if available) ──────────────────────────
    if bundle:
        st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
        model = bundle["model"]
        feat_names = bundle.get("feature_columns", [])
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is not None and len(feat_names) == len(importances):
            fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)
            fi_fig = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color="#1E6FDB", opacity=0.8,
            ))
            fi_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0A1628",
                font={"color": "#E8ECF0", "family": "IBM Plex Sans"},
                xaxis={"gridcolor": "#1A2E4A"}, yaxis={"gridcolor": "#1A2E4A"},
                height=500, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700;'>About This Platform</h2>
    <div class='metric-card' style='margin-top:1.5rem;'>
        <div class='section-header'>Technology Stack</div>
        <p style='color:#C0CDD8; line-height:1.9; font-size:0.92rem;'>
            <strong style='color:#E8ECF0;'>Machine Learning:</strong> XGBoost, Random Forest, Logistic Regression (Scikit-learn)<br>
            <strong style='color:#E8ECF0;'>Explainability:</strong> SHAP (SHapley Additive exPlanations)<br>
            <strong style='color:#E8ECF0;'>Data Processing:</strong> Pandas, NumPy, StandardScaler, LabelEncoder<br>
            <strong style='color:#E8ECF0;'>Visualisation:</strong> Plotly, Matplotlib<br>
            <strong style='color:#E8ECF0;'>UI Framework:</strong> Streamlit<br>
            <strong style='color:#E8ECF0;'>Model Persistence:</strong> Joblib<br>
        </p>
    </div>
    <div class='metric-card' style='margin-top:1rem;'>
        <div class='section-header'>Scoring Methodology</div>
        <p style='color:#C0CDD8; line-height:1.7; font-size:0.92rem;'>
            The platform generates a credit score in the 300–900 range, mirroring India's CIBIL scoring system.
            The formula maps the model's predicted default probability directly to a score:<br><br>
            <code style='background:#0A1628; padding:0.3rem 0.6rem; border-radius:4px; font-size:0.9rem;'>
                score = 300 + (1 − P(default)) × 600
            </code><br><br>
            A probability of 0% maps to 900 (excellent), while 100% maps to 300 (high risk).
        </p>
    </div>
    """, unsafe_allow_html=True)
