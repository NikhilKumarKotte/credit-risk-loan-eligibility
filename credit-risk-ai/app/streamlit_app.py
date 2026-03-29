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
from datetime import datetime
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

:root {
    --bg-0: #05070B;
    --bg-1: #0A0F16;
    --bg-2: #0F1622;
    --gold-1: #C8A24A;
    --gold-2: #E3C168;
    --gold-3: #8A6B2B;
    --text-0: #F2F2F2;
    --text-1: #B8C0CC;
    --border-0: #1A2430;
    --shadow-1: 10px 10px 24px rgba(0,0,0,0.65);
    --shadow-2: -8px -8px 18px rgba(255,255,255,0.03);
    --inset-1: inset 6px 6px 12px rgba(0,0,0,0.55);
    --inset-2: inset -6px -6px 12px rgba(255,255,255,0.04);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: var(--bg-0);
    color: var(--text-0);
}
.stApp {
    background: radial-gradient(1200px 800px at 15% 10%, #0F1724 0%, #06090F 55%, #05070B 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B111A 0%, #070A10 100%);
    border-right: 1px solid var(--border-0);
    box-shadow: var(--shadow-1);
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #0D131D 0%, #0B1018 100%);
    border: 1px solid rgba(200,162,74,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: var(--shadow-1), var(--shadow-2);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.metric-card:before {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: 14px;
    box-shadow: var(--inset-1), var(--inset-2);
    pointer-events: none;
}
.metric-card h2 { margin: 0 0 0.2rem 0; font-family: 'IBM Plex Mono', monospace; color: var(--gold-2); }
.metric-card p  { margin: 0; color: var(--text-1); font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(227,193,104,0.45);
    box-shadow: 14px 14px 30px rgba(0,0,0,0.7), -8px -8px 18px rgba(255,255,255,0.05), 0 0 28px rgba(227,193,104,0.18);
}

/* Score badge */
.score-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.6rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    color: var(--gold-2);
    text-shadow: 0 6px 18px rgba(200,162,74,0.25);
}
.section-header {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-1);
    margin-bottom: 0.8rem;
    font-weight: 600;
    text-shadow: 0 0 18px rgba(200,162,74,0.35), 0 0 32px rgba(200,162,74,0.18);
}
/* Form inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input {
    background: #0B111A !important;
    border: 1px solid rgba(200,162,74,0.2) !important;
    color: var(--text-0) !important;
    border-radius: 12px !important;
    box-shadow: var(--inset-1), var(--inset-2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(145deg, #1A1F27 0%, #0B0F14 100%);
    color: var(--gold-2);
    border: 1px solid rgba(200,162,74,0.35);
    border-radius: 14px;
    padding: 0.8rem 2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-size: 0.9rem;
    transition: all 0.2s ease;
    width: 100%;
    margin-top: 0.5rem;
    box-shadow: var(--shadow-1), var(--shadow-2);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 12px 12px 28px rgba(0,0,0,0.7), -6px -6px 14px rgba(255,255,255,0.05);
    color: #F3D07B;
}

/* Risk pill */
.risk-pill {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
    background: linear-gradient(145deg, #0F151E 0%, #0A0E14 100%);
    border: 1px solid rgba(200,162,74,0.25);
    box-shadow: var(--shadow-1), var(--shadow-2);
}

/* Divider */
.h-line {
    border-top: 1px solid rgba(200,162,74,0.35);
    margin: 1.2rem 0;
    box-shadow: 0 0 18px rgba(200,162,74,0.18);
}

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
if "bank" not in st.session_state:
    st.session_state.bank = None
if "bank_role" not in st.session_state:
    st.session_state.bank_role = None


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model …")
def load_model():
    try:
        return load_bundle()
    except FileNotFoundError:
        return None


DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CREDENTIALS_PATH = DATA_DIR / "credentials.csv"
APPLICATIONS_PATH = DATA_DIR / "applications.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
DOCUMENTS_PATH = DATA_DIR / "documents.csv"
NOTIFICATIONS_PATH = DATA_DIR / "notifications.csv"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"
QUERIES_PATH = DATA_DIR / "queries.csv"
EMAIL_LOG_PATH = LOG_DIR / "emails.log"

BANKS = ["SBI", "Indian Bank", "Axis"]


def _seed_credentials() -> None:
    if CREDENTIALS_PATH.exists():
        return
    seed = pd.DataFrame(
        [
            {"role": "Bank", "bank": "SBI", "bank_role": "Manager", "username": "sbi_admin", "password": "bank123"},
            {"role": "Bank", "bank": "SBI", "bank_role": "Officer", "username": "sbi_officer", "password": "bank123"},
            {"role": "Bank", "bank": "Indian Bank", "bank_role": "Manager", "username": "indian_admin", "password": "bank123"},
            {"role": "Bank", "bank": "Indian Bank", "bank_role": "Officer", "username": "indian_officer", "password": "bank123"},
            {"role": "Bank", "bank": "Axis", "bank_role": "Manager", "username": "axis_admin", "password": "bank123"},
            {"role": "Bank", "bank": "Axis", "bank_role": "Officer", "username": "axis_officer", "password": "bank123"},
            {"role": "Client", "bank": "", "bank_role": "", "username": "client", "password": "client123"},
        ]
    )
    seed.to_csv(CREDENTIALS_PATH, index=False)


def load_credentials() -> pd.DataFrame:
    _seed_credentials()
    df = pd.read_csv(CREDENTIALS_PATH)
    if "bank_role" not in df.columns:
        df["bank_role"] = ""
        df.to_csv(CREDENTIALS_PATH, index=False)
    return df


def verify_login(
    role: str,
    username: str,
    password: str,
    bank: str | None = None,
    bank_role: str | None = None,
) -> bool:
    df = load_credentials()
    if role == "Client":
        row = df[
            (df["role"] == role)
            & (df["username"] == username)
            & (df["password"] == password)
        ]
    else:
        bank_val = bank or ""
        role_val = bank_role or ""
        row = df[
            (df["role"] == role)
            & (df["username"] == username)
            & (df["password"] == password)
            & (df["bank"] == bank_val)
            & (df["bank_role"] == role_val)
        ]
    return not row.empty


def create_client_account(username: str, password: str, full_name: str = "", email: str = "", phone: str = "") -> tuple[bool, str]:
    df = load_credentials()
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."
    if not df[df["username"] == username].empty:
        return False, "Username already exists. Please choose another."
    df.loc[len(df)] = {
        "role": "Client",
        "bank": "",
        "bank_role": "",
        "username": username,
        "password": password,
    }
    df.to_csv(CREDENTIALS_PATH, index=False)
    customers = load_customers()
    if customers[customers["client_username"] == username].empty:
        customers.loc[len(customers)] = {
            "client_username": username,
            "full_name": full_name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "address": "",
            "kyc_status": "Pending",
            "risk_flags": "",
        }
        customers.to_csv(CUSTOMERS_PATH, index=False)
    return True, "Account created successfully."


def _ensure_csv(path: Path, columns: list[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def init_data_files() -> None:
    _ensure_csv(APPLICATIONS_PATH, [
        "application_id", "client_username", "bank", "submitted_at",
        "status", "decision_at", "decision_by", "payload_json",
    ])
    _ensure_csv(CUSTOMERS_PATH, [
        "client_username", "full_name", "email", "phone", "address",
        "kyc_status", "risk_flags",
    ])
    _ensure_csv(DOCUMENTS_PATH, [
        "client_username", "doc_type", "file_path", "uploaded_at",
    ])
    _ensure_csv(NOTIFICATIONS_PATH, [
        "client_username", "message", "created_at", "read",
    ])
    _ensure_csv(AUDIT_LOG_PATH, [
        "event_time", "actor", "bank", "action", "application_id", "notes",
    ])
    _ensure_csv(QUERIES_PATH, [
        "client_username", "message", "submitted_at", "status",
    ])
    if not PRODUCTS_PATH.exists():
        products = pd.DataFrame([
            {"product": "Personal Loan", "tenure_months": "12-60", "interest_rate": "10-18%", "max_amount": "₹10,00,000"},
            {"product": "Home Loan", "tenure_months": "60-240", "interest_rate": "7-10%", "max_amount": "₹1,00,00,000"},
            {"product": "Auto Loan", "tenure_months": "12-84", "interest_rate": "8-12%", "max_amount": "₹15,00,000"},
            {"product": "Business Loan", "tenure_months": "12-120", "interest_rate": "12-20%", "max_amount": "₹50,00,000"},
        ])
        products.to_csv(PRODUCTS_PATH, index=False)


def load_customers() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(CUSTOMERS_PATH)


def save_customers(df: pd.DataFrame) -> None:
    df.to_csv(CUSTOMERS_PATH, index=False)


def load_documents() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(DOCUMENTS_PATH)


def save_documents(df: pd.DataFrame) -> None:
    df.to_csv(DOCUMENTS_PATH, index=False)


def load_notifications() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(NOTIFICATIONS_PATH)


def add_notification(client_username: str, message: str) -> None:
    notes = load_notifications()
    notes.loc[len(notes)] = {
        "client_username": client_username,
        "message": message,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read": False,
    }
    notes.to_csv(NOTIFICATIONS_PATH, index=False)


def load_audit_log() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(AUDIT_LOG_PATH)


def append_audit(actor: str, bank: str, action: str, application_id: str, notes: str = "") -> None:
    log = load_audit_log()
    log.loc[len(log)] = {
        "event_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actor": actor,
        "bank": bank,
        "action": action,
        "application_id": application_id,
        "notes": notes,
    }
    log.to_csv(AUDIT_LOG_PATH, index=False)


def load_products() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(PRODUCTS_PATH)


def load_queries() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(QUERIES_PATH)


def save_queries(df: pd.DataFrame) -> None:
    df.to_csv(QUERIES_PATH, index=False)


def load_applications() -> pd.DataFrame:
    init_data_files()
    return pd.read_csv(APPLICATIONS_PATH)


def save_applications(df: pd.DataFrame) -> None:
    df.to_csv(APPLICATIONS_PATH, index=False)


def log_demo_email(to_email: str, subject: str, body: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EMAIL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] TO: {to_email} | SUBJECT: {subject}\n{body}\n---\n")


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
        if st.session_state.role == "Bank":
            role_label = f"{st.session_state.bank} • {st.session_state.bank_role}"
        else:
            role_label = "Client"
        st.markdown(
            f"<div style='font-size:0.72rem; color:#4A7FA5; letter-spacing:0.12em;'>SIGNED IN AS</div>"
            f"<div style='font-size:0.95rem; color:#E8ECF0; font-weight:600; margin-top:0.3rem;'>{role_label}</div>",
            unsafe_allow_html=True,
        )

        if st.button("Log out"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.bank = None
            st.session_state.bank_role = None
            st.session_state.prediction_result = None
            st.session_state.applicant_data = None
            st.rerun()

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)

        if st.session_state.role == "Bank":
            bank_role = st.session_state.get("bank_role", "Officer")
            manager_pages = [
                "🏠 Home",
                "📈 Market Watch",
                "🔍 Applicant Evaluation",
                "👤 Client Portal",
                "📨 Applications",
                "👥 Customer Profiles",
                "📦 Loan Products",
                "✅ Approval Workflow",
                "⚖️ Credit Policy",
                "🧮 Rate & EMI Calculator",
                "📉 Risk Monitoring",
                "🧾 Audit Log",
                "🔔 Notifications",
                "📊 Model Analytics",
                "ℹ️ About",
            ]
            officer_pages = [
                "🏠 Home",
                "📈 Market Watch",
                "🔍 Applicant Evaluation",
                "👤 Client Portal",
                "📨 Applications",
                "👥 Customer Profiles",
                "📦 Loan Products",
                "✅ Approval Workflow",
                "🧮 Rate & EMI Calculator",
                "🔔 Notifications",
                "ℹ️ About",
            ]
            page_list = manager_pages if bank_role == "Manager" else officer_pages
            page = st.radio(
                "Navigate",
                page_list,
                label_visibility="collapsed",
            )
        else:
            page = st.radio(
                "Navigate",
                ["👤 Client Home", "📈 Market Watch", "👤 Client Portal", "🔔 Notifications", "ℹ️ About"],
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

    st.markdown("<div class='section-header'>Login</div>", unsafe_allow_html=True)
    role = st.selectbox("Portal", ["Bank", "Client"])

    if role == "Bank":
        with st.form("bank_login_form"):
            bank = st.selectbox("Bank", BANKS)
            bank_role = st.selectbox("Role", ["Manager", "Officer"])
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
        if submitted:
            if verify_login(role, username, password, bank, bank_role):
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username
                st.session_state.bank = bank
                st.session_state.bank_role = bank_role
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    else:
        access_mode = st.radio("Client Access", ["Existing User", "New User"], horizontal=True)
        if access_mode == "Existing User":
            with st.form("client_login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In")
            if submitted:
                if verify_login("Client", username, password):
                    st.session_state.authenticated = True
                    st.session_state.role = "Client"
                    st.session_state.username = username
                    st.session_state.bank = None
                    st.session_state.bank_role = None
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        else:
            with st.form("client_register_form"):
                st.markdown("<div class='section-header'>Create Account</div>", unsafe_allow_html=True)
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                username = st.text_input("Choose Username")
                password = st.text_input("Choose Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account")
            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_client_account(username, password, full_name, email, phone)
                    if ok:
                        st.success(msg)
                        st.session_state.authenticated = True
                        st.session_state.role = "Client"
                        st.session_state.username = username.strip()
                        st.session_state.bank = None
                        st.session_state.bank_role = None
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown("""
    <div class='metric-card' style='margin-top:1rem;'>
        <div class='section-header'>Demo Credentials</div>
        <p style='color:#C0CDD8; font-size:0.9rem; line-height:1.7;'>
            SBI Manager: <span style='font-family:monospace;'>sbi_admin / bank123</span><br>
            SBI Officer: <span style='font-family:monospace;'>sbi_officer / bank123</span><br>
            Indian Bank Manager: <span style='font-family:monospace;'>indian_admin / bank123</span><br>
            Indian Bank Officer: <span style='font-family:monospace;'>indian_officer / bank123</span><br>
            Axis Manager: <span style='font-family:monospace;'>axis_admin / bank123</span><br>
            Axis Officer: <span style='font-family:monospace;'>axis_officer / bank123</span><br>
            Client: <span style='font-family:monospace;'>client / client123</span>
        </p>
        <div style='font-size:0.75rem; color:#8899AA; margin-top:0.5rem;'>
            Credentials are stored in CSV for demo purposes only.
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

    st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Client Dashboard</div>", unsafe_allow_html=True)
    apps = load_applications()
    my_apps = apps[apps["client_username"] == st.session_state.username]
    pending = (my_apps["status"] == "Pending").sum()
    accepted = (my_apps["status"] == "Accepted").sum()
    rejected = (my_apps["status"] == "Rejected").sum()
    c1, c2, c3 = st.columns(3)
    for col, (val, label) in zip([c1, c2, c3], [(pending, "Pending"), (accepted, "Accepted"), (rejected, "Rejected")]):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <h2 style='font-size:2rem; color:#1E6FDB;'>{val}</h2>
                <p>{label} Applications</p>
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
        st.session_state.client_calc = {
            "desired_result": desired_result,
            "max_result": max_result,
            "max_loan": max_loan,
            "monthly_income": monthly_income,
            "dti_cap": dti_cap,
            "monthly_capacity": monthly_capacity,
            "dti_limit": dti_limit,
            "desired_emi": desired_emi,
            "desired_dti": desired_dti,
            "portfolio_rows": portfolio_rows,
            "portfolio_value": portfolio_value,
            "tickers": list(tickers),
            "prices": prices,
            "total_savings": total_savings,
            "desired_loan_amount": desired_loan_amount,
            "loan_term": loan_term,
            "interest_rate": interest_rate,
        }

    calc = st.session_state.get("client_calc")
    if not submitted and not calc:
        st.info("Calculate eligibility to see results.")
        st.stop()
    if calc:
        desired_result = calc["desired_result"]
        max_result = calc["max_result"]
        max_loan = calc["max_loan"]
        monthly_income = calc["monthly_income"]
        dti_cap = calc["dti_cap"]
        monthly_capacity = calc["monthly_capacity"]
        dti_limit = calc["dti_limit"]
        desired_emi = calc["desired_emi"]
        desired_dti = calc["desired_dti"]
        portfolio_rows = calc["portfolio_rows"]
        portfolio_value = calc["portfolio_value"]
        tickers = tuple(calc["tickers"])
        prices = calc["prices"]
        total_savings = calc["total_savings"]
        desired_loan_amount = calc["desired_loan_amount"]
        loan_term = calc["loan_term"]
        interest_rate = calc["interest_rate"]

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

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Loan Application</div>", unsafe_allow_html=True)
        st.caption("Submit your application to selected banks or all banks. Demo email logs are saved to logs/emails.log.")

        application_payload = {
            "client_username": st.session_state.username,
            "age": age,
            "income": income,
            "employment_status": employment_status,
            "years_employed": years_employed,
            "credit_history_length": credit_history_length,
            "number_of_credit_cards": number_of_credit_cards,
            "outstanding_loan_amount": outstanding_loan_amount,
            "monthly_expenses": monthly_expenses,
            "previous_defaults": previous_defaults,
            "late_payments": late_payments,
            "loan_amount": desired_loan_amount,
            "loan_term": loan_term,
            "interest_rate": interest_rate,
            "savings_balance": savings_balance,
            "stock_portfolio_value": portfolio_value,
            "default_probability": float(desired_result.default_probability),
            "credit_score": int(desired_result.credit_score),
            "risk_category": desired_result.risk_category,
        }

        if "selected_banks" not in st.session_state:
            st.session_state.selected_banks = BANKS.copy()
        else:
            st.session_state.selected_banks = [
                bank for bank in st.session_state.selected_banks if bank in BANKS
            ]
        if "selected_banks_widget" not in st.session_state:
            st.session_state.selected_banks_widget = st.session_state.selected_banks.copy()

        def _sync_selected_banks() -> None:
            st.session_state.selected_banks = st.session_state.selected_banks_widget.copy()

        c_sel_all, c_clear_all = st.columns(2)
        with c_sel_all:
            if st.button("Select All", key="select_all_banks_btn"):
                st.session_state.selected_banks = BANKS.copy()
                st.session_state.selected_banks_widget = BANKS.copy()
        with c_clear_all:
            if st.button("Clear All", key="clear_all_banks_btn"):
                st.session_state.selected_banks = []
                st.session_state.selected_banks_widget = []
        selected_banks = st.multiselect(
            "Select banks",
            BANKS,
            key="selected_banks_widget",
            on_change=_sync_selected_banks,
        )
        st.session_state.selected_banks = st.session_state.selected_banks_widget.copy()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Submit to Selected Banks"):
                apps = load_applications()
                for bank in selected_banks:
                    app_id = f"APP-{int(datetime.now().timestamp())}-{bank.replace(' ', '').upper()}"
                    apps.loc[len(apps)] = {
                        "application_id": app_id,
                        "client_username": st.session_state.username,
                        "bank": bank,
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "Pending",
                        "decision_at": "",
                        "decision_by": "",
                        "payload_json": json.dumps(application_payload),
                    }
                save_applications(apps)
                customers = load_customers()
                if st.session_state.username not in customers["client_username"].astype(str).tolist():
                    customers.loc[len(customers)] = {
                        "client_username": st.session_state.username,
                        "full_name": st.session_state.username,
                        "email": f"{st.session_state.username}@example.com",
                        "phone": "",
                        "address": "",
                        "kyc_status": "Pending",
                        "risk_flags": "",
                    }
                    save_customers(customers)
                add_notification(st.session_state.username, "Application submitted to selected banks.")
                st.success("Application submitted to selected banks.")
        with col_b:
            if st.button("Submit to All Banks"):
                apps = load_applications()
                for bank in BANKS:
                    app_id = f"APP-{int(datetime.now().timestamp())}-{bank.replace(' ', '').upper()}"
                    apps.loc[len(apps)] = {
                        "application_id": app_id,
                        "client_username": st.session_state.username,
                        "bank": bank,
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "Pending",
                        "decision_at": "",
                        "decision_by": "",
                        "payload_json": json.dumps(application_payload),
                    }
                save_applications(apps)
                customers = load_customers()
                if st.session_state.username not in customers["client_username"].astype(str).tolist():
                    customers.loc[len(customers)] = {
                        "client_username": st.session_state.username,
                        "full_name": st.session_state.username,
                        "email": f"{st.session_state.username}@example.com",
                        "phone": "",
                        "address": "",
                        "kyc_status": "Pending",
                        "risk_flags": "",
                    }
                    save_customers(customers)
                add_notification(st.session_state.username, "Application submitted to all banks.")
                st.success("Application submitted to all banks.")

        st.download_button(
            "Download Your Application",
            data=pd.DataFrame([application_payload]).to_csv(index=False),
            file_name="creditai_application.csv",
            mime="text/csv",
        )

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Document Upload</div>", unsafe_allow_html=True)
        doc_type = st.selectbox("Document Type", ["ID Proof", "Income Proof", "Bank Statement", "Other"])
        upload = st.file_uploader("Upload document", type=["pdf", "png", "jpg", "jpeg"])
        if upload is not None:
            file_name = f"{st.session_state.username}_{int(datetime.now().timestamp())}_{upload.name}"
            file_path = UPLOAD_DIR / file_name
            with open(file_path, "wb") as f:
                f.write(upload.getbuffer())
            docs = load_documents()
            docs.loc[len(docs)] = {
                "client_username": st.session_state.username,
                "doc_type": doc_type,
                "file_path": str(file_path),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_documents(docs)
            add_notification(st.session_state.username, f"Uploaded document: {doc_type}.")
            st.success("Document uploaded successfully.")

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Application History</div>", unsafe_allow_html=True)
        all_apps = load_applications()
        hist = all_apps[all_apps["client_username"] == st.session_state.username]
        if hist.empty:
            st.info("No applications submitted yet.")
        else:
            st.dataframe(hist[["application_id", "bank", "submitted_at", "status", "decision_at"]], use_container_width=True)

        st.markdown("<div class='h-line'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Client Queries</div>", unsafe_allow_html=True)
        query_text = st.text_area("Ask a question to the bank", height=120)
        if st.button("Submit Query"):
            if query_text.strip():
                q = load_queries()
                q.loc[len(q)] = {
                    "client_username": st.session_state.username,
                    "message": query_text.strip(),
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "Open",
                }
                save_queries(q)
                add_notification(st.session_state.username, "Query submitted to bank.")
                st.success("Query submitted.")
            else:
                st.warning("Please enter a query.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NOTIFICATIONS (Client)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Notifications":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Notifications</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Updates about applications, documents, and decisions.
    </p>
    """, unsafe_allow_html=True)
    notes = load_notifications()
    if st.session_state.role == "Bank":
        if notes.empty:
            st.info("No notifications yet.")
        else:
            st.dataframe(notes, use_container_width=True)
    else:
        mine = notes[notes["client_username"] == st.session_state.username]
        if mine.empty:
            st.info("No notifications yet.")
        else:
            st.dataframe(mine[["created_at", "message"]], use_container_width=True)


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
# PAGE: APPLICATIONS (Bank)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📨 Applications":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Applications</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Review client applications and accept or reject them. Accepted applications generate a demo email log.
    </p>
    """, unsafe_allow_html=True)

    bank_name = st.session_state.get("bank", "")
    apps = load_applications()
    if bank_name:
        apps = apps[apps["bank"] == bank_name]

    if apps.empty:
        st.info("No applications found for this bank.")
        st.stop()

    st.dataframe(apps[["application_id", "client_username", "bank", "submitted_at", "status"]], use_container_width=True)

    selected_id = st.selectbox("Select application", apps["application_id"].tolist())
    app_row = apps[apps["application_id"] == selected_id].iloc[0]
    payload = json.loads(app_row["payload_json"])

    st.markdown("<div class='section-header'>Application Details</div>", unsafe_allow_html=True)
    st.json(payload)
    if st.session_state.get("bank_role") != "Manager":
        st.info("Officer role can review only. Manager approval required for accept/reject.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept Application", disabled=st.session_state.get("bank_role") != "Manager"):
            all_apps = load_applications()
            idx = all_apps[all_apps["application_id"] == selected_id].index
            all_apps.loc[idx, "status"] = "Accepted"
            all_apps.loc[idx, "decision_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_apps.loc[idx, "decision_by"] = st.session_state.username
            save_applications(all_apps)
            append_audit(st.session_state.username, app_row["bank"], "Accepted", selected_id)
            add_notification(payload["client_username"], f"Application {selected_id} accepted by {app_row['bank']}.")
            log_demo_email(
                to_email=f"{payload['client_username']}@example.com",
                subject="CreditAI Application Accepted",
                body=f"Your loan application ({selected_id}) has been accepted by {app_row['bank']}.",
            )
            st.success("Application accepted. Demo email logged.")
    with col2:
        if st.button("Reject Application", disabled=st.session_state.get("bank_role") != "Manager"):
            all_apps = load_applications()
            idx = all_apps[all_apps["application_id"] == selected_id].index
            all_apps.loc[idx, "status"] = "Rejected"
            all_apps.loc[idx, "decision_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_apps.loc[idx, "decision_by"] = st.session_state.username
            save_applications(all_apps)
            append_audit(st.session_state.username, app_row["bank"], "Rejected", selected_id)
            add_notification(payload["client_username"], f"Application {selected_id} rejected by {app_row['bank']}.")
            st.warning("Application rejected.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER PROFILES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Profiles":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Customer Profiles</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        KYC summaries and risk flags for clients.
    </p>
    """, unsafe_allow_html=True)
    customers = load_customers()
    if customers.empty:
        st.info("No customer profiles available.")
    else:
        st.dataframe(customers, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LOAN PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Loan Products":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Loan Products</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Bank lending products and standard terms.
    </p>
    """, unsafe_allow_html=True)
    products = load_products()
    st.dataframe(products, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: APPROVAL WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✅ Approval Workflow":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Approval Workflow</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Multi-step workflow: Review → Approve → Disburse.
    </p>
    """, unsafe_allow_html=True)
    apps = load_applications()
    bank_name = st.session_state.get("bank", "")
    if bank_name:
        apps = apps[apps["bank"] == bank_name]
    if apps.empty:
        st.info("No applications found.")
    else:
        st.dataframe(apps[["application_id", "client_username", "status", "submitted_at"]], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CREDIT POLICY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Credit Policy":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Credit Policy Rules</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Auto-review thresholds used for internal policy guidance.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='metric-card'>
        <div class='section-header'>Sample Rules</div>
        <ul style='color:#C0CDD8; line-height:2; padding-left:1.2rem; font-size:0.92rem;'>
            <li>Credit score < 500 ⇒ Auto Reject</li>
            <li>DTI > 0.60 ⇒ Manual Review</li>
            <li>Previous defaults ≥ 2 ⇒ High Risk</li>
            <li>Income < ₹20,000/month ⇒ Reject or collateral required</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RATE & EMI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Rate & EMI Calculator":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Rate & EMI Calculator</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Quick EMI and affordability calculations.
    </p>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        principal = st.number_input("Loan Amount (₹)", min_value=1000, value=300000, step=1000)
    with c2:
        rate = st.number_input("Interest Rate (%)", min_value=3.0, value=10.5, step=0.25)
    with c3:
        months = st.number_input("Tenure (months)", min_value=6, value=36, step=1)
    emi = loan_emi(principal, rate, int(months))
    st.markdown(f"""
    <div class='metric-card' style='text-align:center;'>
        <div class='section-header'>Estimated EMI</div>
        <h2 style='font-size:2rem; color:#1E6FDB;'>₹{emi:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Risk Monitoring":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Risk Monitoring</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Portfolio-level KPIs for active applications.
    </p>
    """, unsafe_allow_html=True)
    apps = load_applications()
    if not apps.empty:
        total = len(apps)
        pending = (apps["status"] == "Pending").sum()
        accepted = (apps["status"] == "Accepted").sum()
        rejected = (apps["status"] == "Rejected").sum()
        c1, c2, c3, c4 = st.columns(4)
        for col, (val, label) in zip([c1, c2, c3, c4], [(total, "Total"), (pending, "Pending"), (accepted, "Accepted"), (rejected, "Rejected")]):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center;'>
                    <h2 style='font-size:2rem; color:#1E6FDB;'>{val}</h2>
                    <p>{label} Apps</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No application data yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧾 Audit Log":
    st.markdown("""
    <h2 style='font-size:1.8rem; font-weight:700; margin-bottom:0.2rem;'>Audit Log</h2>
    <p style='color:#8899AA; font-size:0.9rem; margin-bottom:1.5rem;'>
        Record of approvals and rejections.
    </p>
    """, unsafe_allow_html=True)
    audit = load_audit_log()
    if audit.empty:
        st.info("No audit events yet.")
    else:
        st.dataframe(audit, use_container_width=True)


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
