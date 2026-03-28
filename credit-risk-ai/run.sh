#!/usr/bin/env bash
# run.sh — Full pipeline: generate data → train → launch app

set -e
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AI Credit Risk & Loan Default Platform     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Install dependencies
echo "[1/4] Installing dependencies …"
pip install -r requirements.txt --quiet

# 2. Generate dataset
echo "[2/4] Generating synthetic loan dataset …"
python src/generate_dataset.py

# 3. Train model
echo "[3/4] Training ML models …"
python src/train_model.py

# 4. Launch Streamlit
echo "[4/4] Launching Streamlit dashboard …"
echo ""
echo "  Open → http://localhost:8501"
echo ""
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
