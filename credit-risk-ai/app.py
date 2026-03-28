# app.py — HuggingFace Spaces entrypoint
# This file is automatically detected by HuggingFace Spaces (Streamlit SDK).
# It simply re-exports the Streamlit app from the app/ subdirectory.

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Bootstrap: generate data and train if model doesn't exist
model_path = ROOT / "models" / "credit_model.pkl"
if not model_path.exists():
    print("No model found — running bootstrap pipeline …")
    subprocess.run([sys.executable, str(ROOT / "src" / "generate_dataset.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "src" / "train_model.py")], check=True)

# HuggingFace will run: streamlit run app.py
# We import the actual app to make it discoverable
exec(open(ROOT / "app" / "streamlit_app.py").read())
