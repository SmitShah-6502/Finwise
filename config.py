# config.py
import os
from dotenv import load_dotenv

# Load .env locally (no effect on Streamlit Cloud)
try:
    load_dotenv()
except Exception:
    pass

def get_secret(name: str, default: str = "") -> str:
    """Prefer Streamlit secrets if present, else env/.env. Never raise."""
    try:
        import streamlit as st
        # Accessing st.secrets may raise if no secrets.toml locally:
        try:
            val = st.secrets.get(name, os.getenv(name, default))
        except Exception:
            val = os.getenv(name, default)
    except Exception:
        val = os.getenv(name, default)
    return (val or "").strip()

GROQ_API_KEY = get_secret("GROQ_API_KEY")
# Pick a widely available model; change if you need bigger one
MODEL_NAME = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")
