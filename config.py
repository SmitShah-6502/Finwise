import os
from dotenv import load_dotenv
import streamlit as st

# Load .env file (for local)
load_dotenv()

# Try Streamlit Secrets first (for deployment), else .env / system env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
MODEL_NAME = "llama-3.3-70b-versatile"
