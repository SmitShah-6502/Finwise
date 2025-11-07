import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="🌈 Diversification Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        html, body {
            background: linear-gradient(to right, #ffecd2, #fcb69f);
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.15);
            margin: 2rem auto;
            max-width: 1200px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .badge {
            display: inline-block;
            padding: 0.4em 0.8em;
            font-size: 1em;
            font-weight: bold;
            border-radius: 8px;
            color: white;
        }
        .high-risk { background-color: #e74c3c; }
        .neutral { background-color: #f39c12; }
        .diversified { background-color: #27ae60; }
        .highlight {
            background-color: #ffeaa7;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #fdcb6e;
            margin-top: 1rem;
        }
        .section {
            background-color: #ffffff;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main'>", unsafe_allow_html=True)

# ✅ Local Finance Banner Image
st.image("finance2.jpg", use_container_width=True)

# Title
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🔍 Portfolio Diversification Prediction</h1>", unsafe_allow_html=True)

# Load investment data
investment_df = pd.read_excel("Investment.xlsx", sheet_name="1_TOTAL", engine="openpyxl")
investment_df = investment_df[["Name of Script", "SECTOR"]].dropna()
ticker_sector_map = dict(zip(investment_df["Name of Script"].str.upper(), investment_df["SECTOR"]))

# Generate synthetic training data
random.seed(42)
unique_tickers = list(ticker_sector_map.keys())
synthetic_data = []

for _ in range(300):
    sample_tickers = random.sample(unique_tickers, k=random.randint(1, 6))
    sectors = [ticker_sector_map[t] for t in sample_tickers if t in ticker_sector_map]
    unique_sector_count = len(set(sectors))
    if unique_sector_count < 3:
        label = "High Risk"
    elif unique_sector_count == 3:
        label = "Neutral"
    else:
        label = "Diversified Balanced"
    synthetic_data.append((unique_sector_count, label))

train_df = pd.DataFrame(synthetic_data, columns=["UniqueSectorCount", "DiversificationClass"])
target_map = {"High Risk": 0, "Neutral": 1, "Diversified Balanced": 2}
train_df["Target"] = train_df["DiversificationClass"].map(target_map)

# Train ML pipeline
X = train_df[["UniqueSectorCount"]]
y = train_df["Target"]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline.fit(X, y)
class_decoder = {v: k for k, v in target_map.items()}

# Diversification Prediction Section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("## 🔍 Diversification Prediction")
    user_input = st.text_input("🎯 Enter stock tickers separated by commas (e.g., UPL, INFY, RELIANCE):")

    if user_input:
        tickers = [ticker.strip().upper() for ticker in user_input.split(",")]
        matched_sectors = [ticker_sector_map[t] for t in tickers if t in ticker_sector_map]
        unique_sectors = set(matched_sectors)
        unique_sector_count = len(unique_sectors)

        pred_class = pipeline.predict([[unique_sector_count]])[0]
        diversification_class = class_decoder[pred_class]

        badge_class = {
            "High Risk": "high-risk",
            "Neutral": "neutral",
            "Diversified Balanced": "diversified"
        }[diversification_class]

        st.markdown(f"<h3>🧠 Classification: <span class='badge {badge_class}'>{diversification_class}</span></h3>", unsafe_allow_html=True)

        summary = (
            f"Your portfolio includes stocks from **{unique_sector_count} unique sectors**: "
            f"{', '.join(unique_sectors)}.\n\n"
            f"🔍 Based on this, your portfolio is classified as **{diversification_class}**.\n\n"
            f"📈 Diversification across multiple sectors helps reduce risk and improve stability."
        )
        st.markdown("### 📋 AI Summary")
        st.markdown(f"<div class='highlight'>{summary}</div>", unsafe_allow_html=True)

        st.markdown("### 📊 Sector Breakdown")
        sector_df = pd.DataFrame({"Ticker": tickers, "Sector": matched_sectors})
        st.dataframe(sector_df)
    else:
        st.warning("🚨 Please enter stock tickers to get diversification prediction.")
    st.markdown("</div>", unsafe_allow_html=True)

# End main content container
st.markdown("</div>", unsafe_allow_html=True)
