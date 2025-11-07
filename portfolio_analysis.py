# import streamlit as st
# from agents.stock_agent import analyze_stock_with_portfolio
# from utils.data_fetcher import fetch_stock_data

# st.set_page_config(page_title="Personalized Portfolio", layout="wide")
# st.title("📌 Personalized Portfolio Recommendation")

# ticker_portfolio = st.text_input("Enter stock ticker for portfolio analysis (e.g., RELIANCE.NS):")
# shares = st.number_input("Number of shares you currently own:", min_value=0, step=1, value=0)
# recession_rate = st.number_input("Current global recession rate (in %):", min_value=0.0, max_value=100.0, value=2.5, step=0.1)

# if ticker_portfolio and shares > 0:
#     df_portfolio = fetch_stock_data(ticker_portfolio)
#     if df_portfolio is not None:
#         portfolio_analysis = analyze_stock_with_portfolio(ticker_portfolio, df_portfolio, shares, recession_rate)
#         st.markdown("### 🧾 Personalized Portfolio Analysis Report")
#         st.write(portfolio_analysis)
# else:
#     st.info("Please enter a valid stock ticker and number of shares > 0.")






import streamlit as st
import pandas as pd

# Custom modules
from agents.stock_agent import analyze_stock_with_portfolio
from utils.data_fetcher import fetch_stock_data

# Page config
st.set_page_config(page_title="🌈 Portfolio Analysis", layout="wide")

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
st.image("finance1.jpg", use_container_width=True)

# Title
st.markdown("<h1 style='text-align: center; font-size: 3rem;'>📌 Personalized Portfolio Analysis</h1>", unsafe_allow_html=True)

# Personalized Portfolio Analysis Section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("## 📌 Personalized Portfolio Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        ticker_portfolio = st.text_input("📈 Enter stock ticker for portfolio analysis (e.g., RELIANCE.NS):")
        shares = st.number_input("📦 Number of shares you currently own:", min_value=0, step=1, value=0)
    with col2:
        recession_rate = st.number_input("🌍 Current global recession rate (in %):", min_value=0.0, max_value=100.0, value=2.5, step=0.1)

    if ticker_portfolio and shares > 0:
        df_portfolio = fetch_stock_data(ticker_portfolio)
        if df_portfolio is not None:
            portfolio_analysis = analyze_stock_with_portfolio(ticker_portfolio, df_portfolio, shares, recession_rate)
            st.markdown("### 🧾 Personalized Portfolio Analysis Report")
            st.success("✅ Here is your personalized analysis based on current market conditions:")
            st.write(portfolio_analysis)
    else:
        st.info("ℹ️ Please enter a valid stock ticker and number of shares > 0.")
    st.markdown("</div>", unsafe_allow_html=True)

# End main content container
st.markdown("</div>", unsafe_allow_html=True)
