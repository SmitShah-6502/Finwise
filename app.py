import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from groq import Groq
from yahooquery import Ticker
import requests
import logging
import math
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
from gtts import gTTS
from deep_translator import GoogleTranslator
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Assuming the following imports are available from your project structure
from agents.stock_agent import analyze_stock_with_portfolio
from utils.data_fetcher import fetch_stock_data
from agents.finance_chatbot_agent import multilingual_chatbot, voice_input_to_text

# Unified page config
st.set_page_config(page_title="AI Finance Pro", layout="wide")

# Custom CSS for enhanced, attractive styling
st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), 
                        url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-attachment: fixed;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1a2a44;
        }
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #1a2a44;
            text-align: center;
            margin: 30px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 25px;
            margin: 20px auto;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            max-width: 1200px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #3b82f6, #1d4ed8);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1d4ed8, #1e40af);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stTextInput>div>input, .stNumberInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
            border: 2px solid #3b82f6;
            border-radius: 8px;
            padding: 12px;
            background-color: #f9fafb;
            transition: border-color 0.3s ease;
        }
        .stTextInput>div>input:focus, .stNumberInput>div>input:focus, .stTextArea>div>textarea:focus, .stSelectbox>div>div:focus {
            border-color: #1d4ed8;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        .stSelectbox div[role="listbox"] .option {
            color: #1a2a44 !important;
            background-color: #ffffff !important;
        }
        .stDataFrame, .stPyplot {
            border-radius: 12px;
            padding: 20px;
            background: #f9fafb;
            box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        }
        .highlight {
            background: #e0f2fe;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #3b82f6;
            margin: 15px 0;
            font-size: 16px;
            line-height: 1.6;
        }
        .badge {
            display: inline-block;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 6px;
            color: white;
        }
        .high-risk { background-color: #dc2626; }
        .neutral { background-color: #f59e0b; }
        .diversified { background-color: #16a34a; }
        footer { 
            visibility: visible; 
            text-align: center; 
            padding: 20px; 
            font-size: 14px; 
            color: #6b7280;
            background: rgba(255,255,255,0.9);
            border-top: 1px solid #e5e7eb;
        }
        footer::after {
            content: "Powered by xAI";
        }

        /* Tabbed Navigation Styles */
        .stTabs [role="tablist"] {
            display: flex;
            justify-content: center;
            background: linear-gradient(180deg, #1a2a44, #3b82f6);
            padding: 12px;
            border-radius: 12px;
            margin: 20px auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 1200px;
        }
        .stTabs [role="tab"] {
            background: #ffffff;
            color: #1a2a44;
            padding: 12px 24px;
            margin: 0 8px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            border: 2px solid #3b82f6;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .stTabs [role="tab"]:hover {
            background: #e0f2fe;
            transform: translateY(-2px);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: #3b82f6;
            color: white;
            border: 2px solid #1d4ed8;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .tab-icon {
            font-size: 18px;
        }
        .logo-container {
            text-align: center;
            margin: 30px 0;
        }
        .logo-container img {
            width: 120px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .header-quote {
            font-style: italic;
            color: #1a2a44;
            font-size: 18px;
            text-align: center;
            margin: 20px 0;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# Header with logo and quote
st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
try:
    st.image("finance-logo.jpg", width=120)
except FileNotFoundError:
    st.warning("⚠️ Logo image not found. Please ensure 'finance-logo.jpg' is in the correct directory.")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p class='header-quote'>FINWISE: Finance Assistant</p>", unsafe_allow_html=True)


# # Tabbed Navigation with Icons
tab_names = [
    "📈 Stock Analysis & Forecast",
    "📊 Portfolio Analysis",
    "💬 Finance Chatbot",
    "🔍 Diversification Analysis",
    "💰 Budget Recommendation"
]
tabs = st.tabs(tab_names)

# Function for Stock Analysis
def stock_analysis_page():
    # --- Logging ---
    logging.basicConfig(level=logging.INFO)

    # --- Groq Setup ---
    os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
    MODEL_NAME = "llama-3.3-70b-versatile"
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # --- RAG Pipeline ---
    @st.cache_resource
    def load_vector_db():
        """Load PDFs from data folder, embed, and store in FAISS"""
        pdf_folder = "data"
        documents = []
        if os.path.exists(pdf_folder):
            for file in os.listdir(pdf_folder):
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(pdf_folder, file))
                    documents.extend(loader.load())
        if not documents:
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        return vector_db

    vector_db = load_vector_db()

    def rag_query(user_query: str):
        """Retrieve from vector DB and query Groq with perplexity score"""
        if vector_db:
            retriever = vector_db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_query)
            if docs:
                context_text = "\n".join([doc.page_content for doc in docs])
                prompt = f"""
    You are a financial assistant.
    Use the following context if relevant to answer the question.
    If context is not useful, just answer from your knowledge.

    Context:
    {context_text}

    Question: {user_query}
    Answer in detail:
                """
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=500
                )
                response = completion.choices[0].message.content.strip()
                token_count = len(response.split())
                perplexity = math.exp(-math.log(0.5) * token_count / 500) if token_count > 0 else 0
                return f"{response}\n\n**Perplexity Score**: {perplexity:.2f}"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_query}],
            temperature=0.4,
            max_tokens=500
        )
        response = completion.choices[0].message.content.strip()
        token_count = len(response.split())
        perplexity = math.exp(-math.log(0.5) * token_count / 500) if token_count > 0 else 0
        return f"{response}\n\n**Perplexity Score**: {perplexity:.2f}"

    # --- Stock Functions ---
    def normalize_ticker(ticker: str) -> str:
        """Normalize ticker to handle various formats"""
        ticker = ticker.upper().strip()
        if ticker.endswith(('.NS', '.BO', '.L', '.T')):
            return ticker
        elif ticker in ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'HINDUNILVR', 'ITC', 'LT', 'ICICIBANK', 'BHARTIARTL', 'SBIN']:
            return f"{ticker}.NS"
        return ticker

    def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
        try:
            ticker = normalize_ticker(ticker)
            ticker_obj = Ticker(ticker)
            hist = ticker_obj.history(period=period)
            if hist.empty:
                return None
            hist = hist.reset_index()
            if "symbol" in hist.columns:
                hist = hist[hist["symbol"] == ticker]
            df = hist.rename(columns={
                "date": "ds",
                "open": "Open Price",
                "high": "High Price",
                "low": "Low Price",
                "close": "Close Price",
                "volume": "Volume"
            })
            df = df[["ds", "Open Price", "High Price", "Low Price", "Close Price", "Volume"]]
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
            return df
        except Exception as e:
            logging.error(f"Error fetching stock data for {ticker}: {e}")
            return None

    def predict_ticker_prophet(df: pd.DataFrame, predict_days: int = 5):
        try:
            df_prophet = df[["ds", "Close Price"]].rename(columns={"Close Price": "y"})
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=predict_days)
            forecast = model.predict(future)
            return forecast, model
        except Exception as e:
            logging.error(f"Prophet model error: {e}")
            return None, None

    def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame, ticker: str, currency: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["ds"], df["Close Price"], label=f"Historical Close ({currency})", color="#1a2a44")
        ax.plot(forecast["ds"], forecast["yhat"], label=f"Forecast Close ({currency})", color="#3b82f6")
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="#93c5fd",
            alpha=0.3,
            label="Confidence Interval"
        )
        ax.set_title(f"Stock Price Forecast for {ticker} ({currency})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(f"Price ({currency})", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig

    def ai_stock_analysis(df: pd.DataFrame, ticker: str, currency: str):
        try:
            recent_df = df.tail(15).copy()
            recent_df["ds"] = recent_df["ds"].dt.strftime('%Y-%m-%d')
            recent_text = recent_df.to_string(index=False)
            prompt = f"""
    You are a professional financial analyst. Provide a structured analysis of {ticker}.
    All values are in {currency}.
    Use this structure:
    1. 📈 Trend Analysis
    2. 📊 Volatility
    3. 🏛 Support & Resistance
    4. 💡 Short-term Outlook
    5. ⚠ Risks

    Here is the recent stock data (last 15 rows, {currency}):
    {recent_text}

    Write in clear and concise paragraphs.
            """
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=500
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating stock analysis: {e}"

    def get_currency(ticker: str) -> str:
        """Determine currency based on ticker suffix"""
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            return "INR"
        elif ticker.endswith('.L'):
            return "GBP"
        elif ticker.endswith('.T'):
            return "JPY"
        else:
            return "USD"

    # --- Page Content ---
    st.markdown("<div class='main-title'>💹 AI Stock Analysis & Forecasting</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    ticker = st.text_input("📈 Enter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS, TCS.NS):", placeholder="e.g., AAPL")
    predict_days = st.slider("📅 Days to Predict", 1, 30, 5, help="Select number of days for forecasting")

    if ticker:
        with st.spinner("Fetching data and generating insights..."):
            df_stock = fetch_stock_data(ticker)
            if df_stock is not None and not df_stock.empty:
                currency = get_currency(ticker)
                ticker_info = Ticker(ticker).summary_detail
                if ticker in ticker_info and "currency" in ticker_info[ticker]:
                    currency = ticker_info[ticker]["currency"]

                st.markdown(f"### 📊 Stock Data Preview ({currency})")
                st.dataframe(df_stock.tail(predict_days))

                forecast, model = predict_ticker_prophet(df_stock, predict_days)
                if forecast is not None:
                    st.markdown(f"### 📈 Forecast Plot (Close Price in {currency})")
                    fig = plot_forecast(df_stock, forecast, ticker, currency)
                    st.pyplot(fig)

                    st.markdown(f"### 🧮 Forecasted Prices (Next Days in {currency})")
                    forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(predict_days).copy()
                    forecast_display.columns = ["Date", f"Predicted Close ({currency})", f"Lower Bound ({currency})", f"Upper Bound ({currency})"]
                    st.dataframe(forecast_display)

                    st.markdown(f"### 🧠 AI-Based Stock Analysis ({currency})")
                    stock_summary = ai_stock_analysis(df_stock, ticker, currency)
                    st.markdown(f"<div class='highlight'>{stock_summary}</div>", unsafe_allow_html=True)

                    st.markdown("### 📚 Knowledge Base Insights")
                    rag_answer = rag_query(f"Give me detailed information about {ticker} stock and its financial background.")
                    st.markdown(f"<div class='highlight'>{rag_answer}</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("🔗 **Explore More**: Visit [Value Research Online](https://www.valueresearchonline.com/) for detailed stock analysis.", unsafe_allow_html=True)
            else:
                st.error(f"No stock data available for {ticker}. Please check the ticker format or availability.")
                st.info("""
                **Valid Ticker Examples:**
                - US: AAPL, TSLA, MSFT
                - India (NSE): RELIANCE.NS, TCS.NS, INFY.NS
                - India (BSE): RELIANCE.BO, TCS.BO, INFY.BO
                - UK (LSE): BP.L, VOD.L
                - Japan (TSE): 7203.T, 6758.T
                """)
    st.markdown("</div>", unsafe_allow_html=True)

# Function for Portfolio Analysis
def portfolio_analysis_page():
    # --- Page Content ---
    st.markdown("<div class='main-title'>📌 Personalized Portfolio Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    # Placeholder for banner image (finance-themed)
    # st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_container_width=True)

    st.markdown("### 📌 Portfolio Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        ticker_portfolio = st.text_input("📈 Enter stock ticker for portfolio analysis (e.g., RELIANCE.NS):", placeholder="e.g., RELIANCE.NS")
        shares = st.number_input("📦 Number of shares you currently own:", min_value=0, step=1, value=0)
    with col2:
        recession_rate = st.number_input("🌍 Current global recession rate (in %):", min_value=0.0, max_value=100.0, value=2.5, step=0.1)

    if ticker_portfolio and shares > 0:
        df_portfolio = fetch_stock_data(ticker_portfolio)
        if df_portfolio is not None:
            portfolio_analysis = analyze_stock_with_portfolio(ticker_portfolio, df_portfolio, shares, recession_rate)
            st.markdown("### 🧾 Portfolio Analysis Report")
            st.success("✅ Your personalized analysis based on current market conditions:")
            st.markdown(f"<div class='highlight'>{portfolio_analysis}</div>", unsafe_allow_html=True)
        else:
            st.error("🚨 Unable to fetch portfolio data. Please check the ticker.")
    else:
        st.info("ℹ️ Please enter a valid stock ticker and number of shares > 0.")
    st.markdown("</div>", unsafe_allow_html=True)

# Function for Finance Chatbot
def finance_chatbot_page():
    # --- Page Content ---
    st.markdown("<div class='main-title'>💬 AI Finance Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #1a2a44; font-style: italic;'>Your AI-powered financial advisor at your fingertips!</p>", unsafe_allow_html=True)

    # Language dropdown
    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Gujarati": "gu"
    }
    selected_lang = st.selectbox("🌐 Select Language for Response:", list(lang_map.keys()), help="Choose your preferred language")
    lang_code = lang_map[selected_lang]

    # Translation Helper
    def translate_text(text, target_language):
        try:
            return GoogleTranslator(source='auto', target=target_language).translate(text)
        except Exception as e:
            st.warning(f"⚠️ Translation failed: {e}")
            return text

    # Main function
    def get_answer_in_selected_language(question, lang_code):
        if lang_code == "en":
            return multilingual_chatbot(question)
        else:
            question_in_en = translate_text(question, "en")
            answer_in_en = multilingual_chatbot(question_in_en)
            answer_in_target = translate_text(answer_in_en, lang_code)
            return answer_in_target

    # User text input
    user_question = st.text_area("📝 Ask a finance-related question:", height=150, placeholder="e.g., What is a mutual fund?")

    if st.button("💡 Get Answer"):
        if user_question.strip():
            answer = get_answer_in_selected_language(user_question, lang_code)
            st.markdown("**Answer:**")
            st.markdown(f"<div class='highlight'>{answer}</div>", unsafe_allow_html=True)

            # Text to Speech
            try:
                tts = gTTS(text=answer, lang=lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    audio_bytes = open(tmp_audio.name, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")
                os.remove(tmp_audio.name)
            except Exception as e:
                st.warning("⚠️ Text-to-speech failed.")
        else:
            st.warning("⚠️ Please enter a question.")

    # Voice-based Q&A
    if st.button("🎙️ Use Voice Input"):
        voice_text = voice_input_to_text()
        st.write(f"🎤 You said: {voice_text}")
        if voice_text and not voice_text.startswith("Sorry") and not voice_text.startswith("Speech recognition failed"):
            answer = get_answer_in_selected_language(voice_text, lang_code)
            st.markdown("**Answer:**")
            st.markdown(f"<div class='highlight'>{answer}</div>", unsafe_allow_html=True)

            try:
                tts = gTTS(text=answer, lang=lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    audio_bytes = open(tmp_audio.name, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")
                os.remove(tmp_audio.name)
            except Exception as e:
                st.warning("⚠️ Text-to-speech failed.")
        else:
            st.warning(voice_text)
    st.markdown("</div>", unsafe_allow_html=True)

# Function for Diversification Analysis
def diversification_analysis_page():
    # --- Page Content ---
    st.markdown("<div class='main-title'>🔍 Portfolio Diversification Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    # Placeholder for banner image
    # st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_container_width=True)

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

    # Diversification Prediction
    st.markdown("### 🔍 Diversification Prediction")
    user_input = st.text_input("🎯 Enter stock tickers separated by commas (e.g., UPL, INFY, RELIANCE):", placeholder="e.g., UPL, INFY, RELIANCE")

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

# Function for Budget Recommendation
def budget_recommendation_page():
    # --- Page Content ---
    st.markdown("""
        <style>
            @keyframes rain {
                0% {transform: translateY(-10%);opacity: 1;}
                100% {transform: translateY(110vh);opacity: 0;}
            }
            .emoji-rain {
                position: fixed;
                top: 0;
                left: 0;
                pointer-events: none;
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                z-index: 9999;
            }
            .emoji-rain span {
                position: absolute;
                top: -2em;
                font-size: 20px;
                animation-name: rain;
                animation-timing-function: linear;
                animation-iteration-count: infinite;
                animation-duration: 7s;
                user-select: none;
            }
            .emoji-rain span:nth-child(1) {left: 10%;animation-delay: 0s;}
            .emoji-rain span:nth-child(2) {left: 25%;animation-delay: 1.5s;animation-duration: 6s;}
            .emoji-rain span:nth-child(3) {left: 40%;animation-delay: 3s;animation-duration: 8s;}
            .emoji-rain span:nth-child(4) {left: 55%;animation-delay: 2s;animation-duration: 7s;}
            .emoji-rain span:nth-child(5) {left: 70%;animation-delay: 4s;animation-duration: 6.5s;}
            .emoji-rain span:nth-child(6) {left: 85%;animation-delay: 1s;animation-duration: 7.5s;}
        </style>
       
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>💰 AI Budget Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    monthly_income = st.number_input("💸 Enter your total monthly income (in your currency):", min_value=0.0, format="%.2f", help="Enter your monthly income")
    goals = st.multiselect(
        "🎯 Select your financial goals (choose one or more):",
        options=["Retirement Savings", "Emergency Fund", "Debt Repayment", "Vacation Fund", "Home Purchase", "Education Savings", "Investment Growth", "General Savings"],
        help="Select your financial priorities"
    )

    debt_amount = 0
    if "Debt Repayment" in goals:
        debt_amount = st.number_input("💳 Enter your monthly debt repayment amount (optional):", min_value=0.0, format="%.2f", help="Enter monthly debt repayment amount")

    if st.button("📊 Generate Budget Recommendation"):
        if monthly_income <= 0:
            st.warning("🚨 Please enter a valid monthly income greater than 0.")
        else:
            essentials_pct, savings_pct, discretionary_pct, debt_pct = 50, 30, 20, 0

            if "Debt Repayment" in goals and debt_amount > 0:
                debt_pct = (debt_amount / monthly_income) * 100
                discretionary_pct -= debt_pct
                if discretionary_pct < 5: discretionary_pct = 5
                savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

            if "Retirement Savings" in goals:
                savings_pct += 10; discretionary_pct -= 10
            if "Emergency Fund" in goals:
                savings_pct += 5; discretionary_pct -= 5

            if discretionary_pct < 5:
                discretionary_pct = 5
                savings_pct = 100 - essentials_pct - debt_pct - discretionary_pct

            st.markdown("### 📋 Budget Allocation")
            st.markdown(f"<div class='highlight'>"
                        f"- **Essentials** (housing, food, utilities): **{essentials_pct:.1f}%**<br>"
                        f"- **Savings** (retirement, emergency, investment): **{savings_pct:.1f}%**<br>"
                        f"{f'- **Debt Repayment**: **{debt_pct:.1f}%**<br>' if debt_pct > 0 else ''}"
                        f"- **Discretionary Spending** (leisure, shopping): **{discretionary_pct:.1f}%**"
                        f"</div>", unsafe_allow_html=True)

            st.markdown("### 💵 Monthly Amounts")
            st.markdown(f"<div class='highlight'>"
                        f"- Essentials: {monthly_income * essentials_pct / 100:.2f}<br>"
                        f"- Savings: {monthly_income * savings_pct / 100:.2f}<br>"
                        f"{f'- Debt Repayment: {monthly_income * debt_pct / 100:.2f}<br>' if debt_pct > 0 else ''}"
                        f"- Discretionary: {monthly_income * discretionary_pct / 100:.2f}"
                        f"</div>", unsafe_allow_html=True)

            st.info("ℹ️ This is a basic recommendation. For personalized advice, consider consulting a financial advisor.")
    st.markdown("</div>", unsafe_allow_html=True)

# Render the selected page using tabs
with tabs[0]:
    stock_analysis_page()
with tabs[1]:
    portfolio_analysis_page()
with tabs[2]:
    finance_chatbot_page()
with tabs[3]:
    diversification_analysis_page()
with tabs[4]:
    budget_recommendation_page()


















