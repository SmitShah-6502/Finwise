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

# --- RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Groq Setup ---
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ----------------- RAG PIPELINE -----------------
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
            return f"{response}\n\nPerplexity Score: {perplexity:.2f}"
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_query}],
        temperature=0.4,
        max_tokens=500
    )
    response = completion.choices[0].message.content.strip()
    token_count = len(response.split())
    perplexity = math.exp(-math.log(0.5) * token_count / 500) if token_count > 0 else 0
    return f"{response}\n\nPerplexity Score: {perplexity:.2f}"

# ----------------- STOCK FUNCTIONS -----------------
def normalize_ticker(ticker: str) -> str:
    """Normalize ticker to handle various formats (e.g., RELIANCE.NS, TCS.NS, AAPL)"""
    ticker = ticker.upper().strip()
    if ticker.endswith(('.NS', '.BO', '.L', '.T')):
        return ticker
    elif ticker in ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'HINDUNILVR', 'ITC', 'LT', 'ICICIBANK', 'BHARTIARTL', 'SBIN']:
        return f"{ticker}.NS"  # Default to NSE for Indian stocks
    return ticker  # Return as is for others like AAPL, TSLA

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
    ax.plot(df["ds"], df["Close Price"], label=f"Historical Close ({currency})", color="black")
    ax.plot(forecast["ds"], forecast["yhat"], label=f"Forecast Close ({currency})", color="blue")
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="skyblue",
        alpha=0.3,
        label="Confidence Interval"
    )
    ax.set_title(f"Stock Price Forecast for {ticker} ({currency})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Price ({currency})")
    ax.legend()
    ax.grid(True)
    return fig

def fetch_news(ticker: str, api_key="demo"):
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-09-01&to=2025-09-01&token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()[:5]
        else:
            return []
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return []

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

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="AI Stock Analysis & Forecast", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
       .stApp {
           background: linear-gradient(120deg, #f8f9fa, #e9ecef);
           color: #212529;
       }
       .title {
           font-size: 36px;
           font-weight: 800;
           color: #0d6efd;
           text-align: center;
           padding: 15px;
           border-radius: 12px;
           box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
           margin-bottom: 20px;
       }
       .stButton>button {
           background: linear-gradient(90deg, #0d6efd, #0a58ca);
           color: white;
           font-weight: bold;
           border-radius: 10px;
           padding: 0.6em 1.2em;
           border: none;
           transition: 0.3s;
       }
       .stButton>button:hover {
           background: linear-gradient(90deg, #0a58ca, #003c8f);
           transform: scale(1.05);
       }
       .stMarkdown, .stDataFrame, .stPyplot {
           background: white;
           border-radius: 15px;
           padding: 15px;
           box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
           margin-bottom: 20px;
       }
       footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💹 AI Stock Analysis & Forecasting with RAG</div>", unsafe_allow_html=True)

ticker = st.text_input(" Enter stock ticker (e.g., AAPL, TSLA, RELIANCE.NS, TCS.NS):")
predict_days = st.slider("Days to Predict", 1, 30, 5)

if ticker:
    with st.spinner("Fetching data and generating forecast..."):
        df_stock = fetch_stock_data(ticker)
        if df_stock is not None and not df_stock.empty:
            currency = get_currency(ticker)
            ticker_info = Ticker(ticker).summary_detail
            if ticker in ticker_info and "currency" in ticker_info[ticker]:
                currency = ticker_info[ticker]["currency"]

            st.subheader(f"📊 Stock Data Preview ({currency})")
            st.dataframe(df_stock.tail(predict_days))

            forecast, model = predict_ticker_prophet(df_stock, predict_days)
            if forecast is not None:
                st.subheader(f"📈 Forecast Plot (Close Price in {currency})")
                fig = plot_forecast(df_stock, forecast, ticker, currency)
                st.pyplot(fig)

                st.subheader(f"🧮 Forecasted Prices (Next Days in {currency})")
                forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(predict_days).copy()
                forecast_display.columns = ["Date", f"Predicted Close ({currency})", f"Lower Bound ({currency})", f"Upper Bound ({currency})"]
                st.dataframe(forecast_display)

                st.subheader(f"🧠 AI-Based Stock Analysis ({currency})")
                stock_summary = ai_stock_analysis(df_stock, ticker, currency)
                st.write(stock_summary)

                st.subheader("📚 RAG Knowledge Base Insights")
                rag_answer = rag_query(f"Give me detailed information about {ticker} stock and its financial background.")
                st.write(rag_answer)

                # 🔗 Add Value Research link at the end
                st.markdown("---")
                st.markdown("### 🔗 For more information and detailed stock analysis, visit [Value Research Online](https://www.valueresearchonline.com/)", unsafe_allow_html=True)

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



