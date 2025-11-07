from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(api_key="GROQ_API_KEY", model_name=MODEL_NAME)

# Stock analysis chain
analysis_prompt = PromptTemplate(
    input_variables=["ticker", "data"],
    template="""
You are a stock market analyst. Analyze the following stock data for {ticker}:
{data}

Provide:
1. Market Overview
2. Financial Health
3. Risk Disclosure
4. An insightful summary and prediction based on this data.
"""
)
chain = LLMChain(llm=llm, prompt=analysis_prompt)

def analyze_stock(ticker, df):
    return chain.run({"ticker": ticker, "data": df.describe().to_string()})


# Personalized portfolio recommendation chain (Plain text format)
portfolio_prompt = PromptTemplate(
    input_variables=["ticker", "data", "shares", "recession_rate"],
    template="""
You are a financial expert and stock market analyst.

A user owns {shares} shares of the stock {ticker}.
The global recession rate is currently {recession_rate}%.

Analyze the following stock data:
{data}

Provide a detailed plain-text report covering:
1. Market Overview
2. Financial Health
3. Risk Disclosure
4. Buy/Hold/Sell Recommendation with reasoning
5. End-of-Month Stock Price Forecast
6. A final summary with actionable advice for the user
"""
)

chain_portfolio = LLMChain(llm=llm, prompt=portfolio_prompt)

def analyze_stock_with_portfolio(ticker, df, shares, recession_rate):
    description = df.describe().to_string()
    return chain_portfolio.run({
        "ticker": ticker,
        "data": description,
        "shares": shares,
        "recession_rate": recession_rate
    })
