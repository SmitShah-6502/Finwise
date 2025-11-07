import yfinance as yf

def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d")
        return df
    except Exception as e:
        print("Error fetching stock data:", e)
        return None

def fetch_news(ticker):
    # Placeholder for real news fetching logic
    return f"Latest news headlines related to {ticker} stock."

