from yahooquery import tickers

# Get all available Yahoo Finance tickers
all_tickers = tickers.get_all_tickers()

print("Total tickers:", len(all_tickers))
print(all_tickers[:50])  # Show first 50 tickers
