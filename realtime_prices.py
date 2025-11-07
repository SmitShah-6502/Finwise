import streamlit as st
import yfinance as yf
from forex_python.converter import CurrencyRates

# Function to get price
def get_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d')
        price = data['Close'].iloc[-1]
        return round(price, 2)
    except Exception:
        return None

# Show market indices and exchange rates
def show_realtime_prices():
    st.subheader("📈 Indian Stock Market Indices")

    indices = {
        "Nifty 50": "^NSEI",
        "Bank Nifty": "^NSEBANK",
        "Sensex (BSE)": "^BSESN",
    }

    cols = st.columns(len(indices))
    for col, (name, symbol) in zip(cols, indices.items()):
        price = get_price(symbol)
        col.metric(label=name, value=f"₹ {price}" if price else "N/A")

    st.subheader("💱 Currency Exchange Rates (INR)")

    try:
        c = CurrencyRates()
        usd_inr = round(c.get_rate('USD', 'INR'), 2)
        eur_inr = round(c.get_rate('EUR', 'INR'), 2)
        gbp_inr = round(c.get_rate('GBP', 'INR'), 2)

        col1, col2, col3 = st.columns(3)
        col1.metric("USD to INR", usd_inr)
        col2.metric("EUR to INR", eur_inr)
        col3.metric("GBP to INR", gbp_inr)
    except Exception as e:
        st.error(f"Currency conversion error: {e}")

# Currency converter UI
def currency_converter():
    st.markdown("---")
    st.header("🔁 Currency Converter")

    c = CurrencyRates()
    currencies = sorted(['USD', 'INR', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'NZD'])

    col1, col2 = st.columns(2)
    with col1:
        from_currency = st.selectbox("From Currency", currencies, index=currencies.index('USD'))
    with col2:
        to_currency = st.selectbox("To Currency", currencies, index=currencies.index('INR'))

    amount = st.number_input("Amount to Convert", min_value=0.0, value=1.0, step=0.01)

    if st.button("Convert"):
        try:
            rate = c.get_rate(from_currency, to_currency)
            converted_amount = round(amount * rate, 4)
            st.success(f"{amount} {from_currency} = {converted_amount} {to_currency}")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
