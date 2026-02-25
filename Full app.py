import streamlit as st
import plotly.graph_objs as go
from utils.data import load_stock
from utils.indicators import add_indicators
from utils.lstm_model import train_or_load
from utils.predictor import predict_future
from utils.watchlist import load_watchlist, save_watchlist
import numpy as np

st.set_page_config(layout="wide")
st.title("🚀 AI Deep Learning Stock Predictor")

# ---------------- WATCHLIST ----------------
watchlist = load_watchlist()
new_stock = st.sidebar.text_input("Add to Watchlist")

if st.sidebar.button("Add"):
    if new_stock.upper() not in watchlist:
        watchlist.append(new_stock.upper())
        save_watchlist(watchlist)

st.sidebar.subheader("📌 Watchlist")
for s in watchlist:
    st.sidebar.write(s)

# ---------------- MAIN ----------------
ticker = st.text_input("Stock Ticker", value="AAPL").upper()

timeframe = st.selectbox(
    "Prediction Period",
    {"1 Day":1, "1 Week":7, "30 Days":30, "1 Year":365}
)

if st.button("Run AI Prediction"):

    df = load_stock(ticker)
    df = add_indicators(df)

    model = train_or_load(ticker, df)

    preds = predict_future(model, df, ticker, days=timeframe)

    last_price = df["Close"].iloc[-1]
    future_price = preds[-1]

    signal = "BUY" if future_price > last_price else "SELL"
    confidence = round(abs(future_price-last_price)/last_price*100,2)

    st.subheader(f"Prediction for {ticker}")
    st.metric("Current Price", round(last_price,2))
    st.metric("Predicted Price", round(future_price,2))
    st.metric("Signal", signal)
    st.metric("Confidence %", confidence)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], name="SMA"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], name="EMA"))

    st.plotly_chart(fig, use_container_width=True)

# -------- FRACTIONAL SHARES ----------
st.subheader("💰 Fractional Share Calculator")
investment = st.number_input("Investment Amount ($)", value=1000)
if ticker:
    df = load_stock(ticker)
    if not df.empty:
        price = df["Close"].iloc[-1]
        shares = investment / price
        st.write(f"You can buy **{round(shares,4)} shares** of {ticker}")
