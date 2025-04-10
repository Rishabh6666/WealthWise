import yfinance as yf
import ta
import streamlit as st
import google.generativeai as genai
import plotly.graph_objects as go
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

# 🔹 API Keys (Use Environment Variables!)
GEMINI_AI_KEY = "AIzaSyC3bYYhZSfz6ODuW3oNabL93G2xE66i8w0"
NEWS_API_KEY = "5ccb9046ec144fe8b8a5d825b5a02e94"

if not GEMINI_AI_KEY or not NEWS_API_KEY:
    st.error("API keys not found. Please set GEMINI_AI_KEY and NEWS_API_KEY as environment variables.")
    st.stop()

# 🔹 Configure AI
genai.configure(api_key=GEMINI_AI_KEY)

# 🔹 Popular Stocks Dropdown
POPULAR_STOCKS = {
    "India (NSE/BSE)": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
        "HDFCBANK.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "WIPRO.NS", "LT.NS",
        "ITC.NS", "MARUTI.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "GRASIM.NS", "SUNPHARMA.NS"
    ],
    "US (NASDAQ/NYSE)": [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
        "NVDA", "META", "BRK.B", "JPM", "NFLX",
        "AMD", "INTC", "IBM", "ORCL", "CSCO",
        "XOM", "V", "DIS", "PEP", "PG"
    ],
    "Global (Other)": [
        "BTC-USD", "ETH-USD", "EURUSD=X",
        "GOLD", "SILVER", "USDINR=X",
        "JPY=X", "GBPUSD=X", "CADUSD=X",
        "WTI", "BRENT", "COPPER"
    ],
    "Indexes": [
        "^NSEI", "^BSESN", "^GSPC", "^IXIC", "^DJI",
        "^FTSE", "^GDAXI", "^N225", "^STOXX50E",
        "^GSPTSE", "^MERV", "^JKSE"
    ]
}

# ✅ Fetch stock price with currency formatting
@st.cache_data(ttl=300)
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1d")
        if df.empty:
            return "⚠ No Data"
        price = round(df["Close"].iloc[-1], 2)
        return f"₹{price}" if symbol.endswith(".NS") or symbol.endswith(".BO") or symbol.startswith("^NSEI") or symbol.startswith("^BSESN") else f"${price}"
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return "Error"

# ✅ Fetch historical data
@st.cache_data(ttl=600)
def get_historical_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period).copy()
        if not df.empty:
            df["EMA_9"] = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
            df["EMA_15"] = ta.trend.EMAIndicator(df["Close"], window=15).ema_indicator()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["MACD"] = ta.trend.MACD(df["Close"]).macd()
            df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
            df["BB_UP"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
            df["BB_LOW"] = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
            df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
            df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

# ✅ Enhanced AI Sentiment Analysis
@st.cache_data(ttl=300)
def get_sentiment_analysis(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url).json()
        if "articles" in response and response["articles"]:
            news_articles = response["articles"][:5]
            analyzer = SentimentIntensityAnalyzer()
            sentiments = []
            for article in news_articles:
                vs = analyzer.polarity_scores(article["title"])
                compound = vs['compound']
                if compound >= 0.05:
                    sentiments.append("Positive")
                elif compound <= -0.05:
                    sentiments.append("Negative")
                else:
                    sentiments.append("Neutral")
            return ", ".join(sentiments)
        else:
            return "No news data found."
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return "Error fetching news."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."

def calculate_stop_loss(current_price, atr, multiplier=2):
    """Calculates a stop-loss based on ATR."""
    return current_price - (atr * multiplier)

# ✅ AI-Based Buy/Sell Recommendation
def get_ai_insights(stock, trend, rsi, macd, adx, bb_up, bb_low, obv, atr, current_price):
    stop_loss = calculate_stop_loss(current_price, atr)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = f"""
        📊 Stock Analysis for {stock}
        🔹 Market Trend: {trend}
        🔹 RSI: {rsi}
        🔹 MACD: {macd}
        🔹 ADX: {adx}
        🔹 Bollinger Bands (Upper): {bb_up}
        🔹 Bollinger Bands (Lower): {bb_low}
        🔹 OBV: {obv}
        🔹 ATR: {atr}
        🔹 Sentiment Analysis: {get_sentiment_analysis(stock)}
        🔹 Stop-Loss: {stop_loss}
        💡 Investment Recommendation: Should the user Buy, Sell, or Hold? Explain why, and explain the stop-loss.
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response else "⚠ AI Error"
    except Exception as e:
        st.error(f"AI error: {e}")
        return "AI error."
# 📊 Enhanced Trend Calculation:
def get_accurate_trend(df):
    if df is None or len(df) < 20:
        return "Insufficient Data"

    ema_short = df["EMA_9"].iloc[-1]
    ema_long = df["EMA_15"].iloc[-1]
    ema_trend = "Uptrend" if ema_short > ema_long else "Downtrend"

    adx = df["ADX"].iloc[-1]
    adx_strength = "Strong" if adx > 25 else "Weak"

    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-10]
    price_action = "Uptrend" if last_close > prev_close else "Downtrend"

    macd = df["MACD"].iloc[-1]
    macd_signal = "Uptrend" if macd > 0 else "Downtrend"

    uptrend_count = 0
    downtrend_count = 0

    if ema_trend == "Uptrend":
        uptrend_count += 1
    else:
        downtrend_count += 1

    if price_action == "Uptrend":
        uptrend_count += 1
    else:
        downtrend_count += 1

    if macd_signal == "Uptrend":
        uptrend_count += 1
    else:
        downtrend_count += 1

    if adx_strength == "Strong" and adx > 25:
        if ema_trend == "Uptrend":
            uptrend_count += 1
        else:
            downtrend_count += 1

    if uptrend_count > downtrend_count:
        final_trend = f"Strong {ema_trend}" if adx_strength == "Strong" else ema_trend
    else:
        final_trend = f"Strong {ema_trend}" if adx_strength == "Strong" else ema_trend

    return final_trend

# 📊 Plot Candlestick Charts and Indicators
def plot_candlestick_chart1(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"], mode="lines", name="9-day EMA", line=dict(color="darkblue", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_15"], mode="lines", name="15-day EMA", line=dict(color="orange", width=2)))
    fig.update_layout(
        title="📊 Stock Price Candlestick Chart with EMAs",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        title_font=dict(size=20, color="black"),
        xaxis_title_font=dict(size=16, color="black"),
        yaxis_title_font=dict(size=16, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

def plot_candlestick_chart2(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], mode="lines", name="Bollinger Upper", line=dict(color="darkgrey", width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], mode="lines", name="Bollinger Lower", line=dict(color="blue", width=2)))
    fig.update_layout(
        title="📊 Stock Price Candlestick Chart with Bollinger Bands",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        title_font=dict(size=24, color="black"),
        xaxis_title_font=dict(size=20, color="black"),
        yaxis_title_font=dict(size=20, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

def plot_rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="orange", width=2)))
    fig.add_hline(y=70, line=dict(color="red", dash="dash"), annotation_text="Overbought (70)")
    fig.add_hline(y=30, line=dict(color="green", dash="dash"), annotation_text="Oversold (30)")
    fig.update_layout(
        title="📊 RSI Indicator",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        height=500,
        title_font=dict(size=24, color="black"),
        xaxis_title_font=dict(size=20, color="black"),
        yaxis_title_font=dict(size=20, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

def plot_macd_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="blue", width=2)))
    fig.update_layout(
        title="📊 MACD Indicator",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="MACD Value",
        height=500,
        title_font=dict(size=24, color="black"),
        xaxis_title_font=dict(size=20, color="black"),
        yaxis_title_font=dict(size=20, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

def plot_obv_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], mode="lines", name="OBV", line=dict(color="purple", width=2)))
    fig.update_layout(
        title="📊 OBV Indicator",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="OBV Value",
        height=500,
        title_font=dict(size=24, color="black"),
        xaxis_title_font=dict(size=20, color="black"),
        yaxis_title_font=dict(size=20, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

def plot_atr_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], mode="lines", name="ATR", line=dict(color="green", width=2)))
    fig.update_layout(
        title="📊 Average True Range (ATR)",
        template="plotly_white",
        hovermode="x",
        xaxis_title="Date",
        yaxis_title="ATR Value",
        height=500,
        title_font=dict(size=24, color="black"),
        xaxis_title_font=dict(size=20, color="black"),
        yaxis_title_font=dict(size=20, color="black"),
        legend_font=dict(size=14, color="black"),
        font=dict(color="black"),
        yaxis=dict(tickfont=dict(color="black", size=14)),
        xaxis=dict(tickfont=dict(color="black", size=14))
    )
    return fig

# ✅ Streamlit UI
st.set_page_config(page_title="WealthWise Chatbot", layout="wide")
st.title("📈 WealthWise Stock Chatbot")

# ✅ Sidebar
st.sidebar.header("Stock Selection")
market = st.sidebar.selectbox("Select Market", list(POPULAR_STOCKS.keys()))
symbol = st.sidebar.selectbox("Select a Popular Stock", POPULAR_STOCKS[market])
custom_symbol = st.sidebar.text_input("Or Enter Custom Symbol")
if custom_symbol:
    symbol = custom_symbol

period = st.sidebar.selectbox("Select Time Period", ["3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    **Note:** * `.NS`: National Stock Exchange of India (NSE)
    * `.BO`: Bombay Stock Exchange (BSE)
    * `.TO`: Toronto Stock Exchange (TSX)
    * `.L`: London Stock Exchange (LSE)
    * `^...`: Index values (e.g., ^NSEI for Nifty 50)
    * No suffix: Typically U.S. exchanges (NYSE, NASDAQ)
    """)

# 🔹 Tabs: Overview, Technical Analysis
tab1, tab2 = st.tabs(["📊 Overview", "📈 Technical Analysis"])

# ✅ Combined Analyze Button
if st.sidebar.button("Analyze Stock"):
    with st.spinner("Analyzing Stock..."):
        df = get_historical_data(symbol, period)
        if df is not None:
            trend = get_accurate_trend(df)
            rsi, macd, adx = round(df["RSI"].iloc[-1], 2), round(df["MACD"].iloc[-1], 2), round(df["ADX"].iloc[-1], 2)
            bb_up = round(df["BB_UP"].iloc[-1], 2)
            bb_low = round(df["BB_LOW"].iloc[-1], 2)
            obv = round(df["OBV"].iloc[-1], 2)
            atr = round(df["ATR"].iloc[-1], 2)
            current_price = round(df["Close"].iloc[-1], 2)

            # 📊 Overview Tab
            with tab1:
                st.markdown("### 📊 Stock Overview")
                col1, col2, col3 = st.columns(3)
                col1.metric("💰 Current Price", f"{get_stock_price(symbol)}", delta_color="off")
                col2.metric("📊 ADX", f"{adx}")
                col3.metric("📉 Market Trend", f"{trend}", delta_color="off")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 RSI", f"{rsi}")
                col2.metric("📈 MACD", f"{macd}")
                col3.metric("📈 OBV", f"{obv}")
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Bollinger Upper", f"{bb_up}")
                col2.metric("📉 Bollinger Lower", f"{bb_low}")
                col3.metric("📈 ATR", f"{atr}")

                ai_response = get_ai_insights(symbol, trend, rsi, macd, adx, bb_up, bb_low, obv, atr, current_price)
                st.markdown("### 🤖 AI Insights")
                st.write(ai_response)

            with tab2:
                st.plotly_chart(plot_candlestick_chart1(df))
                st.plotly_chart(plot_candlestick_chart2(df))
                st.plotly_chart(plot_rsi_chart(df))
                st.plotly_chart(plot_macd_chart(df))
                st.plotly_chart(plot_obv_chart(df))
                st.plotly_chart(plot_atr_chart(df))
        else:
            st.error("Could not get stock data")