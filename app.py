import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import yfinance as yf

# -----------------------------
# SET YOUR API KEY HERE
# -----------------------------
ALPHA_VANTAGE_API_KEY = "2Q96IJ3NG4FRN2GX"  # Replace with your key

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")
st.write("Predict stock trends using sentiment and a simple deep learning model.")

# -----------------------------
# Caching: resources vs data
# -----------------------------
@st.cache_resource
def get_hf_pipeline():
    """Return a HuggingFace sentiment pipeline (cached as a resource)."""
    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("sentiment-analysis", device=device)
    except Exception:
        # Fallback to CPU if something goes wrong
        return pipeline("sentiment-analysis", device=-1)


hf_pipeline = get_hf_pipeline()


@st.cache_resource
def get_gnews_client():
    return GNews(language="en", max_results=10)


gnews_client = get_gnews_client()

# -----------------------------
# Fetch News (cached, dynamic company names)
# -----------------------------
@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def fetch_news(symbol: str):
    """
    Fetch news articles for the symbol using GNews.
    Dynamically gets the company name from yfinance.
    Returns a list of dicts with 'title' and 'description' keys.
    """
    try:
        # Get company info from yfinance
        ticker = yf.Ticker(symbol.upper())
        company_name = ticker.info.get("longName") or symbol.upper()
        
        # Fetch news with GNews using company name
        raw = gnews_client.get_news(company_name)
        return raw or []
    except Exception as e:
        st.warning(f"News fetch error: {e}")
        return []

# -----------------------------
# Fetch Stock Data (cached)
# -----------------------------
@st.cache_data(ttl=60 * 10)  # cache for 10 minutes
def fetch_stock_data(symbol: str):
    """
    Fetch daily stock data from Alpha Vantage and return a cleaned DataFrame sorted by date ascending.
    """
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
        # Rename columns to friendly names
        data = data.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        )
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        # Convert to numeric (AlphaVantage returns strings)
        data[["Open", "High", "Low", "Close", "Volume"]] = data[
            ["Open", "High", "Low", "Close", "Volume"]
        ].apply(pd.to_numeric, errors="coerce")
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Failed to fetch stock data for {symbol}: {e}")
        return pd.DataFrame()

# -----------------------------
# Sentiment helpers
# -----------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    """Compute sentiment polarity using TextBlob. Returns float in [-1,1]."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def get_hf_label_value(label: str) -> int:
    """Map HF label to numeric value."""
    if label.upper().startswith("POS"):
        return 1
    if label.upper().startswith("NEG"):
        return -1
    return 0

# -----------------------------
# Simple PyTorch Regression Model (training)
# -----------------------------
def train_model(X: torch.Tensor, y: torch.Tensor, epochs: int = 150, lr: float = 0.01):
    """
    Train a simple linear regressor. X: (N, F), y: (N, 1)
    Returns trained model (nn.Module).
    """
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
    return model

# -----------------------------
# Analyze Stock function
# -----------------------------
def analyze_stock(symbol: str):
    # 1) Get stock data
    df = fetch_stock_data(symbol)
    if df.empty:
        raise RuntimeError("No stock data available.")

    # 2) Get news + sentiment
    news = fetch_news(symbol)
    if not news:
        st.warning("No recent news articles found for this symbol.")

    sentiments = []
    for n in news:
        title = n.get("title") or ""
        desc = n.get("description") or ""
        text = (title + " " + desc).strip()
        if not text:
            continue

        # TextBlob polarity
        tb = get_textblob_sentiment(text)

        # HuggingFace label
        try:
            hf_res = hf_pipeline(text[:512])
            hf_label = hf_res[0]["label"] if hf_res and isinstance(hf_res, list) else "NEUTRAL"
        except Exception:
            hf_label = "NEUTRAL"
        val = get_hf_label_value(hf_label)

        sentiments.append((text, tb, val))

    if len(sentiments) == 0:
        sent_df = pd.DataFrame(columns=["Text", "TextBlob", "HF_Sentiment"])
    else:
        sent_df = pd.DataFrame(sentiments, columns=["Text", "TextBlob", "HF_Sentiment"])

    # Compute average sentiment safely
    if not sent_df.empty:
        avg_sentiment = float(pd.concat([sent_df["TextBlob"], sent_df["HF_Sentiment"]], axis=1).mean().mean())
    else:
        avg_sentiment = 0.0

    # 3) Prepare data for model
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    features = ["Open", "High", "Low", "Volume"]
    for f in features:
        if f not in df.columns:
            raise RuntimeError(f"Missing expected feature column: {f}")

    df["Volume_Scaled"] = df["Volume"] / (df["Volume"].max() + 1e-9)
    X_np = df[["Open", "High", "Low", "Volume_Scaled"]].values.astype(np.float32)
    y_np = df["Close"].values.astype(np.float32).reshape(-1, 1)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # 4) Train model
    model = train_model(X, y)

    # 5) Predict next-day price
    last_row = df.iloc[-1]
    last_feat = np.array(
        [
            last_row["Open"],
            last_row["High"],
            last_row["Low"],
            last_row["Volume"] / (df["Volume"].max() + 1e-9),
        ],
        dtype=np.float32,
    )
    last_tensor = torch.tensor(last_feat.reshape(1, -1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        next_pred = float(model(last_tensor).item())

    last_close = float(df["Close"].iloc[-1])
    suggestion = "📈 Hold"
    if next_pred > last_close and avg_sentiment > 0:
        suggestion = "📈 Buy"
    elif next_pred < last_close and avg_sentiment < 0:
        suggestion = "📉 Sell"
    elif abs(next_pred - last_close) / (last_close + 1e-9) < 0.01:
        suggestion = "🔸 Neutral (small move)"

    return df, sent_df, next_pred, suggestion, avg_sentiment

# -----------------------------
# Streamlit UI
# -----------------------------
st.sidebar.header("Inputs")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    try:
        with st.spinner("Fetching data and running analysis..."):
            df, sent_df, next_pred, suggestion, avg_sentiment = analyze_stock(symbol)

        st.success(f"✅ Analysis Complete for {symbol}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Next-Day Close", f"${next_pred:.2f}")
        col2.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
        col3.metric("Avg Sentiment", f"{avg_sentiment:.3f}")

        st.markdown("### Trading Suggestion")
        st.info(suggestion)

        st.subheader("Recent Stock Prices (Close)")
        st.line_chart(df["Close"])

        st.subheader("Sentiment Summary (recent articles)")
        st.dataframe(sent_df)

        if not sent_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(sent_df["TextBlob"], bins=10, kde=True, ax=ax)
            ax.set_title("TextBlob Polarity Distribution")
            st.pyplot(fig)
        else:
            st.write("No recent news articles found to compute sentiment.")

    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.info("Enter a symbol and click Analyze (left).")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and HuggingFace.")
