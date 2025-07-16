import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pytz
from datetime import datetime
import xgboost as xgb
import os

# For sentiment
from cryptopanic_API_Wrapper.api_gather import CryptoPanicAPI
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For on-chain metrics
from glassnode import GlassnodeClient

import yfinance as yf

st.set_page_config(page_title="Crypto Trend Predictor", layout="centered")

# 1. Data sources – Cached for performance
@st.cache_data
def fetch_price(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": 180, "interval": "daily"}
    df = pd.DataFrame(requests.get(url, params=params).json()['prices'], columns=["ts", "price"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("timestamp")[["price"]]

@st.cache_data
def fetch_sentiment(symbol):
    cp = CryptoPanicAPI(os.getenv("CRYPTOPANIC_API_KEY"))
    df = cp.get_dataframe(symbol=symbol, limit=20)
    sid = SentimentIntensityAnalyzer()
    return df['title'].map(lambda x: sid.polarity_scores(x)['compound']).mean()

@st.cache_data
def fetch_onchain(asset):
    client = GlassnodeClient(api_key=os.getenv("GLASSNODE_API_KEY"))
    series = client.get("entities", "active_count", {"a": asset, "i": "24h"})
    return series.pct_change().iloc[-1]

@st.cache_data
def fetch_dxy_mom():
    df = yf.download("DX-Y.NYB", period="60d", interval="1d")['Close']
    return df.pct_change().rolling(5).mean().iloc[-1]

# 2. Feature engineering
def add_features(df, symbol):
    df['SMA_10'] = df['price'].rolling(10).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['price']).rsi()
    df['MACD'] = ta.trend.MACD(df['price']).macd_diff()
    bb = ta.volatility.BollingerBands(df['price'])
    df['BB_upper'], df['BB_lower'] = bb.bollinger_hband(), bb.bollinger_lband()
    df['EMA_10'] = ta.trend.EMAIndicator(df['price'], 10).ema_indicator()
    df['momentum'] = ta.momentum.ROCIndicator(df['price']).roc()
    df['rolling_vol'] = df['price'].pct_change().rolling(5).std()
    df['news_sentiment'] = fetch_sentiment(symbol)
    df['onchain_pct'] = fetch_onchain(symbol[:3].upper())
    df['dxy_mom'] = fetch_dxy_mom()
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    return df.dropna()

# 3. Modeling pipeline
def train_model(df):
    feature_cols = ["SMA_10","RSI","MACD","BB_upper","BB_lower","EMA_10","momentum","rolling_vol",
                    "news_sentiment","onchain_pct","dxy_mom"]
    X = df[feature_cols]; y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test); probs = model.predict_proba(X_test)[:,1]
    df_test = df.loc[X_test.index].copy()
    df_test['pred_shift'] = preds.shift(1)
    df_test['ret'] = df_test['price'].pct_change().shift(-1)
    df_test['strat_ret'] = df_test['ret'] * (df_test['pred_shift']==1)
    df_test['strat_cum'] = (1+df_test['strat_ret']).cumprod()
    df_test['hold_cum'] = (1+df_test['ret']).cumprod()
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, preds, output_dict=True)
    importance = model.feature_importances_
    next_pred = "UP" if preds[-1] == 1 else "DOWN"
    confidence = probs[-1]
    return df_test, next_pred, confidence, acc, cm, fpr, tpr, roc_auc, report, importance, feature_cols

# 4. Plotting
def plot_returns(df): ...
# (same as before)
def plot_feature_importance(importance, names):
    fig, ax = plt.subplots(figsize=(8,5))
    pd.Series(importance, index=names).sort_values().plot(kind="barh", ax=ax, color='teal')
    return fig

# 5. Streamlit UI
st.title("Crypto Trend Predictor 🔥")
coins = {"Bitcoin":"bitcoin","Ethereum":"ethereum","Dogecoin":"dogecoin","XRP":"ripple"}
sel = st.selectbox("Crypto", list(coins.keys()))
if st.button("Run"):
    df = fetch_price(coins[sel])
    df = add_features(df, coins[sel])
    df_test, pred, conf, acc, cm, fpr, tpr, roc_auc, report, imp, names = train_model(df)
    st.success(f"Next 24h: **{pred}** (confidence: {conf:.0%})")
    st.markdown(f"Accuracy: **{acc:.2%}**")
    st.pyplot(plot_returns(df_test))
    st.pyplot(plot_feature_importance(imp, names))
    # confusion, ROC, report display here...
    # show macro metrics used:
    st.write(f"Sentiment: {df['news_sentiment'].iat[-1]:.2f}, On-chain pct: {df['onchain_pct']:.2f}, USD momentum: {df['dxy_mom']:.4f}")
    tz = pytz.timezone("US/Eastern")
    st.caption(f"Run at {datetime.now(tz).strftime('%Y-%m-%d %I:%M %p %Z')}")
