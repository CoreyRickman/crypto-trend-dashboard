import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import seaborn as sns
import pytz
from datetime import datetime

st.set_page_config(page_title="Crypto Predictor", layout="centered")

# --- Functions ---
@st.cache_data
def fetch_data(coin):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": 180, "interval": "daily"}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data['prices'], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def add_features(df):
    df["SMA_10"] = df["price"].rolling(10).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["price"]).rsi()
    df["MACD"] = ta.trend.MACD(close=df["price"]).macd_diff()
    bb = ta.volatility.BollingerBands(close=df["price"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["EMA_10"] = ta.trend.EMAIndicator(close=df["price"], window=10).ema_indicator()
    df["momentum"] = ta.momentum.ROCIndicator(close=df["price"]).roc()
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
    return df.dropna()

def train_model(df):
    features = ["SMA_10", "RSI", "MACD", "BB_upper", "BB_lower", "EMA_10", "momentum"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    df_test = df.loc[X_test.index].copy()
    df_test["pred"] = preds
    df_test["pred_shift"] = df_test["pred"].shift(1)
    df_test["ret"] = df_test["price"].pct_change().shift(-1)
    df_test["strat_ret"] = df_test["ret"] * (df_test["pred_shift"] == 1)
    df_test["strat_cum"] = (1 + df_test["strat_ret"]).cumprod()
    df_test["hold_cum"] = (1 + df_test["ret"]).cumprod()

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, preds, output_dict=True)
    next_pred = "UP" if model.predict(X_test.iloc[[-1]])[0] == 1 else "DOWN"

    return df_test, next_pred, acc, cm, fpr, tpr, roc_auc, report

def plot_returns(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["strat_cum"], label="Strategy")
    ax.plot(df.index, df["hold_cum"], label="Buy & Hold", linestyle="--")
    ax.set_title("Cumulative Returns")
    ax.legend()
    ax.grid(True)
    return fig

# --- UI ---
st.title("📊 Crypto Trend Predictor")
st.markdown("Predicts next 24h trend using technical indicators.")

coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Dogecoin": "dogecoin",
    "XRP": "ripple"
}

coin_name = st.selectbox("Choose Crypto", list(coins.keys()))
run = st.button("Run Prediction")

if run:
    with st.spinner("Loading and processing..."):
        df = fetch_data(coins[coin_name])
        df = add_features(df)
        df_test, prediction, acc, cm, fpr, tpr, roc_auc, report = train_model(df)

        st.success(f"📈 Prediction for next 24h: **{prediction}**")
        st.pyplot(plot_returns(df_test))

        st.markdown(f"### Model Accuracy: `{acc:.2%}`")

        st.markdown("#### Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.markdown("#### ROC Curve")
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig_roc)

        st.markdown("#### Classification Report Summary")
        st.json(report)

    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz).strftime("%Y-%m-%d %I:%M %p %Z")
    st.caption(f"Prediction run at {now}")
