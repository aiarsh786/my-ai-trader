"""
ai_trader.py
MODIFIED FOR GITHUB ACTIONS (VERSION 2)
- Removed 'while True:' loop to run on a schedule.
"""

import os
import time
import requests
from datetime import datetime, timedelta
import logging
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ---------------- env & logging ----------------
load_dotenv()

# Logging: both to console and file
LOGFILE = "ai_trader.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)

def notify_console(msg, level="info"):
    """Write to both console and log file."""
    if level == "info":
        logging.info(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    else:
        logging.debug(msg)

# ---------------- Config (easy to edit) ----------------
STOCK_SYMBOLS = ["AAPL", "MSFT"]
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT"]
FOREX_PAIRS = [("USD", "INR"), ("EUR", "USD")]

# SLEEP_SECONDS is no longer needed
ALERT_CONFIDENCE = 0.85      # threshold for 'strong' alerts

MODEL_FOLDER = "models"
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Notification config from .env (or GitHub Secrets)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")
FOREX_API_KEY = os.getenv("FOREX_API_KEY")

# ---------------- Safety controls ----------------
PER_SYMBOL_COOLDOWN = 60 * 60  # 1 hour by default
MAX_SIGNALS_PER_DAY = 3

last_alert_time = {}  
alerts_today = {}    

def can_alert(symbol):
    now = datetime.utcnow()
    last = last_alert_time.get(symbol)
    if last and (now - last).total_seconds() < PER_SYMBOL_COOLDOWN:
        return False
    today = now.strftime("%Y-%m-%d")
    counts = alerts_today.get(symbol, {})
    if counts.get(today, 0) >= MAX_SIGNALS_PER_DAY:
        return False
    return True

def record_alert(symbol):
    now = datetime.utcnow()
    last_alert_time[symbol] = now
    today = now.strftime("%Y-%m-%d")
    if symbol not in alerts_today:
        alerts_today[symbol] = {}
    alerts_today[symbol][today] = alerts_today[symbol].get(today, 0) + 1

# ---------------- Notification functions ----------------
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        notify_console("Telegram keys not found. Skipping alert.", "warning")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
        notify_console("Telegram alert sent", "info")
        return True
    except Exception as e:
        notify_console(f"Telegram send error: {e}", "warning")
        return False

def send_email(subject, body):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and TO_EMAIL):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = TO_EMAIL
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [TO_EMAIL], msg.as_string())
        server.quit()
        notify_console("Email alert sent", "info")
        return True
    except Exception as e:
        notify_console(f"Email send error: {e}", "warning")
        return False

# ---------------- Technical indicators ----------------
def SMA(series, period):
    return series.rolling(window=period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ---------------- yfinance helpers ----------------
def ensure_df_from_yf(obj):
    if isinstance(obj, pd.Series):
        return pd.DataFrame({"Close": obj.astype(float)})
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            obj.columns = obj.columns.droplevel(0)
        if "Close" not in obj.columns and "Adj Close" in obj.columns:
            obj = obj.rename(columns={"Adj Close": "Close"})
        return obj
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

def get_close_series(df):
    if isinstance(df, pd.Series):
        s = pd.to_numeric(df, errors="coerce")
        s.index = pd.to_datetime(s.index)
        return s
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        close = df["Close"]
    else:
        for col in df.columns:
            if "close" in str(col).lower():
                close = df[col]
                break
        else:
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 1:
                close = numeric.iloc[:, -1]
            else:
                close = df.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    close.index = pd.to_datetime(close.index)
    return close

# ---------------- Data fetchers ----------------
def fetch_stock_history(symbol, period="365d", interval="1d"):
    notify_console(f"Fetching stock history for {symbol} ...")
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    df = ensure_df_from_yf(raw)
    if df is None or (hasattr(df, "empty") and df.empty):
        raise RuntimeError(f"No data for {symbol}")
    if isinstance(df, pd.DataFrame):
        df = df.dropna(how='all')
    else:
        df = df.dropna()
    return df

def fetch_crypto_ticker(exchange, pair):
    try:
        return exchange.fetch_ticker(pair)
    except Exception as e:
        notify_console(f"Crypto fetch error {pair}: {e}", "warning")
        return None

def fetch_forex_rate(base, quote):
    try:
        url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
        if FOREX_API_KEY:
            url += f"&access_key={FOREX_API_KEY}"
        r = requests.get(url, timeout=8)
        j = r.json()
        if isinstance(j, dict) and "rates" in j and j["rates"].get(quote):
            return float(j["rates"][quote])
    except Exception:
        pass
    try:
        r2 = requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=8)
        j2 = r2.json()
        if isinstance(j2, dict) and "rates" in j2 and j2["rates"].get(quote):
            return float(j2["rates"][quote])
    except Exception:
        pass
    return None

# ---------------- Feature builder & ML ----------------
def build_features_from_df(df):
    close = get_close_series(df)
    if close.empty or close.isna().all():
        raise ValueError("No Close data")
    vol = 0
    if isinstance(df, pd.DataFrame):
        if "Volume" in df.columns:
            vol = df["Volume"].reindex(close.index).fillna(0)
        else:
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] >= 1:
                vol = numeric.iloc[:, -1].reindex(close.index).fillna(0)
    data = pd.DataFrame({"Close": close})
    data["Volume"] = vol
    data["sma_10"] = SMA(data["Close"], 10)
    data["sma_50"] = SMA(data["Close"], 50)
    data["ema_12"] = EMA(data["Close"], 12)
    data["rsi_14"] = RSI(data["Close"], 14)
    macd_line, sig_line, hist = MACD(data["Close"])
    data["macd_hist"] = hist
    data = data.dropna()
    return data

def prepare_ml_data(df):
    df = df.copy()
    df["future_close"] = df["Close"].shift(-1)
    df = df.dropna()
    df["target"] = (df["future_close"] > df["Close"]).astype(int)
    features = ["sma_10", "sma_50", "ema_12", "rsi_14", "macd_hist", "Volume"]
    X = df[features]
    y = df["target"]
    return X, y, df

def load_model_if_exists(symbol):
    path = os.path.join(MODEL_FOLDER, f"model_{symbol}.pkl")
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            notify_console(f"Loaded saved model for {symbol}")
            return model
        except Exception:
            notify_console(f"Failed loading model file for {symbol}", "warning")
    return None

def train_and_persist_model(symbol, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    notify_console(f"Model trained for {symbol} — accuracy: {acc:.3f}")
    joblib.dump(model, os.path.join(MODEL_FOLDER, f"model_{symbol}.pkl"))
    return model

def build_or_load_model_for_stock(symbol):
    # Note: GitHub Actions servers are temporary. They won't find a saved model.
    # This will train a new model every time, which is what we want.
    model = load_model_if_exists(symbol)
    if model is not None:
        return model
    hist = fetch_stock_history(symbol, period="365d", interval="1d")
    features = build_features_from_df(hist)
    X, y, _ = prepare_ml_data(features)
    if X.empty:
        raise RuntimeError("No training data")
    model = train_and_persist_model(symbol, X, y)
    return model

def get_latest_features(symbol):
    hist = fetch_stock_history(symbol, period="60d", interval="1d")
    features = build_features_from_df(hist)
    return features.iloc[-1]

def assess_stock(symbol, model):
    last = get_latest_features(symbol)
    feature_cols = ["sma_10", "sma_50", "ema_12", "rsi_14", "macd_hist", "Volume"]
    X_live = last[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(np.nan_to_num(X_live))[0]
    prob_up = float(prob[1])
    technical_support = (last["Close"] > last["sma_50"]) and (last["rsi_14"] < 70)
    if prob_up > ALERT_CONFIDENCE and technical_support:
        signal = "STRONG_BUY"
    elif prob_up > 0.6 and technical_support:
        signal = "BUY"
    elif prob_up < 1 - ALERT_CONFIDENCE and last["rsi_14"] > 30:
        signal = "STRONG_SELL"
    else:
        signal = "HOLD"
    return {
        "symbol": symbol,
        "close": float(last["Close"]),
        "prob_up": prob_up,
        "signal": signal,
        "rsi": float(last["rsi_14"]),
    }

def assess_crypto(exchange, pair):
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["close"] = df["close"].astype(float)
        df["ema20"] = EMA(df["close"], 20)
        last = df.iloc[-1]
        price = float(last["close"])
        ema20 = float(last["ema20"])
        if price > ema20 * 1.01:
            return {"pair": pair, "signal":"BUY", "price": price}
        elif price < ema20 * 0.99:
            return {"pair": pair, "signal":"SELL", "price": price}
        else:
            return {"pair": pair, "signal":"HOLD", "price": price}
    except Exception as e:
        notify_console("Crypto error: " + str(e), "warning")
        return {"pair": pair, "signal":"ERROR"}

# ---------------- Main loop (NO LONGER A LOOP) ----------------
def main_run():
    notify_console("Starting AI trading assistant run...")
    stock_models = {}
    for s in STOCK_SYMBOLS:
        try:
            stock_models[s] = build_or_load_model_for_stock(s)
        except Exception as e:
            notify_console(f"Skipping {s} (model error): {e}", "warning")

    exchange = ccxt.binance({"enableRateLimit": True})
    
    # Stocks
    for s, model in stock_models.items():
        if model is None:
            continue
        try:
            res = assess_stock(s, model)
            if res["signal"] in ("STRONG_BUY", "STRONG_SELL"):
                # We don't need 'can_alert' checks on a 15-min schedule
                msg = f"{res['signal']} {s} @ {res['close']:.2f} (p_up={res['prob_up']:.2f}, rsi={res['rsi']:.1f})"
                notify_console(msg, "info")
                send_telegram(msg)
                send_email(f"ALERT {s} {res['signal']}", msg)
                record_alert(s) # This is good to keep
            else:
                notify_console(f"Signal for {s} is {res['signal']}. No alert.", "info")
        except Exception as e:
            notify_console(f"Error assessing {s}: {e}", "warning")

    # Crypto
    for pair in CRYPTO_SYMBOLS:
        try:
            r = assess_crypto(exchange, pair)
            if r.get("signal") in ("BUY", "SELL"):
                symbol = pair.replace("/", "_")
                msg = f"{r['pair']} {r['signal']} price={r.get('price')}"
                notify_console(msg, "info")
                send_telegram(msg)
                send_email(f"ALERT {pair} {r['signal']}", msg)
                record_alert(symbol)
        except Exception as e:
            notify_console(f"Crypto assess error for {pair}: {e}", "warning")

    # Forex
    for base, quote in FOREX_PAIRS:
        rate = fetch_forex_rate(base, quote)
        if rate:
            notify_console(f"FX {base}/{quote} = {rate:.4f}", "info")

    notify_console("AI trading assistant run finished.")


if __name__ == "__main__":
    main_run()