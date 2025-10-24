import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore") # To hide common warnings

# ---
# THIS IS THE VERSION 3 CHECK.
# ---
st.text("APP VERSION 3 - THE REAL FIX")
# ---

# -----------------------------
# 1️⃣ Helper functions
# -----------------------------

def SMA(series, period): 
    return series.rolling(window=period).mean()

def EMA(series, period): 
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()
        rs = ma_up / (ma_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        raise type(e)(f"Error in RSI function: {e}")

def MACD(series, fast=12, slow=26, signal=9):
    try:
        ema_fast = EMA(series, fast)
        ema_slow = EMA(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = EMA(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    except Exception as e:
        raise type(e)(f"Error in MACD function: {e}")

# ---
# THIS IS THE NEW, FIXED FUNCTION (VERSION 3)
# ---
def ensure_df_from_yf(obj):
    """
    This new function correctly handles messy 'MultiIndex' columns
    that yfinance sends.
    
    It fixes the "Duplicate column names" error by taking the
    FIRST part of the column name (e.g., 'Open') instead of the
    last part (e.g., 'AAPL').
    """
    if isinstance(obj, pd.Series):
        return pd.DataFrame({"Close": obj.astype(float)})
    
    if isinstance(obj, pd.DataFrame):
        # The main fix! Check if columns are a MultiIndex
        if isinstance(obj.columns, pd.MultiIndex):
            # This is the new logic:
            new_cols = []
            for col in obj.columns:
                # col might be ('Open', 'AAPL')
                # We will join them with an underscore, e.g., 'Open_AAPL'
                # Or just take the first part:
                new_cols.append(col[0]) # Takes 'Open' from ('Open', 'AAPL')
            
            obj.columns = new_cols
        
        # Check for duplicates *after* our fix
        if obj.columns.duplicated().any():
            # If there are *still* duplicates, we must remove them
            obj = obj.loc[:, ~obj.columns.duplicated()]

        if "Close" not in obj.columns and "Adj Close" in obj.columns:
            obj = obj.rename(columns={"Adj Close": "Close"})
            
        return obj
    
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()
# ---
# END OF NEW, FIXED FUNCTION
# ---

def get_close_series(df):
    try:
        if isinstance(df, pd.Series):
            s = pd.to_numeric(df, errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s
        if df is None or (hasattr(df, "empty") and df.empty):
            return pd.Series(dtype=float)

        cols_lower_map = {str(c).lower(): str(c) for c in df.columns}
        close_col_name = None

        if "close" in cols_lower_map:
            close_col_name = cols_lower_map["close"]
        elif "adj close" in cols_lower_map:
            close_col_name = cols_lower_map["adj close"]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            price_col = None
            for col in numeric_cols:
                if "volume" not in str(col).lower():
                    price_col = col
                    break
            if price_col is not None:
                close_col_name = price_col
            elif len(numeric_cols) > 0:
                close_col_name = numeric_cols[0]
            else:
                raise ValueError("Could not find any 'Close' or numeric price data")
        
        if close_col_name is None:
             raise ValueError("Failed to find a suitable price column.")
             
        close = df[close_col_name]
        
        if not isinstance(close, (pd.Series, np.ndarray, list, tuple)):
             raise TypeError(f"Column '{close_col_name}' was found, but it is not a Series (it is a {type(close)}). This is unexpected.")
             
        close_numeric = pd.to_numeric(close, errors="coerce")
        close_numeric.index = pd.to_datetime(close.index)
        return close_numeric
    
    except Exception as e:
        raise type(e)(f"Error in get_close_series: {e}")

def fetch_stock_history(symbol, period="60d", interval="1d"):
    try:
        raw = yf.download(symbol, period=period, interval=interval, progress=False)
        df = ensure_df_from_yf(raw) # This now uses the NEW V3 fixed function
        if df is None or (hasattr(df, "empty") and df.empty):
            raise RuntimeError(f"No data for {symbol} (download may have failed or returned empty).")
        if isinstance(df, pd.DataFrame):
            df = df.dropna(how='all')
        else:
            df = df.dropna()
        return df
    except Exception as e:
        raise type(e)(f"Error in fetch_stock_history: {e}")


def build_features_from_df(df):
    try:
        # ---
        # FIXED THE TYPO HERE (was 'get_get_close_series')
        # ---
        close = get_close_series(df) 
        if close.empty or close.isna().all():
            raise ValueError("No valid 'Close' data found to analyze.")
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
    except Exception as e:
        raise type(e)(f"Error in build_features_from_df: {e}")


def load_model(symbol):
    path = f"models/model_{symbol}.pkl"
    try:
        return joblib.load(path)
    except Exception:
        return None

# -----------------------------
# 2️⃣ Streamlit UI (The Website)
# -----------------------------
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📈 AI Trading Dashboard (Beginner Friendly)")

debug_mode = st.checkbox("Show Debug Information")

symbol = st.selectbox("Choose symbol", ["AAPL", "MSFT", "BTC-USD"])

try:
    hist = fetch_stock_history(symbol, period="60d", interval="1d")
    
    if debug_mode:
        st.subheader("Debug: Raw Data (Cleaned)")
        st.write("This is the data after being cleaned by the new V3 function.")
        st.dataframe(hist) # This should finally work
            
    df = build_features_from_df(hist)
    
    if debug_mode:
        st.subheader("Debug: Processed Data with Features")
        st.write("This is the final data after calculating RSI, MACD, etc.")
        st.dataframe(df)

    latest_data = df.iloc[-1]
    latest_price = latest_data["Close"]
    latest_rsi = latest_data["rsi_14"]
    
    col1, col2 = st.columns(2)
    col1.metric(label=f"Latest Price ({symbol})", value=f"{latest_price:,.2f}")
    col2.metric(label="Latest RSI (14-day)", value=f"{latest_rsi:,.1f}")

    st.line_chart(df["Close"])
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (IST)")

except Exception as e:
    st.error(f"⚠️ Error loading data for {symbol}: {e}")
    if debug_mode:
        st.exception(e) 
    st.stop() 


# -----------------------------
# 3️⃣ Model predictions
# -----------------------------
model_symbol = symbol if symbol in ["AAPL", "MSFT"] else None

if model_symbol:
    model = load_model(model_symbol)
    if model is None:
        st.warning(f"⚠️ No trained model found for {symbol}. Run your ai_trader.py first to create one.")
    else:
        last = df.iloc[-1]
        
        feature_cols = ["sma_10", "sma_50", "ema_12", "rsi_14", "macd_hist", "Volume"]
        
        X = last[feature_cols].values.reshape(1, -1)
        proba = model.predict_proba(np.nan_to_num(X))[0]
        prob_up = float(proba[1])

        st.subheader("🤖 AI Model Analysis")
        col1, col2 = st.columns([2, 1])
        
        col1.metric("Predicted ↑ Probability", f"{prob_up:.2f}")

        signal = "HOLD"
        if prob_up > 0.85 and last["Close"] > last["sma_50"] and last["rsi_14"] < 70:
            signal = "🚀 STRONG BUY"
        elif prob_up > 0.6 and last["Close"] > last["sma_50"]:
            signal = "BUY"
        elif prob_up < 0.15:
            signal = "🔻 STRONG SELL"

        col2.markdown(f"### Signal: **{signal}**")

        if signal in ("🚀 STRONG BUY", "🔻 STRONG SELL"):
            row = {
                "time_utc": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "signal": signal,
                "prob_up": prob_up,
                "price": float(last["Close"])
            }
            log_df = pd.DataFrame([row])
            log_path = "signals_history.csv"
            if not os.path.exists(log_path):
                log_df.to_csv(log_path, index=False)
            else:
                log_df.to_csv(log_path, mode="a", index=False, header=False)
            st.success(f"Logged {signal} to signals_history.csv")
else:
    st.info(f"AI Model analysis is not available for {symbol}. (Models are only built for AAPL and MSFT).")

# -----------------------------
# 4️⃣ Signal history
# -----------------------------
if os.path.exists("signals_history.csv"):
    hist_df = pd.read_csv("signals_history.csv")
    st.subheader("🕓 Recent AI Signals")
    st.dataframe(hist_df.sort_values("time_utc", ascending=False).head(20))