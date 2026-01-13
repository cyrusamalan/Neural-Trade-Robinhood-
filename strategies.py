import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import IsolationForest
from scipy.stats import linregress

# --- 0. HELPER: DETECT MARKET REGIME ---
def get_regime(df):
    """Returns 1 for Uptrend, -1 for Downtrend"""
    if len(df) < 200: return 0
    
    sma200 = ta.sma(df['Close'], length=200)
    if sma200 is None or pd.isna(sma200.iloc[-1]): return 0
    
    price = df['Close'].iloc[-1]
    if price > sma200.iloc[-1]: return 1
    return -1

# --- 1. STATISTICAL ANOMALY (Z-Score) ---
def strategy_ml_anomaly(df):
    """Uses Z-Score to find statistical anomalies instantly."""
    if len(df) < 50: return 0
    
    vol = df['Volume']
    mean_vol = vol.rolling(20).mean()
    std_vol = vol.rolling(20).std()
    
    if std_vol.iloc[-1] == 0: return 0 
    
    z_score_vol = (vol.iloc[-1] - mean_vol.iloc[-1]) / std_vol.iloc[-1]
    
    if z_score_vol > 3:
        if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
            return 2  # Anomalous Pump
        else:
            return -2 # Anomalous Dump
            
    return 0

# --- 2. NEW: MACD HISTOGRAM REVERSAL (Replaces Linear Regression) ---
def strategy_macd_histogram(df):
    """
    Votes BUY when the Histogram curls UP (Momentum Shift), even if trend is down.
    This fixes the conflict with RSI.
    """
    # Calculate MACD (Fast=12, Slow=26, Signal=9)
    if 'MACD_12_26_9' not in df.columns:
        macd = ta.macd(df['Close'])
        if macd is None: return 0
        df = pd.concat([df, macd], axis=1)
    
    # The Histogram is usually the 2nd column in pandas_ta output
    # It represents the distance between MACD line and Signal line
    hist_col = df.columns[-2] 
    hist = df[hist_col]
    
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    prev2 = hist.iloc[-3]
    
    # Bullish Curl: Histogram was moving down, now moving up
    # Logic: It's 'less red' than before, meaning selling is drying up
    if curr > prev and prev > prev2:
        return 2 # Strong Buy Signal (Momentum is turning)
        
    # Bearish Curl: Histogram was moving up, now moving down
    if curr < prev and prev < prev2:
        return -2 # Strong Sell Signal
        
    return 0

# --- 3. ADAPTIVE RSI (High Confidence) ---
def strategy_adaptive_rsi(df):
    """
    1. Checks Trend (Regime).
    2. Calculates Dynamic Percentiles (Bottom 5%).
    3. Only votes Strong Buy (2) if we are in an Uptrend AND oversold.
    """
    if 'RSI' not in df.columns: df['RSI'] = ta.rsi(df['Close'], length=14)
    
    last_rsi = df['RSI'].iloc[-1]
    history = df['RSI'].iloc[-100:].dropna()
    
    if len(history) < 50: return 0
    
    regime = get_regime(df)
    
    low_threshold = np.percentile(history, 5)
    high_threshold = np.percentile(history, 95)
    
    if regime == 1:
        if last_rsi < low_threshold: return 2
    elif regime == -1:
        if last_rsi > high_threshold: return -2
    else:
        if last_rsi < 20: return 2
        
    return 0

# --- 4. VWAP REJECTION ---
def strategy_vwap(df):
    """Buys when price touches VWAP in an uptrend"""
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    price = df['Close'].iloc[-1]
    vwap = df['VWAP'].iloc[-1]
    
    sma = ta.sma(df['Close'], length=50)
    if sma is None: return 0
    
    if price > sma.iloc[-1]: 
        dist = (price - vwap) / vwap
        if abs(dist) < 0.002: return 1 
        
    return 0

# --- VOTING BOOTH ---
def get_council_votes(df):
    if 'RSI' not in df.columns: df['RSI'] = ta.rsi(df['Close'], length=14)
    
    return {
        "RSI_Vote": strategy_adaptive_rsi(df),
        "Breakout_Vote": strategy_ml_anomaly(df),
        "VWAP_Vote": strategy_vwap(df),
        "MACD_Vote": strategy_macd_histogram(df) # <--- UPDATED to use Histogram
    }