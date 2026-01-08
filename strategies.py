import pandas as pd
import pandas_ta as ta

# --- 1. RSI (Looser Rules) ---
def strategy_rsi_reversion(df):
    """Votes BUY if RSI < 40 (was 30), SELL if > 60 (was 70)"""
    last_rsi = df['RSI'].iloc[-1]
    if last_rsi < 30: return 2   # Strong Buy
    if last_rsi < 40: return 1   # Weak Buy
    if last_rsi > 70: return -2  # Strong Sell
    if last_rsi > 60: return -1  # Weak Sell
    return 0

# --- 2. BREAKOUT (Faster) ---
def strategy_breakout(df):
    """Checks 10-candle High/Low instead of 20"""
    current_price = df['Close'].iloc[-1]
    recent_high = df['High'].iloc[-11:-1].max() # Last 10 candles
    recent_low = df['Low'].iloc[-11:-1].min()
    
    if current_price > recent_high: return 1
    if current_price < recent_low: return -1
    return 0

# --- 3. NEW: MACD CROSSOVER (Trend) ---
def strategy_macd(df):
    """Votes based on MACD line crossing Signal line"""
    # Calculate MACD if missing
    if 'MACD_12_26_9' not in df.columns:
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
    
    macd_val = df['MACD_12_26_9'].iloc[-1]
    signal_val = df['MACDs_12_26_9'].iloc[-1]
    
    if macd_val > signal_val: return 1  # Bullish Trend
    if macd_val < signal_val: return -1 # Bearish Trend
    return 0

# --- 4. HEIKIN ASHI (Keep as is) ---
def strategy_heikin_ashi(df):
    ha = ta.ha(df['Open'], df['High'], df['Low'], df['Close'])
    curr_green = ha['HA_close'].iloc[-1] > ha['HA_open'].iloc[-1]
    prev_green = ha['HA_close'].iloc[-2] > ha['HA_open'].iloc[-2]
    if curr_green and not prev_green: return 1
    if not curr_green and prev_green: return -1
    return 0

# --- VOTING BOOTH ---
def get_council_votes(df):
    if 'RSI' not in df.columns: df['RSI'] = ta.rsi(df['Close'], length=14)
    
    return {
        "RSI_Vote": strategy_rsi_reversion(df),
        "Breakout_Vote": strategy_breakout(df),
        "Heikin_Vote": strategy_heikin_ashi(df),
        "MACD_Vote": strategy_macd(df) # Replaces Fib
    }