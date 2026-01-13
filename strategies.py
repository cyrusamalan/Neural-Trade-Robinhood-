import pandas_ta as ta
import pandas as pd
import numpy as np

def get_council_votes(df):
    """
    Analyzes the dataframe and returns 'votes' (-1, 0, 1) from 4 different strategies.
    Expected by day_engine.py and app.py.
    """
    
    # 1. Initialize Default Votes (Neutral)
    votes = {
        'RSI_Vote': 0,
        'Breakout_Vote': 0,
        'VWAP_Vote': 0,
        'MACD_Vote': 0
    }

    # SAFETY: If data is too short, return neutral votes immediately to prevent crash
    if df is None or len(df) < 20:
        return pd.Series(votes)

    # COPY: Work on a copy to not affect the main loop's dataframe
    df = df.copy()

    # --- STRATEGY 1: RSI (Reversal) ---
    try:
        # If RSI isn't calculated yet, do it here
        if 'RSI' not in df.columns:
            df['RSI'] = ta.rsi(df['Close'], length=14)
        
        current_rsi = df['RSI'].iloc[-1]
        
        if current_rsi < 30: votes['RSI_Vote'] = 1   # Oversold -> Buy
        elif current_rsi > 70: votes['RSI_Vote'] = -1 # Overbought -> Sell
    except: pass

    # --- STRATEGY 2: BREAKOUT (Bollinger Bands) ---
    try:
        # Calculate Bollinger Bands (20, 2)
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None:
            lower_band = bb[bb.columns[0]] # Lower
            upper_band = bb[bb.columns[2]] # Upper
            
            current_price = df['Close'].iloc[-1]
            
            # If price closes above upper band -> Breakout Buy
            if current_price > upper_band.iloc[-1]: 
                votes['Breakout_Vote'] = 1
            # If price closes below lower band -> Breakdown Sell
            elif current_price < lower_band.iloc[-1]: 
                votes['Breakout_Vote'] = -1
    except: pass

    # --- STRATEGY 3: VWAP (Trend) ---
    # Note: day_engine.py maps 'heikin' to 'VWAP_Vote', so we use VWAP logic here.
    try:
        # Standard VWAP Calculation
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        current_price = df['Close'].iloc[-1]
        current_vwap = df['VWAP'].iloc[-1]
        
        # Price above VWAP = Bullish
        if current_price > current_vwap: votes['VWAP_Vote'] = 1
        else: votes['VWAP_Vote'] = -1
    except: pass

    # --- STRATEGY 4: MACD (Momentum) ---
    try:
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            # MACD Line: macd.columns[0], Signal Line: macd.columns[2]
            macd_line = macd[macd.columns[0]]
            signal_line = macd[macd.columns[2]]
            
            # Crossover Check
            if macd_line.iloc[-1] > signal_line.iloc[-1]: 
                votes['MACD_Vote'] = 1 # Bullish Cross
            elif macd_line.iloc[-1] < signal_line.iloc[-1]: 
                votes['MACD_Vote'] = -1 # Bearish Cross
    except: pass

    return pd.Series(votes)