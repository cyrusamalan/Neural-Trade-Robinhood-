import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import time, datetime
from sklearn.ensemble import RandomForestClassifier # <--- Added ML Library
import joblib # <--- Added for saving brain
from models import db, Trade, ModelDecision

# --- SETTINGS ---
INTERVAL = "5m" 
PERIOD = "59d"
FORCE_CLOSE_TIME = time(15, 55)

def compute_rsi(series, period=14):
    return ta.rsi(series, length=period)

def download_intraday_data(ticker):
    print(f"📥 Fetching 60 days of 5-minute data for {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL)
        if df.empty: return None
            
        df.ffill(inplace=True); df.bfill(inplace=True)
        
        # --- FEATURE ENGINEERING ---
        # These are the inputs the "Brain" will learn from
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['RSI'] = compute_rsi(df['Close'], 14)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'] # How far are we from VWAP?
        df['Volatility'] = df['Close'].pct_change().rolling(5).std() # Is the market crazy right now?
        
        # Create Target (Did price go up in the next 3 candles?)
        df['Future_Return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0.001).astype(int) # Target: > 0.1% profit
        
        return df.dropna()
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_day_simulation(app, ticker):
    with app.app_context():
        df = download_intraday_data(ticker)
        if df is None: return {"error": "No data"}

        # Define the Features for the AI
        features = ['RSI', 'Dist_VWAP', 'Volatility', 'Volume']
        
        # --- TRAIN THE AI (Walk-Forward) ---
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        # Split data: Train on first 70%, Test on last 30%
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        
        # Train
        X_train = train_df[features]
        y_train = train_df['Target']
        model.fit(X_train, y_train)
        print("🧠 Day Trading AI Trained!")

        # --- SIMULATION ---
        results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}
        capital = 25000.0
        shares = 0
        in_trade = False
        active_trade = None
        
        try:
            Trade.query.filter_by(symbol=ticker, mode='DAY_BACKTEST').delete()
            db.session.commit()
        except: pass

        print(f"☀️ Starting AI Day Trade Sim for {ticker}...")

        # Run simulation on the TEST data (Unseen data)
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            current_time = row.name.time()
            current_price = float(row['Close'])
            
            # Prediction
            feat_row = row[features].values.reshape(1, -1)
            prob = model.predict_proba(feat_row)[0][1] # Confidence Score (0.0 to 1.0)

            # 1. FORCE CLOSE
            if in_trade and current_time >= FORCE_CLOSE_TIME:
                pnl = (current_price - active_trade['entry_price']) * active_trade['quantity']
                capital += pnl
                in_trade = False
                results['logs'].append({"date": str(row.name), "msg": "Force Close", "type": "loss"})
                continue

            # 2. AI BUY LOGIC
            if not in_trade:
                # Safe Hours + High AI Confidence (> 55%)
                if time(10,0) < current_time < time(15,0):
                    if prob > 0.55: 
                        shares = 25000 / current_price
                        active_trade = {
                            "entry_price": current_price,
                            "quantity": shares,
                            "entry_time": row.name.to_pydatetime()
                        }
                        in_trade = True
                        results['logs'].append({"date": str(row.name), "msg": f"AI BUY (Conf: {prob:.2f})", "type": "buy"})

            # 3. EXIT LOGIC
            elif in_trade:
                # Standard Scalp Exits
                if current_price >= active_trade['entry_price'] * 1.01:
                    pnl = (current_price - active_trade['entry_price']) * active_trade['quantity']
                    capital += pnl
                    in_trade = False
                    results['logs'].append({"date": str(row.name), "msg": "Profit Target", "type": "profit"})
                elif current_price <= active_trade['entry_price'] * 0.995:
                    pnl = (current_price - active_trade['entry_price']) * active_trade['quantity']
                    capital += pnl
                    in_trade = False
                    results['logs'].append({"date": str(row.name), "msg": "Stop Loss", "type": "loss"})
            
            # Record Data
            results['dates'].append(row.name.strftime('%Y-%m-%d %H:%M'))
            results['stock_price'].append(current_price)
            results['bot_balance'].append(capital)

        # --- SAVE THE BRAIN ---
        # We save it as 'day_brain.pkl' so it doesn't overwrite the swing brain
        brain_package = {
            "model": model,
            "features": features,
            "ticker": ticker,
            "type": "DAY_TRADING_5M"
        }
        joblib.dump(brain_package, "day_brain.pkl")
        print("💾 Day Trading Brain Saved to 'day_brain.pkl'")

        return results