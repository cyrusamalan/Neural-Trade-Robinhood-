import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import time, timedelta
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
        if df.empty:
            print("❌ No data found.")
            return None
            
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Calculate VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['RSI'] = compute_rsi(df['Close'], 14)
        
        return df.dropna()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def execute_buy(ticker, price, time_val, rsi_val):
    """ Creates a trade dictionary to track in memory """
    # FIX 1: Ensure time_val is a standard python datetime, not Pandas Timestamp
    if hasattr(time_val, 'to_pydatetime'):
        entry_dt = time_val.to_pydatetime().replace(tzinfo=None)
    else:
        entry_dt = time_val

    return {
        "entry_time": entry_dt,
        "entry_price": float(price),
        "quantity": float(25000 / price),
        "rsi_at_entry": float(rsi_val),
        "status": "OPEN"
    }

def execute_sell(ticker, active_trade, exit_price, exit_time, reason):
    """ Saves the completed trade to the Database """
    try:
        pnl = float((exit_price - active_trade['entry_price']) * active_trade['quantity'])
        pnl_pct = float((exit_price - active_trade['entry_price']) / active_trade['entry_price'])

        # FIX 2: Ensure exit_time is a standard python datetime without timezone
        if hasattr(exit_time, 'to_pydatetime'):
            exit_dt = exit_time.to_pydatetime().replace(tzinfo=None)
        else:
            exit_dt = exit_time

        new_trade = Trade(
            symbol=ticker,
            mode='DAY_BACKTEST',
            entry_time=active_trade['entry_time'], # Already fixed in execute_buy
            entry_price=float(active_trade['entry_price']),
            quantity=float(active_trade['quantity']),
            direction="LONG",
            exit_time=exit_dt, # Uses the fixed time
            exit_price=float(exit_price),
            pnl_dollar=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason
        )
        
        decision = ModelDecision(
            confidence_score=0.99,
            model_version="v1_day_vwap",
            rsi_at_entry=active_trade['rsi_at_entry'],
            decision_type="RULE_BASED"
        )
        
        new_trade.decision = decision
        db.session.add(new_trade)
        db.session.commit()
        return True
    except Exception as e:
        print(f"❌ DB Error: {e}")
        db.session.rollback()
        return False

def run_day_simulation(app, ticker):
    with app.app_context():
        df = download_intraday_data(ticker)
        if df is None: return {"error": "No data"}

        in_trade = False
        active_trade = None
        
        try:
            Trade.query.filter_by(symbol=ticker, mode='DAY_BACKTEST').delete()
            db.session.commit()
        except: pass

        results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}
        capital = 25000.0

        print(f"☀️ Starting Day Trade Sim for {ticker}...")

        for i in range(20, len(df)):
            row = df.iloc[i]
            current_time = row.name.time()
            current_price = row['Close']
            current_rsi = row['RSI']
            current_vwap = row['VWAP']
            
            if in_trade and current_time >= FORCE_CLOSE_TIME:
                execute_sell(ticker, active_trade, current_price, row.name, "EOD Force Close")
                capital += (current_price - active_trade['entry_price']) * active_trade['quantity']
                in_trade = False
                active_trade = None
                results['logs'].append({"date": str(row.name), "msg": "Force Close", "type": "loss"})
                continue

            if not in_trade:
                if time(10,0) < current_time < time(15,0):
                    if current_price > current_vwap and current_rsi < 60:
                        active_trade = execute_buy(ticker, current_price, row.name, current_rsi)
                        in_trade = True
                        results['logs'].append({"date": str(row.name), "msg": f"BUY @ {current_price:.2f}", "type": "buy"})

            elif in_trade:
                if current_price >= active_trade['entry_price'] * 1.015:
                    execute_sell(ticker, active_trade, current_price, row.name, "Scalp Target")
                    capital += (current_price - active_trade['entry_price']) * active_trade['quantity']
                    in_trade = False
                    results['logs'].append({"date": str(row.name), "msg": f"PROFIT @ {current_price:.2f}", "type": "profit"})
                
                elif current_price <= active_trade['entry_price'] * 0.99:
                    execute_sell(ticker, active_trade, current_price, row.name, "Stop Loss")
                    capital += (current_price - active_trade['entry_price']) * active_trade['quantity']
                    in_trade = False
                    results['logs'].append({"date": str(row.name), "msg": f"STOP @ {current_price:.2f}", "type": "loss"})

            results['dates'].append(row.name.strftime('%Y-%m-%d %H:%M'))
            results['stock_price'].append(float(current_price)) 
            results['bot_balance'].append(float(capital))

        return results