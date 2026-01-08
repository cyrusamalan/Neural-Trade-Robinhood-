from flask import Flask, render_template, jsonify, request
import bot_engine
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import robin_stocks.robinhood as r
from robin_stocks.robinhood import helper
import threading
import time
from datetime import datetime
import numpy as np
import joblib
import day_engine

# --- IMPORT DATABASE MODELS ---
from models import db, Trade, ModelDecision

app = Flask(__name__)

# --- DATABASE CONFIGURATION ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres@localhost:5432/neuraltraderobinhood'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

# --- CONFIGURATION ---
DATA_DIR = "datasets"  
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
JOINT_ACCT_NUM = "116729674659"

# --- GLOBAL TRADING STATE ---
live_state = {
    "active": False,
    "status": "Idle",
    "logs": [],
    "last_update": None,
    "in_trade": False,
    "entry_price": 0.0,
    "ticker": "INTC",
    "trade_amount": 10.00,
    "interval": 60,
    "mode": "swing",
    "paper_mode": True  # <--- NEW: Default to Paper Trading
}

def login_robinhood():
    if live_state["paper_mode"]: return True # Skip login for Paper Mode
    try:
        if not os.path.exists("token.txt"): return False
        with open("token.txt", "r") as f: token = f.read().strip()
        helper.update_session("Authorization", token)
        helper.update_session("X-Robinhood-Account-Id", JOINT_ACCT_NUM)
        helper.set_login_state(True)
        return True
    except: return False

def background_trader():
    print("🚀 Background Trader Started")
    
    # Login Check
    if not login_robinhood():
        live_state["active"] = False
        live_state["logs"].append(f"❌ Login Failed (Real Mode). Stopping.")
        return

    # --- 1. DETERMINE MODE & FILES ---
    current_mode = live_state.get("mode", "swing")
    brain_file = "day_brain.pkl" if current_mode == 'day' else "brain.pkl"
    data_interval = "5m" if current_mode == 'day' else "1h"
    
    # --- 2. LOAD BRAIN ---
    model = None
    features = []
    
    if os.path.exists(brain_file):
        try:
            brain_pkg = joblib.load(brain_file)
            if brain_pkg.get('ticker') == live_state["ticker"]:
                model = brain_pkg['model']
                features = brain_pkg['features']
                live_state["logs"].append(f"🧠 Brain Loaded: {current_mode.upper()} mode")
            else:
                live_state["logs"].append(f"⚠️ Brain Mismatch: Expected {live_state['ticker']}")
        except Exception as e:
            live_state["logs"].append(f"❌ Corrupt Brain: {e}")
    else:
        live_state["logs"].append(f"⚠️ Brain missing. Run Simulation first.")

    # --- 3. CHECK HOLDINGS (Real Mode Only) ---
    if not live_state["paper_mode"]:
        try:
            positions = r.get_open_stock_positions(account_number=JOINT_ACCT_NUM)
            for pos in positions:
                ins = r.get_instrument_by_url(pos['instrument'])
                if ins['symbol'] == live_state["ticker"]:
                    live_state["in_trade"] = True
                    live_state["entry_price"] = float(pos['average_buy_price'])
                    live_state["logs"].append(f"ℹ️ Found Real Position: {live_state['ticker']}")
        except: pass

    # --- 4. TRADING LOOP ---
    while live_state["active"]:
        try:
            symbol = live_state["ticker"]
            live_state["status"] = f"Scanning ({current_mode})..."
            
            # A. DOWNLOAD DATA
            period = "5d" if current_mode == 'day' else "59d"
            df = yf.Ticker(symbol).history(period=period, interval=data_interval)
            
            if df.empty:
                time.sleep(60); continue
            
            # B. CALCULATE FEATURES
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            if current_mode == 'day':
                df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
                df['Dist_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
                df['Volatility'] = df['Close'].pct_change().rolling(5).std()
                df.ffill(inplace=True); df.bfill(inplace=True)
            else:
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()
                df['SP500_Close'] = df['Close']; df['VIX_Close'] = 20.0
                df = df.fillna(0)

            last_row = df.iloc[-1]
            current_price = last_row['Close']
            
            # C. AI PREDICTION
            ai_approved = False
            prob = 0.0
            
            if model:
                try:
                    feat_values = [last_row.get(f, 0) for f in features]
                    prob = model.predict_proba([feat_values])[0][1]
                    threshold = 0.55 if current_mode == 'day' else 0.51
                    
                    if prob > threshold:
                        ai_approved = True
                        live_state["logs"].append(f"🤖 AI SIGNAL: {prob:.2f}")
                except Exception as e:
                    live_state["logs"].append(f"⚠️ Prediction Error: {e}")
            else:
                if last_row['RSI'] < 30: ai_approved = True

            # D. EXECUTE TRADES
            risk_amount = live_state.get("trade_amount", 10.00)
            prefix = "📝 PAPER" if live_state["paper_mode"] else "🚀 REAL"
            
            if not live_state["in_trade"]:
                # BUY LOGIC
                if ai_approved:
                    if live_state["paper_mode"]:
                        # --- PAPER BUY ---
                        live_state["in_trade"] = True
                        live_state["entry_price"] = current_price
                        live_state["logs"].append(f"{prefix} BUY {symbol} @ ${current_price:.2f} (Conf: {prob:.2f})")
                    else:
                        # --- REAL BUY ---
                        try:
                            r.order_buy_fractional_by_price(symbol, risk_amount, account_number=JOINT_ACCT_NUM)
                            live_state["in_trade"] = True
                            live_state["entry_price"] = current_price
                            live_state["logs"].append(f"{prefix} BUY ORDER SENT: {symbol} @ ${current_price:.2f}")
                        except Exception as e: live_state["logs"].append(f"❌ Buy Failed: {e}")
            
            else:
                # SELL LOGIC
                stop_pct = 0.995 if current_mode == 'day' else 0.95
                profit_pct = 1.01 if current_mode == 'day' else 1.05
                
                # Day Trade Force Close
                is_force_close = False
                if current_mode == 'day':
                    now = datetime.now()
                    if now.hour == 15 and now.minute >= 55: is_force_close = True

                reason = None
                if current_price < live_state["entry_price"] * stop_pct: reason = "Stop Loss"
                elif current_price > live_state["entry_price"] * profit_pct: reason = "Take Profit"
                elif is_force_close: reason = "EOD Force Close"

                if reason:
                    if live_state["paper_mode"]:
                        # --- PAPER SELL ---
                        live_state["in_trade"] = False
                        pnl = (current_price - live_state["entry_price"]) / live_state["entry_price"] * 100
                        live_state["logs"].append(f"{prefix} SELL ({reason}) @ ${current_price:.2f} (PnL: {pnl:.2f}%)")
                    else:
                        # --- REAL SELL ---
                        try:
                            positions = r.get_open_stock_positions(account_number=JOINT_ACCT_NUM)
                            for pos in positions:
                                ins = r.get_instrument_by_url(pos['instrument'])
                                if ins['symbol'] == symbol:
                                    r.order_sell_fractional_by_quantity(symbol, float(pos['quantity']), account_number=JOINT_ACCT_NUM)
                            live_state["in_trade"] = False
                            live_state["logs"].append(f"{prefix} SELL ORDER SENT ({reason})")
                        except Exception as e: live_state["logs"].append(f"❌ Sell Failed: {e}")

            # E. HEARTBEAT UPDATE
            live_state["last_update"] = f"${current_price:.2f} | RSI: {last_row['RSI']:.1f}"
            time.sleep(live_state["interval"])

        except Exception as e:
            live_state["logs"].append(f"⚠️ Loop Error: {e}")
            time.sleep(60)

# --- ROUTES ---
@app.route('/')
def home():
    token_status = "❌ Missing"
    if os.path.exists("token.txt"):
        with open("token.txt", "r") as f:
            if len(f.read().strip()) > 10: token_status = "✅ Active"
    return render_template('index.html', token_status=token_status)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    action = request.json.get('action')
    ticker = request.json.get('ticker', 'INTC').upper()
    mode = request.json.get('mode', 'swing')
    paper = request.json.get('paper', True) # <--- GET PAPER SETTING
    
    if action == 'start':
        if not live_state["active"]:
            live_state["active"] = True
            live_state["ticker"] = ticker
            live_state["mode"] = mode
            live_state["paper_mode"] = paper # <--- SAVE IT
            
            icon = "📝" if paper else "🚀"
            live_state["logs"].append(f"▶️ STARTING {icon} {mode.upper()} BOT on {ticker}")
            
            t = threading.Thread(target=background_trader); t.daemon = True; t.start()
    else:
        live_state["active"] = False; live_state["logs"].append("🛑 STOPPING Bot...")
    
    return jsonify({"success": True, "active": live_state["active"]})

@app.route('/get_live_status', methods=['GET'])
def get_live_status(): return jsonify(live_state)

@app.route('/run_sim', methods=['POST'])
def run_sim():
    try:
        ticker = request.json.get('ticker', 'INTC').upper()
        mode = request.json.get('mode', 'swing')

        if mode == 'day':
            data = day_engine.run_day_simulation(app, ticker)
        else:
            filename = f"{ticker}_3y_enriched_data.csv"
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(file_path): 
                # Auto-download if missing
                dl_res = bot_engine.download_fresh_data(ticker) # Assuming you have this helper available or handle error
                if not os.path.exists(file_path): return jsonify({"error": "Data missing."})
                
            data = bot_engine.get_simulation_data(app, ticker, file_path)
        
        return jsonify(data)
    except Exception as e: return jsonify({"error": str(e)})

# --- MISSING HELPER ROUTES ---

@app.route('/get_balance', methods=['GET'])
def get_balance():
    try:
        if not os.path.exists("token.txt"): return jsonify({"success": False, "error": "Token missing"})
        with open("token.txt", "r") as f: token = f.read().strip()
        
        # Authenticate
        helper.update_session("Authorization", token)
        helper.update_session("X-Robinhood-Account-Id", JOINT_ACCT_NUM)
        helper.set_login_state(True)
        
        # Fetch Profile Data
        data = helper.request_get(f"https://api.robinhood.com/accounts/{JOINT_ACCT_NUM}/", "regular")
        
        if data: 
            return jsonify({"success": True, "cash": float(data.get('buying_power', 0))})
        return jsonify({"success": False, "error": "Fetch failed"})
    except Exception as e: 
        return jsonify({"success": False, "error": str(e)})

@app.route('/save_token', methods=['POST'])
def save_token():
    token = request.json.get('token', '').strip()
    if not token.startswith("Bearer "): return jsonify({"success": False, "error": "Invalid Token"})
    with open("token.txt", "w") as f: f.write(token)
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)