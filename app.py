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
import strategies

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
    # Day mode needs less data (5d), Swing needs more for 200 SMA (300d+)
    period = "5d" if current_mode == 'day' else "1y" 
    data_interval = "5m" if current_mode == 'day' else "1h"
    
    # --- 2. LOAD BRAIN ---
    model = None
    features = []
    
    if os.path.exists(brain_file):
        try:
            brain_pkg = joblib.load(brain_file)
            # Verify ticker match
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
            df = yf.Ticker(symbol).history(period=period, interval=data_interval)
            
            if df.empty:
                time.sleep(60); continue
            
            # --- B. FEATURE ENGINEERING (STRICTLY MATCHING ENGINES) ---
            
            # === DAY TRADING MODE (Matches day_engine.py) ===
            if current_mode == 'day':
                # 1. Calculate Base Indicators required by Council
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['Volatility'] = df['Close'].pct_change().rolling(5).std()
                df['Volume_Trend'] = df['Volume'] / df['Volume'].rolling(20).mean()
                
                # 2. Get Council Votes (CRITICAL: This was missing)
                votes = strategies.get_council_votes(df)
                
                # 3. Add Raw Features manually (as done in day_engine.py)
                votes['Raw_RSI'] = df['RSI'].iloc[-1]
                votes['Raw_Volatility'] = df['Volatility'].iloc[-1]
                votes['Raw_Volume'] = df['Volume_Trend'].iloc[-1]
                
                # 4. Prepare Feature Vector
                try:
                    feat_values = [votes.get(f, 0) for f in features]
                except Exception as e:
                    print(f"Feature mapping error: {e}")
                    feat_values = []

                current_price = df['Close'].iloc[-1]
                last_rsi = df['RSI'].iloc[-1]

            # === SWING TRADING MODE (Matches bot_engine.py) ===
            else:
                # 1. Fetch External Market Data (SPY, VIX, TNX)
                try:
                    spy = yf.Ticker("SPY").history(period="5d", interval="1h")['Close']
                    vix = yf.Ticker("^VIX").history(period="5d", interval="1h")['Close']
                    tnx = yf.Ticker("^TNX").history(period="5d", interval="1h")['Close']
                    
                    # Align indices (basic ffill to match stock timestamps)
                    df['SP500_Close'] = spy.reindex(df.index, method='ffill')
                    df['VIX_Close'] = vix.reindex(df.index, method='ffill')
                    df['10Y_Yield'] = tnx.reindex(df.index, method='ffill')
                    
                    df = df.ffill().fillna(0) # Safety fill
                except:
                    # Fallback if external data fails
                    live_state["logs"].append("⚠️ Market Data Fail. Using Fallbacks.")
                    df['SP500_Close'] = df['Close']
                    df['VIX_Close'] = 20.0
                    df['10Y_Yield'] = 4.0

                # 2. Calculate Technical Indicators
                df['RSI'] = ta.rsi(close=df['Close'], length=14)
                df['SMA_50'] = ta.sma(close=df['Close'], length=50)
                df['SMA_200'] = ta.sma(close=df['Close'], length=200)
                df['ATR'] = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)
                
                bb = ta.bbands(close=df['Close'], length=20, std=2)
                if bb is not None: df['BBP'] = bb[bb.columns[0]]
                
                adx = ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=14)
                if adx is not None: df['ADX'] = adx[adx.columns[0]]
                
                df['Rel_Strength_SP500'] = df['Close'] / df['SP500_Close']
                df['Semi_Sector_Close'] = df['Close'] 
                df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()
                
                df = df.fillna(0)
                
                # 3. Prepare Feature Vector
                last_row = df.iloc[-1]
                feat_values = [last_row.get(f, 0) for f in features]
                
                current_price = last_row['Close']
                last_rsi = last_row['RSI']

            # --- C. AI PREDICTION ---
            ai_approved = False
            prob = 0.0
            
            if model and len(feat_values) > 0:
                try:
                    # SKLearn expects a 2D array: [[val1, val2, ...]]
                    prob = model.predict_proba([feat_values])[0][1]
                    
                    # Thresholds match your UI sliders defaults
                    threshold = 0.55 if current_mode == 'day' else 0.51
                    
                    if prob > threshold:
                        ai_approved = True
                        live_state["logs"].append(f"🤖 AI SIGNAL: {prob:.2f}")
                except Exception as e:
                    live_state["logs"].append(f"⚠️ Prediction Error: {e}")
            else:
                # Fallback only if model is broken
                if last_rsi < 30: ai_approved = True

            # --- D. EXECUTE TRADES ---
            risk_amount = live_state.get("trade_amount", 10.00)
            prefix = "📝 PAPER" if live_state["paper_mode"] else "🚀 REAL"
            
            if not live_state["in_trade"]:
                # BUY LOGIC
                if ai_approved:
                    if live_state["paper_mode"]:
                        live_state["in_trade"] = True
                        live_state["entry_price"] = current_price
                        live_state["logs"].append(f"{prefix} BUY {symbol} @ ${current_price:.2f} (Conf: {prob:.2f})")
                    else:
                        try:
                            r.order_buy_fractional_by_price(symbol, risk_amount, account_number=JOINT_ACCT_NUM)
                            live_state["in_trade"] = True
                            live_state["entry_price"] = current_price
                            live_state["logs"].append(f"{prefix} BUY ORDER SENT: {symbol} @ ${current_price:.2f}")
                        except Exception as e: live_state["logs"].append(f"❌ Buy Failed: {e}")
            
            else:
                # --- NEW: DYNAMIC SELL LOGIC (Matches User Settings) ---
                reason = None
                
                # 1. Retrieve Dynamic Risk Settings (Default: 2% SL, 4% TP)
                sl_pct = live_state.get("stop_loss_pct", 0.02)
                tp_pct = live_state.get("take_profit_pct", 0.04)
                
                # 2. Calculate Price Targets
                stop_price = live_state["entry_price"] * (1 - sl_pct)
                target_price = live_state["entry_price"] * (1 + tp_pct)
                
                # 3. Check for Exits
                if current_price <= stop_price: 
                    reason = f"Stop Loss (-{sl_pct*100}%)"
                elif current_price >= target_price: 
                    reason = f"Take Profit (+{tp_pct*100}%)"
                
                # 4. Special Day Trade Force Close (3:55 PM)
                if current_mode == 'day':
                    now = datetime.now()
                    if now.hour == 15 and now.minute >= 55: 
                        reason = "EOD Force Close"

                if reason:
                    if live_state["paper_mode"]:
                        live_state["in_trade"] = False
                        pnl = (current_price - live_state["entry_price"]) / live_state["entry_price"] * 100
                        live_state["logs"].append(f"{prefix} SELL ({reason}) @ ${current_price:.2f} (PnL: {pnl:.2f}%)")
                    else:
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
            live_state["last_update"] = f"${current_price:.2f} | RSI: {last_rsi:.1f}"
            time.sleep(live_state["interval"])

        except Exception as e:
            live_state["logs"].append(f"⚠️ Loop Error: {e}")
            print(f"Loop Error: {e}")
            time.sleep(60)

# --- ROUTES ---
@app.route('/')
def home():
    token_status = "❌ Missing"
    if os.path.exists("token.txt"):
        with open("token.txt", "r") as f:
            if len(f.read().strip()) > 10: token_status = "✅ Active"
    return render_template('index.html', token_status=token_status)

# In app.py

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    action = request.json.get('action')
    ticker = request.json.get('ticker', 'INTC').upper()
    mode = request.json.get('mode', 'swing')
    paper = request.json.get('paper', True)
    
    # Existing: Get amount (Default to $10 if missing)
    try:
        amount = float(request.json.get('amount', 10.0))
    except:
        amount = 10.0

    # NEW: Get Risk Settings (Default to 2% SL and 4% TP if missing)
    try:
        # Convert user input (e.g., 2.0) into decimal (0.02)
        sl_pct = float(request.json.get('stop_loss', 2.0)) / 100.0 
        tp_pct = float(request.json.get('take_profit', 4.0)) / 100.0
    except:
        sl_pct = 0.02
        tp_pct = 0.04
    
    if action == 'start':
        if not live_state["active"]:
            live_state["active"] = True
            live_state["ticker"] = ticker
            live_state["mode"] = mode
            live_state["paper_mode"] = paper
            live_state["trade_amount"] = amount
            
            # NEW: Save Risk Settings to Global State
            live_state["stop_loss_pct"] = sl_pct
            live_state["take_profit_pct"] = tp_pct
            
            icon = "📝" if paper else "🚀"
            # Update log to show risk parameters
            live_state["logs"].append(f"▶️ STARTING {icon} {mode.upper()} BOT on {ticker} (${amount:.2f}/trade)")
            live_state["logs"].append(f"🛡️ RISK: Stop Loss -{sl_pct*100}% | Take Profit +{tp_pct*100}%")
            
            t = threading.Thread(target=background_trader); t.daemon = True; t.start()
    else:
        live_state["active"] = False; live_state["logs"].append("🛑 STOPPING Bot...")
    
    return jsonify({"success": True, "active": live_state["active"]})

@app.route('/get_live_status', methods=['GET'])
def get_live_status(): return jsonify(live_state)

# ... inside run_sim function in app.py ...

@app.route('/run_sim', methods=['POST'])
def run_sim():
    try:
        ticker = request.json.get('ticker', 'INTC').upper()
        mode = request.json.get('mode', 'swing')
        train_days = int(request.json.get('train_days', 365))
        test_days = int(request.json.get('test_days', 10))
        
        # 1. Get Confidence
        raw_conf = int(request.json.get('min_conf', 51))
        min_conf = raw_conf / 100.0

        # 2. NEW: Get Risk Settings (Defaults match your old hardcoded values)
        # We divide by 100 because the UI sends "2.0" for 2%
        stop_loss = float(request.json.get('stop_loss', 2.0)) / 100.0
        take_profit = float(request.json.get('take_profit', 4.0)) / 100.0

        if mode == 'day':
            # Pass new args to day engine
            data = day_engine.run_day_simulation(
                app, ticker, train_days, test_days, min_conf, stop_loss, take_profit
            )
        else:
            filename = f"{ticker}_3y_enriched_data.csv"
            file_path = os.path.join(DATA_DIR, filename)
            
            if not os.path.exists(file_path): 
                dl_res = bot_engine.download_fresh_data(ticker)
                if not os.path.exists(file_path): return jsonify({"error": "Data missing."})
                
            # Pass new args to swing engine
            data = bot_engine.get_simulation_data(
                app, ticker, file_path, train_days, test_days, min_conf, stop_loss, take_profit
            )
        
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