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
from models import db, Trade, ModelDecision, MarketEvent 

app = Flask(__name__)

# --- DATABASE CONFIGURATION (UPDATED) ---
# Format: postgresql://user@localhost:5432/database_name
# (No password needed, just 'postgres@')
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres@localhost:5432/neuraltraderobinhood'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the DB with this app
db.init_app(app)

# Create tables automatically if they don't exist
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
    "interval": 60 
}

def login_robinhood():
    try:
        if not os.path.exists("token.txt"): return False
        with open("token.txt", "r") as f: token = f.read().strip()
        helper.update_session("Authorization", token)
        helper.update_session("X-Robinhood-Account-Id", JOINT_ACCT_NUM)
        helper.set_login_state(True)
        return True
    except: return False

# --- LIVE TRADING BOT ---
def background_trader():
    print("🚀 Background Trader Started")
    if not login_robinhood():
        live_state["active"] = False
        live_state["logs"].append(f"❌ Login Failed. Stopping.")
        return

    # 1. SECURE BRAIN LOAD
    model = None
    features = []
    
    if os.path.exists("brain.pkl"):
        try:
            brain_pkg = joblib.load("brain.pkl")
            brain_ticker = brain_pkg.get('ticker', 'UNKNOWN')
            
            if brain_ticker == live_state["ticker"]:
                model = brain_pkg['model']
                features = brain_pkg['features']
                live_state["logs"].append(f"🧠 Brain Loaded: Trained on {brain_ticker}")
            else:
                live_state["logs"].append(f"⚠️ BRAIN MISMATCH: File is for {brain_ticker}, but you are trading {live_state['ticker']}.")
                live_state["logs"].append("⚠️ Disabling AI. Reverting to 'Dumb Mode' (RSI Only).")
                model = None
        except Exception as e:
            live_state["logs"].append(f"❌ Corrupt Brain File: {e}")
            model = None
    else:
        live_state["logs"].append("⚠️ No brain.pkl found. Running in Dumb Mode.")

    # Check Positions
    try:
        positions = r.get_open_stock_positions(account_number=JOINT_ACCT_NUM)
        for pos in positions:
            ins = r.get_instrument_by_url(pos['instrument'])
            if ins['symbol'] == live_state["ticker"]:
                live_state["in_trade"] = True
                live_state["entry_price"] = float(pos['average_buy_price'])
                live_state["logs"].append(f"ℹ️ Holding {live_state['ticker']} @ ${live_state['entry_price']:.2f}")
    except: pass

    # 2. MAIN LOOP
    while live_state["active"]:
        try:
            symbol = live_state["ticker"]
            live_state["status"] = "Scanning Market..."
            
            # Fetch Data
            df = yf.Ticker(symbol).history(period="59d", interval="1h")
            if df.empty:
                time.sleep(60); continue
            
            df.index = df.index.tz_localize(None)
            
            # Macros
            try:
                spy = yf.Ticker("SPY").history(period="59d", interval="1h")['Close']
                vix = yf.Ticker("^VIX").history(period="59d", interval="1h")['Close']
                tnx = yf.Ticker("^TNX").history(period="59d", interval="1h")['Close']
                df['SP500_Close'] = spy; df['VIX_Close'] = vix; df['10Y_Yield'] = tnx
                df = df.ffill().bfill()
            except:
                df['SP500_Close'] = df['Close']; df['VIX_Close'] = 20.0; df['10Y_Yield'] = 4.0

            # Indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None: df['BBP'] = bb[bb.columns[0]]
            
            df['Rel_Strength_SP500'] = df['Close'] / df['SP500_Close']
            df['Semi_Sector_Close'] = df['Close']
            df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()
            df = df.fillna(0)

            # Snapshot
            last_row = df.iloc[-1]
            current_price = last_row['Close']
            current_rsi = last_row['RSI']
            status_msg = f"{datetime.now().strftime('%H:%M')} | ${current_price:.2f} | RSI: {current_rsi:.1f}"
            live_state["last_update"] = status_msg

            # Trading Logic
            risk_amount = live_state.get("trade_amount", 10.00)
            
            if not live_state["in_trade"]:
                if current_rsi < 45: 
                    ai_approved = False
                    if model:
                        try:
                            feat_df = pd.DataFrame([last_row[features].values], columns=features)
                            prob = model.predict_proba(feat_df)[0][1]
                            if prob > 0.51:
                                ai_approved = True
                                live_state["logs"].append(f"🤖 AI APPROVED: Buy Conf {prob:.2f}")
                            else:
                                live_state["logs"].append(f"🛡️ AI BLOCKED: Buy Conf {prob:.2f}")
                        except: ai_approved = True 
                    else: ai_approved = True 
                    
                    if ai_approved:
                        try:
                            r.order_buy_fractional_by_price(symbol, risk_amount, account_number=JOINT_ACCT_NUM)
                            live_state["in_trade"] = True
                            live_state["entry_price"] = current_price
                            live_state["logs"].append(f"🚀 ORDER SENT: Bought ${risk_amount} of {symbol}")
                        except Exception as e: live_state["logs"].append(f"❌ BUY FAILED: {str(e)}")
            else:
                if current_price < live_state["entry_price"] * 0.95:
                    live_state["logs"].append(f"🛑 STOP LOSS triggered at ${current_price:.2f}")
                    try:
                        positions = r.get_open_stock_positions(account_number=JOINT_ACCT_NUM)
                        for pos in positions:
                            ins = r.get_instrument_by_url(pos['instrument'])
                            if ins['symbol'] == symbol:
                                r.order_sell_fractional_by_quantity(symbol, float(pos['quantity']), account_number=JOINT_ACCT_NUM)
                        live_state["in_trade"] = False
                        live_state["logs"].append(f"📉 SOLD all {symbol}")
                    except Exception as e: live_state["logs"].append(f"❌ SELL FAILED: {str(e)}")
                elif current_rsi > 55:
                    live_state["logs"].append(f"💰 TAKE PROFIT triggered at ${current_price:.2f}")
                    try:
                        positions = r.get_open_stock_positions(account_number=JOINT_ACCT_NUM)
                        for pos in positions:
                            ins = r.get_instrument_by_url(pos['instrument'])
                            if ins['symbol'] == symbol:
                                r.order_sell_fractional_by_quantity(symbol, float(pos['quantity']), account_number=JOINT_ACCT_NUM)
                        live_state["in_trade"] = False
                        live_state["logs"].append(f"🤑 SOLD all {symbol}")
                    except Exception as e: live_state["logs"].append(f"❌ SELL FAILED: {str(e)}")

            if len(live_state["logs"]) > 50: live_state["logs"] = live_state["logs"][-50:]
            
            # --- DYNAMIC SLEEP ---
            sleep_time = live_state.get("interval", 60)
            live_state["status"] = f"Sleeping ({sleep_time}s) (Last: ${current_price:.2f})"
            time.sleep(sleep_time) 
            
        except Exception as e:
            live_state["logs"].append(f"⚠️ Loop Error: {str(e)}")
            time.sleep(60)
    live_state["status"] = "Stopped"

# --- DATA DOWNLOADER ---
def download_fresh_data(symbol):
    print(f"\n📥 Manager: Downloading history for {symbol}...")
    period = "730d"
    try:
        df = yf.Ticker(symbol).history(period=period, interval="1h")
        if df.empty: return {"success": False, "error": f"Yahoo has no data for {symbol}."}
        df.index = df.index.tz_localize(None); df.index.name = 'Date'
        try:
            spy = yf.Ticker("SPY").history(period=period, interval="1h")['Close']
            vix = yf.Ticker("^VIX").history(period=period, interval="1h")['Close']
            tnx = yf.Ticker("^TNX").history(period="59d", interval="1h")['Close']
            df['SP500_Close'] = spy; df['VIX_Close'] = vix; df['10Y_Yield'] = tnx
            df = df.ffill().bfill().fillna(0)
        except: pass
        try:
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
            df.dropna(inplace=True)
        except Exception as e: return {"success": False, "error": f"Indicator Error: {e}"}
        filename = f"{symbol}_3y_enriched_data.csv"
        file_path = os.path.join(DATA_DIR, filename)
        df.to_csv(file_path)
        return {"success": True, "rows": len(df), "period_used": period}
    except Exception as e: return {"success": False, "error": str(e)}

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
    
    try: amount = float(request.json.get('amount', 10.0))
    except: amount = 10.0
    
    try:
        interval_min = float(request.json.get('interval', 1.0))
        interval_sec = int(interval_min * 60)
    except: interval_sec = 60

    if action == 'start':
        if not live_state["active"]:
            live_state["active"] = True
            live_state["ticker"] = ticker
            live_state["trade_amount"] = amount
            live_state["interval"] = interval_sec
            live_state["logs"].append(f"▶️ STARTING AI TRADER on {ticker} (${amount}, Every {interval_min}m)")
            t = threading.Thread(target=background_trader); t.daemon = True; t.start()
    else:
        live_state["active"] = False; live_state["logs"].append("🛑 STOPPING Bot...")
    return jsonify({"success": True, "active": live_state["active"]})

@app.route('/get_live_status', methods=['GET'])
def get_live_status(): return jsonify(live_state)

@app.route('/get_balance', methods=['GET'])
def get_balance():
    try:
        if not os.path.exists("token.txt"): return jsonify({"success": False, "error": "Token missing"})
        with open("token.txt", "r") as f: token = f.read().strip()
        helper.update_session("Authorization", token)
        helper.update_session("X-Robinhood-Account-Id", JOINT_ACCT_NUM)
        helper.set_login_state(True)
        data = helper.request_get(f"https://api.robinhood.com/accounts/{JOINT_ACCT_NUM}/", "regular")
        if data: return jsonify({"success": True, "cash": float(data.get('buying_power', 0))})
        return jsonify({"success": False, "error": "Fetch failed"})
    except Exception as e: return jsonify({"success": False, "error": str(e)})

@app.route('/save_token', methods=['POST'])
def save_token():
    token = request.json.get('token', '').strip()
    if not token.startswith("Bearer "): return jsonify({"success": False, "error": "Invalid Token"})
    with open("token.txt", "w") as f: f.write(token)
    return jsonify({"success": True})

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
    ticker = request.json.get('ticker', 'INTC').upper()
    result = download_fresh_data(ticker)
    return jsonify(result)

@app.route('/run_sim', methods=['POST'])
def run_sim():
    try:
        # Get parameters from the frontend request
        ticker = request.json.get('ticker', 'INTC').upper()
        mode = request.json.get('mode', 'swing')  # Default to 'swing' if not specified

        if mode == 'day':
            # --- MODE A: DAY TRADING (New) ---
            # Call the function in day_engine.py
            print(f"☀️ Running Day Trading Sim for {ticker}...")
            data = day_engine.run_day_simulation(app, ticker)
        
        else:
            # --- MODE B: SWING TRADING (Original) ---
            print(f"🌊 Running Swing Trading Sim for {ticker}...")
            filename = f"{ticker}_3y_enriched_data.csv"
            file_path = os.path.join(DATA_DIR, filename)
            
            if not os.path.exists(file_path): 
                return jsonify({"error": "Data missing. Click Download button first."})
            
            # Call the function in bot_engine.py
            data = bot_engine.get_simulation_data(app, ticker, file_path)
        
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)