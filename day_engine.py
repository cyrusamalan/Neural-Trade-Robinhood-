import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import time, datetime
from sklearn.ensemble import RandomForestClassifier
import joblib
from models import db, Trade, ModelDecision, StrategyVote
import strategies  # <--- This loads your Council of Strategies

# --- SETTINGS ---
INTERVAL = "5m" 
PERIOD = "59d"
FORCE_CLOSE_TIME = time(15, 55)

# Helper to pretty-print votes
def get_vote_emoji(vote):
    if vote >= 1: return "🟢 BUY"
    if vote <= -1: return "🔴 SELL"
    return "⚪ WAIT"

def download_intraday_data(ticker):
    print(f"📥 Fetching 60 days of 5-minute data for {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL)
        if df.empty: return None
            
        df.ffill(inplace=True); df.bfill(inplace=True)
        
        # --- FIX: Calculate Indicators Globally ---
        # This prevents the "KeyError: RSI"
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Calculate MACD for the new specialist
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1) 
        
        # Helper Features
        df['Volatility'] = df['Close'].pct_change().rolling(5).std()
        df['Volume_Trend'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Create Target (Did price go up in next 3 candles?)
        df['Future_Return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0.001).astype(int)
        
        return df.dropna()
    except Exception as e:
        print(f"Error: {e}")
        return None

# ... inside day_engine.py ...

def run_day_simulation(app, ticker):
    with app.app_context():
        df = download_intraday_data(ticker)
        if df is None: return {"error": "No data"}

        print("🗳️ Polling the Council of Strategies...")

        # 1. BUILD VOTE DATASET
        vote_history = []
        valid_indices = []
        for i in range(50, len(df)):
            market_slice = df.iloc[:i+1]
            votes = strategies.get_council_votes(market_slice)
            votes['Raw_RSI'] = df['RSI'].iloc[i]
            votes['Raw_Volatility'] = df['Volatility'].iloc[i]
            votes['Raw_Volume'] = df['Volume_Trend'].iloc[i]
            vote_history.append(votes)
            valid_indices.append(df.index[i])

        X_full = pd.DataFrame(vote_history, index=valid_indices)
        y_full = df.loc[valid_indices, 'Target']
        
        # --- FIX: DEFINE FEATURES HERE ---
        features = list(X_full.columns)
        # ---------------------------------
        
        # 2. TRAIN MANAGER
        split = int(len(X_full) * 0.7)
        X_train = X_full.iloc[:split]
        y_train = y_full.iloc[:split]
        X_test = X_full.iloc[split:]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # 3. RUN SIMULATION
        results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}
        capital = 25000.0
        shares = 0
        in_trade = False
        active_trade = None
        
        # Clear old sim data safely
        try:
            Trade.query.filter_by(symbol=ticker, mode='DAY_COUNCIL').delete()
            db.session.commit()
        except: db.session.rollback()

        print(f"☀️ Starting Sim for {ticker}...")

        for i in range(len(X_test)):
            current_votes = X_test.iloc[i]
            current_time = current_votes.name.time()
            current_price = df.loc[current_votes.name]['Close']
            
            feat_row = current_votes.values.reshape(1, -1)
            prob = model.predict_proba(feat_row)[0][1]

            # --- DECISION LOGIC ---
            if not in_trade:
                # BUY SIGNAL
                if time(10,0) < current_time < time(15,0) and prob > 0.51:
                    shares = 25000 / current_price
                    active_trade = {
                        "entry_price": current_price,
                        "quantity": shares,
                        "entry_time": current_votes.name.to_pydatetime(),
                        "confidence": prob,
                        "votes": {
                            "rsi": int(current_votes['RSI_Vote']),
                            "breakout": int(current_votes['Breakout_Vote']),
                            "heikin": int(current_votes['Heikin_Vote']),
                            "macd": int(current_votes['MACD_Vote'])
                        }
                    }
                    in_trade = True
                    
                    msg = f"BUY @ ${current_price:.2f} (Conf: {prob:.2f})"
                    results['logs'].append({"date": str(current_votes.name), "msg": msg, "type": "buy"})
                
                # HOLD SIGNAL
                else:
                    if prob > 0.40: 
                        votes_row = StrategyVote(
                            vote_rsi=int(current_votes['RSI_Vote']),
                            vote_breakout=int(current_votes['Breakout_Vote']),
                            vote_heikin=int(current_votes['Heikin_Vote']),
                            vote_fib=int(current_votes['MACD_Vote']) 
                        )
                        decision = ModelDecision(
                            confidence_score=float(prob),
                            model_version="v2_council", 
                            decision_type="HOLD",
                            symbol=ticker,
                            rsi_at_entry=float(current_votes['Raw_RSI']),
                            features_used="Council Votes"
                        )
                        decision.votes = votes_row 
                        try:
                            db.session.add(decision)
                            db.session.commit()
                        except: db.session.rollback()

            # EXIT LOGIC
            elif in_trade:
                should_sell = False
                reason = ""
                
                if current_price >= active_trade['entry_price'] * 1.01:
                    should_sell = True; reason = "TAKE PROFIT"
                elif current_price <= active_trade['entry_price'] * 0.995:
                    should_sell = True; reason = "STOP LOSS"
                elif current_time >= FORCE_CLOSE_TIME:
                    should_sell = True; reason = "FORCE CLOSE"
                
                if should_sell:
                    pnl = (current_price - active_trade['entry_price']) * active_trade['quantity']
                    pnl_pct = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                    capital += pnl
                    
                    new_trade = Trade(
                        symbol=ticker, mode='DAY_COUNCIL',
                        entry_time=active_trade['entry_time'], entry_price=float(active_trade['entry_price']),
                        quantity=float(active_trade['quantity']), direction="LONG",
                        exit_time=current_votes.name.to_pydatetime(), exit_price=float(current_price),
                        pnl_dollar=float(pnl), pnl_percent=float(pnl_pct), exit_reason=reason
                    )
                    decision = ModelDecision(
                        confidence_score=float(active_trade['confidence']),
                        model_version="v2_council", decision_type="AI_ENSEMBLE", symbol=ticker,
                        rsi_at_entry=float(active_trade['votes']['rsi']),
                        features_used="Council Votes"
                    )
                    decision.votes = StrategyVote(
                        vote_rsi=int(active_trade['votes']['rsi']),
                        vote_breakout=int(active_trade['votes']['breakout']),
                        vote_heikin=int(active_trade['votes']['heikin']),
                        vote_fib=int(active_trade['votes']['macd']) 
                    )
                    new_trade.decision = decision 
                    try:
                        db.session.add(new_trade)
                        db.session.commit()
                    except: db.session.rollback()
                    
                    in_trade = False
                    log_type = "profit" if pnl > 0 else "loss"
                    
                    msg = f"{reason} @ ${current_price:.2f}"
                    results['logs'].append({"date": str(current_votes.name), "msg": msg, "type": log_type})

            results['dates'].append(current_votes.name.strftime('%Y-%m-%d %H:%M'))
            results['stock_price'].append(current_price)
            results['bot_balance'].append(capital)
        
        # Save Brain
        brain_package = {"model": model, "features": features, "ticker": ticker, "type": "COUNCIL_MANAGER_5M"}
        joblib.dump(brain_package, "day_brain.pkl")

        return results