import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import os
from datetime import datetime
import yfinance as yf
import pandas_ta as ta
from models import db, Trade, ModelDecision 

warnings.filterwarnings("ignore")

def get_simulation_data(app, ticker, file_path, train_days=365, test_days=10, min_conf=0.51, stop_loss=0.05, take_profit=0.04):
    """
    Runs a Walk-Forward simulation on the LAST 'test_days' of data,
    using a model trained on the 'train_days' prior to that.
    """
    
    with app.app_context():
        results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}

        # --- 1. Load & Clean Data ---
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        except Exception as e:
            return {"error": str(e)}

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # Predict 1 day ahead (Standard Swing Target)
        df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        df['Is_Setup'] = (df['RSI'] < 55)

        feature_candidates = [
            'Volume', 'Semi_Sector_Close', 'SP500_Close', '10Y_Yield', 'VIX_Close', 
            'RSI', 'SMA_50', 'SMA_200', 'ATR', 'OBV', 
            'Rel_Strength_SP500', 'Volatility_20d', 'ADX', 'BBP'
        ]
        features = [f for f in feature_candidates if f in df.columns]

        # --- 2. Database Prep ---
        try:
            Trade.query.filter_by(symbol=ticker, mode='BACKTEST').delete()
            db.session.commit()
        except Exception as e:
            # print(f"⚠️ Warning: Database clean failed: {e}")
            db.session.rollback()

        # --- 3. Determine Split Points ---
        total_rows = len(df)
        
        # We want to simulate the LAST 'test_days'
        start_index = total_rows - test_days
        
        # Ensure we don't start before we have data
        if start_index < 50: start_index = 50 

        capital = 10000.0
        shares = 0
        in_trade = False
        active_trade = None 
        current_model = None
        
        # How often to re-train the brain (e.g., every 5 days)
        retrain_interval = 5 

        print(f"🌊 Swing Sim: Training on {train_days}d window, Testing last {test_days}d...")

        # --- 4. The Simulation Loop ---
        for i in range(start_index, len(df)):
            current_date = df.index[i]
            current_row = df.iloc[i]
            price = float(current_row['Close'])
            date_str = current_date.strftime('%Y-%m-%d')
            
            # A. Retraining Logic
            # We train on the window: [Current Day - Train Days] -> [Current Day]
            if i % retrain_interval == 0 or current_model is None:
                train_start_idx = max(0, i - train_days)
                historical_data = df.iloc[train_start_idx:i]
                
                # Filter for Setup days (Smart Training)
                training_set = historical_data[historical_data['Is_Setup'] == True]
                
                if len(training_set) > 20:
                    X_train = training_set[features]
                    y_train = training_set['Target']
                    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                    model.fit(X_train, y_train)
                    current_model = model
                else:
                    current_model = None 

            # B. Trading Logic
            if not in_trade:
                if current_row['Is_Setup'] and current_model:
                    try:
                        feat_vals = current_row[features].values.reshape(1, -1)
                        prob = current_model.predict_proba(feat_vals)[0][1]
                        
                        # --- USE NEW THRESHOLD HERE ---
                        if prob > min_conf:
                            shares = capital / price
                            capital = 0
                            in_trade = True
                            
                            active_trade = {
                                "entry_time": current_date,
                                "entry_price": float(price),
                                "quantity": float(shares),
                                "confidence": float(prob),
                                "rsi": float(current_row.get('RSI', 0)),
                                "sma50": float(current_row.get('SMA_50', 0)),
                                "sma200": float(current_row.get('SMA_200', 0)),
                                "vix": float(current_row.get('VIX_Close', 0)),
                                "sp500": float(current_row.get('SP500_Close', 0))
                            }
                            results['logs'].append({"date": date_str, "msg": f"BUY @ ${price:.2f} (Conf: {prob:.2f})", "type": "buy"})
                    except Exception as e: 
                        print(f"Buy Error: {e}")
            
            else:
                exit_triggered = False
                reason = ""
                
                # 1. Calculate Dynamic Thresholds based on your settings
                # (Assumes stop_loss and take_profit are passed into the function)
                stop_price = active_trade['entry_price'] * (1 - stop_loss)
                target_price = active_trade['entry_price'] * (1 + take_profit)

                # 2. Check Stop Loss
                if price < stop_price:
                    exit_triggered = True; reason = "Stop Loss"
                    results['logs'].append({"date": date_str, "msg": f"STOP LOSS (-{stop_loss*100:.1f}%) @ ${price:.2f}", "type": "loss"})
                
                # 3. Check Price Target (NEW)
                elif price >= target_price:
                    exit_triggered = True; reason = "Take Profit"
                    results['logs'].append({"date": date_str, "msg": f"TAKE PROFIT (+{take_profit*100:.1f}%) @ ${price:.2f}", "type": "profit"})

                # 4. Check RSI Target (Keep this as a backup "smart" exit)
                elif current_row['RSI'] > 75: 
                    exit_triggered = True; reason = "Take Profit (RSI)"
                    results['logs'].append({"date": date_str, "msg": f"TAKE PROFIT (RSI > 75) @ ${price:.2f}", "type": "profit"})
                if exit_triggered:
                    capital = shares * price
                    shares = 0
                    in_trade = False
                    
                    try:
                        pnl = float((price - active_trade['entry_price']) * active_trade['quantity'])
                        pnl_pct = float((price - active_trade['entry_price']) / active_trade['entry_price'])
                        
                        new_trade = Trade(
                            symbol=ticker, mode='BACKTEST',
                            entry_time=active_trade['entry_time'], entry_price=float(active_trade['entry_price']),
                            quantity=float(active_trade['quantity']), direction="LONG",
                            exit_time=current_date, exit_price=float(price),
                            pnl_dollar=pnl, pnl_percent=pnl_pct, exit_reason=reason
                        )
                        
                        decision = ModelDecision(
                            confidence_score=float(active_trade['confidence']),
                            model_version="v2_walk_forward",
                            symbol=ticker, decision_type="AI_SWING",
                            rsi_at_entry=float(active_trade['rsi']),
                            sma50_at_entry=float(active_trade['sma50']),
                            sma200_at_entry=float(active_trade['sma200']),
                            vix_at_entry=float(active_trade['vix']),
                            sp500_at_entry=float(active_trade['sp500'])
                        )
                        new_trade.decision = decision
                        db.session.add(new_trade)
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()

            current_val = capital + (shares * price)
            results['dates'].append(date_str)
            results['stock_price'].append(price)
            results['bot_balance'].append(current_val)

        # --- 5. Final Future Forecast ---
        # Train one last time on the MOST RECENT window to be ready for tomorrow
        final_train_start = max(0, len(df) - train_days)
        final_train_set = df.iloc[final_train_start:]
        final_train_set = final_train_set[final_train_set['Is_Setup'] == True]
        
        if len(final_train_set) > 20:
            final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            final_model.fit(final_train_set[features], final_train_set['Target'])
            
            brain_package = {
                "model": final_model, "features": features,
                "ticker": ticker, "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            joblib.dump(brain_package, "brain.pkl")
            
            # Predict Tomorrow
            try:
                last_row = df.iloc[-1]
                feat_vals = last_row[features].values.reshape(1, -1)
                prob = final_model.predict_proba(feat_vals)[0][1]
                signal = "BULLISH 🟢" if prob > 0.51 else "BEARISH 🔴"
                msg = f"🔮 FORECAST (Next Day): {signal} (Conf: {prob:.2f})"
                results['logs'].append({"date": "FUTURE", "msg": msg, "type": "forecast"})
            except: pass

        return results

# --- DATA DOWNLOADER (Required) ---
def download_fresh_data(symbol):
    print(f"\n📥 Manager: Downloading history for {symbol}...")
    DATA_DIR = "datasets"  
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
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