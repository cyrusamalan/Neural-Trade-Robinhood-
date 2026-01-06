import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from datetime import datetime
from models import db, Trade, ModelDecision 

warnings.filterwarnings("ignore")

def get_simulation_data(app, ticker, file_path):
    """
    Runs a Walk-Forward simulation AND saves results to Postgres.
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
            print(f"🧹 Clearing old backtest records for {ticker}...")
            Trade.query.filter_by(symbol=ticker, mode='BACKTEST').delete()
            db.session.commit()
        except Exception as e:
            print(f"⚠️ Warning: Database clean failed: {e}")

        # --- 3. Walk-Forward Loop ---
        capital = 10000.0
        shares = 0
        in_trade = False
        active_trade = None 
        start_index = 300 
        retrain_interval = 24 
        current_model = None

        print(f"Starting Walk-Forward Simulation for {ticker}...")

        for i in range(start_index, len(df)):
            current_date = df.index[i]
            current_row = df.iloc[i]
            price = float(current_row['Close'])
            date_str = current_date.strftime('%Y-%m-%d')
            
            # A. Retraining Logic
            if i % retrain_interval == 0 or current_model is None:
                historical_data = df.iloc[:i]
                training_set = historical_data[historical_data['Is_Setup'] == True]
                
                if len(training_set) > 20:
                    X_train = training_set[features]
                    y_train = training_set['Target']
                    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
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
                        
                        if prob > 0.51:
                            shares = capital / price
                            capital = 0
                            in_trade = True
                            
                            # FIX: Convert everything to standard python float()
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
                
                if price < active_trade['entry_price'] * 0.95:
                    exit_triggered = True; reason = "Stop Loss"
                    results['logs'].append({"date": date_str, "msg": f"STOP LOSS @ ${price:.2f}", "type": "loss"})
                elif current_row['RSI'] > 55:
                    exit_triggered = True; reason = "Take Profit"
                    results['logs'].append({"date": date_str, "msg": f"TAKE PROFIT @ ${price:.2f}", "type": "profit"})

                if exit_triggered:
                    capital = shares * price
                    shares = 0
                    in_trade = False
                    
                    try:
                        # FIX: Explicit floats for PnL calculations
                        pnl = float((price - active_trade['entry_price']) * active_trade['quantity'])
                        pnl_pct = float((price - active_trade['entry_price']) / active_trade['entry_price'])
                        
                        new_trade = Trade(
                            symbol=ticker,
                            mode='BACKTEST',
                            entry_time=active_trade['entry_time'],
                            entry_price=float(active_trade['entry_price']),
                            quantity=float(active_trade['quantity']),
                            direction="LONG",
                            exit_time=current_date,
                            exit_price=float(price),
                            pnl_dollar=pnl,
                            pnl_percent=pnl_pct,
                            exit_reason=reason
                        )
                        
                        decision = ModelDecision(
                            confidence_score=float(active_trade['confidence']),
                            model_version="v2_walk_forward",
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
                        print(f"❌ DB Error: {e}")
                        db.session.rollback()

            current_val = capital + (shares * price)
            results['dates'].append(date_str)
            results['stock_price'].append(price)
            results['bot_balance'].append(current_val)

        # --- 4. Final Live Training ---
        final_train_set = df[df['Is_Setup'] == True]
        if len(final_train_set) > 20:
            final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            final_model.fit(final_train_set[features], final_train_set['Target'])
            
            brain_package = {
                "model": final_model,
                "features": features,
                "ticker": ticker,
                "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M')
            }
            joblib.dump(brain_package, "brain.pkl")
            results['logs'].append({"date": "SYSTEM", "msg": f"🧠 Final Brain Saved for Live Trading", "type": "profit"})

        return results