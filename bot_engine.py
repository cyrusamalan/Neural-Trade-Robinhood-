import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def get_simulation_data(ticker, file_path):
    """
    Runs simulation AND saves the trained model WITH METADATA.
    Handles 'Not enough data' by being less aggressive with cleaning.
    """
    results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}

    # 1. Load Data
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    except Exception as e:
        return {"error": str(e)}

    # === 🧹 ROBUST SANITIZER (FIXED) ===
    # 1. Replace Infinity with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Fill NaNs instead of dropping them (Crucial for SMA_200)
    df.fillna(method='ffill', inplace=True) # Forward fill (use yesterday's data)
    df.fillna(0, inplace=True)              # Fill start gaps with 0
    # ===================================

    # Target & Setup
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    # Auto-Relax Logic (Strict -> Moderate -> Loose)
    # Level 1: Strict
    df['Is_Setup'] = (df['BBP'] < 0.30) | (df['RSI'] < 45)
    setup_data = df[df['Is_Setup'] == True].copy()
    
    # Level 2: Moderate (If < 10 rows)
    if len(setup_data) < 10:
        results['logs'].append({"date": "SYSTEM", "msg": "⚠️ Strict dips rare. Relaxing to RSI < 55.", "type": "loss"})
        df['Is_Setup'] = df['RSI'] < 55
        setup_data = df[df['Is_Setup'] == True].copy()

    # Level 3: Loose (If still < 10 rows, just use any RSI < 60)
    if len(setup_data) < 10:
        results['logs'].append({"date": "SYSTEM", "msg": "⚠️ Still low data. Force-relaxing to RSI < 60.", "type": "loss"})
        df['Is_Setup'] = df['RSI'] < 60
        setup_data = df[df['Is_Setup'] == True].copy()

    # Final Check
    if len(setup_data) < 5: # Lowered requirement from 10 to 5
        return {"error": f"Not enough data. Only found {len(setup_data)} potential trade setups in history."}

    # 2. Train Model
    feature_candidates = [
        'Volume', 'Semi_Sector_Close', 'SP500_Close', '10Y_Yield', 'VIX_Close', 
        'RSI', 'SMA_50', 'SMA_200', 'ATR', 'OBV', 
        'Rel_Strength_SP500', 'Volatility_20d', 'ADX', 'BBP'
    ]
    features = [f for f in feature_candidates if f in df.columns]
    
    # Clean training set one last time
    clean_setup = setup_data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_target = setup_data.loc[clean_setup.index, 'Target']

    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    model.fit(clean_setup, clean_target)

    # --- SAVE THE BRAIN PACKAGE ---
    brain_package = {
        "model": model,
        "features": features,
        "ticker": ticker,
        "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    joblib.dump(brain_package, "brain.pkl")
    results['logs'].append({"date": "SYSTEM", "msg": f"🧠 Brain Saved: Locked to {ticker}", "type": "profit"})
    # ----------------------------

    # 3. Run Backtest
    capital = 10000.0
    shares = 0
    in_trade = False
    entry_price = 0
    
    for i in range(len(df)):
        date = df.index[i].strftime('%Y-%m-%d')
        price = float(df.iloc[i]['Close'])
        row = df.iloc[i]

        if i > 20: 
            if row['Is_Setup'] and not in_trade:
                try:
                    feat_vals = row[features].values.reshape(1, -1)
                    # Use the robust data (NaNs are already 0)
                    feat_row = pd.DataFrame(feat_vals, columns=features)
                    prob = model.predict_proba(feat_row)[0][1]
                    
                    if prob > 0.51:
                        shares = capital / price
                        capital = 0
                        in_trade = True
                        entry_price = price
                        results['logs'].append({"date": date, "msg": f"BUY @ ${price:.2f} (Conf: {prob:.2f})", "type": "buy"})
                except: pass

            elif in_trade:
                if price < entry_price * 0.95:
                    capital = shares * price
                    shares = 0
                    in_trade = False
                    results['logs'].append({"date": date, "msg": f"STOP LOSS @ ${price:.2f}", "type": "loss"})
                elif row['RSI'] > 55:
                    capital = shares * price
                    shares = 0
                    in_trade = False
                    results['logs'].append({"date": date, "msg": f"TAKE PROFIT @ ${price:.2f}", "type": "profit"})

        current_val = capital + (shares * price)
        results['dates'].append(date)
        results['stock_price'].append(price)
        results['bot_balance'].append(current_val)

    return results