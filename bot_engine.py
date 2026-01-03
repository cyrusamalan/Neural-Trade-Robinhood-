import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def get_simulation_data(ticker, file_path):
    """
    Runs a Walk-Forward (Rolling Window) simulation.
    1. Trains on past data (e.g., first 500 hours).
    2. Predicts the NEXT hour.
    3. Retrains with the new hour added to history.
    """
    results = {"dates": [], "stock_price": [], "bot_balance": [], "logs": []}

    # --- 1. Load & Clean Data ---
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    except Exception as e:
        return {"error": str(e)}

    # Robust Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    # Define Target (Future Return)
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)

    # Define Setup (The "Dip")
    df['Is_Setup'] = (df['RSI'] < 55)

    # Feature List
    feature_candidates = [
        'Volume', 'Semi_Sector_Close', 'SP500_Close', '10Y_Yield', 'VIX_Close', 
        'RSI', 'SMA_50', 'SMA_200', 'ATR', 'OBV', 
        'Rel_Strength_SP500', 'Volatility_20d', 'ADX', 'BBP'
    ]
    features = [f for f in feature_candidates if f in df.columns]

    # --- 2. Walk-Forward Loop Setup ---
    capital = 10000.0
    shares = 0
    in_trade = False
    entry_price = 0
    
    # MINIMUM_TRAIN: Need at least 300 hours of history before guessing
    start_index = 300 
    
    # Re-train every 24 hours to keep simulation fast
    retrain_interval = 24 
    current_model = None

    print(f"Starting Walk-Forward Simulation for {ticker}...")

    # Iterate through the timeline
    for i in range(start_index, len(df)):
        
        # Current Market State
        current_date = df.index[i]
        current_row = df.iloc[i]
        price = float(current_row['Close'])
        date_str = current_date.strftime('%Y-%m-%d')
        
        # --- A. RETRAINING LOGIC ---
        if i % retrain_interval == 0 or current_model is None:
            # Slice history: Everything strictly BEFORE now
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

        # --- B. TRADING LOGIC ---
        if not in_trade:
            if current_row['Is_Setup']:
                if current_model:
                    try:
                        feat_vals = current_row[features].values.reshape(1, -1)
                        prob = current_model.predict_proba(feat_vals)[0][1]
                        
                        if prob > 0.51:
                            shares = capital / price
                            capital = 0
                            in_trade = True
                            entry_price = price
                            results['logs'].append({"date": date_str, "msg": f"BUY @ ${price:.2f} (Conf: {prob:.2f})", "type": "buy"})
                    except: 
                        pass 
        
        else:
            # Stop Loss
            if price < entry_price * 0.95:
                capital = shares * price
                shares = 0
                in_trade = False
                results['logs'].append({"date": date_str, "msg": f"STOP LOSS @ ${price:.2f}", "type": "loss"})
            
            # Take Profit
            elif current_row['RSI'] > 55:
                capital = shares * price
                shares = 0
                in_trade = False
                results['logs'].append({"date": date_str, "msg": f"TAKE PROFIT @ ${price:.2f}", "type": "profit"})

        current_val = capital + (shares * price)
        results['dates'].append(date_str)
        results['stock_price'].append(price)
        results['bot_balance'].append(current_val)

    # --- 3. Final Save (For Live Trading) ---
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