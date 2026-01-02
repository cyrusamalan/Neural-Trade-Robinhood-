import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Mute the warnings for clean output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
FILE_NAME = "INTC_3y_enriched_data.csv" # Change this to whatever file you use
MIN_CONFIDENCE = 0.51 
STOP_LOSS_PCT = 0.05  # 5% Max Risk per trade

def load_and_prep_data(filepath):
    print(f"📂 Loading {filepath}...")
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{filepath}'.")
        return None

    # Target: Price Up next bar
    df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    
    # Setup: Dips (BBP < 0.3) or Oversold (RSI < 45)
    df['Is_Setup'] = (df['BBP'] < 0.30) | (df['RSI'] < 45)
    
    df.dropna(inplace=True)
    return df

def train_hybrid_model(df):
    setup_data = df[df['Is_Setup'] == True].copy()
    
    if len(setup_data) < 10:
        print("❌ Not enough dip data to train.")
        return None, None
        
    print("🧠 Training Hybrid AI (Random Forest)...")
    
    features = [
        'Volume', 'Semi_Sector_Close', 'SP500_Close', '10Y_Yield', 'VIX_Close',
        'RSI', 'SMA_50', 'SMA_200', 'ATR', 'OBV', 
        'Rel_Strength_SP500', 'Rel_Strength_Sector', 'Volatility_20d', 'ADX'
    ]
    
    # Ensure we only use columns that exist
    features = [f for f in features if f in df.columns]
    
    X = setup_data[features]
    y = setup_data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Model Accuracy on Pullbacks: {acc:.1%}")
    
    return model, features

def run_backtest(df, model, features):
    print("\n⚔️ RUNNING BACKTEST (Starting with $10,000)...")
    
    capital = 10000.0
    shares = 0
    in_trade = False
    entry_price = 0
    
    for i in range(len(df)):
        if i < 20: continue 
        
        row = df.iloc[i]
        price = row['Close']
        date = df.index[i].date()
        
        # 1. ENTRY LOGIC
        if row['Is_Setup'] and not in_trade:
            # Fix: Create a DataFrame for prediction to avoid warnings
            current_feat_df = pd.DataFrame([row[features].values], columns=features)
            prob_up = model.predict_proba(current_feat_df)[0][1]
            
            if prob_up > MIN_CONFIDENCE:
                shares = capital / price
                capital = 0
                in_trade = True
                entry_price = price
                print(f"[{date}] 🟢 BUY  @ ${price:.2f} | AI Conf: {prob_up:.2f}")
        
        # 2. EXIT LOGIC
        elif in_trade:
            # STOP LOSS: If we lose 5%, GET OUT.
            if price < entry_price * (1 - STOP_LOSS_PCT):
                capital = shares * price
                shares = 0
                in_trade = False
                print(f"[{date}] 🛡️ STOP LOSS TRIGGERED @ ${price:.2f} (-5%)")
            
            # TAKE PROFIT: RSI recovers
            elif row['RSI'] > 55:
                capital = shares * price
                shares = 0
                in_trade = False
                print(f"[{date}] 💰 TAKE PROFIT @ ${price:.2f}")

    # Final tally
    if in_trade:
        capital = shares * df.iloc[-1]['Close']
        
    print(f"\n💰 FINAL BALANCE: ${capital:.2f}")

if __name__ == "__main__":
    data = load_and_prep_data(FILE_NAME)
    if data is not None:
        ai, feats = train_hybrid_model(data)
        if ai:
            run_backtest(data, ai, feats)