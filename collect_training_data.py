import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
SYMBOL = "NVDA"
YEARS_BACK = 25
INTERVAL = "1d"

def fetch_and_process_data(symbol):
    print(f"📥 Downloading core data for {symbol}...")
    
    start_date = (datetime.now() - timedelta(days=YEARS_BACK*365)).strftime('%Y-%m-%d')
    
    # 1. FETCH MAIN SYMBOL
    df = yf.download(symbol, start=start_date, interval=INTERVAL, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print("❌ Error: No data found.")
        return

    # 2. FETCH MACRO CONTEXT (The "Peripheral Vision")
    print("🌍 Fetching Macro Data (SPY, VIX, Yields, Sector)...")
    
    # ^GSPC = S&P 500 (Market Health)
    # ^VIX  = Volatility Index (Fear Gauge)
    # ^TNX  = 10-Year Treasury Yield (Interest Rates hurt Tech stocks)
    # SOXX  = Semiconductor ETF (Is the whole sector moving or just NVDA?)
    macro_tickers = ["^GSPC", "^VIX", "^TNX", "SOXX"]
    
    macro_data = yf.download(macro_tickers, start=start_date, interval=INTERVAL, auto_adjust=True, progress=False)
    
    # Flatten MultiIndex columns from macro download
    # Structure is usually (Price, Ticker). We want just Close prices.
    if isinstance(macro_data.columns, pd.MultiIndex):
        macro_closes = macro_data['Close']
    else:
        macro_closes = macro_data

    # Rename and Merge
    # We use 'Close' prices for context
    macro_closes = macro_closes.rename(columns={
        "^GSPC": "SP500_Close",
        "^VIX": "VIX_Close",
        "^TNX": "10Y_Yield",
        "SOXX": "Semi_Sector_Close"
    })
    
    # Merge macro data into main df (Left join ensures we keep NVDA rows)
    df = df.join(macro_closes)

    # Fill missing macro data (e.g. holidays) with previous day's value
    df.ffill(inplace=True)

    print("🧮 Calculating Advanced Indicators...")
    
    # --- 3. STANDARD INDICATORS ---
    df['RSI'] = df.ta.rsi(length=14)
    df['SMA_50'] = df.ta.sma(length=50)
    df['SMA_200'] = df.ta.sma(length=200)
    df['ATR'] = df.ta.atr(length=14)
    df['OBV'] = df.ta.obv()

    # --- 4. RELATIVE STRENGTH (Alpha) ---
    # Is NVDA beating the market? (Ratio Analysis)
    df['Rel_Strength_SP500'] = df['Close'] / df['SP500_Close']
    df['Rel_Strength_Sector'] = df['Close'] / df['Semi_Sector_Close']

    # --- 5. TARGET ENGINEERING (Machine Learning Food) ---
    # ML models hate raw prices ($100 vs $1000). They love Percentages (Returns).
    
    # "Velocity": How fast is price changing?
    df['Log_Ret'] = df.ta.log_return(append=False)
    
    # "Volatility": Rolling Standard Deviation
    df['Volatility_20d'] = df['Log_Ret'].rolling(20).std()

    # --- 6. ADVANCED INDICATORS (Fixed) ---
    # ADX
    adx_df = df.ta.adx(length=14)
    if adx_df is not None:
        adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
        df['ADX'] = adx_df[adx_col]
    
    # Bollinger %B
    bbands_df = df.ta.bbands(length=20, std=2)
    if bbands_df is not None:
        bbp_cols = [c for c in bbands_df.columns if c.startswith('BBP')]
        if bbp_cols:
            df['BBP'] = bbands_df[bbp_cols[0]]

    # --- CLEANUP ---
    df.dropna(inplace=True)
    
    filename = f"{symbol}_{YEARS_BACK}y_enriched_data.csv"
    df.to_csv(filename)
    
    print(f"\n✨ DONE! Saved to: {filename}")
    print(f"📊 Total Rows: {len(df)}")
    print("👀 Preview of New Data:")
    print(df[['Close', 'SP500_Close', 'VIX_Close', 'Rel_Strength_Sector']].tail(3))

if __name__ == "__main__":
    fetch_and_process_data(SYMBOL)