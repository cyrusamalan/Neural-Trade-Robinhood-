import yfinance as yf
import pandas_ta as ta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import datetime as dt

# --- CONFIGURATION ---
SYMBOL = "NVDA"         
SMA_PERIOD = 200
REFRESH_RATE = 1000    # Update every 1000ms (1 second)

# Setup the Plot Style
style.use('dark_background')
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 1, 1)

# Global Storage for Live Data
xs = [] # Time
ys = [] # Price
sma_level = 0.0 # The Strategy Line

def initialize_data():
    """Calculates the 200 SMA and gets recent minute history so the graph isn't empty."""
    global sma_level, xs, ys
    
    print(f"📥 Initializing Strategy for {SYMBOL}...")
    
    # 1. Get Daily Data for the 200 SMA (The Strategy Line)
    daily_df = yf.download(SYMBOL, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.get_level_values(0)
    
    daily_df['SMA_200'] = daily_df.ta.sma(length=SMA_PERIOD)
    sma_level = daily_df['SMA_200'].iloc[-1]
    print(f"   🔹 Strategy Line (200 SMA): ${sma_level:.2f}")

    # 2. Get Intraday Data (Last 60 mins) to fill the graph history
    # Note: 1m data might be delayed 15m depending on Yahoo, but it gives context.
    intraday_df = yf.download(SYMBOL, period="1d", interval="1m", progress=False, auto_adjust=True)
    if isinstance(intraday_df.columns, pd.MultiIndex):
        intraday_df.columns = intraday_df.columns.get_level_values(0)
    
    # Take the last 50 points
    recent = intraday_df.tail(50)
    xs = list(range(len(recent))) # Simple index for X-axis
    ys = recent['Close'].tolist()

def animate(i):
    """This function runs every second to update the graph."""
    global xs, ys
    
    # 1. Fetch Live Price (Fast Method)
    ticker = yf.Ticker(SYMBOL)
    try:
        # 'fast_info' is the new realtime method in yfinance 1.0+
        current_price = ticker.fast_info['lastPrice']
    except:
        return # Skip frame if connection fails
    
    # 2. Append to Data
    xs.append(xs[-1] + 1)
    ys.append(current_price)
    
    # Keep only last 100 points to keep graph moving
    if len(xs) > 100:
        xs = xs[-100:]
        ys = ys[-100:]
    
    # 3. Draw the Graph
    ax1.clear()
    
    # Determine Color (Green if Bullish, Red if Bearish)
    if current_price > sma_level:
        line_color = '#00ff00' # Bright Green
        status = "BULL (INVESTED)"
    else:
        line_color = '#ff0000' # Bright Red
        status = "BEAR (CASH)"

    # Plot Price Line
    ax1.plot(xs, ys, color=line_color, linewidth=2, label='Live Price')
    
    # Plot Strategy Line (SMA 200)
    ax1.axhline(y=sma_level, color='cyan', linestyle='--', linewidth=1, label=f'Strategy Line (${sma_level:.2f})')
    
    # Labels & Annotations
    ax1.set_title(f"🔴 LIVE TRADING BOT: {SYMBOL} | {status}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # Show current price on graph
    ax1.text(xs[-1], ys[-1], f"${current_price:.2f}", color='white', fontsize=12, fontweight='bold')

if __name__ == "__main__":
    # Pre-load data
    initialize_data()
    
    # Start the "Heartbeat" Animation
    ani = animation.FuncAnimation(fig, animate, interval=REFRESH_RATE)
    plt.show()