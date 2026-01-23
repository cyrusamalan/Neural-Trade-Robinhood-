Here is a professional, GitHub-ready `README.md` file for your project. It incorporates all the advanced features we discussed (The Council, Market Replay, Hybrid Data) and presents the project as a serious engineering portfolio piece.

---

# 📈 NeuralTrade: Quantitative Trading Ecosystem

**A full-stack automated trading platform that bridges the gap between quantitative research and live execution.**

---

## 📖 Executive Summary

**NeuralTrade** is an algorithmic trading system built to solve the problem of "Model Overfitting" in financial markets. Instead of relying on a single rigid strategy, it employs an **Ensemble Machine Learning Architecture** dubbed **"The Council."**

The system aggregates real-time signals from multiple distinct technical algorithms (Trend, Mean Reversion, Momentum) and feeds them into a **Random Forest Meta-Model**. This model learns which strategies are currently profitable for a specific asset and adapts its confidence score dynamically.

The platform includes a **Hybrid Data Pipeline** to bypass API limitations (merging Stock/Crypto API data with Futures CSV data) and a **Visual Analytics Dashboard** for post-trade analysis.

---

## 🧠 Core Architecture: "The Council"

The decision-making engine operates in two tiers to ensure high-precision entries.

### Tier 1: The Council (Feature Extraction)

Market data is first analyzed by four distinct "Specialist" algorithms defined in `strategies.py`. Each casts a vote `(-1, 0, 1)`:

1. **Reversal Specialist (RSI):** Detects overbought/oversold extremes.
2. **Trend Follower (VWAP):** Confirms institutional control via Volume-Weighted Average Price.
3. **Volatility Trader (Bollinger):** Identifies price expansion and breakouts.
4. **Momentum Gauge (MACD):** Tracks shifts in price velocity.

### Tier 2: The Neural Manager (Random Forest)

A **Random Forest Classifier** ingests the Council's votes alongside macro-features (Volatility, SP500 Correlation). It outputs a **Probabilistic Confidence Score** (e.g., `0.85`).

* **Dynamic Retraining:** The model retrains every 5 days on a rolling window to adapt to shifting market regimes.

---

## 🚀 Key Features

### 🛠️ Hybrid Data Ingestion

* **Multi-Asset Sync:** Automated API polling for **Stocks & Crypto** via `robin_stocks`.
* **Futures Support:** Custom CSV ingestion engine to track **Robinhood Legend (Futures)** trades, overcoming the lack of public API endpoints for this asset class.
* **Orphan Handling:** FIFO logic to reconcile "Buy" and "Sell" orders with mismatched timestamps.

### 🧪 Advanced Simulation Engines

* **Intraday Simulator:** A high-frequency engine (`day_engine.py`) processing 5-minute candles with "Force Close" logic at 3:55 PM.
* **Hindsight Analytics:** Calculates "Opportunity Cost" by flagging missed pumps (>1%) where the AI confidence was too low.
* **Walk-Forward Backtesting:** Strictly prevents data leakage by training on past data and testing on future data (`bot_engine.py`).

### 📊 Interactive Dashboard

* **Market Replay Mode:** A JavaScript tool to replay historical trading days candle-by-candle, visualizing exactly when and why the AI triggered a trade.
* **P&L Calendar:** A heat-map visualization of daily trading performance.
* **Real-Time Monitor:** Streams live price action and AI confidence scores via WebSocket-like polling.

---

## 📂 Project Structure

```bash
NeuralTrade/
├── app.py                 # Main Flask Application & Route Handler
├── bot_engine.py          # Swing Trading & Walk-Forward Backtester
├── day_engine.py          # Intraday Simulator & Hindsight Analytics
├── sync_trades.py         # Hybrid Data Ingestion (API + CSV)
├── monitor.py             # Dashboard Backend API
├── strategies.py          # "The Council" Technical Logic
├── models.py              # SQLAlchemy Database Models
├── templates/
│   └── monitor.html       # Frontend Dashboard (Tailwind + Plotly)
└── datasets/              # Historical Data & Futures CSVs

```

---

## 🛠️ Installation & Setup

### 1. Prerequisites

* Python 3.10+
* PostgreSQL (Local or Cloud)
* Robinhood Account

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NeuralTrade.git
cd NeuralTrade

# Install dependencies
pip install -r requirements.txt

```

### 3. Database Setup

Create a PostgreSQL database named `neuraltraderobinhood` and configure your credentials in `app.py` and `sync_trades.py`.

### 4. Authentication

To generate your Robinhood token (valid for 24 hours):

1. Log in to Robinhood Web.
2. Open Developer Tools -> Application -> Local Storage.
3. Copy the Authorization Bearer token.
4. Save it to `token.txt` in the root directory.

---

## 🖥️ Usage

### 1. Start the Dashboard

```bash
python monitor.py

```

* Access the dashboard at `http://localhost:5001` to view the **P&L Calendar** and **Market Replay**.

### 2. Run the Data Sync (Background Service)

```bash
python sync_trades.py

```

* This script polls for new trades and ingests `futures_data.csv` if present.

### 3. Run a Simulation

Send a POST request to `/run_sim` or use the dashboard UI to trigger a backtest:

* **Swing Mode:** Uses `bot_engine.py` (Daily candles).
* **Day Mode:** Uses `day_engine.py` (5-min candles).

---

## ⚖️ Disclaimer

*This software is for educational and research purposes only. Algorithmic trading involves significant risk of financial loss. The developers are not responsible for any losses incurred by using this software.*
