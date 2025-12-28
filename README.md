<<<<<<< HEAD
# 📈 Neural-Trade-Robinhood

A full-stack algorithmic trading platform that automates technical analysis and machine learning strategies on Robinhood. This system features a "Strategy Tournament" engine that logs the performance of multiple algorithms (Trend Following, Mean Reversion, Gradient Boosting) into a PostgreSQL database to statistically determine the best trading logic for different market regimes.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Database](https://img.shields.io/badge/PostgreSQL-14+-elephant.svg)
![Framework](https://img.shields.io/badge/Flask-Backend-green.svg)
![Status](https://img.shields.io/badge/Status-Live_Simulation-orange.svg)

---

## 📖 Overview

**Neural-Trade-Robinhood** is not just a trading bot; it is a quantitative research platform. It runs a "Squad" of distinct trading algorithms side-by-side:
1.  **The Trend Surfer:** Captures massive bull runs (Best for NVDA, BTC).
2.  **The Rubber Band:** Profits from sideways volatility and crashes (Best for NIO, ETH).
3.  **The ML Sniper:** (Research) Uses Gradient Boosting to find volume-based anomalies.

Instead of guessing which strategy works, this application logs every signal from every bot into a centralized database. It then tracks the *actual* stock movement over the next 1-5 days to statistically grade each model's "Predictive Power."

---

## 🏗️ Architecture & Data Pipeline

The core differentiator of this project is its **Analysis-First** architecture.

### 1. The Decision Engine (Python)
* Runs daily before market open.
* Fetches live data via `yfinance`.
* Calculates technical indicators (RSI, Bollinger Bands, SMA 200, OBV).
* Generates a vote (BUY/SELL/HOLD) from every strategy in the "Squad."

### 2. The Persistence Layer (PostgreSQL)
We do not just execute trades; we record the *intent*.
* **Input Logging:** Every model's decision is saved to `strategy_logs` *before* any trade is made.
* **Outcome Tagging:** A nightly cron job updates the database with the *actual* market return for that day.

### 3. The Analytics Dashboard (Flask)
* Queries the database to calculate real-time **Precision**, **Win Rate**, and **Sharpe Ratio** for each bot.
* Visualizes which strategy is currently "Hot" (winning) and which is "Cold" (losing).

---

## 💾 Database Schema for Statistical Analysis

We use a relational schema to separate "Signals" (Predictions) from "Reality" (Market Performance).

### Table 1: `strategy_logs`
*Records the raw prediction signal from a model at a specific point in time.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | SERIAL | Primary Key |
| `timestamp` | TIMESTAMP | Exact time of analysis |
| `ticker` | VARCHAR(10) | e.g., "NIO", "NVDA" |
| `strategy_name` | VARCHAR(50) | e.g., "Rubber_Band_v1", "Trend_SMA200" |
| `signal` | VARCHAR(10) | "BUY", "SELL", "HOLD" |
| `confidence` | FLOAT | 0.0 - 1.0 (ML Confidence Score) |
| `market_price` | FLOAT | Price at the moment of signal generation |
| `indicators_json` | JSONB | Snapshot of RSI, ADX, SMA at that moment (for debugging) |

### Table 2: `trade_outcomes`
*Records the actual result of that signal (populated 24h - 5 days later).*

| Column | Type | Description |
| :--- | :--- | :--- |
| `log_id` | INT | Foreign Key to `strategy_logs` |
| `actual_return_1d` | FLOAT | % Stock movement after 24 hours |
| `actual_return_5d` | FLOAT | % Stock movement after 5 days |
| `is_correct` | BOOLEAN | TRUE if Signal matched Market Direction |
| `profit_loss_theoretical` | FLOAT | Dollar value P/L if $1000 was traded |

**✨ Why this matters:**
This schema allows us to run SQL queries like:
> *"Show me the Win Rate of the 'Rubber Band' strategy on NIO specifically when RSI was below 30."*

---

## 🚀 Key Strategies (The Squad)

### 🌊 The Trend Surfer (`live_trader.py`)
* **Logic:** Buy when Price > 200-Day SMA. Sell when Price < 200-Day SMA.
* **Performance:** ~1696x return in backtests on volatile growth stocks.
* **Best Use:** Bull Markets.

### 🧶 The Rubber Band (`live_rubber_band.py`)
* **Logic:** Mean Reversion using Bollinger Bands (20, 2). Buy Low, Sell High.
* **Performance:** The *only* strategy that profited on NIO during the 2022-2024 crash.
* **Best Use:** Bear Markets & Sideways Chop.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10+
* PostgreSQL installed locally or via cloud (AWS RDS / Heroku).
* Robinhood Account.

### 2. Install Dependencies
```bash
git clone [https://github.com/your-username/Neural-Trade-Robinhood.git](https://github.com/your-username/Neural-Trade-Robinhood.git)
cd Neural-Trade-Robinhood
pip install -r requirements.txt
=======
📈 Neural-Trade-RobinhoodA full-stack algorithmic trading platform that automates technical analysis and machine learning strategies on Robinhood. This system features a "Strategy Tournament" engine that logs the performance of multiple algorithms (Trend Following, Mean Reversion, Gradient Boosting) into a PostgreSQL database to statistically determine the best trading logic for different market regimes.🏗️ Tech StackLanguage: Python 3.13+Trading API: robin_stocks (Robinhood), yfinance (Data Feeds)Data Science: pandas, pandas_ta, numpy, scikit-learnDatabase: PostgreSQLBackend Framework: Flask (for dashboard and API endpoints)🚀 Key Features1. Multi-Strategy ArchitectureThe bot does not rely on a single algorithm. It supports a "Squad" of strategies:The Trend Surfer: Uses 200-day SMA logic to capture long-term bull runs.The Rubber Band: Uses Bollinger Band Reversion to trade sideways/choppy markets (e.g., NIO).The ML Sniper: (Experimental) Uses Gradient Boosting & Random Forest to predict short-term moves based on volume and momentum.2. Live Trading EngineSafety First: Checks available cash (not margin) and reserves a buffer before every trade.Regime Detection: Automatically detects if a stock is in a "Bull" or "Bear" phase to switch strategies.Execution: Places fractional share orders directly via the Robinhood API.3. 📊 Performance Analysis & Database LoggingCrucially, this project does not just "fire and forget." It stores detailed analytics of every decision into PostgreSQL for audit and refinement.The Data Pipeline:Daily Prediction Log: Every morning, the system runs all strategies (even the inactive ones) and logs their "vote" (Buy/Sell/Hold) to the database.Outcome Tagging: A scheduled cron job runs 24 hours later to fetch the actual stock movement and updates the previous day's log.Statistical Scoring: Queries run against the DB to calculate real-time Sharpe Ratios, Precision, and Drawdown for each strategy.💾 Database SchemaThe application uses a relational schema to track model drift and accuracy over time.strategy_logs TableStores the raw decision from every model, every day.ColumnTypeDescriptionidSERIALPrimary KeytimestampTIMESTAMPTime of analysistickerVARCHAR(10)e.g., "NIO", "NVDA"strategy_nameVARCHAR(50)e.g., "Trend_SMA200", "ML_GradientBoost"signalVARCHAR(10)"BUY", "SELL", "HOLD"confidenceFLOAT0.0 - 1.0 (For ML models)market_priceFLOATPrice at the time of signalperformance_metrics TableUsed for the "Tournament" scoring.ColumnTypeDescriptionlog_idINTForeign Key to strategy_logsactual_return_1dFLOAT% Change of stock next dayactual_return_1wFLOAT% Change of stock next weekis_correctBOOLEANDid the signal match the market direction?🛠️ Installation & Setup1. PrerequisitesPython 3.10+ installed.PostgreSQL installed and running locally or on the cloud.Robinhood Account (Cash account recommended).2. Clone and InstallBashgit clone https://github.com/your-username/Neural-Trade-Robinhood.git
cd Neural-Trade-Robinhood
pip install -r requirements.txt
3. ConfigurationCreate a .env file for your database credentials:BashDB_HOST=localhost
DB_NAME=trading_bot
DB_USER=your_username
DB_PASS=your_password
4. AuthenticationTo generate your Robinhood token (valid for 24 hours):Login to Robinhood Web.Open DevTools -> Application -> Local Storage.Copy the Authorization token.Run:Bashpython update_token.py
(Paste the token when prompted).🏃‍♂️ UsageRunning the Live BotTo execute the active strategy (defaults to "Trend Surfer" for active stocks, "Rubber Band" for choppy ones):Bashpython live_trader.py
Running the Strategy TournamentTo backtest all models and generate a scorecard:Bashpython ml_tournament.py
Visualizing PerformanceTo generate charts showing entries, exits, and wealth curves:Bashpython visualize_strategy.py
⚠️ DisclaimerThis software is for educational purposes only. Do not risk money you cannot afford to lose. The authors are not responsible for any financial losses incurred by using this software.
>>>>>>> e45f9c263b638007496db63d46252ddd5878a57c
