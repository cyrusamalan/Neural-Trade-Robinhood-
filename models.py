from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# --- 1. TRADES TABLE ---
class Trade(db.Model):
    __tablename__ = 'trades'
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    mode = db.Column(db.String(20))
    
    entry_time = db.Column(db.DateTime)
    entry_price = db.Column(db.Float)
    quantity = db.Column(db.Float)
    direction = db.Column(db.String(10))
    
    exit_time = db.Column(db.DateTime)
    exit_price = db.Column(db.Float)
    pnl_dollar = db.Column(db.Float)
    pnl_percent = db.Column(db.Float)
    exit_reason = db.Column(db.String(50))

    # Relationship to Decisions (One Trade can have One Decision)
    # This allows us to say: trade.decision
    decision = db.relationship('ModelDecision', backref='trade', uselist=False, cascade="all, delete-orphan")

# --- 2. DECISIONS TABLE (Parent) ---
class ModelDecision(db.Model):
    __tablename__ = 'model_decisions'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # --- NEW: SYMBOL COLUMN ---
    symbol = db.Column(db.String(10)) 
    
    confidence_score = db.Column(db.Float)
    model_version = db.Column(db.String(50))
    decision_type = db.Column(db.String(20)) # 'BUY', 'HOLD', 'SELL'
    
    # Context Columns
    rsi_at_entry = db.Column(db.Float, nullable=True)
    sma50_at_entry = db.Column(db.Float, nullable=True)
    sma200_at_entry = db.Column(db.Float, nullable=True)
    vix_at_entry = db.Column(db.Float, nullable=True)
    sp500_at_entry = db.Column(db.Float, nullable=True)
    features_used = db.Column(db.Text, nullable=True)

    # Foreign Key to Trade (Optional, for Holds)
    trade_id = db.Column(db.Integer, db.ForeignKey('trades.id'), nullable=True)

    # Relationship to Votes (One Decision has One Vote Set)
    votes = db.relationship('StrategyVote', backref='decision', uselist=False, cascade="all, delete-orphan")

# --- 3. VOTES TABLE (Child) ---
class StrategyVote(db.Model):
    __tablename__ = 'strategy_votes'
    id = db.Column(db.Integer, primary_key=True)
    
    # --- NEW: LINK TO DECISION ---
    decision_id = db.Column(db.Integer, db.ForeignKey('model_decisions.id'))
    
    vote_rsi = db.Column(db.Integer)
    vote_breakout = db.Column(db.Integer)
    vote_heikin = db.Column(db.Integer)
    vote_fib = db.Column(db.Integer)   # We use this for MACD in Python currently
    vote_macd = db.Column(db.Integer, nullable=True) # Optional extra