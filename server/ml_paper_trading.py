#!/usr/bin/env python3
"""
ML Paper Trading Service - 3 Strategy Comparison

실시간으로 3가지 전략 비교:
1. No ML: 모든 신호 진입
2. ML Entry Only: Entry 필터링 (threshold=0.7)
3. ML Combined: Entry 필터링 + Dynamic Exit

실행: python ml_paper_trading.py
SSH에서 백그라운드: nohup python ml_paper_trading.py > paper_trading.log 2>&1 &
"""

import sqlite3
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'cryptoflow.db')
PAPER_DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'ml_paper_trading.db')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'backtest', 'models')

# Configuration
SYMBOL = "BINANCE:BTCUSDT"
HTF = "1h"   # 채널 감지
LTF = "15m"  # 진입
SERVICE_PORT = 5003

# Strategy parameters
TOUCH_THRESHOLD = 0.003
SL_BUFFER_PCT = 0.0008  # Match backtest settings
ENTRY_THRESHOLD = 0.7

# Paper trading parameters
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.015
MAX_LEVERAGE = 15
FEE_PCT = 0.0004

# Labels
EXIT_AT_TP1 = 0
HOLD_FOR_TP2 = 1


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int
    resistance_touches: int
    lowest_low: float
    highest_high: float
    confirmed: bool


@dataclass
class Signal:
    timestamp: str
    direction: str  # LONG or SHORT
    setup_type: str  # BOUNCE or FAKEOUT
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    channel_support: float
    channel_resistance: float
    entry_prob: float = 0.0  # ML Entry probability


@dataclass
class Trade:
    signal_id: int
    strategy: str  # NO_ML, ML_ENTRY, ML_COMBINED
    timestamp: int  # milliseconds
    direction: str
    setup_type: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    channel_support: float
    channel_resistance: float
    status: str = 'ACTIVE'  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT, BE_HIT
    exit_decision: str = ''  # EXIT or HOLD (for ML_COMBINED)
    pnl_pct: float = 0.0
    closed_at: int = 0  # milliseconds
    db_id: int = 0
    tp1_profit_taken: bool = False  # Track if TP1 profit has been added to capital
    entry_candle_time: int = 0  # Candle start time at entry (for lookahead bias prevention)


@dataclass
class StrategyState:
    name: str
    capital: float = INITIAL_CAPITAL
    peak_capital: float = INITIAL_CAPITAL
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    active_trades: List[Trade] = field(default_factory=list)


class MLPaperTradingService:
    def __init__(self):
        self.running = False
        self.channels: Dict[int, Channel] = {}
        self.pending_fakeouts: Dict[str, dict] = {}

        # 3 strategies
        self.strategies = {
            'NO_ML': StrategyState(name='No ML (All Signals)'),
            'ML_ENTRY': StrategyState(name='ML Entry Only (0.7)'),
            'ML_COMBINED': StrategyState(name='ML Entry + Dynamic Exit')
        }

        # Load ML models
        self.entry_model = None
        self.entry_scaler = None
        self.exit_model = None
        self.exit_scaler = None
        self._load_models()

        # Initialize database
        self._init_db()

        # Load state from database (capital, trades, etc.)
        self._load_state()

        # Signal counter
        self.signal_counter = 0

        # Track last processed candle time for candle-close entry (like backtest)
        self.last_processed_candle_time = 0

    def _load_models(self):
        """Load trained ML models."""
        try:
            self.entry_model = joblib.load(os.path.join(MODELS_DIR, 'entry_model.joblib'))
            self.entry_scaler = joblib.load(os.path.join(MODELS_DIR, 'entry_scaler.joblib'))
            self.exit_model = joblib.load(os.path.join(MODELS_DIR, 'exit_model.joblib'))
            self.exit_scaler = joblib.load(os.path.join(MODELS_DIR, 'exit_scaler.joblib'))
            print(f"[ML] Models loaded from {MODELS_DIR}")
        except Exception as e:
            print(f"[ML] Failed to load models: {e}")
            print("[ML] Running without ML models (NO_ML only)")

    def _init_db(self):
        """Initialize SQLite database for paper trading."""
        os.makedirs(os.path.dirname(PAPER_DB_PATH), exist_ok=True)

        conn = sqlite3.connect(PAPER_DB_PATH)
        c = conn.cursor()

        # Signals table
        c.execute('''CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            direction TEXT,
            setup_type TEXT,
            entry_price REAL,
            sl_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            channel_support REAL,
            channel_resistance REAL,
            entry_prob REAL
        )''')

        # Trades table
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            strategy TEXT,
            timestamp INTEGER,
            direction TEXT,
            setup_type TEXT,
            entry_price REAL,
            sl_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            channel_support REAL,
            channel_resistance REAL,
            status TEXT,
            exit_decision TEXT,
            pnl_pct REAL,
            closed_at INTEGER,
            tp1_profit_taken INTEGER DEFAULT 0,
            entry_candle_time INTEGER DEFAULT 0,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )''')

        # Add entry_candle_time column if it doesn't exist (for existing DBs)
        try:
            c.execute('ALTER TABLE trades ADD COLUMN entry_candle_time INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Strategy states table
        c.execute('''CREATE TABLE IF NOT EXISTS strategy_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            strategy TEXT,
            capital REAL,
            peak_capital REAL,
            total_trades INTEGER,
            wins INTEGER,
            losses INTEGER,
            total_pnl REAL,
            max_drawdown REAL
        )''')

        conn.commit()
        conn.close()
        print(f"[DB] Initialized: {PAPER_DB_PATH}")

    def _load_state(self):
        """Load strategy state and active trades from database."""
        try:
            conn = sqlite3.connect(PAPER_DB_PATH)
            c = conn.cursor()

            # Check if tables exist
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_states'")
            if c.fetchone():
                # Load strategy stats from last snapshot
                for strategy_name in self.strategies:
                    c.execute('''SELECT capital, peak_capital, total_trades, wins, losses, total_pnl, max_drawdown
                                 FROM strategy_states WHERE strategy = ? ORDER BY timestamp DESC LIMIT 1''',
                              (strategy_name,))
                    row = c.fetchone()
                    if row:
                        state = self.strategies[strategy_name]
                        state.capital = row[0]
                        state.peak_capital = row[1]
                        state.total_trades = row[2]
                        state.wins = row[3]
                        state.losses = row[4]
                        state.total_pnl = row[5]
                        state.max_drawdown = row[6]
                        print(f"[LOAD] {strategy_name}: capital=${state.capital:,.2f}, trades={state.total_trades}")
            else:
                print("[LOAD] No strategy_states table yet, starting fresh")

            # Check if trades table exists
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if c.fetchone():
                # Load active trades (status != closed)
                for strategy_name, state in self.strategies.items():
                    c.execute('''SELECT id, signal_id, strategy, timestamp, direction, setup_type,
                                        entry_price, sl_price, tp1_price, tp2_price,
                                        channel_support, channel_resistance,
                                        status, exit_decision, pnl_pct, closed_at, tp1_profit_taken,
                                        entry_candle_time
                                 FROM trades WHERE strategy = ? AND status IN ('ACTIVE', 'TP1_HIT')''',
                              (strategy_name,))
                    for row in c.fetchall():
                        trade = Trade(
                            signal_id=row[1],
                            strategy=row[2],
                            timestamp=row[3] or 0,
                            direction=row[4],
                            setup_type=row[5],
                            entry_price=row[6],
                            sl_price=row[7],
                            tp1_price=row[8],
                            tp2_price=row[9],
                            channel_support=row[10] or 0.0,
                            channel_resistance=row[11] or 0.0,
                            status=row[12],
                            exit_decision=row[13] or '',
                            pnl_pct=row[14] or 0.0,
                            closed_at=row[15] or 0,
                            db_id=row[0],
                            tp1_profit_taken=bool(row[16]) if row[16] is not None else False,
                            entry_candle_time=row[17] or 0
                        )
                        state.active_trades.append(trade)
                        print(f"[LOAD] Restored active trade: {trade.direction} {trade.setup_type} @ {trade.entry_price:.2f} ({trade.status})")
            else:
                print("[LOAD] No trades table yet, starting fresh")

            conn.close()
        except Exception as e:
            print(f"[LOAD] Error loading state: {e}, starting fresh")

    def _save_signal(self, signal: Signal) -> int:
        """Save signal to database and return ID."""
        conn = sqlite3.connect(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO signals
            (timestamp, direction, setup_type, entry_price, sl_price, tp1_price, tp2_price,
             channel_support, channel_resistance, entry_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (signal.timestamp, signal.direction, signal.setup_type, signal.entry_price,
             signal.sl_price, signal.tp1_price, signal.tp2_price,
             signal.channel_support, signal.channel_resistance, signal.entry_prob))
        signal_id = c.lastrowid
        conn.commit()
        conn.close()
        return signal_id if signal_id is not None else 0

    def _save_trade(self, trade: Trade) -> int:
        """Save trade to database."""
        conn = sqlite3.connect(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO trades
            (signal_id, strategy, timestamp, direction, setup_type, entry_price, sl_price,
             tp1_price, tp2_price, channel_support, channel_resistance, status,
             exit_decision, pnl_pct, closed_at, tp1_profit_taken, entry_candle_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (trade.signal_id, trade.strategy, trade.timestamp, trade.direction,
             trade.setup_type, trade.entry_price, trade.sl_price, trade.tp1_price,
             trade.tp2_price, trade.channel_support, trade.channel_resistance, trade.status,
             trade.exit_decision, trade.pnl_pct, trade.closed_at,
             1 if trade.tp1_profit_taken else 0, trade.entry_candle_time))
        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        return trade_id if trade_id is not None else 0

    def _update_trade(self, trade: Trade):
        """Update trade in database."""
        conn = sqlite3.connect(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute('''UPDATE trades SET status=?, exit_decision=?, pnl_pct=?, closed_at=?, tp1_profit_taken=?
            WHERE id=?''',
            (trade.status, trade.exit_decision, trade.pnl_pct, trade.closed_at,
             1 if trade.tp1_profit_taken else 0, trade.db_id))
        conn.commit()
        conn.close()

    def _save_strategy_state(self, strategy: str, state: StrategyState):
        """Save strategy state snapshot."""
        conn = sqlite3.connect(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO strategy_states
            (timestamp, strategy, capital, peak_capital, total_trades, wins, losses, total_pnl, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (datetime.now().isoformat(), strategy, state.capital, state.peak_capital,
             state.total_trades, state.wins, state.losses, state.total_pnl, state.max_drawdown))
        conn.commit()
        conn.close()

    def _load_candles(self, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Load candles from database."""
        # Convert timeframe to minutes for table name
        tf_to_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        tf_minutes = tf_to_minutes.get(timeframe, timeframe)

        conn = sqlite3.connect(DB_PATH)
        query = f'''
            SELECT time, open, high, low, close, volume, delta
            FROM candles_{tf_minutes}
            WHERE symbol = ?
            ORDER BY time DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(SYMBOL, limit))
        conn.close()

        if len(df) == 0:
            return pd.DataFrame()

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def _extract_entry_features(self, df_15m: pd.DataFrame, idx: int, channel: Channel,
                                 direction: str, setup_type: str, fakeout_extreme: Optional[float] = None) -> np.ndarray:
        """Extract features for entry model."""
        closes = df_15m['close'].values
        highs = df_15m['high'].values
        lows = df_15m['low'].values
        volumes = df_15m['volume'].values
        deltas = df_15m['delta'].values if 'delta' in df_15m.columns else np.zeros(len(df_15m))

        current_close = closes[idx]
        channel_width = (channel.resistance - channel.support) / channel.support
        price_in_channel = (current_close - channel.support) / (channel.resistance - channel.support)

        # Volume/Delta ratios
        vol_20 = np.mean(volumes[max(0,idx-20):idx]) if idx >= 20 else np.mean(volumes[:idx+1])
        delta_20 = np.mean(deltas[max(0,idx-20):idx]) if idx >= 20 else np.mean(deltas[:idx+1])
        volume_ratio = volumes[idx] / (vol_20 + 1e-10)
        delta_ratio = deltas[idx] / (np.abs(delta_20) + 1e-10)
        cvd_recent = np.sum(deltas[max(0,idx-5):idx+1])

        # ATR
        if idx >= 14:
            tr = np.maximum(highs[idx-14:idx] - lows[idx-14:idx],
                           np.abs(highs[idx-14:idx] - closes[idx-15:idx-1]))
            atr_14 = np.mean(tr)
        else:
            atr_14 = highs[idx] - lows[idx]
        atr_ratio = atr_14 / (current_close + 1e-10)

        # Momentum & RSI
        momentum_5 = (closes[idx] - closes[max(0,idx-5)]) / (closes[max(0,idx-5)] + 1e-10)
        momentum_20 = (closes[idx] - closes[max(0,idx-20)]) / (closes[max(0,idx-20)] + 1e-10)

        # Simple RSI calculation
        if idx >= 15:
            price_changes = np.diff(closes[idx-14:idx+1])
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-10)
            rsi_14 = 100 - (100 / (1 + rs))
        else:
            rsi_14 = 50.0

        # Candle info
        body = abs(closes[idx] - df_15m['open'].values[idx])
        body_size_pct = body / current_close
        candle_range = highs[idx] - lows[idx]
        wick_ratio = (candle_range - body) / (candle_range + 1e-10)
        is_bullish = 1 if closes[idx] > df_15m['open'].values[idx] else 0

        # Time features
        ts = df_15m['time'].iloc[idx] if 'time' in df_15m.columns else df_15m.index[idx]
        if hasattr(ts, 'hour'):
            hour = ts.hour
            day_of_week = ts.dayofweek
        else:
            hour = 12
            day_of_week = 0

        # Fakeout depth
        if fakeout_extreme is not None:
            fakeout_depth = abs(fakeout_extreme - current_close) / current_close
        else:
            fakeout_depth = 0.0

        is_bounce = 1 if setup_type == 'BOUNCE' else 0
        is_long = 1 if direction == 'LONG' else 0

        features = np.array([[
            channel_width, channel.support_touches, channel.resistance_touches,
            channel.support_touches + channel.resistance_touches,
            price_in_channel, volume_ratio, delta_ratio, cvd_recent,
            vol_20, delta_20, atr_14, atr_ratio,
            momentum_5, momentum_20, rsi_14, is_bounce, is_long,
            body_size_pct, wick_ratio, is_bullish, hour, day_of_week,
            fakeout_depth
        ]])

        return features

    def _extract_exit_features(self, df_15m: pd.DataFrame, entry_idx: int, tp1_idx: int,
                                entry_price: float, tp1_price: float, tp2_price: float,
                                channel_width: float, is_long: bool, is_fakeout: bool) -> np.ndarray:
        """Extract features for exit model at TP1 hit."""
        closes = df_15m['close'].values
        highs = df_15m['high'].values
        lows = df_15m['low'].values
        opens = df_15m['open'].values
        volumes = df_15m['volume'].values
        deltas = df_15m['delta'].values if 'delta' in df_15m.columns else np.zeros(len(df_15m))

        candles_to_tp1 = tp1_idx - entry_idx
        time_to_tp1 = candles_to_tp1 * 15

        # Cumulative delta/volume during trade
        cumulative_delta = np.sum(deltas[entry_idx+1:tp1_idx+1])
        cumulative_volume = np.sum(volumes[entry_idx+1:tp1_idx+1])

        # Average before entry
        lookback = 20
        if entry_idx >= lookback:
            avg_delta = np.mean(np.abs(deltas[entry_idx-lookback:entry_idx]))
            avg_volume = np.mean(volumes[entry_idx-lookback:entry_idx])
        else:
            avg_delta = 1
            avg_volume = 1

        delta_ratio = cumulative_delta / (avg_delta * candles_to_tp1 + 1e-10)
        volume_ratio = cumulative_volume / (avg_volume * candles_to_tp1 + 1e-10)

        # Max favorable excursion
        if is_long:
            max_favorable = np.max((highs[entry_idx+1:tp1_idx+1] - entry_price) / entry_price)
        else:
            max_favorable = np.max((entry_price - lows[entry_idx+1:tp1_idx+1]) / entry_price)

        # Momentum at TP1
        if tp1_idx >= 5:
            momentum_at_tp1 = (closes[tp1_idx] - closes[tp1_idx-5]) / closes[tp1_idx-5]
        else:
            momentum_at_tp1 = 0

        # RSI at TP1
        if tp1_idx >= 15:
            price_changes = np.diff(closes[tp1_idx-14:tp1_idx+1])
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            rs = np.mean(gains) / (np.mean(losses) + 1e-10)
            rsi_at_tp1 = 100 - (100 / (1 + rs))
        else:
            rsi_at_tp1 = 50.0

        # ATR at TP1
        if tp1_idx >= 14:
            tr = np.maximum(highs[tp1_idx-14:tp1_idx] - lows[tp1_idx-14:tp1_idx],
                           np.abs(highs[tp1_idx-14:tp1_idx] - closes[tp1_idx-15:tp1_idx-1]))
            atr_at_tp1 = np.mean(tr)
        else:
            atr_at_tp1 = 0

        price_vs_tp1 = (closes[tp1_idx] - tp1_price) / tp1_price

        if is_long:
            distance_to_tp2 = (tp2_price - tp1_price) / tp1_price
        else:
            distance_to_tp2 = (tp1_price - tp2_price) / tp1_price

        last_body = abs(closes[tp1_idx] - opens[tp1_idx])
        last_body_pct = last_body / closes[tp1_idx]
        last_is_bullish = 1 if closes[tp1_idx] > opens[tp1_idx] else 0

        features = np.array([[
            candles_to_tp1, time_to_tp1, cumulative_delta, cumulative_volume,
            delta_ratio, volume_ratio, momentum_at_tp1, rsi_at_tp1,
            atr_at_tp1, max_favorable, price_vs_tp1, distance_to_tp2,
            channel_width, last_body_pct, last_is_bullish,
            1 if is_long else 0, 1 if is_fakeout else 0
        ]])

        return features

    def _predict_entry(self, features: np.ndarray) -> Tuple[bool, float]:
        """Predict entry decision using ML model."""
        if self.entry_model is None or self.entry_scaler is None:
            return True, 1.0

        features_scaled = self.entry_scaler.transform(features)
        prob = self.entry_model.predict_proba(features_scaled)[0, 1]
        take = prob >= ENTRY_THRESHOLD
        return take, prob

    def _predict_exit(self, features: np.ndarray) -> str:
        """Predict exit decision using ML model."""
        if self.exit_model is None or self.exit_scaler is None:
            return 'HOLD'

        features_scaled = self.exit_scaler.transform(features)
        pred = self.exit_model.predict(features_scaled)[0]
        return 'EXIT' if pred == EXIT_AT_TP1 else 'HOLD'

    def _process_signal(self, signal: Signal, entry_candle_time: int = 0):
        """Process a new signal for all strategies."""
        self.signal_counter += 1
        signal_id = self._save_signal(signal)

        now_ms = int(datetime.now().timestamp() * 1000)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*60}")
        print(f"[SIGNAL #{self.signal_counter}] {signal.direction} {signal.setup_type}")
        print(f"  Time: {now_str}")
        print(f"  Entry: {signal.entry_price:.2f}")
        print(f"  SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f} | TP2: {signal.tp2_price:.2f}")
        print(f"  ML Entry Prob: {signal.entry_prob:.1%}")
        print(f"{'='*60}")

        # Strategy 1: NO_ML - Always enter
        trade_no_ml = Trade(
            signal_id=signal_id,
            strategy='NO_ML',
            timestamp=now_ms,
            direction=signal.direction,
            setup_type=signal.setup_type,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            channel_support=signal.channel_support,
            channel_resistance=signal.channel_resistance,
            entry_candle_time=entry_candle_time
        )
        trade_no_ml.db_id = self._save_trade(trade_no_ml)
        self.strategies['NO_ML'].active_trades.append(trade_no_ml)
        self.strategies['NO_ML'].total_trades += 1
        print(f"  [NO_ML] Trade opened")

        # Strategy 2 & 3: Check ML Entry
        if signal.entry_prob >= ENTRY_THRESHOLD:
            # ML_ENTRY
            trade_ml_entry = Trade(
                signal_id=signal_id,
                strategy='ML_ENTRY',
                timestamp=now_ms,
                direction=signal.direction,
                setup_type=signal.setup_type,
                entry_price=signal.entry_price,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                channel_support=signal.channel_support,
                channel_resistance=signal.channel_resistance,
                entry_candle_time=entry_candle_time
            )
            trade_ml_entry.db_id = self._save_trade(trade_ml_entry)
            self.strategies['ML_ENTRY'].active_trades.append(trade_ml_entry)
            self.strategies['ML_ENTRY'].total_trades += 1
            print(f"  [ML_ENTRY] Trade opened (prob={signal.entry_prob:.1%} >= {ENTRY_THRESHOLD:.1%})")

            # ML_COMBINED
            trade_ml_combined = Trade(
                signal_id=signal_id,
                strategy='ML_COMBINED',
                timestamp=now_ms,
                direction=signal.direction,
                setup_type=signal.setup_type,
                entry_price=signal.entry_price,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                channel_support=signal.channel_support,
                channel_resistance=signal.channel_resistance,
                entry_candle_time=entry_candle_time
            )
            trade_ml_combined.db_id = self._save_trade(trade_ml_combined)
            self.strategies['ML_COMBINED'].active_trades.append(trade_ml_combined)
            self.strategies['ML_COMBINED'].total_trades += 1
            print(f"  [ML_COMBINED] Trade opened")
        else:
            print(f"  [ML_ENTRY] SKIPPED (prob={signal.entry_prob:.1%} < {ENTRY_THRESHOLD:.1%})")
            print(f"  [ML_COMBINED] SKIPPED")

    def _update_trades(self, current_price: float, current_high: float, current_low: float,
                       df_15m: Optional[pd.DataFrame] = None, current_idx: Optional[int] = None):
        """Update all active trades with current price."""
        now_ms = int(datetime.now().timestamp() * 1000)

        # Get current 15m candle time for lookahead bias prevention
        current_candle_time = 0
        if df_15m is not None and len(df_15m) > 0:
            current_candle_time = int(df_15m['time'].iloc[-1].timestamp() * 1000)

        for strategy_name, state in self.strategies.items():
            trades_to_remove = []

            for trade in state.active_trades:
                is_long = trade.direction == 'LONG'
                closed = False
                pnl_pct = 0.0

                # Lookahead bias prevention: skip TP/SL check if still in entry candle
                if trade.entry_candle_time > 0 and current_candle_time <= trade.entry_candle_time:
                    # Still in the entry candle, skip TP/SL check to prevent lookahead bias
                    continue

                if trade.status == 'ACTIVE':
                    # Check SL
                    if is_long and current_low <= trade.sl_price:
                        trade.status = 'SL_HIT'
                        pnl_pct = (trade.sl_price - trade.entry_price) / trade.entry_price
                        closed = True
                    elif not is_long and current_high >= trade.sl_price:
                        trade.status = 'SL_HIT'
                        pnl_pct = (trade.entry_price - trade.sl_price) / trade.entry_price
                        closed = True
                    # Check TP1
                    elif is_long and current_high >= trade.tp1_price:
                        trade.status = 'TP1_HIT'
                        # Immediately take 50% profit at TP1
                        if not trade.tp1_profit_taken:
                            tp1_pnl_pct = 0.5 * (trade.tp1_price - trade.entry_price) / trade.entry_price
                            sl_dist = abs(trade.entry_price - trade.sl_price) / trade.entry_price
                            leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
                            position = state.capital * leverage
                            tp1_pnl_dollar = position * tp1_pnl_pct - position * FEE_PCT  # Only 1 fee for partial close
                            state.capital += tp1_pnl_dollar
                            state.total_pnl += tp1_pnl_dollar
                            trade.tp1_profit_taken = True
                            print(f"\n[{strategy_name}] TP1 HIT - 50% profit taken!")
                            print(f"  TP1 PnL: {tp1_pnl_pct*100:+.2f}% (${tp1_pnl_dollar:+.2f})")
                            print(f"  Capital: ${state.capital:,.2f}")
                        # For ML_COMBINED, decide exit at TP1
                        if strategy_name == 'ML_COMBINED' and self.exit_model is not None and df_15m is not None:
                            trade.exit_decision = 'HOLD'  # Default to HOLD for now
                    elif not is_long and current_low <= trade.tp1_price:
                        trade.status = 'TP1_HIT'
                        # Immediately take 50% profit at TP1
                        if not trade.tp1_profit_taken:
                            tp1_pnl_pct = 0.5 * (trade.entry_price - trade.tp1_price) / trade.entry_price
                            sl_dist = abs(trade.entry_price - trade.sl_price) / trade.entry_price
                            leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
                            position = state.capital * leverage
                            tp1_pnl_dollar = position * tp1_pnl_pct - position * FEE_PCT
                            state.capital += tp1_pnl_dollar
                            state.total_pnl += tp1_pnl_dollar
                            trade.tp1_profit_taken = True
                            print(f"\n[{strategy_name}] TP1 HIT - 50% profit taken!")
                            print(f"  TP1 PnL: {tp1_pnl_pct*100:+.2f}% (${tp1_pnl_dollar:+.2f})")
                            print(f"  Capital: ${state.capital:,.2f}")
                        if strategy_name == 'ML_COMBINED' and self.exit_model is not None:
                            trade.exit_decision = 'HOLD'

                elif trade.status == 'TP1_HIT':
                    # Check TP2 - only remaining 50% profit
                    if is_long and current_high >= trade.tp2_price:
                        trade.status = 'TP2_HIT'
                        pnl_pct = 0.5 * (trade.tp2_price - trade.entry_price) / trade.entry_price  # Only remaining 50%
                        closed = True
                    elif not is_long and current_low <= trade.tp2_price:
                        trade.status = 'TP2_HIT'
                        pnl_pct = 0.5 * (trade.entry_price - trade.tp2_price) / trade.entry_price  # Only remaining 50%
                        closed = True
                    # Check BE (breakeven stop after TP1) - remaining 50% exits at 0
                    elif is_long and current_low <= trade.entry_price:
                        trade.status = 'BE_HIT'
                        pnl_pct = 0  # Remaining 50% exits at breakeven
                        closed = True
                    elif not is_long and current_high >= trade.entry_price:
                        trade.status = 'BE_HIT'
                        pnl_pct = 0  # Remaining 50% exits at breakeven
                        closed = True

                if closed:
                    trade.pnl_pct = pnl_pct
                    trade.closed_at = now_ms

                    # Update capital for remaining position
                    sl_dist = abs(trade.entry_price - trade.sl_price) / trade.entry_price
                    leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
                    position = state.capital * leverage

                    # Fee calculation: 1 fee if TP1 was hit (partial close already paid 1 fee), else 2 fees for full close
                    fees = FEE_PCT if trade.tp1_profit_taken else FEE_PCT * 2
                    pnl_dollar = position * pnl_pct - position * fees

                    state.capital += pnl_dollar
                    state.capital = max(state.capital, 0)
                    state.total_pnl += pnl_dollar

                    # Win/Loss counting: if TP1 was hit, it's a win regardless of final exit
                    if trade.tp1_profit_taken:
                        state.wins += 1  # TP1 hit = win (even if BE after)
                    elif pnl_dollar > 0:
                        state.wins += 1
                    else:
                        state.losses += 1

                    if state.capital > state.peak_capital:
                        state.peak_capital = state.capital

                    dd = (state.peak_capital - state.capital) / state.peak_capital if state.peak_capital > 0 else 0
                    state.max_drawdown = max(state.max_drawdown, dd)

                    self._update_trade(trade)
                    trades_to_remove.append(trade)

                    win_rate = state.wins / (state.wins + state.losses) * 100 if (state.wins + state.losses) > 0 else 0
                    print(f"\n[{strategy_name}] Trade closed: {trade.status}")
                    print(f"  Remaining PnL: {pnl_pct*100:+.2f}% (${pnl_dollar:+.2f})")
                    print(f"  Capital: ${state.capital:,.2f} | WR: {win_rate:.1f}% | MaxDD: {state.max_drawdown*100:.1f}%")

            for trade in trades_to_remove:
                state.active_trades.remove(trade)

    def _find_swing_points(self, candles: pd.DataFrame, confirm_candles: int = 3):
        """Find swing highs and lows."""
        if len(candles) < confirm_candles + 1:
            return [], []

        highs = candles['high'].values
        lows = candles['low'].values
        times = candles['time'].values if 'time' in candles.columns else candles.index.values

        swing_highs = []
        swing_lows = []

        potential_high_idx = 0
        potential_high_price = highs[0]
        candles_since_high = 0

        potential_low_idx = 0
        potential_low_price = lows[0]
        candles_since_low = 0

        for i in range(1, len(candles)):
            if highs[i] > potential_high_price:
                potential_high_idx = i
                potential_high_price = highs[i]
                candles_since_high = 0
            else:
                candles_since_high += 1
                if candles_since_high == confirm_candles:
                    swing_highs.append({'idx': potential_high_idx, 'price': potential_high_price, 'time': times[potential_high_idx]})

            if lows[i] < potential_low_price:
                potential_low_idx = i
                potential_low_price = lows[i]
                candles_since_low = 0
            else:
                candles_since_low += 1
                if candles_since_low == confirm_candles:
                    swing_lows.append({'idx': potential_low_idx, 'price': potential_low_price, 'time': times[potential_low_idx]})

            if candles_since_high >= confirm_candles:
                potential_high_price = highs[i]
                potential_high_idx = i
                candles_since_high = 0

            if candles_since_low >= confirm_candles:
                potential_low_price = lows[i]
                potential_low_idx = i
                candles_since_low = 0

        return swing_highs, swing_lows

    def _update_channel(self, df_1h: pd.DataFrame) -> Optional[Channel]:
        """Update channel detection."""
        if len(df_1h) < 20:
            return None

        swing_highs, swing_lows = self._find_swing_points(df_1h)

        print(f"[CHANNEL] Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        if swing_highs:
            print(f"[CHANNEL] Highs: {[round(s['price']) for s in swing_highs[-5:]]}")
        if swing_lows:
            print(f"[CHANNEL] Lows: {[round(s['price']) for s in swing_lows[-5:]]}")

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            print(f"[CHANNEL] Not enough swing points")
            return None

        current_close = df_1h['close'].iloc[-1]
        best_channel = None
        best_score = -1

        for sh in swing_highs[-10:]:
            for sl in swing_lows[-10:]:
                if sh['price'] <= sl['price']:
                    continue

                width_pct = (sh['price'] - sl['price']) / sl['price']
                if width_pct < 0.008 or width_pct > 0.05:
                    continue

                if current_close < sl['price'] * 0.98 or current_close > sh['price'] * 1.02:
                    continue

                # Use 0.4% tolerance for touches (matches ML training)
                touch_tolerance = 0.004
                support_touches = sum(1 for s in swing_lows if abs(s['price'] - sl['price']) / sl['price'] < touch_tolerance)
                resistance_touches = sum(1 for s in swing_highs if abs(s['price'] - sh['price']) / sh['price'] < touch_tolerance)

                confirmed = support_touches >= 2 and resistance_touches >= 2

                if support_touches >= 1 and resistance_touches >= 1:
                    print(f"[CHANNEL] Candidate: S={sl['price']:.0f}({support_touches}) R={sh['price']:.0f}({resistance_touches}) width={width_pct*100:.1f}% confirmed={confirmed}")

                if confirmed:
                    score = support_touches + resistance_touches
                    if score > best_score:
                        best_score = score
                        lowest = min(s['price'] for s in swing_lows[-20:]) if swing_lows else sl['price']
                        highest = max(s['price'] for s in swing_highs[-20:]) if swing_highs else sh['price']
                        best_channel = Channel(
                            support=sl['price'],
                            resistance=sh['price'],
                            support_touches=support_touches,
                            resistance_touches=resistance_touches,
                            lowest_low=lowest,
                            highest_high=highest,
                            confirmed=True
                        )

        return best_channel

    def _check_fakeout(self, df_1h: pd.DataFrame, channel: Channel) -> Optional[dict]:
        """Check for fakeout signals."""
        if not channel or not channel.confirmed:
            return None

        current_close = df_1h['close'].iloc[-1]
        current_high = df_1h['high'].iloc[-1]
        current_low = df_1h['low'].iloc[-1]

        # Check pending fakeouts
        for key in list(self.pending_fakeouts.keys()):
            pf = self.pending_fakeouts[key]
            candles_since = len(df_1h) - 1 - pf['break_idx']

            if candles_since > 5:
                del self.pending_fakeouts[key]
                continue

            if pf['type'] == 'bear':
                pf['extreme'] = min(pf['extreme'], current_low)
                if current_close > channel.support:
                    del self.pending_fakeouts[key]
                    return {'type': 'bear', 'extreme': pf['extreme'], 'channel': channel}
            else:
                pf['extreme'] = max(pf['extreme'], current_high)
                if current_close < channel.resistance:
                    del self.pending_fakeouts[key]
                    return {'type': 'bull', 'extreme': pf['extreme'], 'channel': channel}

        # Check new breakouts
        if current_close < channel.support * 0.997:
            if 'bear' not in self.pending_fakeouts:
                self.pending_fakeouts['bear'] = {
                    'type': 'bear',
                    'break_idx': len(df_1h) - 1,
                    'extreme': current_low
                }
        elif current_close > channel.resistance * 1.003:
            if 'bull' not in self.pending_fakeouts:
                self.pending_fakeouts['bull'] = {
                    'type': 'bull',
                    'break_idx': len(df_1h) - 1,
                    'extreme': current_high
                }

        return None

    def _scan_for_signals(self):
        """Scan for new trading signals."""
        df_1h = self._load_candles(HTF, 200)
        df_15m = self._load_candles(LTF, 500)

        print(f"[SCAN] Loaded {len(df_1h)} 1h candles, {len(df_15m)} 15m candles")

        if len(df_1h) < 20 or len(df_15m) < 20:
            print(f"[SCAN] Not enough data (need 20+)")
            return

        # Update channel
        current_price = df_1h['close'].iloc[-1]

        # Invalidate old channel if price moved too far (>3% away)
        if hasattr(self, 'current_channel') and self.current_channel:
            ch = self.current_channel
            if current_price < ch.support * 0.97 or current_price > ch.resistance * 1.03:
                print(f"[CHANNEL] Price {current_price:.0f} out of range, invalidating old channel {ch.support:.0f}-{ch.resistance:.0f}")
                self.current_channel = None

        channel = self._update_channel(df_1h)
        if channel:
            self.current_channel = channel
            print(f"[CHANNEL] Detected: {channel.support:.0f} - {channel.resistance:.0f} (touches: S={channel.support_touches}, R={channel.resistance_touches})")

        if not hasattr(self, 'current_channel') or self.current_channel is None:
            print(f"[SCAN] No channel detected yet")
            return

        channel = self.current_channel
        current_close = df_15m['close'].iloc[-1]
        current_high = df_15m['high'].iloc[-1]
        current_low = df_15m['low'].iloc[-1]
        current_candle_time = int(df_15m['time'].iloc[-1].timestamp() * 1000)

        # Save current price for API
        self.last_price = current_close

        # Update existing trades with current candle data
        self._update_trades(current_close, current_high, current_low, df_15m, len(df_15m)-1)

        # ===== CANDLE-CLOSE ENTRY (like backtest) =====
        # Only check for new signals when a NEW candle starts
        # This means the previous candle just completed
        if current_candle_time == self.last_processed_candle_time:
            # Same candle, skip signal detection
            return

        # First scan after startup - just record current candle time, don't process signals
        if self.last_processed_candle_time == 0:
            print(f"[SCAN] First scan - recording candle time, waiting for next candle close")
            self.last_processed_candle_time = current_candle_time
            return

        # New candle detected - check the COMPLETED candle (index -2)
        if len(df_15m) < 3:
            self.last_processed_candle_time = current_candle_time
            return

        # Use the completed candle (previous candle, index -2)
        completed_idx = len(df_15m) - 2
        completed_close = df_15m['close'].iloc[completed_idx]
        completed_high = df_15m['high'].iloc[completed_idx]
        completed_low = df_15m['low'].iloc[completed_idx]
        completed_candle_time = int(df_15m['time'].iloc[completed_idx].timestamp() * 1000)

        print(f"[SCAN] New candle detected! Checking completed candle at {df_15m['time'].iloc[completed_idx]}")

        mid_price = (channel.resistance + channel.support) / 2

        # Check for bounce signals on COMPLETED candle
        signal_key = f"{round(channel.support)}_{round(channel.resistance)}_{completed_candle_time}"
        if not hasattr(self, 'recent_signals'):
            self.recent_signals = set()

        if signal_key not in self.recent_signals:
            # Support bounce → LONG (on completed candle)
            if completed_low <= channel.support * (1 + TOUCH_THRESHOLD) and completed_close > channel.support:
                entry = completed_close  # Enter at completed candle's close
                sl = channel.support * (1 - SL_BUFFER_PCT)
                tp1 = mid_price
                tp2 = channel.resistance * 0.998

                if entry > sl and tp1 > entry:
                    features = self._extract_entry_features(df_15m, completed_idx, channel, 'LONG', 'BOUNCE', None)
                    take, prob = self._predict_entry(features)

                    signal = Signal(
                        timestamp=str(completed_candle_time),
                        direction='LONG',
                        setup_type='BOUNCE',
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        channel_support=channel.support,
                        channel_resistance=channel.resistance,
                        entry_prob=prob
                    )
                    # Use current candle time for entry_candle_time (TP/SL checked from next candle)
                    self._process_signal(signal, current_candle_time)
                    self.recent_signals.add(signal_key)

            # Resistance bounce → SHORT (on completed candle)
            elif completed_high >= channel.resistance * (1 - TOUCH_THRESHOLD) and completed_close < channel.resistance:
                entry = completed_close  # Enter at completed candle's close
                sl = channel.resistance * (1 + SL_BUFFER_PCT)
                tp1 = mid_price
                tp2 = channel.support * 1.002

                if sl > entry and entry > tp1:
                    features = self._extract_entry_features(df_15m, completed_idx, channel, 'SHORT', 'BOUNCE', None)
                    take, prob = self._predict_entry(features)

                    signal = Signal(
                        timestamp=str(completed_candle_time),
                        direction='SHORT',
                        setup_type='BOUNCE',
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        channel_support=channel.support,
                        channel_resistance=channel.resistance,
                        entry_prob=prob
                    )
                    self._process_signal(signal, current_candle_time)
                    self.recent_signals.add(signal_key)

        # Update last processed candle time
        self.last_processed_candle_time = current_candle_time

        # Clean old signal keys
        if len(self.recent_signals) > 100:
            self.recent_signals = set(list(self.recent_signals)[-50:])

    def run(self):
        """Main service loop."""
        self.running = True
        print(f"\n{'='*60}")
        print("  ML PAPER TRADING SERVICE STARTED")
        print(f"  Strategies: NO_ML, ML_ENTRY (0.7), ML_COMBINED")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"{'='*60}\n")

        # Start HTTP server in background
        server_thread = threading.Thread(target=self._run_http_server, daemon=True)
        server_thread.start()
        print(f"[HTTP] Status server started on port {SERVICE_PORT}")
        print(f"       http://localhost:{SERVICE_PORT}/status")

        scan_interval = 60  # 1 minute
        last_scan = 0

        while self.running:
            try:
                now = time.time()

                if now - last_scan >= scan_interval:
                    self._scan_for_signals()
                    last_scan = now

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n[SERVICE] Shutting down...")
                self.running = False
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        # Save final states
        for name, state in self.strategies.items():
            self._save_strategy_state(name, state)

        print("[SERVICE] Stopped")

    def _run_http_server(self):
        """Run HTTP server for status endpoint."""
        service = self

        class StatusHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logs

            def do_GET(self):
                if self.path.startswith('/status'):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    status = {
                        'timestamp': datetime.now().isoformat(),
                        'channel': None,
                        'current_price': None,
                        'strategies': {}
                    }

                    # Add channel info if available
                    if hasattr(service, 'current_channel') and service.current_channel:
                        ch = service.current_channel
                        status['channel'] = {
                            'support': round(ch.support, 2),
                            'resistance': round(ch.resistance, 2),
                            'support_touches': ch.support_touches,
                            'resistance_touches': ch.resistance_touches,
                            'width_pct': round((ch.resistance - ch.support) / ch.support * 100, 2)
                        }

                    # Add current price if available
                    if hasattr(service, 'last_price') and service.last_price:
                        status['current_price'] = round(service.last_price, 2)

                    for name, state in service.strategies.items():
                        win_rate = state.wins / (state.wins + state.losses) * 100 if (state.wins + state.losses) > 0 else 0

                        # Include active trade details
                        active_trade_details = []
                        for trade in state.active_trades:
                            # Calculate unrealized P&L
                            current_price = service.last_price if hasattr(service, 'last_price') and service.last_price else trade.entry_price
                            if trade.direction == 'LONG':
                                unrealized_pnl = (current_price - trade.entry_price) / trade.entry_price * 100
                            else:
                                unrealized_pnl = (trade.entry_price - current_price) / trade.entry_price * 100

                            active_trade_details.append({
                                'signal_id': trade.signal_id,
                                'direction': trade.direction,
                                'setup_type': trade.setup_type,
                                'entry_price': round(trade.entry_price, 2),
                                'sl_price': round(trade.sl_price, 2),
                                'tp1_price': round(trade.tp1_price, 2),
                                'tp2_price': round(trade.tp2_price, 2),
                                'channel_support': round(trade.channel_support, 2),
                                'channel_resistance': round(trade.channel_resistance, 2),
                                'status': trade.status,
                                'timestamp': trade.timestamp,
                                'exit_decision': trade.exit_decision,
                                'pnl_pct': round(trade.pnl_pct, 4),
                                'tp1_profit_taken': trade.tp1_profit_taken,
                                'unrealized_pnl': round(unrealized_pnl, 2)
                            })

                        status['strategies'][name] = {
                            'name': state.name,
                            'capital': round(state.capital, 2),
                            'return_pct': round((state.capital / INITIAL_CAPITAL - 1) * 100, 2),
                            'total_trades': state.total_trades,
                            'wins': state.wins,
                            'losses': state.losses,
                            'win_rate': round(win_rate, 1),
                            'max_drawdown': round(state.max_drawdown * 100, 1),
                            'active_trades': len(state.active_trades),
                            'active_trade_details': active_trade_details
                        }

                    # Add recent signals from database
                    try:
                        conn = sqlite3.connect(PAPER_DB_PATH)
                        c = conn.cursor()
                        c.execute('''SELECT timestamp, direction, setup_type, entry_price, sl_price, tp1_price, tp2_price, entry_prob, channel_support, channel_resistance
                                     FROM signals ORDER BY id DESC LIMIT 10''')
                        recent_signals = []
                        for row in c.fetchall():
                            # Convert timestamp to int (milliseconds) for frontend
                            ts = row[0]
                            try:
                                ts = int(ts)
                            except (ValueError, TypeError):
                                ts = 0
                            recent_signals.append({
                                'timestamp': ts,
                                'direction': row[1],
                                'setup_type': row[2],
                                'entry_price': row[3],
                                'sl_price': row[4],
                                'tp1_price': row[5],
                                'tp2_price': row[6],
                                'entry_prob': row[7],
                                'channel_support': row[8],
                                'channel_resistance': row[9]
                            })
                        conn.close()
                        status['recent_signals'] = recent_signals
                    except Exception as e:
                        status['recent_signals'] = []

                    self.wfile.write(json.dumps(status, indent=2).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer(('', SERVICE_PORT), StatusHandler)
        server.serve_forever()

    def print_status(self):
        """Print current status of all strategies."""
        print(f"\n{'='*70}")
        print(f"  PAPER TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"{'Strategy':<25} {'Capital':>12} {'Return':>10} {'Trades':>8} {'WR':>8} {'MaxDD':>8}")
        print(f"{'-'*70}")

        for name, state in self.strategies.items():
            win_rate = state.wins / (state.wins + state.losses) * 100 if (state.wins + state.losses) > 0 else 0
            ret = (state.capital / INITIAL_CAPITAL - 1) * 100
            print(f"{state.name:<25} ${state.capital:>10,.2f} {ret:>+9.1f}% {state.total_trades:>8} {win_rate:>7.1f}% {state.max_drawdown*100:>7.1f}%")

        print(f"{'='*70}\n")


def main():
    service = MLPaperTradingService()
    service.run()


if __name__ == "__main__":
    main()
