#!/usr/bin/env python3
"""
Paper Trading Service - NO_ML Only

실시간으로 No ML 전략만 실행합니다.

실행: python ml_paper_trading.py
SSH에서 백그라운드: nohup python ml_paper_trading.py > paper_trading.log 2>&1 &
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import os
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared.channel_builder import build_channels as _build_htf_map
from shared.channel_builder import Channel as SharedChannel

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "data", "cryptoflow.db")
PAPER_DB_PATH = os.path.join(SCRIPT_DIR, "data", "ml_paper_trading.db")

# DB connection helper with timeout to prevent corruption
# Keep timeout short for real-time trading - retry on next scan if busy
DB_TIMEOUT = 2  # seconds - unified across all services


def get_db_connection(db_path, readonly=False):
    """Create DB connection with proper timeout and WAL mode

    Args:
        db_path: Path to the database file
        readonly: If True, open in read-only mode (safer for shared DBs)
    """
    if readonly:
        # Read-only URI mode - prevents write locks and corruption
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=DB_TIMEOUT)
    else:
        conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT)
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=2000")  # 2s - unified across all services
    return conn


def get_readonly_connection(db_path):
    """Convenience function for read-only DB access (use for cryptoflow.db)"""
    return get_db_connection(db_path, readonly=True)


# Configuration
SYMBOL = "BINANCE:BTCUSDT"
HTF = "1h"  # 채널 감지
LTF = "15m"  # 진입
SERVICE_PORT = 5003

# Strategy parameters
TOUCH_THRESHOLD = 0.003
SL_BUFFER_PCT = 0.0008  # Match backtest settings

# Paper trading parameters
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.015
MAX_LEVERAGE = 20
FEE_PCT = 0.0004
MAX_HOLD_CANDLES = 150


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
    entry_prob: float = 0.0  # Entry probability (unused)


@dataclass
class Trade:
    signal_id: int
    strategy: str  # NO_ML
    timestamp: int  # milliseconds
    direction: str
    setup_type: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    channel_support: float
    channel_resistance: float
    status: str = "ACTIVE"  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT, BE_HIT
    exit_decision: str = ""
    pnl_pct: float = 0.0
    closed_at: int = 0  # milliseconds
    db_id: int = 0
    tp1_profit_taken: bool = False  # Track if TP1 was hit (backtest accounting)
    entry_candle_time: int = (
        0  # Candle start time at entry (for lookahead bias prevention)
    )
    entry_capital: float = 0.0  # Capital at entry (backtest accounting)


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
        self.pending_fakeouts: Dict[str, dict] = {}

        # Strategy (NO_ML only)
        self.strategies = {"NO_ML": StrategyState(name="No ML (All Signals)")}

        # Initialize database
        self._init_db()

        # Load state from database (capital, trades, etc.)
        self._load_state()

        # Signal counter
        self.signal_counter = 0

        # Track last processed candle time for candle-close entry (like backtest)
        self.last_processed_candle_time = 0

        # Load recent signals from DB to prevent duplicate entries after restart
        self.recent_signals = self._load_recent_signal_keys()

    def _init_db(self):
        """Initialize SQLite database for paper trading."""
        os.makedirs(os.path.dirname(PAPER_DB_PATH), exist_ok=True)

        conn = get_db_connection(PAPER_DB_PATH)
        c = conn.cursor()

        # Signals table
        c.execute("""CREATE TABLE IF NOT EXISTS signals (
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
        )""")

        # Trades table
        c.execute("""CREATE TABLE IF NOT EXISTS trades (
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
            entry_capital REAL DEFAULT 0,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )""")

        # Add entry_candle_time column if it doesn't exist (for existing DBs)
        try:
            c.execute(
                "ALTER TABLE trades ADD COLUMN entry_candle_time INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            c.execute("ALTER TABLE trades ADD COLUMN entry_capital REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Strategy states table
        c.execute("""CREATE TABLE IF NOT EXISTS strategy_states (
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
        )""")

        conn.commit()
        conn.close()
        print(f"[DB] Initialized: {PAPER_DB_PATH}")

    def _load_state(self):
        """Load strategy state and active trades from database."""
        try:
            conn = get_db_connection(PAPER_DB_PATH)
            c = conn.cursor()

            # Check if tables exist
            c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_states'"
            )
            if c.fetchone():
                # Load strategy stats from last snapshot
                for strategy_name in self.strategies:
                    c.execute(
                        """SELECT capital, peak_capital, total_trades, wins, losses, total_pnl, max_drawdown
                                 FROM strategy_states WHERE strategy = ? ORDER BY timestamp DESC LIMIT 1""",
                        (strategy_name,),
                    )
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
                        print(
                            f"[LOAD] {strategy_name}: capital=${state.capital:,.2f}, trades={state.total_trades}"
                        )
            else:
                print("[LOAD] No strategy_states table yet, starting fresh")

            # Check if trades table exists
            c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
            )
            if c.fetchone():
                # Load active trades (status != closed)
                for strategy_name, state in self.strategies.items():
                    c.execute(
                        """SELECT id, signal_id, strategy, timestamp, direction, setup_type,
                                        entry_price, sl_price, tp1_price, tp2_price,
                                        channel_support, channel_resistance,
                                        status, exit_decision, pnl_pct, closed_at, tp1_profit_taken,
                                        entry_candle_time, entry_capital
                                 FROM trades WHERE strategy = ? AND status IN ('ACTIVE', 'TP1_HIT')""",
                        (strategy_name,),
                    )
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
                            exit_decision=row[13] or "",
                            pnl_pct=row[14] or 0.0,
                            closed_at=row[15] or 0,
                            db_id=row[0],
                            tp1_profit_taken=bool(row[16])
                            if row[16] is not None
                            else False,
                            entry_candle_time=row[17] or 0,
                            entry_capital=row[18]
                            if len(row) > 18 and row[18] is not None
                            else 0.0,
                        )
                        if trade.entry_capital <= 0:
                            trade.entry_capital = state.capital
                        state.active_trades.append(trade)
                        print(
                            f"[LOAD] Restored active trade: {trade.direction} {trade.setup_type} @ {trade.entry_price:.2f} ({trade.status})"
                        )
            else:
                print("[LOAD] No trades table yet, starting fresh")

            conn.close()
        except Exception as e:
            print(f"[LOAD] Error loading state: {e}, starting fresh")

    def _load_recent_signal_keys(self) -> set:
        """Load recent signal keys from DB to prevent duplicate entries after restart."""
        SIGNAL_COOLDOWN_MS = 20 * 15 * 60 * 1000  # 5 hours
        recent_keys = set()

        try:
            conn = get_db_connection(PAPER_DB_PATH)
            c = conn.cursor()
            c.execute(
                """SELECT timestamp, channel_support, channel_resistance 
                   FROM signals ORDER BY id DESC LIMIT 100"""
            )
            for row in c.fetchall():
                try:
                    ts = int(row[0])
                    support = round(row[1])
                    resistance = round(row[2])
                    key = f"{support}_{resistance}_{ts // SIGNAL_COOLDOWN_MS}"
                    recent_keys.add(key)
                except (ValueError, TypeError):
                    pass
            conn.close()
            print(f"[LOAD] Loaded {len(recent_keys)} recent signal keys")
        except Exception as e:
            print(f"[LOAD] Error loading signal keys: {e}")

        return recent_keys

    def _save_signal(self, signal: Signal) -> int:
        """Save signal to database and return ID. Returns -1 if duplicate."""
        conn = get_db_connection(PAPER_DB_PATH)
        c = conn.cursor()

        # Check for duplicate signal (same timestamp and direction)
        c.execute(
            """SELECT id FROM signals
                     WHERE timestamp = ? AND direction = ? AND entry_price = ?""",
            (signal.timestamp, signal.direction, signal.entry_price),
        )
        existing = c.fetchone()
        if existing:
            conn.close()
            print(
                f"[SIGNAL] Duplicate detected, skipping (timestamp={signal.timestamp})"
            )
            return -1  # Signal duplicate marker

        c.execute(
            """INSERT INTO signals
            (timestamp, direction, setup_type, entry_price, sl_price, tp1_price, tp2_price,
             channel_support, channel_resistance, entry_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.timestamp,
                signal.direction,
                signal.setup_type,
                signal.entry_price,
                signal.sl_price,
                signal.tp1_price,
                signal.tp2_price,
                signal.channel_support,
                signal.channel_resistance,
                signal.entry_prob,
            ),
        )
        signal_id = c.lastrowid
        conn.commit()
        conn.close()
        return signal_id if signal_id is not None else 0

    def _save_trade(self, trade: Trade) -> int:
        """Save trade to database."""
        conn = get_db_connection(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT INTO trades
            (signal_id, strategy, timestamp, direction, setup_type, entry_price, sl_price,
             tp1_price, tp2_price, channel_support, channel_resistance, status,
             exit_decision, pnl_pct, closed_at, tp1_profit_taken, entry_candle_time, entry_capital)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.signal_id,
                trade.strategy,
                trade.timestamp,
                trade.direction,
                trade.setup_type,
                trade.entry_price,
                trade.sl_price,
                trade.tp1_price,
                trade.tp2_price,
                trade.channel_support,
                trade.channel_resistance,
                trade.status,
                trade.exit_decision,
                trade.pnl_pct,
                trade.closed_at,
                1 if trade.tp1_profit_taken else 0,
                trade.entry_candle_time,
                trade.entry_capital,
            ),
        )
        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        return trade_id if trade_id is not None else 0

    def _update_trade(self, trade: Trade):
        """Update trade in database."""
        conn = get_db_connection(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute(
            """UPDATE trades SET status=?, exit_decision=?, pnl_pct=?, closed_at=?, tp1_profit_taken=?, entry_capital=?
            WHERE id=?""",
            (
                trade.status,
                trade.exit_decision,
                trade.pnl_pct,
                trade.closed_at,
                1 if trade.tp1_profit_taken else 0,
                trade.entry_capital,
                trade.db_id,
            ),
        )
        conn.commit()
        conn.close()

    def _save_strategy_state(self, strategy: str, state: StrategyState):
        """Save strategy state snapshot."""
        conn = get_db_connection(PAPER_DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT INTO strategy_states
            (timestamp, strategy, capital, peak_capital, total_trades, wins, losses, total_pnl, max_drawdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                strategy,
                state.capital,
                state.peak_capital,
                state.total_trades,
                state.wins,
                state.losses,
                state.total_pnl,
                state.max_drawdown,
            ),
        )
        conn.commit()
        conn.close()

    def _load_candles(self, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Load candles from database."""
        # Convert timeframe to minutes for table name
        tf_to_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        tf_minutes = tf_to_minutes.get(timeframe, timeframe)

        conn = get_readonly_connection(DB_PATH)  # Read-only to prevent corruption
        query = f"""
            SELECT time, open, high, low, close, volume, delta
            FROM candles_{tf_minutes}
            WHERE symbol = ?
            ORDER BY time DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=[SYMBOL, limit])
        conn.close()

        if len(df) == 0:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def _process_signal(self, signal: Signal, entry_candle_time: int = 0):
        """Process a new signal for the NO_ML strategy."""
        signal_id = self._save_signal(signal)

        # Skip if duplicate signal
        if signal_id == -1:
            return

        self.signal_counter += 1
        now_ms = int(datetime.now().timestamp() * 1000)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 60}")
        print(f"[SIGNAL #{self.signal_counter}] {signal.direction} {signal.setup_type}")
        print(f"  Time: {now_str}")
        print(f"  Entry: {signal.entry_price:.2f}")
        print(
            f"  SL: {signal.sl_price:.2f} | TP1: {signal.tp1_price:.2f} | TP2: {signal.tp2_price:.2f}"
        )
        print(f"{'=' * 60}")

        # Strategy 1: NO_ML - Always enter
        trade_no_ml = Trade(
            signal_id=signal_id,
            strategy="NO_ML",
            timestamp=now_ms,
            direction=signal.direction,
            setup_type=signal.setup_type,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            channel_support=signal.channel_support,
            channel_resistance=signal.channel_resistance,
            entry_candle_time=entry_candle_time,
            entry_capital=self.strategies["NO_ML"].capital,
        )
        trade_no_ml.db_id = self._save_trade(trade_no_ml)
        self.strategies["NO_ML"].active_trades.append(trade_no_ml)
        self.strategies["NO_ML"].total_trades += 1
        print(f"  [NO_ML] Trade opened")

    def _update_trades(
        self,
        current_price: float,
        current_high: float,
        current_low: float,
        df_15m: Optional[pd.DataFrame] = None,
        current_idx: Optional[int] = None,
    ):
        """Update all active trades with current price."""
        now_ms = int(datetime.now().timestamp() * 1000)

        # Get current 15m candle time for lookahead bias prevention
        current_candle_time = 0
        if df_15m is not None and len(df_15m) > 0:
            current_candle_time = int(df_15m["time"].iloc[-1].timestamp() * 1000)

        for strategy_name, state in self.strategies.items():
            trades_to_remove = []

            for trade in state.active_trades:
                is_long = trade.direction == "LONG"
                closed = False
                pnl_pct = 0.0
                timed_out = False

                # Lookahead bias prevention: skip TP/SL check if still in entry candle
                if (
                    trade.entry_candle_time > 0
                    and current_candle_time <= trade.entry_candle_time
                ):
                    # Still in the entry candle, skip TP/SL check to prevent lookahead bias
                    continue

                if (
                    trade.entry_candle_time > 0
                    and current_candle_time > trade.entry_candle_time
                ):
                    tf_to_minutes = {
                        "1m": 1,
                        "5m": 5,
                        "15m": 15,
                        "30m": 30,
                        "1h": 60,
                        "4h": 240,
                        "1d": 1440,
                    }
                    ltf_minutes = tf_to_minutes.get(LTF, 15)
                    candle_ms = ltf_minutes * 60 * 1000
                    candles_since_entry = int(
                        (current_candle_time - trade.entry_candle_time) // candle_ms
                    )
                    if candles_since_entry >= MAX_HOLD_CANDLES:
                        trade.status = "TIMEOUT"
                        closed = True
                        timed_out = True
                        print(
                            f"\n[{strategy_name}] Trade timed out after {candles_since_entry} candles"
                        )

                if not timed_out and (
                    trade.status == "ACTIVE" or trade.status == "TP1_HIT"
                ):
                    # Check SL (original SL, even after TP1 - no BE stop)
                    if is_long and current_low <= trade.sl_price:
                        trade.status = "SL_HIT"
                        closed = True
                    elif not is_long and current_high >= trade.sl_price:
                        trade.status = "SL_HIT"
                        closed = True
                    # Check TP1 (only if not already hit)
                    elif not trade.tp1_profit_taken:
                        if is_long and current_high >= trade.tp1_price:
                            trade.status = "TP1_HIT"
                            trade.tp1_profit_taken = True
                            self._update_trade(trade)  # Persist to DB immediately
                            print(
                                f"\n[{strategy_name}] TP1 HIT - tracking partial profit"
                            )
                        elif not is_long and current_low <= trade.tp1_price:
                            trade.status = "TP1_HIT"
                            trade.tp1_profit_taken = True
                            self._update_trade(trade)  # Persist to DB immediately
                            print(
                                f"\n[{strategy_name}] TP1 HIT - tracking partial profit"
                            )
                    # Check TP2 (after TP1 hit)
                    if not closed and trade.tp1_profit_taken:
                        if is_long and current_high >= trade.tp2_price:
                            trade.status = "TP2_HIT"
                            closed = True
                        elif not is_long and current_low <= trade.tp2_price:
                            trade.status = "TP2_HIT"
                            closed = True

                if closed:
                    trade.closed_at = now_ms

                    if is_long:
                        sl_pct = (
                            trade.sl_price - trade.entry_price
                        ) / trade.entry_price
                        tp1_pct = (
                            trade.tp1_price - trade.entry_price
                        ) / trade.entry_price
                        tp2_pct = (
                            trade.tp2_price - trade.entry_price
                        ) / trade.entry_price
                    else:
                        sl_pct = (
                            trade.entry_price - trade.sl_price
                        ) / trade.entry_price
                        tp1_pct = (
                            trade.entry_price - trade.tp1_price
                        ) / trade.entry_price
                        tp2_pct = (
                            trade.entry_price - trade.tp2_price
                        ) / trade.entry_price

                    if trade.status == "SL_HIT":
                        if trade.tp1_profit_taken:
                            pnl_pct = 0.5 * tp1_pct + 0.5 * sl_pct
                        else:
                            pnl_pct = sl_pct
                    elif trade.status == "TP2_HIT":
                        pnl_pct = 0.5 * tp1_pct + 0.5 * tp2_pct
                    elif trade.status == "TIMEOUT":
                        if trade.tp1_profit_taken:
                            exit_pct = (
                                (current_price - trade.entry_price) / trade.entry_price
                                if is_long
                                else (trade.entry_price - current_price)
                                / trade.entry_price
                            )
                            pnl_pct = 0.5 * tp1_pct + 0.5 * exit_pct
                        else:
                            pnl_pct = (
                                (current_price - trade.entry_price) / trade.entry_price
                                if is_long
                                else (trade.entry_price - current_price)
                                / trade.entry_price
                            )
                    else:
                        pnl_pct = 0.0

                    trade.pnl_pct = pnl_pct

                    # Update capital for remaining position
                    entry_capital = (
                        trade.entry_capital
                        if trade.entry_capital > 0
                        else state.capital
                    )
                    sl_dist = (
                        abs(trade.entry_price - trade.sl_price) / trade.entry_price
                    )
                    leverage = (
                        min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
                    )
                    position = entry_capital * leverage
                    fees = position * FEE_PCT * 2
                    pnl_dollar = position * pnl_pct - fees

                    state.capital += pnl_dollar
                    state.capital = max(state.capital, 0)
                    state.total_pnl += pnl_dollar

                    if pnl_dollar > 0:
                        state.wins += 1
                    else:
                        state.losses += 1

                    if state.capital > state.peak_capital:
                        state.peak_capital = state.capital

                    dd = (
                        (state.peak_capital - state.capital) / state.peak_capital
                        if state.peak_capital > 0
                        else 0
                    )
                    state.max_drawdown = max(state.max_drawdown, dd)

                    self._update_trade(trade)
                    trades_to_remove.append(trade)

                    win_rate = (
                        state.wins / (state.wins + state.losses) * 100
                        if (state.wins + state.losses) > 0
                        else 0
                    )
                    print(f"\n[{strategy_name}] Trade closed: {trade.status}")
                    print(f"  Trade PnL: {pnl_pct * 100:+.2f}% (${pnl_dollar:+.2f})")
                    print(
                        f"  Capital: ${state.capital:,.2f} | WR: {win_rate:.1f}% | MaxDD: {state.max_drawdown * 100:.1f}%"
                    )

            for trade in trades_to_remove:
                state.active_trades.remove(trade)

    def _find_swing_points(self, candles: pd.DataFrame, confirm_candles: int = 3):
        """Find swing highs and lows."""
        if len(candles) < confirm_candles + 1:
            return [], []

        highs = candles["high"].values
        lows = candles["low"].values
        times = (
            candles["time"].values
            if "time" in candles.columns
            else candles.index.values
        )

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
                    swing_highs.append(
                        {
                            "idx": potential_high_idx,
                            "price": potential_high_price,
                            "time": times[potential_high_idx],
                        }
                    )

            if lows[i] < potential_low_price:
                potential_low_idx = i
                potential_low_price = lows[i]
                candles_since_low = 0
            else:
                candles_since_low += 1
                if candles_since_low == confirm_candles:
                    swing_lows.append(
                        {
                            "idx": potential_low_idx,
                            "price": potential_low_price,
                            "time": times[potential_low_idx],
                        }
                    )

            if candles_since_high >= confirm_candles:
                potential_high_price = highs[i]
                potential_high_idx = i
                candles_since_high = 0

            if candles_since_low >= confirm_candles:
                potential_low_price = lows[i]
                potential_low_idx = i
                candles_since_low = 0

        return swing_highs, swing_lows

    def _initialize_channels(self, df_1h: pd.DataFrame) -> Optional[Channel]:
        """
        Initialize channels from historical data (called once at startup).
        Simulates the backtest's for-loop through all candles to build channels.
        """
        if len(df_1h) < 20:
            return None

        print(f"[CHANNEL] Initializing from {len(df_1h)} historical candles...")

        # Configuration - min_channel_width 0.015 based on backtest (narrow channels had 50% WR)
        max_channel_width = 0.05
        min_channel_width = 0.015
        touch_threshold = 0.004

        # Find all swing points
        swing_highs, swing_lows = self._find_swing_points(df_1h)
        self.all_swing_highs = swing_highs
        self.all_swing_lows = swing_lows

        print(
            f"[CHANNEL] Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows"
        )

        highs = df_1h["high"].values
        lows = df_1h["low"].values
        closes = df_1h["close"].values

        # Clear existing channels
        self.active_channels = {}

        # Simulate backtest loop - process each candle
        for current_idx in range(len(df_1h)):
            current_close = closes[current_idx]

            # Find new swing points confirmed at this index
            new_high = None
            new_low = None
            for sh in swing_highs:
                if sh["idx"] + 3 == current_idx:
                    new_high = sh
                    break
            for sl in swing_lows:
                if sl["idx"] + 3 == current_idx:
                    new_low = sl
                    break

            # Valid swing points confirmed by current index
            valid_swing_lows = [sl for sl in swing_lows if sl["idx"] + 3 <= current_idx]
            valid_swing_highs = [
                sh for sh in swing_highs if sh["idx"] + 3 <= current_idx
            ]

            # Create NEW channels from new swing points
            if new_high:
                for sl in valid_swing_lows[-30:]:
                    if sl["idx"] < new_high["idx"] - 100:
                        continue
                    if new_high["price"] > sl["price"]:
                        width_pct = (new_high["price"] - sl["price"]) / sl["price"]
                        if min_channel_width <= width_pct <= max_channel_width:
                            key = (new_high["idx"], sl["idx"])
                            if key not in self.active_channels:
                                self.active_channels[key] = Channel(
                                    support=sl["price"],
                                    resistance=new_high["price"],
                                    support_touches=1,
                                    resistance_touches=1,
                                    lowest_low=sl["price"],
                                    highest_high=new_high["price"],
                                    confirmed=False,
                                )

            if new_low:
                for sh in valid_swing_highs[-30:]:
                    if sh["idx"] < new_low["idx"] - 100:
                        continue
                    if sh["price"] > new_low["price"]:
                        width_pct = (sh["price"] - new_low["price"]) / new_low["price"]
                        if min_channel_width <= width_pct <= max_channel_width:
                            key = (sh["idx"], new_low["idx"])
                            if key not in self.active_channels:
                                self.active_channels[key] = Channel(
                                    support=new_low["price"],
                                    resistance=sh["price"],
                                    support_touches=1,
                                    resistance_touches=1,
                                    lowest_low=new_low["price"],
                                    highest_high=sh["price"],
                                    confirmed=False,
                                )

            # Update existing channels
            keys_to_remove = []
            for key, channel in self.active_channels.items():
                if (
                    current_close < channel.lowest_low * 0.96
                    or current_close > channel.highest_high * 1.04
                ):
                    keys_to_remove.append(key)
                    continue

                # Update with new swing points (evolving logic)
                if new_low and new_low["price"] < channel.resistance:
                    if new_low["price"] < channel.lowest_low:
                        channel.lowest_low = new_low["price"]
                        channel.support = new_low["price"]
                        channel.support_touches = 1
                    elif (
                        new_low["price"] > channel.lowest_low
                        and new_low["price"] < channel.support
                    ):
                        channel.support = new_low["price"]
                        channel.support_touches += 1
                    elif (
                        abs(new_low["price"] - channel.support) / channel.support
                        < touch_threshold
                    ):
                        channel.support_touches += 1

                if new_high and new_high["price"] > channel.support:
                    if new_high["price"] > channel.highest_high:
                        channel.highest_high = new_high["price"]
                        channel.resistance = new_high["price"]
                        channel.resistance_touches = 1
                    elif (
                        new_high["price"] < channel.highest_high
                        and new_high["price"] > channel.resistance
                    ):
                        channel.resistance = new_high["price"]
                        channel.resistance_touches += 1
                    elif (
                        abs(new_high["price"] - channel.resistance) / channel.resistance
                        < touch_threshold
                    ):
                        channel.resistance_touches += 1

                # Check confirmation
                if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                    channel.confirmed = True

                # Check width
                width_pct = (channel.resistance - channel.support) / channel.support
                if width_pct > max_channel_width or width_pct < min_channel_width:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.active_channels[key]

        # Find best confirmed channel
        current_close = closes[-1]
        best_channel = None
        best_score = -1

        for key, channel in self.active_channels.items():
            if not channel.confirmed:
                continue
            if (
                current_close < channel.support * 0.98
                or current_close > channel.resistance * 1.02
            ):
                continue
            score = channel.support_touches + channel.resistance_touches
            if score > best_score:
                best_score = score
                best_channel = channel

        confirmed_count = sum(1 for c in self.active_channels.values() if c.confirmed)
        print(
            f"[CHANNEL] Initialized: Active={len(self.active_channels)}, Confirmed={confirmed_count}"
        )
        if best_channel:
            print(
                f"[CHANNEL] Best: S={best_channel.support:.0f}({best_channel.support_touches}) R={best_channel.resistance:.0f}({best_channel.resistance_touches})"
            )

        self.current_channel = best_channel
        self.last_htf_idx = len(df_1h) - 1
        return best_channel

    def _update_channel(self, df_1h: pd.DataFrame) -> Optional[Channel]:
        """
        Update channel detection using EVOLVING channel logic (matches backtest).

        Key differences from static approach:
        1. Track multiple active channels
        2. Update channels when new swing points form
        3. Channels can evolve (support/resistance can change)
        4. Remove invalidated channels
        """
        if len(df_1h) < 20:
            return None

        current_idx = len(df_1h) - 1
        current_close = df_1h["close"].iloc[-1]
        current_high = df_1h["high"].iloc[-1]
        current_low = df_1h["low"].iloc[-1]

        # Configuration - min_channel_width 0.015 based on backtest (narrow channels had 50% WR)
        max_channel_width = 0.05
        min_channel_width = 0.015
        touch_threshold = 0.004

        # Find all swing points up to current index
        swing_highs, swing_lows = self._find_swing_points(df_1h)
        self.all_swing_highs = swing_highs
        self.all_swing_lows = swing_lows

        # Find NEW swing points (confirmed at current index, i.e., swing at idx where idx+3 == current_idx)
        new_high = None
        new_low = None
        for sh in swing_highs:
            if sh["idx"] + 3 == current_idx:
                new_high = sh
                break
        for sl in swing_lows:
            if sl["idx"] + 3 == current_idx:
                new_low = sl
                break

        # Valid swing points (confirmed by current index)
        valid_swing_lows = [sl for sl in swing_lows if sl["idx"] + 3 <= current_idx]
        valid_swing_highs = [sh for sh in swing_highs if sh["idx"] + 3 <= current_idx]

        # Create NEW channels from new swing points
        if new_high:
            for sl in valid_swing_lows[-30:]:
                if sl["idx"] < new_high["idx"] - 100:
                    continue
                if new_high["price"] > sl["price"]:
                    width_pct = (new_high["price"] - sl["price"]) / sl["price"]
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (new_high["idx"], sl["idx"])
                        if key not in self.active_channels:
                            self.active_channels[key] = Channel(
                                support=sl["price"],
                                resistance=new_high["price"],
                                support_touches=1,
                                resistance_touches=1,
                                lowest_low=sl["price"],
                                highest_high=new_high["price"],
                                confirmed=False,
                            )

        if new_low:
            for sh in valid_swing_highs[-30:]:
                if sh["idx"] < new_low["idx"] - 100:
                    continue
                if sh["price"] > new_low["price"]:
                    width_pct = (sh["price"] - new_low["price"]) / new_low["price"]
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (sh["idx"], new_low["idx"])
                        if key not in self.active_channels:
                            self.active_channels[key] = Channel(
                                support=new_low["price"],
                                resistance=sh["price"],
                                support_touches=1,
                                resistance_touches=1,
                                lowest_low=new_low["price"],
                                highest_high=sh["price"],
                                confirmed=False,
                            )

        # UPDATE existing channels (evolving logic from backtest)
        keys_to_remove = []
        for key, channel in self.active_channels.items():
            # Remove if price broke through significantly
            if (
                current_close < channel.lowest_low * 0.96
                or current_close > channel.highest_high * 1.04
            ):
                keys_to_remove.append(key)
                continue

            # Update with new swing points (evolving logic)
            if new_low and new_low["price"] < channel.resistance:
                if new_low["price"] < channel.lowest_low:
                    # New lower low - channel expands down
                    channel.lowest_low = new_low["price"]
                    channel.support = new_low["price"]
                    channel.support_touches = 1
                elif (
                    new_low["price"] > channel.lowest_low
                    and new_low["price"] < channel.support
                ):
                    # Higher low but below current support - channel tightens
                    channel.support = new_low["price"]
                    channel.support_touches += 1
                elif (
                    abs(new_low["price"] - channel.support) / channel.support
                    < touch_threshold
                ):
                    # Touch near support
                    channel.support_touches += 1

            if new_high and new_high["price"] > channel.support:
                if new_high["price"] > channel.highest_high:
                    # New higher high - channel expands up
                    channel.highest_high = new_high["price"]
                    channel.resistance = new_high["price"]
                    channel.resistance_touches = 1
                elif (
                    new_high["price"] < channel.highest_high
                    and new_high["price"] > channel.resistance
                ):
                    # Lower high but above current resistance - channel tightens
                    channel.resistance = new_high["price"]
                    channel.resistance_touches += 1
                elif (
                    abs(new_high["price"] - channel.resistance) / channel.resistance
                    < touch_threshold
                ):
                    # Touch near resistance
                    channel.resistance_touches += 1

            # Check confirmation
            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            # Check width still valid
            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        # Remove invalidated channels
        for key in keys_to_remove:
            del self.active_channels[key]

        # Find BEST confirmed channel using NARROW tiebreaker (matches backtest baseline)
        # Collect all valid candidates with scores
        candidates = []

        for key, channel in self.active_channels.items():
            if not channel.confirmed:
                continue

            # Price should be inside channel
            if (
                current_close < channel.support * 0.98
                or current_close > channel.resistance * 1.02
            ):
                continue

            # Score by total touches
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        best_channel = None
        if candidates:
            # Find max score
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]

            # Apply NARROW tiebreaker: select narrowest channel among tied scores
            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            else:
                # Select narrowest channel (min width)
                best_channel = min(top_candidates, key=lambda c: c[1])[2]

        # Log channel status
        confirmed_count = sum(1 for c in self.active_channels.values() if c.confirmed)
        print(
            f"[CHANNEL] Active: {len(self.active_channels)}, Confirmed: {confirmed_count}, Candidates: {len(candidates)}"
        )
        if best_channel:
            width_pct = (
                (best_channel.resistance - best_channel.support)
                / best_channel.support
                * 100
            )
            print(
                f"[CHANNEL] Best (NARROW): S={best_channel.support:.0f}({best_channel.support_touches}) R={best_channel.resistance:.0f}({best_channel.resistance_touches}) W={width_pct:.2f}%"
            )

        self.current_channel = best_channel
        return best_channel

    def _check_fakeout(self, df_1h: pd.DataFrame, channel: Channel) -> Optional[dict]:
        """Check for fakeout signals."""
        if not channel or not channel.confirmed:
            return None

        current_close = df_1h["close"].iloc[-1]
        current_high = df_1h["high"].iloc[-1]
        current_low = df_1h["low"].iloc[-1]

        # Check pending fakeouts
        for key in list(self.pending_fakeouts.keys()):
            pf = self.pending_fakeouts[key]
            candles_since = len(df_1h) - 1 - pf["break_idx"]

            if candles_since > 5:
                del self.pending_fakeouts[key]
                continue

            if pf["type"] == "bear":
                pf["extreme"] = min(pf["extreme"], current_low)
                if current_close > channel.support:
                    del self.pending_fakeouts[key]
                    return {
                        "type": "bear",
                        "extreme": pf["extreme"],
                        "channel": channel,
                    }
            else:
                pf["extreme"] = max(pf["extreme"], current_high)
                if current_close < channel.resistance:
                    del self.pending_fakeouts[key]
                    return {
                        "type": "bull",
                        "extreme": pf["extreme"],
                        "channel": channel,
                    }

        # Check new breakouts
        if current_close < channel.support * 0.997:
            if "bear" not in self.pending_fakeouts:
                self.pending_fakeouts["bear"] = {
                    "type": "bear",
                    "break_idx": len(df_1h) - 1,
                    "extreme": current_low,
                }
        elif current_close > channel.resistance * 1.003:
            if "bull" not in self.pending_fakeouts:
                self.pending_fakeouts["bull"] = {
                    "type": "bull",
                    "break_idx": len(df_1h) - 1,
                    "extreme": current_high,
                }

        return None

    def _scan_for_signals(self):
        """Scan for new trading signals using stateless htf_map rebuild (matches backtest exactly)."""
        df_1h = self._load_candles(HTF, 1440)  # 60 days to match backtest
        df_15m = self._load_candles(LTF, 500)

        print(f"[SCAN] Loaded {len(df_1h)} 1h candles, {len(df_15m)} 15m candles")

        if len(df_1h) < 21 or len(df_15m) < 20:
            print(f"[SCAN] Not enough data (need 21+ 1h, 20+ 15m)")
            return

        # Drop last (incomplete) HTF candle
        df_1h_closed = df_1h.iloc[:-1].copy()
        current_htf_idx = len(df_1h_closed) - 1
        print(f"[SCAN] Using {len(df_1h_closed)} closed 1h candles for channel")

        # Stateless rebuild: build_channels from scratch every scan (matches backtest exactly)
        htf_map = _build_htf_map(
            df_1h_closed["high"].values,
            df_1h_closed["low"].values,
            df_1h_closed["close"].values,
        )

        # Use htf_map[current_htf_idx - 1] (matches backtest's htf_idx - 1)
        channel = htf_map.get(current_htf_idx - 1)

        htf_map_size = len(htf_map)
        if channel:
            width_pct = (channel.resistance - channel.support) / channel.support * 100
            print(
                f"[CHANNEL] htf_map has {htf_map_size} entries. Using idx {current_htf_idx - 1}: "
                f"S={channel.support:.0f}({channel.support_touches}) R={channel.resistance:.0f}({channel.resistance_touches}) W={width_pct:.1f}%"
            )
        else:
            print(
                f"[CHANNEL] htf_map has {htf_map_size} entries. No channel at idx {current_htf_idx - 1}"
            )
            # Still update trades even without a channel
            current_close = df_15m["close"].iloc[-1]
            current_high = df_15m["high"].iloc[-1]
            current_low = df_15m["low"].iloc[-1]
            self.last_price = current_close
            self._update_trades(
                current_close, current_high, current_low, df_15m, len(df_15m) - 1
            )
            return

        current_close = df_15m["close"].iloc[-1]
        current_high = df_15m["high"].iloc[-1]
        current_low = df_15m["low"].iloc[-1]
        current_candle_time = int(df_15m["time"].iloc[-1].timestamp() * 1000)

        self.last_price = current_close

        self._update_trades(
            current_close, current_high, current_low, df_15m, len(df_15m) - 1
        )

        # Only check for new signals when a NEW 15m candle starts
        if current_candle_time == self.last_processed_candle_time:
            return

        if self.last_processed_candle_time == 0:
            print(
                f"[SCAN] First scan - recording candle time, waiting for next candle close"
            )
            self.last_processed_candle_time = current_candle_time
            return

        if len(df_15m) < 3:
            self.last_processed_candle_time = current_candle_time
            return

        completed_idx = len(df_15m) - 2
        completed_close = df_15m["close"].iloc[completed_idx]
        completed_high = df_15m["high"].iloc[completed_idx]
        completed_low = df_15m["low"].iloc[completed_idx]
        completed_candle_time = int(
            df_15m["time"].iloc[completed_idx].timestamp() * 1000
        )

        print(
            f"[SCAN] New candle detected! Checking completed candle at {df_15m['time'].iloc[completed_idx]}"
        )

        channel_width_pct = (channel.resistance - channel.support) / channel.support
        if channel_width_pct < 0.015:
            print(
                f"[SCAN] Skipping signal - channel too narrow: {channel_width_pct * 100:.2f}%"
            )
            self.last_processed_candle_time = current_candle_time
            return

        mid_price = (channel.resistance + channel.support) / 2

        SIGNAL_COOLDOWN_MS = 20 * 15 * 60 * 1000
        signal_key = f"{round(channel.support)}_{round(channel.resistance)}_{completed_candle_time // SIGNAL_COOLDOWN_MS}"

        if signal_key not in self.recent_signals:
            if (
                completed_low <= channel.support * (1 + TOUCH_THRESHOLD)
                and completed_close > channel.support
            ):
                entry = completed_close
                sl = channel.support * (1 - SL_BUFFER_PCT)
                tp1 = mid_price
                tp2 = channel.resistance * 0.998

                if entry > sl and tp1 > entry:
                    signal = Signal(
                        timestamp=str(completed_candle_time),
                        direction="LONG",
                        setup_type="BOUNCE",
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        channel_support=channel.support,
                        channel_resistance=channel.resistance,
                        entry_prob=0.0,
                    )
                    self._process_signal(signal, completed_candle_time)
                    self.recent_signals.add(signal_key)

            elif (
                completed_high >= channel.resistance * (1 - TOUCH_THRESHOLD)
                and completed_close < channel.resistance
            ):
                entry = completed_close
                sl = channel.resistance * (1 + SL_BUFFER_PCT)
                tp1 = mid_price
                tp2 = channel.support * 1.002

                if sl > entry and entry > tp1:
                    signal = Signal(
                        timestamp=str(completed_candle_time),
                        direction="SHORT",
                        setup_type="BOUNCE",
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        channel_support=channel.support,
                        channel_resistance=channel.resistance,
                        entry_prob=0.0,
                    )
                    self._process_signal(signal, completed_candle_time)
                    self.recent_signals.add(signal_key)

        self.last_processed_candle_time = current_candle_time

        if len(self.recent_signals) > 100:
            self.recent_signals = set(list(self.recent_signals)[-50:])

    def run(self):
        """Main service loop."""
        self.running = True
        print(f"\n{'=' * 60}")
        print("  PAPER TRADING SERVICE STARTED")
        print("  Strategy: NO_ML (All Signals)")
        print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"{'=' * 60}\n")

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
                if self.path.startswith("/status"):
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    status = {
                        "timestamp": datetime.now().isoformat(),
                        "channel": None,
                        "current_price": None,
                        "strategies": {},
                    }

                    # Add channel info if available
                    if hasattr(service, "current_channel") and service.current_channel:
                        ch = service.current_channel
                        status["channel"] = {
                            "support": round(ch.support, 2),
                            "resistance": round(ch.resistance, 2),
                            "support_touches": ch.support_touches,
                            "resistance_touches": ch.resistance_touches,
                            "width_pct": round(
                                (ch.resistance - ch.support) / ch.support * 100, 2
                            ),
                        }

                    # Add current price if available
                    if hasattr(service, "last_price") and service.last_price:
                        status["current_price"] = round(service.last_price, 2)

                    for name, state in service.strategies.items():
                        win_rate = (
                            state.wins / (state.wins + state.losses) * 100
                            if (state.wins + state.losses) > 0
                            else 0
                        )

                        # Include active trade details
                        active_trade_details = []
                        for trade in state.active_trades:
                            # Calculate unrealized P&L
                            current_price = (
                                service.last_price
                                if hasattr(service, "last_price") and service.last_price
                                else trade.entry_price
                            )
                            if trade.direction == "LONG":
                                unrealized_pnl = (
                                    (current_price - trade.entry_price)
                                    / trade.entry_price
                                    * 100
                                )
                            else:
                                unrealized_pnl = (
                                    (trade.entry_price - current_price)
                                    / trade.entry_price
                                    * 100
                                )

                            active_trade_details.append(
                                {
                                    "signal_id": trade.signal_id,
                                    "direction": trade.direction,
                                    "setup_type": trade.setup_type,
                                    "entry_price": round(trade.entry_price, 2),
                                    "sl_price": round(trade.sl_price, 2),
                                    "tp1_price": round(trade.tp1_price, 2),
                                    "tp2_price": round(trade.tp2_price, 2),
                                    "channel_support": round(trade.channel_support, 2),
                                    "channel_resistance": round(
                                        trade.channel_resistance, 2
                                    ),
                                    "status": trade.status,
                                    "timestamp": trade.timestamp,
                                    "exit_decision": trade.exit_decision,
                                    "pnl_pct": round(trade.pnl_pct, 4),
                                    "tp1_profit_taken": trade.tp1_profit_taken,
                                    "unrealized_pnl": round(unrealized_pnl, 2),
                                }
                            )

                        status["strategies"][name] = {
                            "name": state.name,
                            "capital": round(state.capital, 2),
                            "return_pct": round(
                                (state.capital / INITIAL_CAPITAL - 1) * 100, 2
                            ),
                            "total_trades": state.total_trades,
                            "wins": state.wins,
                            "losses": state.losses,
                            "win_rate": round(win_rate, 1),
                            "max_drawdown": round(state.max_drawdown * 100, 1),
                            "active_trades": len(state.active_trades),
                            "active_trade_details": active_trade_details,
                        }

                    # Add recent signals from database (with short timeout to prevent blocking)
                    try:
                        # Use short timeout (2s) to prevent HTTP blocking
                        conn = sqlite3.connect(PAPER_DB_PATH, timeout=2)
                        conn.execute("PRAGMA busy_timeout=2000")
                        c = conn.cursor()
                        c.execute("""SELECT timestamp, direction, setup_type, entry_price, sl_price, tp1_price, tp2_price, entry_prob, channel_support, channel_resistance
                                     FROM signals ORDER BY id DESC LIMIT 10""")
                        recent_signals = []
                        for row in c.fetchall():
                            # Convert timestamp to int (milliseconds) for frontend
                            ts = row[0]
                            try:
                                ts = int(ts)
                            except (ValueError, TypeError):
                                ts = 0
                            recent_signals.append(
                                {
                                    "timestamp": ts,
                                    "direction": row[1],
                                    "setup_type": row[2],
                                    "entry_price": row[3],
                                    "sl_price": row[4],
                                    "tp1_price": row[5],
                                    "tp2_price": row[6],
                                    "entry_prob": row[7],
                                    "channel_support": row[8],
                                    "channel_resistance": row[9],
                                }
                            )
                        conn.close()
                        status["recent_signals"] = recent_signals
                    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                        # DB busy or locked - return empty signals instead of blocking
                        status["recent_signals"] = []
                        status["db_status"] = "busy"
                    except Exception as e:
                        status["recent_signals"] = []

                    self.wfile.write(json.dumps(status, indent=2).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer(("", SERVICE_PORT), StatusHandler)
        server.serve_forever()

    def print_status(self):
        """Print current status of all strategies."""
        print(f"\n{'=' * 70}")
        print(
            f"  PAPER TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'=' * 70}")
        print(
            f"{'Strategy':<25} {'Capital':>12} {'Return':>10} {'Trades':>8} {'WR':>8} {'MaxDD':>8}"
        )
        print(f"{'-' * 70}")

        for name, state in self.strategies.items():
            win_rate = (
                state.wins / (state.wins + state.losses) * 100
                if (state.wins + state.losses) > 0
                else 0
            )
            ret = (state.capital / INITIAL_CAPITAL - 1) * 100
            print(
                f"{state.name:<25} ${state.capital:>10,.2f} {ret:>+9.1f}% {state.total_trades:>8} {win_rate:>7.1f}% {state.max_drawdown * 100:>7.1f}%"
            )

        print(f"{'=' * 70}\n")


def main():
    service = MLPaperTradingService()
    service.run()


if __name__ == "__main__":
    main()
