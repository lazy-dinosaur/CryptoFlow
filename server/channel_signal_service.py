#!/usr/bin/env python3
"""
Channel Strategy Signal Service

Real-time signal generator for MTF Channel Strategy:
- HTF (1H): Channel detection
- LTF (15m): Entry signals (BOUNCE + FAKEOUT)

Features:
- Real-time channel tracking
- Signal generation on entry conditions
- Paper trading simulation
- Signal history tracking
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cryptoflow.db')
SIGNAL_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'channel_signals.db')

# Configuration
SYMBOL = "BTCUSDT"
HTF = "1h"
LTF = "15m"
SERVICE_PORT = 5002

# Strategy parameters
TOUCH_THRESHOLD = 0.003
SL_BUFFER_PCT = 0.0008
MAX_FAKEOUT_WAIT = 5  # HTF candles

# Paper trading parameters
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.015  # 1.5% risk per trade
MAX_LEVERAGE = 15
FEE_PCT = 0.0004  # 0.04% taker fee (round trip 0.08%)


@dataclass
class SwingPoint:
    idx: int
    timestamp: str
    price: float
    type: str  # 'high' or 'low'


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int
    resistance_touches: int
    lowest_low: float
    highest_high: float
    confirmed: bool
    created_at: str
    last_updated: str


@dataclass
class Signal:
    timestamp: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    setup_type: str  # 'BOUNCE' or 'FAKEOUT'
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    channel_support: float
    channel_resistance: float
    status: str = 'ACTIVE'  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT, CANCELLED
    pnl_percent: float = 0.0
    position_size: float = 0.0
    pnl_dollar: float = 0.0
    closed_at: str = ''
    db_id: int = 0  # Database ID for updates


@dataclass
class PaperTradingState:
    """Track paper trading capital and performance."""
    capital: float = INITIAL_CAPITAL
    peak_capital: float = INITIAL_CAPITAL
    total_trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0


# Global state
active_channels: Dict[str, Channel] = {}
pending_fakeouts: List[dict] = []
active_signals: List[Signal] = []
signal_history: List[Signal] = []
paper_state: PaperTradingState = PaperTradingState()


def init_signal_db():
    """Initialize signal tracking database."""
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channel_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            direction TEXT,
            setup_type TEXT,
            entry_price REAL,
            sl_price REAL,
            tp1_price REAL,
            tp2_price REAL,
            channel_support REAL,
            channel_resistance REAL,
            status TEXT DEFAULT 'ACTIVE',
            tp1_hit_at TEXT,
            tp2_hit_at TEXT,
            sl_hit_at TEXT,
            closed_at TEXT,
            pnl_percent REAL DEFAULT 0,
            position_size REAL DEFAULT 0,
            pnl_dollar REAL DEFAULT 0,
            capital_after REAL DEFAULT 0,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS active_channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            support REAL,
            resistance REAL,
            support_touches INTEGER,
            resistance_touches INTEGER,
            confirmed INTEGER,
            created_at TEXT,
            last_updated TEXT
        )
    """)

    # Paper trading state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trading_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            capital REAL,
            peak_capital REAL,
            total_trades INTEGER,
            wins INTEGER,
            total_pnl REAL,
            max_drawdown REAL
        )
    """)

    # Equity curve for visualization
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equity_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            capital REAL,
            trade_id INTEGER
        )
    """)

    conn.commit()
    conn.close()
    print("Signal database initialized")

    # Load existing paper trading state
    load_paper_state()


def load_paper_state():
    """Load paper trading state from database."""
    global paper_state
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT capital, peak_capital, total_trades, wins, total_pnl, max_drawdown
        FROM paper_trading_state
        ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()

    if row:
        paper_state = PaperTradingState(
            capital=row[0],
            peak_capital=row[1],
            total_trades=row[2],
            wins=row[3],
            total_pnl=row[4],
            max_drawdown=row[5]
        )
        print(f"Loaded paper trading state: ${paper_state.capital:.2f}")
    else:
        paper_state = PaperTradingState()
        save_paper_state()
        print(f"Starting fresh paper trading: ${paper_state.capital:.2f}")

    conn.close()


def save_paper_state():
    """Save paper trading state to database."""
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO paper_trading_state
        (timestamp, capital, peak_capital, total_trades, wins, total_pnl, max_drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        paper_state.capital,
        paper_state.peak_capital,
        paper_state.total_trades,
        paper_state.wins,
        paper_state.total_pnl,
        paper_state.max_drawdown
    ))

    conn.commit()
    conn.close()


def save_equity_point(trade_id: int):
    """Save equity curve point after trade."""
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO equity_curve (timestamp, capital, trade_id)
        VALUES (?, ?, ?)
    """, (datetime.now().isoformat(), paper_state.capital, trade_id))

    conn.commit()
    conn.close()


def load_htf_candles(limit: int = 500) -> pd.DataFrame:
    """Load recent HTF candles from database."""
    conn = sqlite3.connect(DB_PATH)

    table_name = f"candles_{HTF}"
    query = f"""
        SELECT timestamp, open, high, low, close, volume, delta
        FROM {table_name}
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(SYMBOL, limit))
    conn.close()

    if len(df) == 0:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def load_ltf_candles(limit: int = 100) -> pd.DataFrame:
    """Load recent LTF candles from database."""
    conn = sqlite3.connect(DB_PATH)

    table_name = f"candles_{LTF}"
    query = f"""
        SELECT timestamp, open, high, low, close, volume, delta
        FROM {table_name}
        WHERE symbol = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(SYMBOL, limit))
    conn.close()

    if len(df) == 0:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find swing highs and lows."""
    if len(candles) < confirm_candles + 1:
        return [], []

    highs = candles['high'].values
    lows = candles['low'].values
    timestamps = candles['timestamp'].values

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
                swing_highs.append(SwingPoint(
                    idx=potential_high_idx,
                    timestamp=str(timestamps[potential_high_idx]),
                    price=potential_high_price,
                    type='high'
                ))

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append(SwingPoint(
                    idx=potential_low_idx,
                    timestamp=str(timestamps[potential_low_idx]),
                    price=potential_low_price,
                    type='low'
                ))

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def update_channels(htf_candles: pd.DataFrame) -> Optional[Channel]:
    """Update channel detection with latest HTF data."""
    global active_channels

    if len(htf_candles) < 20:
        return None

    swing_highs, swing_lows = find_swing_points(htf_candles)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    current_close = htf_candles['close'].iloc[-1]
    current_time = str(htf_candles['timestamp'].iloc[-1])

    # Try to find/update channels
    best_channel = None
    best_score = -1

    # Check recent swing points for channel formation
    for sh in swing_highs[-10:]:
        for sl in swing_lows[-10:]:
            if sh.price <= sl.price:
                continue

            width_pct = (sh.price - sl.price) / sl.price
            if width_pct < 0.008 or width_pct > 0.05:
                continue

            # Check if price is inside
            if current_close < sl.price * 0.98 or current_close > sh.price * 1.02:
                continue

            # Count touches
            support_touches = sum(1 for s in swing_lows if abs(s.price - sl.price) / sl.price < 0.004)
            resistance_touches = sum(1 for s in swing_highs if abs(s.price - sh.price) / sh.price < 0.004)

            confirmed = support_touches >= 2 and resistance_touches >= 2

            if confirmed:
                score = support_touches + resistance_touches
                if score > best_score:
                    best_score = score
                    best_channel = Channel(
                        support=sl.price,
                        resistance=sh.price,
                        support_touches=support_touches,
                        resistance_touches=resistance_touches,
                        lowest_low=min(s.price for s in swing_lows[-20:]),
                        highest_high=max(s.price for s in swing_highs[-20:]),
                        confirmed=True,
                        created_at=current_time,
                        last_updated=current_time
                    )

    if best_channel:
        key = f"{SYMBOL}_{HTF}"
        active_channels[key] = best_channel

    return best_channel


def check_fakeout_signals(htf_candles: pd.DataFrame, channel: Channel) -> List[dict]:
    """Check for fakeout signals on HTF."""
    global pending_fakeouts

    if not channel or not channel.confirmed:
        return []

    current_close = htf_candles['close'].iloc[-1]
    current_high = htf_candles['high'].iloc[-1]
    current_low = htf_candles['low'].iloc[-1]
    current_time = str(htf_candles['timestamp'].iloc[-1])

    fakeout_signals = []

    # Process pending fakeouts
    for pf in pending_fakeouts[:]:
        candles_since = len(htf_candles) - pf['break_idx']

        if candles_since > MAX_FAKEOUT_WAIT:
            pending_fakeouts.remove(pf)
            continue

        if pf['type'] == 'bear':
            pf['extreme'] = min(pf['extreme'], current_low)
            if current_close > channel.support:
                fakeout_signals.append({
                    'type': 'bear',
                    'extreme': pf['extreme'],
                    'channel': channel,
                    'timestamp': current_time
                })
                pending_fakeouts.remove(pf)
        else:
            pf['extreme'] = max(pf['extreme'], current_high)
            if current_close < channel.resistance:
                fakeout_signals.append({
                    'type': 'bull',
                    'extreme': pf['extreme'],
                    'channel': channel,
                    'timestamp': current_time
                })
                pending_fakeouts.remove(pf)

    # Check for new breakouts
    if current_close < channel.support * 0.997:
        already = any(pf['type'] == 'bear' for pf in pending_fakeouts)
        if not already:
            pending_fakeouts.append({
                'type': 'bear',
                'break_idx': len(htf_candles) - 1,
                'extreme': current_low
            })

    elif current_close > channel.resistance * 1.003:
        already = any(pf['type'] == 'bull' for pf in pending_fakeouts)
        if not already:
            pending_fakeouts.append({
                'type': 'bull',
                'break_idx': len(htf_candles) - 1,
                'extreme': current_high
            })

    return fakeout_signals


def calculate_position_size(entry_price: float, sl_price: float) -> float:
    """Calculate position size based on risk percentage."""
    risk_amount = paper_state.capital * RISK_PCT
    sl_distance_pct = abs(entry_price - sl_price) / entry_price

    if sl_distance_pct < 0.001:
        sl_distance_pct = 0.001

    # Position size without leverage
    position_value = risk_amount / sl_distance_pct

    # Apply max leverage limit
    max_position = paper_state.capital * MAX_LEVERAGE
    position_value = min(position_value, max_position)

    # Position size in BTC
    position_size = position_value / entry_price

    return position_size


def generate_signals(channel: Channel, ltf_candles: pd.DataFrame, fakeout_signals: List[dict]) -> List[Signal]:
    """Generate trading signals based on current conditions."""
    if not channel or not channel.confirmed or len(ltf_candles) == 0:
        return []

    signals = []

    current_close = ltf_candles['close'].iloc[-1]
    current_high = ltf_candles['high'].iloc[-1]
    current_low = ltf_candles['low'].iloc[-1]
    current_time = str(ltf_candles['timestamp'].iloc[-1])

    mid_price = (channel.resistance + channel.support) / 2

    # Check for BOUNCE signals
    # Support touch → LONG
    if current_low <= channel.support * (1 + TOUCH_THRESHOLD) and current_close > channel.support:
        sl_price = channel.support * (1 - SL_BUFFER_PCT)
        position_size = calculate_position_size(current_close, sl_price)
        signal = Signal(
            timestamp=current_time,
            symbol=SYMBOL,
            direction='LONG',
            setup_type='BOUNCE',
            entry_price=current_close,
            sl_price=sl_price,
            tp1_price=mid_price,
            tp2_price=channel.resistance * 0.998,
            channel_support=channel.support,
            channel_resistance=channel.resistance,
            position_size=position_size
        )
        signals.append(signal)

    # Resistance touch → SHORT
    elif current_high >= channel.resistance * (1 - TOUCH_THRESHOLD) and current_close < channel.resistance:
        sl_price = channel.resistance * (1 + SL_BUFFER_PCT)
        position_size = calculate_position_size(current_close, sl_price)
        signal = Signal(
            timestamp=current_time,
            symbol=SYMBOL,
            direction='SHORT',
            setup_type='BOUNCE',
            entry_price=current_close,
            sl_price=sl_price,
            tp1_price=mid_price,
            tp2_price=channel.support * 1.002,
            channel_support=channel.support,
            channel_resistance=channel.resistance,
            position_size=position_size
        )
        signals.append(signal)

    # Check for FAKEOUT signals
    for fs in fakeout_signals:
        f_channel = fs['channel']
        f_mid = (f_channel.resistance + f_channel.support) / 2

        if fs['type'] == 'bear':
            sl_price = fs['extreme'] * (1 - SL_BUFFER_PCT)
            position_size = calculate_position_size(current_close, sl_price)
            signal = Signal(
                timestamp=current_time,
                symbol=SYMBOL,
                direction='LONG',
                setup_type='FAKEOUT',
                entry_price=current_close,
                sl_price=sl_price,
                tp1_price=f_mid,
                tp2_price=f_channel.resistance * 0.998,
                channel_support=f_channel.support,
                channel_resistance=f_channel.resistance,
                position_size=position_size
            )
            signals.append(signal)
        else:
            sl_price = fs['extreme'] * (1 + SL_BUFFER_PCT)
            position_size = calculate_position_size(current_close, sl_price)
            signal = Signal(
                timestamp=current_time,
                symbol=SYMBOL,
                direction='SHORT',
                setup_type='FAKEOUT',
                entry_price=current_close,
                sl_price=sl_price,
                tp1_price=f_mid,
                tp2_price=f_channel.support * 1.002,
                channel_support=f_channel.support,
                channel_resistance=f_channel.resistance,
                position_size=position_size
            )
            signals.append(signal)

    return signals


def save_signal(signal: Signal) -> int:
    """Save signal to database and return the signal ID."""
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO channel_signals
        (timestamp, symbol, direction, setup_type, entry_price, sl_price,
         tp1_price, tp2_price, channel_support, channel_resistance, status,
         position_size, capital_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal.timestamp, signal.symbol, signal.direction, signal.setup_type,
        signal.entry_price, signal.sl_price, signal.tp1_price, signal.tp2_price,
        signal.channel_support, signal.channel_resistance, signal.status,
        signal.position_size, paper_state.capital
    ))

    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id


def update_signal_in_db(signal: Signal, signal_id: int):
    """Update completed signal in database."""
    conn = sqlite3.connect(SIGNAL_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE channel_signals
        SET status = ?, pnl_percent = ?, pnl_dollar = ?, closed_at = ?, capital_after = ?
        WHERE id = ?
    """, (signal.status, signal.pnl_percent, signal.pnl_dollar, signal.closed_at, paper_state.capital, signal_id))

    conn.commit()
    conn.close()


def close_signal(signal: Signal, exit_price: float, current_time: str, status: str, pnl_multiplier: float = 1.0):
    """Close a signal and update paper trading state."""
    global paper_state

    signal.status = status
    signal.closed_at = current_time

    # Calculate PnL
    position_value = signal.position_size * signal.entry_price
    fee = position_value * FEE_PCT * 2  # Entry + exit fee

    if signal.direction == 'LONG':
        if status == 'TP2_HIT':
            # 50% at TP1 + 50% at TP2
            pnl = 0.5 * signal.position_size * (signal.tp1_price - signal.entry_price)
            pnl += 0.5 * signal.position_size * (signal.tp2_price - signal.entry_price)
        elif status == 'SL_BE':
            # Only TP1 profit (SL hit at breakeven for remaining 50%)
            pnl = 0.5 * signal.position_size * (signal.tp1_price - signal.entry_price)
        else:  # SL_HIT
            pnl = signal.position_size * (signal.sl_price - signal.entry_price)
    else:  # SHORT
        if status == 'TP2_HIT':
            pnl = 0.5 * signal.position_size * (signal.entry_price - signal.tp1_price)
            pnl += 0.5 * signal.position_size * (signal.entry_price - signal.tp2_price)
        elif status == 'SL_BE':
            pnl = 0.5 * signal.position_size * (signal.entry_price - signal.tp1_price)
        else:  # SL_HIT
            pnl = signal.position_size * (signal.entry_price - signal.sl_price)

    signal.pnl_dollar = pnl - fee
    signal.pnl_percent = (signal.pnl_dollar / paper_state.capital) * 100

    # Update paper trading state
    paper_state.capital += signal.pnl_dollar
    paper_state.total_trades += 1
    paper_state.total_pnl += signal.pnl_dollar

    if signal.pnl_dollar > 0:
        paper_state.wins += 1

    if paper_state.capital > paper_state.peak_capital:
        paper_state.peak_capital = paper_state.capital

    drawdown = (paper_state.peak_capital - paper_state.capital) / paper_state.peak_capital * 100
    if drawdown > paper_state.max_drawdown:
        paper_state.max_drawdown = drawdown

    # Save to database
    if signal.db_id > 0:
        update_signal_in_db(signal, signal.db_id)
    save_paper_state()
    save_equity_point(paper_state.total_trades)


def update_signal_status(ltf_candles: pd.DataFrame):
    """Update status of active signals based on price action."""
    global active_signals

    if len(ltf_candles) == 0:
        return

    current_high = ltf_candles['high'].iloc[-1]
    current_low = ltf_candles['low'].iloc[-1]
    current_time = str(ltf_candles['timestamp'].iloc[-1])

    for signal in active_signals[:]:
        if signal.status not in ['ACTIVE', 'TP1_HIT']:
            continue

        if signal.direction == 'LONG':
            # Check SL
            if current_low <= signal.sl_price:
                if signal.status == 'TP1_HIT':
                    close_signal(signal, signal.entry_price, current_time, 'SL_BE')
                else:
                    close_signal(signal, signal.sl_price, current_time, 'SL_HIT')
                signal_history.append(signal)
                active_signals.remove(signal)
                print(f"  Signal {signal.direction} {signal.setup_type} closed: {signal.status}")
                print(f"    PnL: ${signal.pnl_dollar:+.2f} ({signal.pnl_percent:+.2f}%)")
                print(f"    Capital: ${paper_state.capital:.2f}")

            # Check TP1
            elif signal.status == 'ACTIVE' and current_high >= signal.tp1_price:
                signal.status = 'TP1_HIT'
                signal.sl_price = signal.entry_price  # Move SL to breakeven
                print(f"  Signal {signal.direction} TP1 hit, SL moved to breakeven")

            # Check TP2
            elif signal.status == 'TP1_HIT' and current_high >= signal.tp2_price:
                close_signal(signal, signal.tp2_price, current_time, 'TP2_HIT')
                signal_history.append(signal)
                active_signals.remove(signal)
                print(f"  Signal {signal.direction} {signal.setup_type} closed: TP2")
                print(f"    PnL: ${signal.pnl_dollar:+.2f} ({signal.pnl_percent:+.2f}%)")
                print(f"    Capital: ${paper_state.capital:.2f}")

        else:  # SHORT
            if current_high >= signal.sl_price:
                if signal.status == 'TP1_HIT':
                    close_signal(signal, signal.entry_price, current_time, 'SL_BE')
                else:
                    close_signal(signal, signal.sl_price, current_time, 'SL_HIT')
                signal_history.append(signal)
                active_signals.remove(signal)
                print(f"  Signal {signal.direction} {signal.setup_type} closed: {signal.status}")
                print(f"    PnL: ${signal.pnl_dollar:+.2f} ({signal.pnl_percent:+.2f}%)")
                print(f"    Capital: ${paper_state.capital:.2f}")

            elif signal.status == 'ACTIVE' and current_low <= signal.tp1_price:
                signal.status = 'TP1_HIT'
                signal.sl_price = signal.entry_price
                print(f"  Signal {signal.direction} TP1 hit, SL moved to breakeven")

            elif signal.status == 'TP1_HIT' and current_low <= signal.tp2_price:
                close_signal(signal, signal.tp2_price, current_time, 'TP2_HIT')
                signal_history.append(signal)
                active_signals.remove(signal)
                print(f"  Signal {signal.direction} {signal.setup_type} closed: TP2")
                print(f"    PnL: ${signal.pnl_dollar:+.2f} ({signal.pnl_percent:+.2f}%)")
                print(f"    Capital: ${paper_state.capital:.2f}")


def signal_loop():
    """Main signal generation loop."""
    print("\n" + "="*60)
    print("  Channel Signal Service Started")
    print(f"  HTF: {HTF}, LTF: {LTF}")
    print("="*60 + "\n")

    last_htf_time = None
    last_ltf_time = None

    while True:
        try:
            # Load candles
            htf_candles = load_htf_candles(500)
            ltf_candles = load_ltf_candles(100)

            if len(htf_candles) == 0 or len(ltf_candles) == 0:
                print("Waiting for data...")
                time.sleep(10)
                continue

            current_htf_time = htf_candles['timestamp'].iloc[-1]
            current_ltf_time = ltf_candles['timestamp'].iloc[-1]

            # Update on new candle
            if last_htf_time != current_htf_time:
                last_htf_time = current_htf_time
                print(f"\n[{current_htf_time}] New HTF candle")

                # Update channels
                channel = update_channels(htf_candles)
                if channel:
                    print(f"  Active Channel: {channel.support:.2f} - {channel.resistance:.2f}")
                    print(f"  Touches: S={channel.support_touches}, R={channel.resistance_touches}")

                # Check fakeouts
                fakeout_signals = check_fakeout_signals(htf_candles, channel)
                if fakeout_signals:
                    print(f"  Fakeout signals: {len(fakeout_signals)}")

            if last_ltf_time != current_ltf_time:
                last_ltf_time = current_ltf_time

                # Get current channel
                key = f"{SYMBOL}_{HTF}"
                channel = active_channels.get(key)

                # Update active signals
                update_signal_status(ltf_candles)

                # Generate new signals
                if channel:
                    fakeout_signals = check_fakeout_signals(htf_candles, channel)
                    new_signals = generate_signals(channel, ltf_candles, fakeout_signals)

                    for signal in new_signals:
                        # Avoid duplicate signals
                        is_duplicate = any(
                            s.direction == signal.direction and
                            s.setup_type == signal.setup_type and
                            abs(s.entry_price - signal.entry_price) / signal.entry_price < 0.001
                            for s in active_signals
                        )

                        if not is_duplicate:
                            signal.db_id = save_signal(signal)
                            active_signals.append(signal)
                            print(f"\n  *** NEW SIGNAL ***")
                            print(f"  {signal.direction} {signal.setup_type}")
                            print(f"  Entry: {signal.entry_price:.2f}")
                            print(f"  SL: {signal.sl_price:.2f}")
                            print(f"  TP1: {signal.tp1_price:.2f}")
                            print(f"  TP2: {signal.tp2_price:.2f}")
                            print(f"  Position: {signal.position_size:.6f} BTC")
                            print(f"  Value: ${signal.position_size * signal.entry_price:.2f}")

            time.sleep(5)  # Check every 5 seconds

        except Exception as e:
            print(f"Error in signal loop: {e}")
            time.sleep(10)


class SignalHandler(BaseHTTPRequestHandler):
    """HTTP handler for signal API."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/status':
            self.send_json({
                'status': 'running',
                'htf': HTF,
                'ltf': LTF,
                'active_signals': len(active_signals),
                'completed_signals': len(signal_history),
                'active_channels': len(active_channels)
            })

        elif path == '/signals':
            self.send_json({
                'active': [asdict(s) for s in active_signals],
                'history': [asdict(s) for s in signal_history[-20:]]
            })

        elif path == '/channels':
            self.send_json({
                'channels': {k: asdict(v) if hasattr(v, '__dict__') else {
                    'support': v.support,
                    'resistance': v.resistance,
                    'support_touches': v.support_touches,
                    'resistance_touches': v.resistance_touches,
                    'confirmed': v.confirmed
                } for k, v in active_channels.items()}
            })

        elif path == '/stats':
            win_rate = paper_state.wins / paper_state.total_trades * 100 if paper_state.total_trades > 0 else 0
            total_return = (paper_state.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            self.send_json({
                'paper_trading': {
                    'initial_capital': INITIAL_CAPITAL,
                    'current_capital': paper_state.capital,
                    'peak_capital': paper_state.peak_capital,
                    'total_return_pct': total_return,
                    'total_pnl_dollar': paper_state.total_pnl,
                    'max_drawdown_pct': paper_state.max_drawdown
                },
                'trades': {
                    'total': paper_state.total_trades,
                    'wins': paper_state.wins,
                    'losses': paper_state.total_trades - paper_state.wins,
                    'win_rate': win_rate
                },
                'settings': {
                    'risk_pct': RISK_PCT * 100,
                    'max_leverage': MAX_LEVERAGE,
                    'fee_pct': FEE_PCT * 100
                },
                'active_signals': len(active_signals)
            })

        elif path == '/equity':
            # Return equity curve data
            conn = sqlite3.connect(SIGNAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, capital FROM equity_curve ORDER BY id")
            rows = cursor.fetchall()
            conn.close()

            self.send_json({
                'equity_curve': [{'timestamp': r[0], 'capital': r[1]} for r in rows]
            })

        elif path == '/reset':
            # Reset paper trading (use POST in production)
            global paper_state
            paper_state = PaperTradingState()
            save_paper_state()
            self.send_json({'message': 'Paper trading reset', 'capital': paper_state.capital})

        else:
            self.send_error(404)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def log_message(self, format, *args):
        pass  # Suppress logging


def start_api_server():
    """Start HTTP API server."""
    server = HTTPServer(('0.0.0.0', SERVICE_PORT), SignalHandler)
    print(f"API server running on port {SERVICE_PORT}")
    server.serve_forever()


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   Channel Strategy Signal Service                         ║
║   Real-time BOUNCE + FAKEOUT signal generation            ║
║   Paper Trading Mode                                       ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Initialize database
    init_signal_db()

    print(f"""
Paper Trading Settings:
  Initial Capital: ${INITIAL_CAPITAL:,.2f}
  Current Capital: ${paper_state.capital:,.2f}
  Risk per Trade:  {RISK_PCT*100:.1f}%
  Max Leverage:    {MAX_LEVERAGE}x
  Fee:             {FEE_PCT*100:.2f}% per trade

API Endpoints:
  GET /status   - Service status
  GET /signals  - Active and recent signals
  GET /channels - Current channels
  GET /stats    - Paper trading statistics
  GET /equity   - Equity curve data
  GET /reset    - Reset paper trading
""")

    # Start API server in background
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()

    # Run signal loop
    signal_loop()


if __name__ == "__main__":
    main()
