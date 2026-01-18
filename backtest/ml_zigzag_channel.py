"""
Zigzag Channel Strategy Backtest
Based on newMethod.pine logic

Channel: 1H Zigzag Structure
Entry: 15m Touch + Close Confirmation
Setups: Bounce + Fakeout
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os

# === DATA CLASSES ===
@dataclass
class SwingPoint:
    idx: int
    price: float
    is_high: bool
    time: int

@dataclass
class Channel:
    high: float
    low: float
    mid: float
    width_pct: float
    swings: List[SwingPoint]
    formed_idx: int

@dataclass
class Signal:
    timestamp: int
    direction: str  # 'LONG' or 'SHORT'
    setup_type: str  # 'BOUNCE' or 'FAKEOUT'
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    channel_high: float
    channel_low: float

@dataclass
class Trade:
    signal: Signal
    entry_time: int
    exit_time: int
    exit_price: float
    exit_reason: str  # 'TP1', 'TP2', 'SL', 'OPPOSITE'
    pnl_pct: float
    partial_pnl: float = 0  # TP1 partial close PnL

# === CONFIGURATION ===
SWING_STRENGTH = 3  # Candles on each side
SWINGS_EACH_SIDE = 2  # Find 2 highs + 2 lows
MIN_CHANNEL_WIDTH = 0.003  # 0.3%
SL_BUFFER = 0.0005  # 0.05% buffer below/above wick
TP1_RATIO = 0.5  # 50% of channel
TP1_QTY_PCT = 0.5  # Close 50% at TP1

# === DATA FETCHING ===
def fetch_binance_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """Fetch klines from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        current_start = data[-1][0] + 1

        if len(data) < 1000:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['time'] = pd.to_numeric(df['time'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])

    return df

# === SWING DETECTION ===
def is_swing_high(highs: np.ndarray, idx: int, strength: int) -> bool:
    """Check if index is a swing high"""
    if idx < strength or idx >= len(highs) - strength:
        return False

    current = highs[idx]
    for i in range(1, strength + 1):
        if highs[idx - i] >= current or highs[idx + i] >= current:
            return False
    return True

def is_swing_low(lows: np.ndarray, idx: int, strength: int) -> bool:
    """Check if index is a swing low"""
    if idx < strength or idx >= len(lows) - strength:
        return False

    current = lows[idx]
    for i in range(1, strength + 1):
        if lows[idx - i] <= current or lows[idx + i] <= current:
            return False
    return True

def find_next_swing_high(highs: np.ndarray, times: np.ndarray, start_idx: int, strength: int) -> Optional[SwingPoint]:
    """Find next swing high from start_idx"""
    max_idx = len(highs) - strength
    for i in range(start_idx, max_idx):
        if is_swing_high(highs, i, strength):
            return SwingPoint(idx=i, price=highs[i], is_high=True, time=times[i])
    return None

def find_next_swing_low(lows: np.ndarray, times: np.ndarray, start_idx: int, strength: int) -> Optional[SwingPoint]:
    """Find next swing low from start_idx"""
    max_idx = len(lows) - strength
    for i in range(start_idx, max_idx):
        if is_swing_low(lows, i, strength):
            return SwingPoint(idx=i, price=lows[i], is_high=False, time=times[i])
    return None

# === ZIGZAG STRUCTURE ===
def find_zigzag_structure(htf_df: pd.DataFrame, current_close: float, strength: int, swings_each: int) -> List[SwingPoint]:
    """
    Find zigzag structure based on current price position
    If close >= recent high -> look for LOW first
    If close < recent high -> look for HIGH first
    """
    highs = htf_df['high'].values
    lows = htf_df['low'].values
    times = htf_df['time'].values

    if len(highs) < strength * 2 + 1:
        return []

    # Determine search direction based on current close
    recent_high = highs[0]  # Most recent candle
    look_for_low = current_close >= recent_high

    swings = []
    search_idx = strength
    highs_found = 0
    lows_found = 0
    max_search = len(highs) - strength
    iterations = 0

    while (highs_found < swings_each or lows_found < swings_each) and search_idx < max_search and iterations < 100:
        iterations += 1

        if look_for_low and lows_found < swings_each:
            swing = find_next_swing_low(lows, times, search_idx, strength)
            if swing:
                swings.append(swing)
                lows_found += 1
                search_idx = swing.idx + 1
                look_for_low = False
            else:
                search_idx += 1
                if search_idx >= max_search:
                    look_for_low = False

        elif not look_for_low and highs_found < swings_each:
            swing = find_next_swing_high(highs, times, search_idx, strength)
            if swing:
                swings.append(swing)
                highs_found += 1
                search_idx = swing.idx + 1
                look_for_low = True
            else:
                search_idx += 1
                if search_idx >= max_search:
                    look_for_low = True
        else:
            if lows_found >= swings_each and highs_found < swings_each:
                look_for_low = False
            elif highs_found >= swings_each and lows_found < swings_each:
                look_for_low = True
            else:
                break

    return swings

def build_channel(swings: List[SwingPoint], min_width: float) -> Optional[Channel]:
    """Build channel from zigzag swings"""
    if len(swings) < 2:
        return None

    prices = [s.price for s in swings]
    ch_high = max(prices)
    ch_low = min(prices)

    if ch_high == ch_low:
        return None

    width_pct = (ch_high - ch_low) / ch_low
    if width_pct < min_width:
        return None

    formed_idx = max(s.idx for s in swings)

    return Channel(
        high=ch_high,
        low=ch_low,
        mid=(ch_high + ch_low) / 2,
        width_pct=width_pct,
        swings=swings,
        formed_idx=formed_idx
    )

# === SIGNAL DETECTION ===
def detect_signals(ltf_df: pd.DataFrame, channel: Channel, htf_idx: int) -> List[Signal]:
    """
    Detect bounce and fakeout signals on LTF
    Entry: Touch + Close confirmation
    SL: Based on candle wick (tail) + small buffer
    """
    signals = []

    if channel is None:
        return signals

    ch_high = channel.high
    ch_low = channel.low
    ch_height = ch_high - ch_low

    # Track fakeout state
    broke_support = False
    broke_resistance = False
    fakeout_extreme_low = None
    fakeout_extreme_high = None

    for i in range(1, len(ltf_df)):
        row = ltf_df.iloc[i]

        low = row['low']
        high = row['high']
        close = row['close']
        timestamp = row['time']

        # === FAKEOUT TRACKING ===
        # Broke below support?
        if close < ch_low * 0.997:  # 0.3% margin
            broke_support = True
            fakeout_extreme_low = low if fakeout_extreme_low is None else min(fakeout_extreme_low, low)

        # Broke above resistance?
        if close > ch_high * 1.003:
            broke_resistance = True
            fakeout_extreme_high = high if fakeout_extreme_high is None else max(fakeout_extreme_high, high)

        # === LONG SIGNALS ===
        # Bounce: Touch support, close above
        if low <= ch_low and close > ch_low and not broke_support:
            # SL = candle's low (wick) - buffer
            sl = low * (1 - SL_BUFFER)
            tp1 = ch_low + ch_height * TP1_RATIO
            tp2 = ch_high

            signals.append(Signal(
                timestamp=timestamp,
                direction='LONG',
                setup_type='BOUNCE',
                entry_price=close,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                channel_high=ch_high,
                channel_low=ch_low
            ))

        # Fakeout: Was below, now closed back above support
        if broke_support and close > ch_low:
            # SL = fakeout extreme low - buffer
            sl = fakeout_extreme_low * (1 - SL_BUFFER) if fakeout_extreme_low else low * (1 - SL_BUFFER)
            tp1 = ch_low + ch_height * TP1_RATIO
            tp2 = ch_high

            signals.append(Signal(
                timestamp=timestamp,
                direction='LONG',
                setup_type='FAKEOUT',
                entry_price=close,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                channel_high=ch_high,
                channel_low=ch_low
            ))
            broke_support = False
            fakeout_extreme_low = None

        # === SHORT SIGNALS ===
        # Bounce: Touch resistance, close below
        if high >= ch_high and close < ch_high and not broke_resistance:
            # SL = candle's high (wick) + buffer
            sl = high * (1 + SL_BUFFER)
            tp1 = ch_high - ch_height * TP1_RATIO
            tp2 = ch_low

            signals.append(Signal(
                timestamp=timestamp,
                direction='SHORT',
                setup_type='BOUNCE',
                entry_price=close,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                channel_high=ch_high,
                channel_low=ch_low
            ))

        # Fakeout: Was above, now closed back below resistance
        if broke_resistance and close < ch_high:
            # SL = fakeout extreme high + buffer
            sl = fakeout_extreme_high * (1 + SL_BUFFER) if fakeout_extreme_high else high * (1 + SL_BUFFER)
            tp1 = ch_high - ch_height * TP1_RATIO
            tp2 = ch_low

            signals.append(Signal(
                timestamp=timestamp,
                direction='SHORT',
                setup_type='FAKEOUT',
                entry_price=close,
                sl_price=sl,
                tp1_price=tp1,
                tp2_price=tp2,
                channel_high=ch_high,
                channel_low=ch_low
            ))
            broke_resistance = False
            fakeout_extreme_high = None

    return signals

# === TRADE SIMULATION ===
def simulate_trade(signal: Signal, ltf_df: pd.DataFrame, signal_idx: int) -> Optional[Trade]:
    """Simulate trade execution with TP1/TP2/SL
    After TP1 hit: Move SL to breakeven (entry price)
    """
    direction = signal.direction
    entry = signal.entry_price
    sl = signal.sl_price
    tp1 = signal.tp1_price
    tp2 = signal.tp2_price

    hit_tp1 = False
    partial_pnl = 0
    remaining_qty = 1.0

    for i in range(signal_idx + 1, len(ltf_df)):
        row = ltf_df.iloc[i]
        high = row['high']
        low = row['low']
        close = row['close']
        timestamp = row['time']

        if direction == 'LONG':
            # Check SL (after TP1: breakeven, before TP1: original SL)
            current_sl = entry if hit_tp1 else sl
            if low <= current_sl:
                if hit_tp1:
                    # Breakeven exit - only keep TP1 profit
                    pnl = partial_pnl
                    return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                               exit_price=entry, exit_reason='BE', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)
                else:
                    pnl = ((sl - entry) / entry) * remaining_qty
                    return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                               exit_price=sl, exit_reason='SL', pnl_pct=pnl * 100, partial_pnl=0)

            # Check TP1
            if not hit_tp1 and high >= tp1:
                partial_pnl = ((tp1 - entry) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True
                # SL is now moved to entry (breakeven)

            # Check TP2
            if high >= tp2:
                pnl = ((tp2 - entry) / entry) * remaining_qty + partial_pnl
                return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                           exit_price=tp2, exit_reason='TP2' if hit_tp1 else 'TP2_FULL', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)

            # Opposite signal (touch resistance)
            if high >= signal.channel_high and close < signal.channel_high:
                pnl = ((close - entry) / entry) * remaining_qty + partial_pnl
                return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                           exit_price=close, exit_reason='OPPOSITE', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)

        else:  # SHORT
            # Check SL (after TP1: breakeven, before TP1: original SL)
            current_sl = entry if hit_tp1 else sl
            if high >= current_sl:
                if hit_tp1:
                    # Breakeven exit - only keep TP1 profit
                    pnl = partial_pnl
                    return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                               exit_price=entry, exit_reason='BE', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)
                else:
                    pnl = ((entry - sl) / entry) * remaining_qty
                    return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                               exit_price=sl, exit_reason='SL', pnl_pct=pnl * 100, partial_pnl=0)

            # Check TP1
            if not hit_tp1 and low <= tp1:
                partial_pnl = ((entry - tp1) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True
                # SL is now moved to entry (breakeven)

            # Check TP2
            if low <= tp2:
                pnl = ((entry - tp2) / entry) * remaining_qty + partial_pnl
                return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                           exit_price=tp2, exit_reason='TP2' if hit_tp1 else 'TP2_FULL', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)

            # Opposite signal (touch support)
            if low <= signal.channel_low and close > signal.channel_low:
                pnl = ((entry - close) / entry) * remaining_qty + partial_pnl
                return Trade(signal=signal, entry_time=signal.timestamp, exit_time=timestamp,
                           exit_price=close, exit_reason='OPPOSITE', pnl_pct=pnl * 100, partial_pnl=partial_pnl * 100)

    return None

# === MAIN BACKTEST ===
def run_backtest(symbol: str = "BTCUSDT", days: int = 30):
    """Run backtest"""
    print(f"\n{'='*60}")
    print(f"Zigzag Channel Strategy Backtest")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {days} days")
    print(f"HTF: 1H (channel detection)")
    print(f"LTF: 15m (entry)")
    print(f"Swing Strength: {SWING_STRENGTH}")
    print(f"Swings Each Side: {SWINGS_EACH_SIDE}")
    print(f"Min Channel Width: {MIN_CHANNEL_WIDTH*100:.1f}%")
    print(f"SL: Candle wick + {SL_BUFFER*100:.2f}% buffer")
    print(f"TP1: {TP1_RATIO*100:.0f}% of channel, close {TP1_QTY_PCT*100:.0f}%")
    print(f"{'='*60}\n")

    # Time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    # Fetch data
    print("Fetching 1H data...")
    htf_df = fetch_binance_klines(symbol, "1h", start_time, end_time)
    print(f"  Got {len(htf_df)} 1H candles")

    print("Fetching 15m data...")
    ltf_df = fetch_binance_klines(symbol, "15m", start_time, end_time)
    print(f"  Got {len(ltf_df)} 15m candles")

    if htf_df.empty or ltf_df.empty:
        print("Error: No data fetched")
        return

    # Process
    all_trades: List[Trade] = []
    channels_found = 0
    signals_generated = 0

    # For each HTF candle, detect channel and find signals
    htf_df = htf_df.sort_values('time').reset_index(drop=True)
    ltf_df = ltf_df.sort_values('time').reset_index(drop=True)

    # Reverse HTF for zigzag detection (most recent first)
    htf_reversed = htf_df.iloc[::-1].reset_index(drop=True)

    print("\nProcessing...")

    for htf_idx in range(SWING_STRENGTH * 2, len(htf_df) - 1):
        htf_row = htf_df.iloc[htf_idx]
        htf_time = htf_row['time']
        htf_close = htf_row['close']

        # Get HTF data up to this point (reversed for zigzag)
        htf_slice = htf_df.iloc[:htf_idx+1].iloc[::-1].reset_index(drop=True)

        # Find zigzag structure
        swings = find_zigzag_structure(htf_slice, htf_close, SWING_STRENGTH, SWINGS_EACH_SIDE)

        if len(swings) < 2:
            continue

        # Build channel
        channel = build_channel(swings, MIN_CHANNEL_WIDTH)
        if channel is None:
            continue

        channels_found += 1

        # Get LTF data for this HTF candle period
        next_htf_time = htf_df.iloc[htf_idx + 1]['time'] if htf_idx + 1 < len(htf_df) else htf_time + 3600000
        ltf_slice = ltf_df[(ltf_df['time'] >= htf_time) & (ltf_df['time'] < next_htf_time)].reset_index(drop=True)

        if ltf_slice.empty:
            continue

        # Detect signals
        signals = detect_signals(ltf_slice, channel, htf_idx)

        for signal in signals:
            signals_generated += 1

            # Find signal index in full LTF data
            signal_idx = ltf_df[ltf_df['time'] == signal.timestamp].index
            if len(signal_idx) == 0:
                continue
            signal_idx = signal_idx[0]

            # Simulate trade
            trade = simulate_trade(signal, ltf_df, signal_idx)
            if trade:
                all_trades.append(trade)

    # === RESULTS ===
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Channels Found: {channels_found}")
    print(f"Signals Generated: {signals_generated}")
    print(f"Trades Executed: {len(all_trades)}")

    if not all_trades:
        print("No trades to analyze")
        return

    # Analyze trades
    wins = [t for t in all_trades if t.pnl_pct > 0]
    losses = [t for t in all_trades if t.pnl_pct <= 0]

    total_pnl = sum(t.pnl_pct for t in all_trades)
    avg_pnl = total_pnl / len(all_trades)
    win_rate = len(wins) / len(all_trades) * 100

    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0

    print(f"\nTotal PnL: {total_pnl:.2f}%")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)}/{len(all_trades)})")
    print(f"Avg Win: {avg_win:.2f}%")
    print(f"Avg Loss: {avg_loss:.2f}%")
    print(f"Avg Trade: {avg_pnl:.2f}%")

    # By setup type
    print(f"\n--- By Setup Type ---")
    for setup in ['BOUNCE', 'FAKEOUT']:
        trades = [t for t in all_trades if t.signal.setup_type == setup]
        if trades:
            pnl = sum(t.pnl_pct for t in trades)
            w = len([t for t in trades if t.pnl_pct > 0])
            print(f"{setup}: {len(trades)} trades, {pnl:.2f}% PnL, {w/len(trades)*100:.1f}% WR")

    # By direction
    print(f"\n--- By Direction ---")
    for direction in ['LONG', 'SHORT']:
        trades = [t for t in all_trades if t.signal.direction == direction]
        if trades:
            pnl = sum(t.pnl_pct for t in trades)
            w = len([t for t in trades if t.pnl_pct > 0])
            print(f"{direction}: {len(trades)} trades, {pnl:.2f}% PnL, {w/len(trades)*100:.1f}% WR")

    # By exit reason
    print(f"\n--- By Exit Reason ---")
    for reason in ['TP2', 'TP2_FULL', 'BE', 'SL', 'OPPOSITE']:
        trades = [t for t in all_trades if t.exit_reason == reason]
        if trades:
            pnl = sum(t.pnl_pct for t in trades)
            print(f"{reason}: {len(trades)} trades, {pnl:.2f}% PnL")

    # Last 10 trades
    print(f"\n--- Last 10 Trades ---")
    for trade in all_trades[-10:]:
        entry_dt = datetime.fromtimestamp(trade.entry_time / 1000).strftime('%m/%d %H:%M')
        print(f"{entry_dt} {trade.signal.direction} {trade.signal.setup_type}: {trade.pnl_pct:+.2f}% ({trade.exit_reason})")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    # 2022-01-01 ~ 2025-01-18 (약 3년)
    run_backtest("BTCUSDT", days=1115)
