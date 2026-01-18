#!/usr/bin/env python3
"""
ML Backtest: 15분봉 채널 + 5분봉 진입
- HTF: 15분봉 (채널 감지)
- LTF: 5분봉 (진입/청산)
- Risk: 1%
- Touch Tolerance: 0.5% (수평 채널)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'btc_data.db')

# Configuration
SYMBOL = "BTCUSDT"
HTF = 15  # 15분봉 for channel detection
LTF = 5   # 5분봉 for entry

# Strategy parameters
TOUCH_TOLERANCE = 0.005  # 0.5% - 수평 채널 인식
CHANNEL_MIN_WIDTH = 0.005  # 최소 0.5% 채널 폭
CHANNEL_MAX_WIDTH = 0.03   # 최대 3% 채널 폭
TOUCH_THRESHOLD = 0.003    # 바운스 신호용

# Risk parameters
RISK_PCT = 0.01  # 1% 리스크
SL_BUFFER_PCT = 0.002
MAX_LEVERAGE = 15
FEE_PCT = 0.0004

# Test period
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int
    resistance_touches: int
    confirmed: bool


@dataclass
class Trade:
    entry_time: str
    direction: str
    setup_type: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    exit_time: str = ''
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    result: str = ''


def load_candles(timeframe: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Load candles from database."""
    conn = sqlite3.connect(DB_PATH)

    query = f'''
        SELECT time, open, high, low, close, volume, delta
        FROM candles_{timeframe}m
        WHERE time >= ? AND time <= ?
        ORDER BY time ASC
    '''

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
    conn.close()

    if len(df) > 0:
        df['time'] = pd.to_datetime(df['time'], unit='ms')

    return df


def find_swing_points(df: pd.DataFrame, confirm_candles: int = 2):
    """Find swing highs and lows."""
    highs = df['high'].values
    lows = df['low'].values
    times = df['time'].values

    swing_highs = []
    swing_lows = []

    for i in range(confirm_candles, len(df) - 1):
        # Swing High
        is_high = True
        for j in range(1, confirm_candles + 1):
            if highs[i] <= highs[i-j] or highs[i] <= highs[i+1]:
                is_high = False
                break
        if is_high:
            swing_highs.append({'idx': i, 'price': highs[i], 'time': times[i]})

        # Swing Low
        is_low = True
        for j in range(1, confirm_candles + 1):
            if lows[i] >= lows[i-j] or lows[i] >= lows[i+1]:
                is_low = False
                break
        if is_low:
            swing_lows.append({'idx': i, 'price': lows[i], 'time': times[i]})

    return swing_highs, swing_lows


def detect_channel(df_htf: pd.DataFrame, current_idx: int, lookback: int = 100) -> Optional[Channel]:
    """Detect horizontal channel from HTF data."""
    start_idx = max(0, current_idx - lookback)
    df_window = df_htf.iloc[start_idx:current_idx+1].copy()

    if len(df_window) < 20:
        return None

    swing_highs, swing_lows = find_swing_points(df_window)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    current_close = df_window['close'].iloc[-1]
    best_channel = None
    best_score = -1

    # Try to find best channel from recent swing points
    for sh in swing_highs[-10:]:
        for sl in swing_lows[-10:]:
            if sh['price'] <= sl['price']:
                continue

            width_pct = (sh['price'] - sl['price']) / sl['price']
            if width_pct < CHANNEL_MIN_WIDTH or width_pct > CHANNEL_MAX_WIDTH:
                continue

            # Check if current price is within channel
            if current_close < sl['price'] * 0.98 or current_close > sh['price'] * 1.02:
                continue

            # Count touches within tolerance
            support_touches = sum(1 for s in swing_lows
                                  if abs(s['price'] - sl['price']) / sl['price'] < TOUCH_TOLERANCE)
            resistance_touches = sum(1 for s in swing_highs
                                     if abs(s['price'] - sh['price']) / sh['price'] < TOUCH_TOLERANCE)

            confirmed = support_touches >= 2 and resistance_touches >= 2

            if confirmed:
                score = support_touches + resistance_touches
                if score > best_score:
                    best_score = score
                    best_channel = Channel(
                        support=sl['price'],
                        resistance=sh['price'],
                        support_touches=support_touches,
                        resistance_touches=resistance_touches,
                        confirmed=True
                    )

    return best_channel


def run_backtest(start_date: str, end_date: str) -> List[Trade]:
    """Run backtest on given period."""
    print(f"\n{'='*60}")
    print(f"Running backtest: {start_date} to {end_date}")
    print(f"HTF: {HTF}m (channel), LTF: {LTF}m (entry)")
    print(f"Risk: {RISK_PCT*100}%, Touch Tolerance: {TOUCH_TOLERANCE*100}%")
    print(f"{'='*60}\n")

    # Load data
    df_htf = load_candles(HTF, start_date, end_date)
    df_ltf = load_candles(LTF, start_date, end_date)

    print(f"Loaded {len(df_htf)} HTF candles, {len(df_ltf)} LTF candles")

    if len(df_htf) < 50 or len(df_ltf) < 50:
        print("Not enough data!")
        return []

    trades = []
    active_trade = None
    current_channel = None
    last_signal_time = None

    # Create time index for LTF to HTF mapping
    htf_times = set(df_htf['time'].values)

    for i in range(50, len(df_ltf)):
        ltf_row = df_ltf.iloc[i]
        current_time = ltf_row['time']
        current_close = ltf_row['close']
        current_high = ltf_row['high']
        current_low = ltf_row['low']

        # Update channel on HTF candle close
        # Find corresponding HTF index
        htf_mask = df_htf['time'] <= current_time
        if htf_mask.sum() > 0:
            htf_idx = htf_mask.sum() - 1

            # Update channel periodically (every HTF candle)
            if htf_idx > 20:
                new_channel = detect_channel(df_htf, htf_idx)
                if new_channel:
                    current_channel = new_channel

        if current_channel is None:
            continue

        # Manage active trade
        if active_trade:
            is_long = active_trade.direction == 'LONG'

            if active_trade.result == '':  # Still active
                # Check SL
                if is_long and current_low <= active_trade.sl_price:
                    active_trade.exit_time = str(current_time)
                    active_trade.exit_price = active_trade.sl_price
                    active_trade.pnl_pct = (active_trade.sl_price - active_trade.entry_price) / active_trade.entry_price
                    active_trade.result = 'SL'
                    trades.append(active_trade)
                    active_trade = None
                elif not is_long and current_high >= active_trade.sl_price:
                    active_trade.exit_time = str(current_time)
                    active_trade.exit_price = active_trade.sl_price
                    active_trade.pnl_pct = (active_trade.entry_price - active_trade.sl_price) / active_trade.entry_price
                    active_trade.result = 'SL'
                    trades.append(active_trade)
                    active_trade = None
                # Check TP1
                elif is_long and current_high >= active_trade.tp1_price:
                    active_trade.exit_time = str(current_time)
                    active_trade.exit_price = active_trade.tp1_price
                    active_trade.pnl_pct = (active_trade.tp1_price - active_trade.entry_price) / active_trade.entry_price
                    active_trade.result = 'TP1'
                    trades.append(active_trade)
                    active_trade = None
                elif not is_long and current_low <= active_trade.tp1_price:
                    active_trade.exit_time = str(current_time)
                    active_trade.exit_price = active_trade.tp1_price
                    active_trade.pnl_pct = (active_trade.entry_price - active_trade.tp1_price) / active_trade.entry_price
                    active_trade.result = 'TP1'
                    trades.append(active_trade)
                    active_trade = None
            continue

        # Look for new signals (no active trade)
        mid_price = (current_channel.resistance + current_channel.support) / 2

        # Prevent duplicate signals
        signal_key = f"{current_channel.support:.0f}_{current_channel.resistance:.0f}_{str(current_time)[:13]}"
        if last_signal_time and (current_time - last_signal_time).total_seconds() < 3600:
            # Skip signals within 1 hour of last signal
            pass
        else:
            # Support bounce -> LONG
            if current_low <= current_channel.support * (1 + TOUCH_THRESHOLD) and current_close > current_channel.support:
                entry = current_close
                sl = current_channel.support * (1 - SL_BUFFER_PCT)
                tp1 = mid_price

                if entry > sl and tp1 > entry:
                    active_trade = Trade(
                        entry_time=str(current_time),
                        direction='LONG',
                        setup_type='BOUNCE',
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=current_channel.resistance * 0.998
                    )
                    last_signal_time = current_time

            # Resistance bounce -> SHORT
            elif current_high >= current_channel.resistance * (1 - TOUCH_THRESHOLD) and current_close < current_channel.resistance:
                entry = current_close
                sl = current_channel.resistance * (1 + SL_BUFFER_PCT)
                tp1 = mid_price

                if sl > entry and entry > tp1:
                    active_trade = Trade(
                        entry_time=str(current_time),
                        direction='SHORT',
                        setup_type='BOUNCE',
                        entry_price=entry,
                        sl_price=sl,
                        tp1_price=tp1,
                        tp2_price=current_channel.support * 1.002
                    )
                    last_signal_time = current_time

    return trades


def calculate_stats(trades: List[Trade], initial_capital: float = 10000.0):
    """Calculate trading statistics."""
    if not trades:
        return None

    capital = initial_capital
    peak_capital = initial_capital
    max_dd = 0.0

    wins = 0
    losses = 0

    for trade in trades:
        # Calculate position size based on risk
        sl_dist = abs(trade.entry_price - trade.sl_price) / trade.entry_price
        leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
        position = capital * leverage

        # Calculate PnL
        pnl_dollar = position * trade.pnl_pct - position * FEE_PCT * 2
        capital += pnl_dollar
        capital = max(capital, 0)

        if pnl_dollar > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak_capital:
            peak_capital = capital

        dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        max_dd = max(max_dd, dd)

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    total_return = (capital / initial_capital - 1) * 100

    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'final_capital': capital,
        'total_return': total_return,
        'max_drawdown': max_dd * 100
    }


def print_results(stats: dict, period: str):
    """Print backtest results."""
    if not stats:
        print(f"\n{period}: No trades")
        return

    print(f"\n{'='*50}")
    print(f" {period} Results")
    print(f"{'='*50}")
    print(f" Total Trades:  {stats['total_trades']}")
    print(f" Wins/Losses:   {stats['wins']}/{stats['losses']}")
    print(f" Win Rate:      {stats['win_rate']:.1f}%")
    print(f" Final Capital: ${stats['final_capital']:,.2f}")
    print(f" Total Return:  {stats['total_return']:+.1f}%")
    print(f" Max Drawdown:  {stats['max_drawdown']:.1f}%")
    print(f"{'='*50}")


def main():
    print("\n" + "="*60)
    print(" ML BACKTEST: 15m Channel + 5m Entry")
    print(" Risk: 1% | Touch Tolerance: 0.5%")
    print("="*60)

    # Check if data exists
    if not os.path.exists(DB_PATH):
        print(f"\nError: Database not found at {DB_PATH}")
        print("Please run data download first.")
        return

    # Run on training period
    print("\n[TRAINING PERIOD]")
    train_trades = run_backtest(TRAIN_START, TRAIN_END)
    train_stats = calculate_stats(train_trades)
    print_results(train_stats, "Training (2024)")

    # Run on test period
    print("\n[TEST PERIOD]")
    test_trades = run_backtest(TEST_START, TEST_END)
    test_stats = calculate_stats(test_trades)
    print_results(test_stats, "Test (2025)")

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    if train_stats and test_stats:
        print(f" Training WR: {train_stats['win_rate']:.1f}% | Test WR: {test_stats['win_rate']:.1f}%")
        print(f" Training DD: {train_stats['max_drawdown']:.1f}% | Test DD: {test_stats['max_drawdown']:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
