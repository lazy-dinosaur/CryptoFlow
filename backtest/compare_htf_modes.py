#!/usr/bin/env python3
"""
Compare CLOSED vs PARTIAL HTF candle usage with realistic streaming simulation.

Simulates real-time behavior:
- 1m candles are processed sequentially
- HTF (1h) candles are aggregated from 1m data in real-time
- PARTIAL: Uses current incomplete 1h candle for channel detection
- CLOSED: Drops incomplete 1h candle (matches corrected paper trading)

Usage:
    python compare_htf_modes.py --year 2024
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int = 1
    resistance_touches: int = 1
    lowest_low: float = 0
    highest_high: float = 0
    confirmed: bool = False


class StreamingChannelDetector:
    """Channel detector that simulates real-time paper trading behavior."""

    def __init__(self, tiebreaker: str = 'narrow'):
        self.active_channels: Dict[tuple, Channel] = {}
        self.current_channel: Optional[Channel] = None
        self.all_swing_highs: List[dict] = []
        self.all_swing_lows: List[dict] = []
        self.tiebreaker = tiebreaker
        self.initialized = False
        self.last_htf_len = 0
        self.htf_channel_map: Dict[int, Channel] = {}  # Store best channel at each HTF idx

        # Config
        self.max_channel_width = 0.05
        self.min_channel_width = 0.008
        self.touch_threshold = 0.004

    def _find_swing_points(self, df_1h: pd.DataFrame, confirm_candles: int = 3):
        """Find swing highs and lows."""
        if len(df_1h) < confirm_candles + 1:
            return [], []

        highs = df_1h['high'].values
        lows = df_1h['low'].values

        swing_highs = []
        swing_lows = []

        potential_high_idx = 0
        potential_high_price = highs[0]
        candles_since_high = 0

        potential_low_idx = 0
        potential_low_price = lows[0]
        candles_since_low = 0

        for i in range(1, len(df_1h)):
            if highs[i] > potential_high_price:
                potential_high_idx = i
                potential_high_price = highs[i]
                candles_since_high = 0
            else:
                candles_since_high += 1
                if candles_since_high == confirm_candles:
                    swing_highs.append({'idx': potential_high_idx, 'price': potential_high_price})

            if lows[i] < potential_low_price:
                potential_low_idx = i
                potential_low_price = lows[i]
                candles_since_low = 0
            else:
                candles_since_low += 1
                if candles_since_low == confirm_candles:
                    swing_lows.append({'idx': potential_low_idx, 'price': potential_low_price})

            if candles_since_high >= confirm_candles:
                potential_high_price = highs[i]
                potential_high_idx = i
                candles_since_high = 0

            if candles_since_low >= confirm_candles:
                potential_low_price = lows[i]
                potential_low_idx = i
                candles_since_low = 0

        return swing_highs, swing_lows

    def initialize_channels(self, df_1h: pd.DataFrame):
        """
        Initialize channels from historical data (matches ml_paper_trading._initialize_channels).
        Simulates the backtest's for-loop through all candles to build channels.
        """
        if len(df_1h) < 20:
            return

        swing_highs, swing_lows = self._find_swing_points(df_1h)
        self.all_swing_highs = swing_highs
        self.all_swing_lows = swing_lows

        closes = df_1h['close'].values
        self.active_channels = {}

        # Simulate backtest loop - process each candle
        for current_idx in range(len(df_1h)):
            current_close = closes[current_idx]

            # Find new swing points confirmed at this index
            new_high = None
            new_low = None
            for sh in swing_highs:
                if sh['idx'] + 3 == current_idx:
                    new_high = sh
                    break
            for sl in swing_lows:
                if sl['idx'] + 3 == current_idx:
                    new_low = sl
                    break

            valid_swing_lows = [sl for sl in swing_lows if sl['idx'] + 3 <= current_idx]
            valid_swing_highs = [sh for sh in swing_highs if sh['idx'] + 3 <= current_idx]

            # Create NEW channels
            if new_high:
                for sl in valid_swing_lows[-30:]:
                    if sl['idx'] < new_high['idx'] - 100:
                        continue
                    if new_high['price'] > sl['price']:
                        width_pct = (new_high['price'] - sl['price']) / sl['price']
                        if self.min_channel_width <= width_pct <= self.max_channel_width:
                            key = (new_high['idx'], sl['idx'])
                            if key not in self.active_channels:
                                self.active_channels[key] = Channel(
                                    support=sl['price'],
                                    resistance=new_high['price'],
                                    support_touches=1,
                                    resistance_touches=1,
                                    lowest_low=sl['price'],
                                    highest_high=new_high['price'],
                                    confirmed=False
                                )

            if new_low:
                for sh in valid_swing_highs[-30:]:
                    if sh['idx'] < new_low['idx'] - 100:
                        continue
                    if sh['price'] > new_low['price']:
                        width_pct = (sh['price'] - new_low['price']) / new_low['price']
                        if self.min_channel_width <= width_pct <= self.max_channel_width:
                            key = (sh['idx'], new_low['idx'])
                            if key not in self.active_channels:
                                self.active_channels[key] = Channel(
                                    support=new_low['price'],
                                    resistance=sh['price'],
                                    support_touches=1,
                                    resistance_touches=1,
                                    lowest_low=new_low['price'],
                                    highest_high=sh['price'],
                                    confirmed=False
                                )

            # Update existing channels
            keys_to_remove = []
            for key, channel in self.active_channels.items():
                if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                    keys_to_remove.append(key)
                    continue

                if new_low and new_low['price'] < channel.resistance:
                    if new_low['price'] < channel.lowest_low:
                        channel.lowest_low = new_low['price']
                        channel.support = new_low['price']
                        channel.support_touches = 1
                    elif new_low['price'] > channel.lowest_low and new_low['price'] < channel.support:
                        channel.support = new_low['price']
                        channel.support_touches += 1
                    elif abs(new_low['price'] - channel.support) / channel.support < self.touch_threshold:
                        channel.support_touches += 1

                if new_high and new_high['price'] > channel.support:
                    if new_high['price'] > channel.highest_high:
                        channel.highest_high = new_high['price']
                        channel.resistance = new_high['price']
                        channel.resistance_touches = 1
                    elif new_high['price'] < channel.highest_high and new_high['price'] > channel.resistance:
                        channel.resistance = new_high['price']
                        channel.resistance_touches += 1
                    elif abs(new_high['price'] - channel.resistance) / channel.resistance < self.touch_threshold:
                        channel.resistance_touches += 1

                if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                    channel.confirmed = True

                width_pct = (channel.resistance - channel.support) / channel.support
                if width_pct > self.max_channel_width or width_pct < self.min_channel_width:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.active_channels[key]

            # Build htf_channel_map IN THE LOOP (like original backtest)
            best = self._find_best_channel(current_close)
            if best:
                self.htf_channel_map[current_idx] = best

        self.initialized = True
        self.last_htf_len = len(df_1h)

    def get_channel_for_htf_idx(self, htf_idx: int) -> Optional[Channel]:
        """Get channel that was best at htf_idx - 1 (like original backtest)."""
        return self.htf_channel_map.get(htf_idx - 1)

    def update_channel(self, df_1h: pd.DataFrame) -> Optional[Channel]:
        """Update channel detection (matches ml_paper_trading._update_channel)."""
        if len(df_1h) < 20:
            return None

        # Initialize if first time or if HTF data grew significantly
        if not self.initialized:
            self.initialize_channels(df_1h)

        current_idx = len(df_1h) - 1
        current_close = df_1h['close'].iloc[-1]

        # Only process if new HTF candle
        if len(df_1h) <= self.last_htf_len:
            # Return current best channel without processing
            return self._find_best_channel(current_close)

        self.last_htf_len = len(df_1h)

        # Find swing points
        swing_highs, swing_lows = self._find_swing_points(df_1h)
        self.all_swing_highs = swing_highs
        self.all_swing_lows = swing_lows

        # Find NEW swing points confirmed at current index
        new_high = None
        new_low = None
        for sh in swing_highs:
            if sh['idx'] + 3 == current_idx:
                new_high = sh
                break
        for sl in swing_lows:
            if sl['idx'] + 3 == current_idx:
                new_low = sl
                break

        valid_swing_lows = [sl for sl in swing_lows if sl['idx'] + 3 <= current_idx]
        valid_swing_highs = [sh for sh in swing_highs if sh['idx'] + 3 <= current_idx]

        # Create NEW channels
        if new_high:
            for sl in valid_swing_lows[-30:]:
                if sl['idx'] < new_high['idx'] - 100:
                    continue
                if new_high['price'] > sl['price']:
                    width_pct = (new_high['price'] - sl['price']) / sl['price']
                    if self.min_channel_width <= width_pct <= self.max_channel_width:
                        key = (new_high['idx'], sl['idx'])
                        if key not in self.active_channels:
                            self.active_channels[key] = Channel(
                                support=sl['price'],
                                resistance=new_high['price'],
                                support_touches=1,
                                resistance_touches=1,
                                lowest_low=sl['price'],
                                highest_high=new_high['price'],
                                confirmed=False
                            )

        if new_low:
            for sh in valid_swing_highs[-30:]:
                if sh['idx'] < new_low['idx'] - 100:
                    continue
                if sh['price'] > new_low['price']:
                    width_pct = (sh['price'] - new_low['price']) / new_low['price']
                    if self.min_channel_width <= width_pct <= self.max_channel_width:
                        key = (sh['idx'], new_low['idx'])
                        if key not in self.active_channels:
                            self.active_channels[key] = Channel(
                                support=new_low['price'],
                                resistance=sh['price'],
                                support_touches=1,
                                resistance_touches=1,
                                lowest_low=new_low['price'],
                                highest_high=sh['price'],
                                confirmed=False
                            )

        # Update existing channels
        keys_to_remove = []
        for key, channel in self.active_channels.items():
            if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                keys_to_remove.append(key)
                continue

            if new_low and new_low['price'] < channel.resistance:
                if new_low['price'] < channel.lowest_low:
                    channel.lowest_low = new_low['price']
                    channel.support = new_low['price']
                    channel.support_touches = 1
                elif new_low['price'] > channel.lowest_low and new_low['price'] < channel.support:
                    channel.support = new_low['price']
                    channel.support_touches += 1
                elif abs(new_low['price'] - channel.support) / channel.support < self.touch_threshold:
                    channel.support_touches += 1

            if new_high and new_high['price'] > channel.support:
                if new_high['price'] > channel.highest_high:
                    channel.highest_high = new_high['price']
                    channel.resistance = new_high['price']
                    channel.resistance_touches = 1
                elif new_high['price'] < channel.highest_high and new_high['price'] > channel.resistance:
                    channel.resistance = new_high['price']
                    channel.resistance_touches += 1
                elif abs(new_high['price'] - channel.resistance) / channel.resistance < self.touch_threshold:
                    channel.resistance_touches += 1

            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > self.max_channel_width or width_pct < self.min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.active_channels[key]

        # Find best channel (NARROW tiebreaker)
        candidates = []
        for key, channel in self.active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        best_channel = None
        if candidates:
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]
            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            elif self.tiebreaker == 'narrow':
                best_channel = min(top_candidates, key=lambda c: c[1])[2]
            else:
                best_channel = top_candidates[0][2]

        self.current_channel = best_channel

        # Update htf_channel_map for current idx
        if best_channel:
            self.htf_channel_map[current_idx] = best_channel

        return best_channel

    def _find_best_channel(self, current_close: float) -> Optional[Channel]:
        """Find best confirmed channel for current price."""
        candidates = []
        for key, channel in self.active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        if not candidates:
            return None

        max_score = max(c[0] for c in candidates)
        top_candidates = [c for c in candidates if c[0] == max_score]
        if len(top_candidates) == 1:
            return top_candidates[0][2]
        elif self.tiebreaker == 'narrow':
            return min(top_candidates, key=lambda c: c[1])[2]
        return top_candidates[0][2]


def simulate_trade(entry_price, sl_price, tp1_price, tp2_price, direction,
                   future_candles: pd.DataFrame) -> dict:
    """Simulate trade outcome using future candles."""
    is_long = direction == 'LONG'
    max_candles = 96  # 24h max hold (15m candles)

    for i in range(min(len(future_candles), max_candles)):
        high = future_candles['high'].iloc[i]
        low = future_candles['low'].iloc[i]

        # Check SL first
        if is_long and low <= sl_price:
            return {'pnl_pct': (sl_price - entry_price) / entry_price, 'outcome': 0, 'exit': 'SL'}
        elif not is_long and high >= sl_price:
            return {'pnl_pct': (entry_price - sl_price) / entry_price, 'outcome': 0, 'exit': 'SL'}

        # Check TP1
        if is_long and high >= tp1_price:
            # Check if TP2 hit after TP1
            for j in range(i, min(len(future_candles), max_candles)):
                h = future_candles['high'].iloc[j]
                l = future_candles['low'].iloc[j]
                if l <= entry_price:  # BE hit
                    pnl = 0.5 * (tp1_price - entry_price) / entry_price
                    return {'pnl_pct': pnl, 'outcome': 0.5, 'exit': 'TP1+BE'}
                if h >= tp2_price:
                    pnl = 0.5 * (tp1_price - entry_price) / entry_price + 0.5 * (tp2_price - entry_price) / entry_price
                    return {'pnl_pct': pnl, 'outcome': 1, 'exit': 'TP2'}
            pnl = 0.5 * (tp1_price - entry_price) / entry_price
            return {'pnl_pct': pnl, 'outcome': 0.5, 'exit': 'TP1+TIMEOUT'}

        elif not is_long and low <= tp1_price:
            for j in range(i, min(len(future_candles), max_candles)):
                h = future_candles['high'].iloc[j]
                l = future_candles['low'].iloc[j]
                if h >= entry_price:  # BE hit
                    pnl = 0.5 * (entry_price - tp1_price) / entry_price
                    return {'pnl_pct': pnl, 'outcome': 0.5, 'exit': 'TP1+BE'}
                if l <= tp2_price:
                    pnl = 0.5 * (entry_price - tp1_price) / entry_price + 0.5 * (entry_price - tp2_price) / entry_price
                    return {'pnl_pct': pnl, 'outcome': 1, 'exit': 'TP2'}
            pnl = 0.5 * (entry_price - tp1_price) / entry_price
            return {'pnl_pct': pnl, 'outcome': 0.5, 'exit': 'TP1+TIMEOUT'}

    # Timeout - exit at current price
    exit_price = future_candles['close'].iloc[min(len(future_candles)-1, max_candles-1)]
    if is_long:
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price
    return {'pnl_pct': pnl, 'outcome': 0.5 if pnl > 0 else 0, 'exit': 'TIMEOUT'}


def run_streaming_backtest(
    candles_1m: pd.DataFrame,
    candles_15m: pd.DataFrame,
    use_partial_htf: bool,
    label: str
) -> dict:
    """
    Run backtest with streaming HTF aggregation.

    Args:
        candles_1m: 1-minute candles
        candles_15m: 15-minute candles (pre-aggregated for entry simulation)
        use_partial_htf: If True, include incomplete 1h candle (PARTIAL)
                         If False, drop incomplete 1h candle (CLOSED)
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  HTF Mode: {'PARTIAL (미완료 1h 포함)' if use_partial_htf else 'CLOSED (완료된 1h만)'}")
    print(f"{'='*70}")

    detector = StreamingChannelDetector(tiebreaker='narrow')

    # Aggregate 1h candles from 1m data
    htf_ohlcv = {
        'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
    }

    # Track current 1h bucket
    current_1h_bucket = None
    current_1h_open = None
    current_1h_high = None
    current_1h_low = None
    current_1h_close = None
    current_1h_volume = 0

    # Track 15m candle completion
    last_15m_time = None

    setups = []
    traded_entries = set()

    touch_threshold = 0.003
    sl_buffer_pct = 0.0008

    for i in tqdm(range(len(candles_1m)), desc=label):
        row = candles_1m.iloc[i]
        ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)

        # Determine 1h bucket
        bucket_1h = ts.floor('1h')

        # New 1h candle started
        if bucket_1h != current_1h_bucket:
            # Save previous completed 1h candle
            if current_1h_bucket is not None:
                htf_ohlcv['time'].append(current_1h_bucket)
                htf_ohlcv['open'].append(current_1h_open)
                htf_ohlcv['high'].append(current_1h_high)
                htf_ohlcv['low'].append(current_1h_low)
                htf_ohlcv['close'].append(current_1h_close)
                htf_ohlcv['volume'].append(current_1h_volume)

            # Start new 1h candle
            current_1h_bucket = bucket_1h
            current_1h_open = row['open']
            current_1h_high = row['high']
            current_1h_low = row['low']
            current_1h_close = row['close']
            current_1h_volume = row['volume']
        else:
            # Update current 1h candle
            current_1h_high = max(current_1h_high, row['high'])
            current_1h_low = min(current_1h_low, row['low'])
            current_1h_close = row['close']
            current_1h_volume += row['volume']

        # Check if 15m candle just completed
        bucket_15m = ts.floor('15min')
        if last_15m_time is not None and bucket_15m != last_15m_time:
            # 15m candle completed - check for entry signals

            # Build HTF dataframe
            if use_partial_htf:
                # Include current incomplete 1h candle
                df_1h = pd.DataFrame({
                    'time': htf_ohlcv['time'] + [current_1h_bucket],
                    'open': htf_ohlcv['open'] + [current_1h_open],
                    'high': htf_ohlcv['high'] + [current_1h_high],
                    'low': htf_ohlcv['low'] + [current_1h_low],
                    'close': htf_ohlcv['close'] + [current_1h_close],
                    'volume': htf_ohlcv['volume'] + [current_1h_volume]
                })
            else:
                # CLOSED: Only completed 1h candles
                if len(htf_ohlcv['time']) == 0:
                    last_15m_time = bucket_15m
                    continue
                df_1h = pd.DataFrame(htf_ohlcv)

            if len(df_1h) < 20:
                last_15m_time = bucket_15m
                continue

            # Update channel detection (builds htf_channel_map)
            detector.update_channel(df_1h)

            # Get completed 15m candle data
            completed_15m_idx = candles_15m.index.get_indexer([last_15m_time], method='ffill')[0]
            if completed_15m_idx < 0 or completed_15m_idx >= len(candles_15m):
                last_15m_time = bucket_15m
                continue

            # Get channel using htf_idx - 1 (like original backtest)
            # htf_idx = LTF_idx // tf_ratio
            htf_idx = completed_15m_idx // 4  # 1h / 15m = 4
            channel = detector.get_channel_for_htf_idx(htf_idx)

            if channel is None:
                last_15m_time = bucket_15m
                continue

            completed_15m = candles_15m.iloc[completed_15m_idx]
            current_close = completed_15m['close']
            current_high = completed_15m['high']
            current_low = completed_15m['low']

            mid_price = (channel.resistance + channel.support) / 2

            # Avoid duplicate entries
            trade_key = (round(channel.support), round(channel.resistance), completed_15m_idx // 20)
            if trade_key in traded_entries:
                last_15m_time = bucket_15m
                continue

            # Check bounce entries
            entry_signal = None

            # Support bounce → LONG
            if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
                entry_price = current_close
                sl_price = channel.support * (1 - sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.resistance * 0.998
                entry_signal = ('LONG', entry_price, sl_price, tp1_price, tp2_price)

            # Resistance bounce → SHORT
            elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
                entry_price = current_close
                sl_price = channel.resistance * (1 + sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.support * 1.002
                entry_signal = ('SHORT', entry_price, sl_price, tp1_price, tp2_price)

            if entry_signal:
                direction, entry_price, sl_price, tp1_price, tp2_price = entry_signal

                risk = abs(entry_price - sl_price)
                reward = abs(tp1_price - entry_price)

                if risk > 0 and reward > 0:
                    # Simulate trade with future 15m candles
                    future_candles = candles_15m.iloc[completed_15m_idx + 1:]
                    if len(future_candles) > 0:
                        result = simulate_trade(entry_price, sl_price, tp1_price, tp2_price, direction, future_candles)
                        result['idx'] = completed_15m_idx
                        result['entry'] = entry_price
                        result['sl'] = sl_price
                        result['direction'] = direction
                        result['channel_support'] = channel.support
                        result['channel_resistance'] = channel.resistance
                        setups.append(result)
                        traded_entries.add(trade_key)

        last_15m_time = bucket_15m

    # Calculate results
    if not setups:
        print("  No setups found!")
        return None

    df = pd.DataFrame(setups)

    # Backtest with position sizing
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade['pnl_pct']
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if trade['outcome'] >= 0.5:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_pnl = df['pnl_pct'].mean() * 100

    print(f"\n  Trades: {len(df)}")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg PnL: {avg_pnl:+.3f}%")
    print(f"  Return: {total_return:+.1f}%")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")

    return {
        'trades': len(df),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'return': total_return,
        'max_dd': max_dd * 100,
        'final_capital': capital,
        'wins': wins,
        'losses': losses
    }


def debug_streaming_index_mapping(candles_1m: pd.DataFrame, candles_15m: pd.DataFrame, year: int = 2024):
    """
    Debug: Check index mapping in streaming simulation.
    """
    print("\n" + "="*70)
    print("  DEBUG: 스트리밍 인덱스 매핑 확인")
    print("="*70)

    # Load pre-built 1h candles for comparison
    htf_candles_preload = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    htf_candles_preload = htf_candles_preload[htf_candles_preload.index.year == year]

    print(f"\n사전 로드된 1h 캔들: {len(htf_candles_preload)}")
    print(f"  첫 캔들 시간: {htf_candles_preload.index[0]}")
    print(f"  마지막 캔들 시간: {htf_candles_preload.index[-1]}")

    print(f"\n사전 로드된 15m 캔들: {len(candles_15m)}")
    print(f"  첫 캔들 시간: {candles_15m.index[0]}")
    print(f"  마지막 캔들 시간: {candles_15m.index[-1]}")

    print(f"\n1m 캔들: {len(candles_1m)}")
    print(f"  첫 캔들 시간: {candles_1m.index[0]}")
    print(f"  마지막 캔들 시간: {candles_1m.index[-1]}")

    # Simulate streaming aggregation for first few hours
    htf_ohlcv = {'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    current_1h_bucket = None
    current_1h_open = None
    current_1h_high = None
    current_1h_low = None
    current_1h_close = None
    current_1h_volume = 0

    last_15m_time = None
    entry_log = []

    # Process first ~500 1m candles (about 8 hours)
    max_1m = min(500, len(candles_1m))

    for i in range(max_1m):
        row = candles_1m.iloc[i]
        ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)

        bucket_1h = ts.floor('1h')

        # New 1h candle started
        if bucket_1h != current_1h_bucket:
            if current_1h_bucket is not None:
                htf_ohlcv['time'].append(current_1h_bucket)
                htf_ohlcv['open'].append(current_1h_open)
                htf_ohlcv['high'].append(current_1h_high)
                htf_ohlcv['low'].append(current_1h_low)
                htf_ohlcv['close'].append(current_1h_close)
                htf_ohlcv['volume'].append(current_1h_volume)

            current_1h_bucket = bucket_1h
            current_1h_open = row['open']
            current_1h_high = row['high']
            current_1h_low = row['low']
            current_1h_close = row['close']
            current_1h_volume = row['volume']
        else:
            current_1h_high = max(current_1h_high, row['high'])
            current_1h_low = min(current_1h_low, row['low'])
            current_1h_close = row['close']
            current_1h_volume += row['volume']

        # Check 15m completion
        bucket_15m = ts.floor('15min')
        if last_15m_time is not None and bucket_15m != last_15m_time:
            # 15m just completed
            completed_15m_idx = candles_15m.index.get_indexer([last_15m_time], method='ffill')[0]
            n_completed_1h = len(htf_ohlcv['time'])

            if i < 200:  # Log first few
                entry_log.append({
                    '1m_idx': i,
                    '15m_time': last_15m_time,
                    '15m_idx': completed_15m_idx,
                    'n_1h': n_completed_1h,
                    'htf_idx': completed_15m_idx // 4,
                    'lookup': (completed_15m_idx // 4) - 1,
                    'last_1h_time': htf_ohlcv['time'][-1] if htf_ohlcv['time'] else None
                })

        last_15m_time = bucket_15m

    # Print log
    print(f"\n스트리밍 인덱스 매핑 (처음 15개):")
    print(f"{'1m_idx':<8} {'15m_time':<22} {'15m_idx':<8} {'n_1h':<6} {'htf_idx':<8} {'lookup':<8} {'last_1h_time'}")
    print("-"*100)
    for e in entry_log[:15]:
        print(f"{e['1m_idx']:<8} {str(e['15m_time']):<22} {e['15m_idx']:<8} {e['n_1h']:<6} {e['htf_idx']:<8} {e['lookup']:<8} {str(e['last_1h_time'])}")

    # Compare aggregated 1h with pre-loaded 1h
    print(f"\n스트리밍 aggregated 1h vs 사전 로드된 1h OHLCV (처음 10개):")
    print(f"{'idx':<4} {'time':<20} {'src':<8} {'open':<10} {'high':<10} {'low':<10} {'close':<10}")
    print("-"*80)
    for idx in range(min(10, len(htf_ohlcv['time']))):
        stream_t = htf_ohlcv['time'][idx]
        stream_o = htf_ohlcv['open'][idx]
        stream_h = htf_ohlcv['high'][idx]
        stream_l = htf_ohlcv['low'][idx]
        stream_c = htf_ohlcv['close'][idx]

        preload_row = htf_candles_preload.iloc[idx]
        preload_o = preload_row['open']
        preload_h = preload_row['high']
        preload_l = preload_row['low']
        preload_c = preload_row['close']

        print(f"{idx:<4} {str(stream_t):<20} {'stream':<8} {stream_o:<10.0f} {stream_h:<10.0f} {stream_l:<10.0f} {stream_c:<10.0f}")
        print(f"{'':<4} {'':<20} {'preload':<8} {preload_o:<10.0f} {preload_h:<10.0f} {preload_l:<10.0f} {preload_c:<10.0f}")

        # Check differences
        diff_o = abs(stream_o - preload_o)
        diff_h = abs(stream_h - preload_h)
        diff_l = abs(stream_l - preload_l)
        diff_c = abs(stream_c - preload_c)

        if diff_o > 0.01 or diff_h > 0.01 or diff_l > 0.01 or diff_c > 0.01:
            print(f"{'':<4} {'':<20} {'DIFF!':<8} {diff_o:<10.2f} {diff_h:<10.2f} {diff_l:<10.2f} {diff_c:<10.2f}")

    # Key insight: when we do completed_15m_idx // 4, what htf index does it correspond to?
    print(f"\n핵심 문제: 15m_idx → htf_idx 매핑")
    print(f"  15m_idx=0 시간: {candles_15m.index[0]}")
    print(f"  15m_idx=4 시간: {candles_15m.index[4]}")
    print(f"  1h_idx=0 시간 (preload): {htf_candles_preload.index[0]}")
    print(f"  1h_idx=1 시간 (preload): {htf_candles_preload.index[1]}")

    # Check if 15m idx=0-3 align with 1h idx=0
    print(f"\n15m idx 0-3은 어떤 1h 캔들에 해당하는가?")
    for idx in range(4):
        t15 = candles_15m.index[idx]
        t1h_bucket = t15.floor('1h')
        print(f"  15m idx={idx}: {t15} → 1h bucket: {t1h_bucket}")

    # Build htf_channel_map from streaming-aggregated 1h and compare with preloaded
    print(f"\n" + "="*70)
    print("  스트리밍 aggregated 1h로 채널 구축 비교")
    print("="*70)

    # Need to aggregate ALL 1m to 1h first
    print("  1m → 1h 전체 aggregation 중...")

    full_htf_ohlcv = {'time': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
    current_1h_bucket = None
    current_1h_open = None
    current_1h_high = None
    current_1h_low = None
    current_1h_close = None
    current_1h_volume = 0

    for i in range(len(candles_1m)):
        row = candles_1m.iloc[i]
        ts = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)
        bucket_1h = ts.floor('1h')

        if bucket_1h != current_1h_bucket:
            if current_1h_bucket is not None:
                full_htf_ohlcv['time'].append(current_1h_bucket)
                full_htf_ohlcv['open'].append(current_1h_open)
                full_htf_ohlcv['high'].append(current_1h_high)
                full_htf_ohlcv['low'].append(current_1h_low)
                full_htf_ohlcv['close'].append(current_1h_close)
                full_htf_ohlcv['volume'].append(current_1h_volume)

            current_1h_bucket = bucket_1h
            current_1h_open = row['open']
            current_1h_high = row['high']
            current_1h_low = row['low']
            current_1h_close = row['close']
            current_1h_volume = row['volume']
        else:
            current_1h_high = max(current_1h_high, row['high'])
            current_1h_low = min(current_1h_low, row['low'])
            current_1h_close = row['close']
            current_1h_volume += row['volume']

    # Add last incomplete candle
    if current_1h_bucket is not None:
        full_htf_ohlcv['time'].append(current_1h_bucket)
        full_htf_ohlcv['open'].append(current_1h_open)
        full_htf_ohlcv['high'].append(current_1h_high)
        full_htf_ohlcv['low'].append(current_1h_low)
        full_htf_ohlcv['close'].append(current_1h_close)
        full_htf_ohlcv['volume'].append(current_1h_volume)

    streaming_df_1h = pd.DataFrame(full_htf_ohlcv).set_index('time')
    print(f"  Streaming aggregated 1h: {len(streaming_df_1h)} candles")
    print(f"  Preloaded 1h: {len(htf_candles_preload)} candles")

    # Compare OHLCV at key indices to find differences
    print(f"\n  Streaming vs Preload 1h OHLCV 차이 (첫 불일치 10개):")
    mismatches = []
    for idx in range(min(len(streaming_df_1h), len(htf_candles_preload))):
        s = streaming_df_1h.iloc[idx]
        p = htf_candles_preload.iloc[idx]

        diff_o = abs(s['open'] - p['open'])
        diff_h = abs(s['high'] - p['high'])
        diff_l = abs(s['low'] - p['low'])
        diff_c = abs(s['close'] - p['close'])

        if diff_o > 0.01 or diff_h > 0.01 or diff_l > 0.01 or diff_c > 0.01:
            mismatches.append({
                'idx': idx,
                'time': streaming_df_1h.index[idx],
                's_o': s['open'], 's_h': s['high'], 's_l': s['low'], 's_c': s['close'],
                'p_o': p['open'], 'p_h': p['high'], 'p_l': p['low'], 'p_c': p['close'],
            })
            if len(mismatches) >= 10:
                break

    if mismatches:
        print(f"  발견된 불일치: {len(mismatches)}개 (더 있을 수 있음)")
        for m in mismatches[:5]:
            print(f"    idx={m['idx']}, time={m['time']}")
            print(f"      stream: O={m['s_o']:.2f} H={m['s_h']:.2f} L={m['s_l']:.2f} C={m['s_c']:.2f}")
            print(f"      preload: O={m['p_o']:.2f} H={m['p_h']:.2f} L={m['p_l']:.2f} C={m['p_c']:.2f}")
    else:
        print(f"  ✓ 모든 OHLCV 일치!")

    # Build channel map from streaming data
    print(f"\n  스트리밍 aggregated 데이터로 채널 구축...")
    streaming_detector = StreamingChannelDetector(tiebreaker='narrow')
    streaming_detector.initialize_channels(streaming_df_1h)

    print(f"  스트리밍 htf_channel_map 길이: {len(streaming_detector.htf_channel_map)}")

    # Compare with original
    from ml_channel_tiebreaker_proper import build_htf_channels
    orig_htf_channel_map, _ = build_htf_channels(htf_candles_preload, tiebreaker='narrow')

    print(f"\n  주요 인덱스에서 채널 비교 (스트리밍 agg vs preload):")
    test_indices = [22, 50, 92, 93, 100, 150, 200]
    for idx in test_indices:
        stream_ch = streaming_detector.htf_channel_map.get(idx)
        orig_ch = orig_htf_channel_map.get(idx)

        stream_str = f"({stream_ch.support:.0f}, {stream_ch.resistance:.0f})" if stream_ch else "None"
        orig_str = f"({orig_ch.support:.0f}, {orig_ch.resistance:.0f})" if orig_ch else "None"

        match = "✓" if stream_str == orig_str else "✗"
        print(f"    idx={idx}: stream={stream_str:<20} orig={orig_str:<20} {match}")


def debug_compare_first_trades(candles_15m: pd.DataFrame, year: int = 2024):
    """
    Debug: Compare htf_channel_map and first trades between original and streaming approaches.
    """
    print("\n" + "="*70)
    print("  DEBUG: 원본 vs 스트리밍 htf_channel_map 비교")
    print("="*70)

    # Load 1h candles (original method)
    htf_candles = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    htf_candles = htf_candles[htf_candles.index.year == year]

    # Import and run original channel building
    from ml_channel_tiebreaker_proper import build_htf_channels
    print("\n원본 방식으로 채널 구축...")
    orig_htf_channel_map, _ = build_htf_channels(htf_candles, tiebreaker='narrow')

    # Build streaming htf_channel_map (final state)
    print("\n스트리밍 방식으로 채널 구축...")
    detector = StreamingChannelDetector(tiebreaker='narrow')
    detector.initialize_channels(htf_candles)  # Use same data for fair comparison
    stream_htf_channel_map = detector.htf_channel_map

    print(f"\n원본 htf_channel_map 길이: {len(orig_htf_channel_map)}")
    print(f"스트리밍 htf_channel_map 길이: {len(stream_htf_channel_map)}")

    # Find first indices with channels
    orig_indices = sorted(orig_htf_channel_map.keys())
    stream_indices = sorted(stream_htf_channel_map.keys())

    print(f"\n원본 첫 10개 인덱스: {orig_indices[:10]}")
    print(f"스트리밍 첫 10개 인덱스: {stream_indices[:10]}")

    # Compare channels at key indices
    test_indices = [22, 50, 92, 100, 150, 200]
    print(f"\n주요 인덱스에서 채널 비교:")
    print(f"{'idx':<6} {'원본 S/R':<25} {'스트리밍 S/R':<25} {'일치?'}")
    print("-"*70)

    for idx in test_indices:
        orig_ch = orig_htf_channel_map.get(idx)
        stream_ch = stream_htf_channel_map.get(idx)

        orig_str = f"({orig_ch.support:.0f}, {orig_ch.resistance:.0f})" if orig_ch else "None"
        stream_str = f"({stream_ch.support:.0f}, {stream_ch.resistance:.0f})" if stream_ch else "None"

        match = "✓" if orig_str == stream_str else "✗"
        print(f"{idx:<6} {orig_str:<25} {stream_str:<25} {match}")

    # Now trace through original backtest to find first trade
    print("\n" + "="*70)
    print("  원본 백테스트 첫 트레이드 찾기")
    print("="*70)

    ltf_highs = candles_15m['high'].values
    ltf_lows = candles_15m['low'].values
    ltf_closes = candles_15m['close'].values

    touch_threshold = 0.003
    tf_ratio = 4  # 1h / 15m

    first_trade_idx = None
    for i in range(len(candles_15m)):
        htf_idx = i // tf_ratio
        channel = orig_htf_channel_map.get(htf_idx - 1)

        if channel is None:
            continue

        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        mid_price = (channel.resistance + channel.support) / 2

        # Check LONG
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            print(f"\n첫 LONG 트레이드 발견!")
            print(f"  LTF idx: {i}")
            print(f"  HTF idx: {htf_idx}")
            print(f"  Channel lookup: htf_channel_map.get({htf_idx - 1})")
            print(f"  Channel: S={channel.support:.0f}, R={channel.resistance:.0f}")
            print(f"  Price: L={current_low:.0f}, C={current_close:.0f}")
            print(f"  Condition: {current_low:.0f} <= {channel.support * (1 + touch_threshold):.0f} and {current_close:.0f} > {channel.support:.0f}")
            first_trade_idx = i
            break

        # Check SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            print(f"\n첫 SHORT 트레이드 발견!")
            print(f"  LTF idx: {i}")
            print(f"  HTF idx: {htf_idx}")
            print(f"  Channel lookup: htf_channel_map.get({htf_idx - 1})")
            print(f"  Channel: S={channel.support:.0f}, R={channel.resistance:.0f}")
            print(f"  Price: H={current_high:.0f}, C={current_close:.0f}")
            print(f"  Condition: {current_high:.0f} >= {channel.resistance * (1 - touch_threshold):.0f} and {current_close:.0f} < {channel.resistance:.0f}")
            first_trade_idx = i
            break

        # Debug: show first few with channel but no entry
        if i < 200 and i % 40 == 0:
            print(f"  idx={i}: ch=({channel.support:.0f}, {channel.resistance:.0f}), price H={current_high:.0f} L={current_low:.0f} C={current_close:.0f}, no entry")

    # Now check what streaming would find at same indices
    print("\n" + "="*70)
    print("  스트리밍 방식으로 동일 인덱스 검증")
    print("="*70)

    # Check early indices to see when streaming finds trades
    for i in [92, 100, 150, 200, 300, first_trade_idx]:
        if i is None or i >= len(candles_15m):
            continue

        htf_idx = i // tf_ratio
        orig_ch = orig_htf_channel_map.get(htf_idx - 1)
        stream_ch = stream_htf_channel_map.get(htf_idx - 1)

        current_low = ltf_lows[i]
        current_high = ltf_highs[i]
        current_close = ltf_closes[i]

        print(f"\nidx={i}, htf_idx={htf_idx}, lookup={htf_idx - 1}:")

        for label, ch in [("원본", orig_ch), ("스트리밍", stream_ch)]:
            if ch:
                # Check LONG
                long_cond = current_low <= ch.support * (1 + touch_threshold) and current_close > ch.support
                # Check SHORT
                short_cond = current_high >= ch.resistance * (1 - touch_threshold) and current_close < ch.resistance

                entry = "LONG" if long_cond else ("SHORT" if short_cond else "None")
                print(f"  {label}: ch=({ch.support:.0f}, {ch.resistance:.0f}), entry={entry}")
            else:
                print(f"  {label}: ch=None")


def run_fast_htf_comparison(htf_candles: pd.DataFrame, ltf_candles: pd.DataFrame,
                            use_partial: bool, label: str) -> dict:
    """
    Fast comparison using pre-loaded data (like original backtest).

    CLOSED: channel = htf_channel_map.get(htf_idx - 1)  # Only completed candles
    PARTIAL: channel = htf_channel_map.get(htf_idx)      # Includes current candle
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    from ml_channel_tiebreaker_proper import build_htf_channels

    htf_channel_map, _ = build_htf_channels(htf_candles, tiebreaker='narrow')

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values
    ltf_opens = ltf_candles['open'].values

    setups = []
    traded_entries = set()

    tf_ratio = 4  # 1h / 15m
    touch_threshold = 0.003
    sl_buffer_pct = 0.0008

    for i in tqdm(range(len(ltf_candles)), desc=label):
        htf_idx = i // tf_ratio

        # KEY DIFFERENCE: CLOSED uses htf_idx - 1, PARTIAL uses htf_idx
        if use_partial:
            channel = htf_channel_map.get(htf_idx)
        else:
            channel = htf_channel_map.get(htf_idx - 1)

        if channel is None:
            continue

        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        mid_price = (channel.resistance + channel.support) / 2

        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        entry_signal = None

        # Support bounce → LONG
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998
            entry_signal = ('LONG', entry_price, sl_price, tp1_price, tp2_price)

        # Resistance bounce → SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002
            entry_signal = ('SHORT', entry_price, sl_price, tp1_price, tp2_price)

        if entry_signal:
            direction, entry_price, sl_price, tp1_price, tp2_price = entry_signal

            risk = abs(entry_price - sl_price)
            reward = abs(tp1_price - entry_price)

            if risk > 0 and reward > 0:
                # Simulate trade
                result = simulate_fast_trade(ltf_candles, i, direction, entry_price, sl_price, tp1_price, tp2_price)
                result['idx'] = i
                result['entry'] = entry_price
                result['sl'] = sl_price
                result['direction'] = direction
                result['channel_support'] = channel.support
                result['channel_resistance'] = channel.resistance
                setups.append(result)
                traded_entries.add(trade_key)

    if not setups:
        print("  No setups found!")
        return None

    df = pd.DataFrame(setups)

    # Backtest with position sizing
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade['pnl_pct']
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if trade['outcome'] >= 0.5:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_pnl = df['pnl_pct'].mean() * 100

    print(f"\n  Trades: {len(df)}")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg PnL: {avg_pnl:+.3f}%")
    print(f"  Return: {total_return:+.1f}%")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")

    # Show first few trades for verification
    print(f"\n  첫 5개 트레이드:")
    for _, t in df.head(5).iterrows():
        print(f"    idx={t['idx']}, {t['direction']}, entry={t['entry']:.0f}, ch=({t['channel_support']:.0f}, {t['channel_resistance']:.0f})")

    return {
        'trades': len(df),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'return': total_return,
        'max_dd': max_dd * 100,
        'final_capital': capital,
        'wins': wins,
        'losses': losses
    }


def simulate_fast_trade(candles, idx, direction, entry_price, sl_price, tp1_price, tp2_price) -> dict:
    """Fast trade simulation using LTF candles."""
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values

    is_long = direction == 'LONG'
    max_candles = 96  # 24h

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    outcome = 0
    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price

    for j in range(idx + 1, min(idx + max_candles, len(candles))):
        if is_long:
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    break
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    outcome = 0.5
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    break
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    outcome = 0.5
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break

    return {'pnl_pct': pnl_pct, 'outcome': outcome}


def main():
    parser = argparse.ArgumentParser(description="Compare CLOSED vs PARTIAL HTF modes")
    parser.add_argument("--year", type=int, default=2024, help="Year to backtest")
    parser.add_argument("--debug", action="store_true", help="Run debug comparison")
    parser.add_argument("--fast", action="store_true", help="Run fast comparison (recommended)")
    args = parser.parse_args()

    if args.debug:
        print("Loading candles for debug...")
        candles_1m = load_candles("BTCUSDT", "1m").to_pandas().set_index('time')
        candles_1m = candles_1m[candles_1m.index.year == args.year]
        candles_15m = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')
        candles_15m = candles_15m[candles_15m.index.year == args.year]

        # First check index mapping
        debug_streaming_index_mapping(candles_1m, candles_15m, args.year)

        # Then compare first trades
        debug_compare_first_trades(candles_15m, args.year)
        return

    if args.fast or True:  # Default to fast mode
        print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   HTF Mode Comparison: CLOSED vs PARTIAL (Fast Mode)                  ║
║                                                                        ║
║   CLOSED: htf_channel_map.get(htf_idx - 1) - 완료된 1h 캔들만         ║
║   PARTIAL: htf_channel_map.get(htf_idx) - 현재 1h 캔들 포함           ║
║                                                                        ║
║   Paper trading 수정 전/후 비교                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

        print(f"Loading 1h candles...")
        htf_candles = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
        htf_candles = htf_candles[htf_candles.index.year == args.year]
        print(f"  Loaded {len(htf_candles):,} candles ({args.year})")

        print(f"\nLoading 15m candles...")
        ltf_candles = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')
        ltf_candles = ltf_candles[ltf_candles.index.year == args.year]
        print(f"  Loaded {len(ltf_candles):,} candles ({args.year})")
        print(f"  Range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}")

        # Run CLOSED mode (like corrected paper trading)
        results_closed = run_fast_htf_comparison(
            htf_candles, ltf_candles,
            use_partial=False,
            label="CLOSED (htf_idx - 1) - 수정된 paper trading"
        )

        # Run PARTIAL mode (like old paper trading)
        results_partial = run_fast_htf_comparison(
            htf_candles, ltf_candles,
            use_partial=True,
            label="PARTIAL (htf_idx) - 이전 paper trading"
        )

        # Summary
        print(f"\n{'='*70}")
        print("  비교 요약")
        print(f"{'='*70}")

        if results_closed and results_partial:
            print(f"\n{'항목':<20} {'CLOSED':>15} {'PARTIAL':>15} {'차이':>15}")
            print("-"*65)
            print(f"{'총 매매':<20} {results_closed['trades']:>15} {results_partial['trades']:>15} {results_partial['trades'] - results_closed['trades']:>+15}")
            print(f"{'승률':<20} {results_closed['win_rate']:>14.1f}% {results_partial['win_rate']:>14.1f}% {results_partial['win_rate'] - results_closed['win_rate']:>+14.1f}%")
            print(f"{'평균 PnL':<20} {results_closed['avg_pnl']:>+14.3f}% {results_partial['avg_pnl']:>+14.3f}% {results_partial['avg_pnl'] - results_closed['avg_pnl']:>+14.3f}%")
            print(f"{'수익률':<20} {results_closed['return']:>+14.1f}% {results_partial['return']:>+14.1f}% {results_partial['return'] - results_closed['return']:>+14.1f}%")
            print(f"{'최대 DD':<20} {results_closed['max_dd']:>14.1f}% {results_partial['max_dd']:>14.1f}% {results_partial['max_dd'] - results_closed['max_dd']:>+14.1f}%")
            print(f"{'최종 자본':<20} ${results_closed['final_capital']:>13,.0f} ${results_partial['final_capital']:>13,.0f} ${results_partial['final_capital'] - results_closed['final_capital']:>+13,.0f}")

            print(f"\n{'='*70}")
            diff_return = results_closed['return'] - results_partial['return']
            if diff_return > 0:
                print(f"  ✅ CLOSED 방식이 {diff_return:.1f}%p 더 좋음")
                print(f"     → 수정된 paper trading이 올바른 방향")
            elif diff_return < 0:
                print(f"  ⚠️  PARTIAL 방식이 {-diff_return:.1f}%p 더 좋음")
                print(f"     → 하지만 lookahead bias 제거를 위해 CLOSED 유지 권장")
            else:
                print(f"  ⚖️  두 방식 동일")
            print(f"{'='*70}")
        return

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   HTF Mode Comparison: CLOSED vs PARTIAL (Streaming Simulation)       ║
║                                                                        ║
║   실시간과 동일하게 1m 캔들에서 1h 캔들을 점진적으로 aggregation       ║
║   CLOSED: 완료된 1h 캔들만 사용 (수정된 paper trading)                 ║
║   PARTIAL: 미완료 1h 캔들 포함 (이전 paper trading)                    ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading 1m candles...")
    candles_1m = load_candles("BTCUSDT", "1m").to_pandas().set_index('time')
    candles_1m = candles_1m[candles_1m.index.year == args.year]
    print(f"  {len(candles_1m):,} candles ({args.year})")

    print("\nLoading 15m candles...")
    candles_15m = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')
    candles_15m = candles_15m[candles_15m.index.year == args.year]
    print(f"  {len(candles_15m):,} candles ({args.year})")

    if len(candles_1m) == 0 or len(candles_15m) == 0:
        print("Error: No data found!")
        return

    print(f"  Range: {candles_1m.index[0]} ~ {candles_1m.index[-1]}")

    # Run CLOSED mode
    results_closed = run_streaming_backtest(
        candles_1m, candles_15m,
        use_partial_htf=False,
        label="CLOSED (htf_idx - 1)"
    )

    # Run PARTIAL mode
    results_partial = run_streaming_backtest(
        candles_1m, candles_15m,
        use_partial_htf=True,
        label="PARTIAL (htf_idx)"
    )

    # Summary
    print(f"\n{'='*70}")
    print("  비교 요약")
    print(f"{'='*70}")

    if results_closed and results_partial:
        print(f"\n{'항목':<20} {'CLOSED':>15} {'PARTIAL':>15} {'차이':>15}")
        print("-"*65)
        print(f"{'총 매매':<20} {results_closed['trades']:>15} {results_partial['trades']:>15} {results_partial['trades'] - results_closed['trades']:>+15}")
        print(f"{'승률':<20} {results_closed['win_rate']:>14.1f}% {results_partial['win_rate']:>14.1f}% {results_partial['win_rate'] - results_closed['win_rate']:>+14.1f}%")
        print(f"{'평균 PnL':<20} {results_closed['avg_pnl']:>+14.3f}% {results_partial['avg_pnl']:>+14.3f}% {results_partial['avg_pnl'] - results_closed['avg_pnl']:>+14.3f}%")
        print(f"{'수익률':<20} {results_closed['return']:>+14.1f}% {results_partial['return']:>+14.1f}% {results_partial['return'] - results_closed['return']:>+14.1f}%")
        print(f"{'최대 DD':<20} {results_closed['max_dd']:>14.1f}% {results_partial['max_dd']:>14.1f}% {results_partial['max_dd'] - results_closed['max_dd']:>+14.1f}%")
        print(f"{'최종 자본':<20} ${results_closed['final_capital']:>13,.0f} ${results_partial['final_capital']:>13,.0f} ${results_partial['final_capital'] - results_closed['final_capital']:>+13,.0f}")

        print(f"\n{'='*70}")
        diff_return = results_closed['return'] - results_partial['return']
        if diff_return > 0:
            print(f"  ✅ CLOSED 방식이 {diff_return:.1f}%p 더 좋음")
            print(f"     → 수정된 paper trading이 올바른 방향")
        elif diff_return < 0:
            print(f"  ⚠️  PARTIAL 방식이 {-diff_return:.1f}%p 더 좋음")
            print(f"     → 하지만 일관성을 위해 CLOSED 유지 권장")
        else:
            print(f"  ⚖️  두 방식 동일")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
