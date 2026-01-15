#!/usr/bin/env python3
"""
Channel Strategy V2 - Two Tests

Test A: ATR-based significant swing detection
Test B: Partial TP at mid-channel + move SL to breakeven
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@dataclass
class SwingLevel:
    idx: int
    price: float
    type: str  # 'high' or 'low'


def calculate_atr(candles: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values

    tr = np.zeros(len(candles))
    for i in range(1, len(candles)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )

    atr = np.zeros(len(candles))
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


def find_swing_points_atr(candles: pd.DataFrame, atr_mult: float = 1.5, confirm_candles: int = 3) -> Tuple[List[SwingLevel], List[SwingLevel]]:
    """
    Test A: ATR-based significant swing detection.

    Only detect swings that are significant (move > ATR * mult from previous swing).
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values

    atr = calculate_atr(candles)

    swing_highs = []
    swing_lows = []

    # Track potential swings
    potential_high_idx = 0
    potential_high_price = highs[0]
    candles_since_high = 0
    last_confirmed_low = lows[0]

    potential_low_idx = 0
    potential_low_price = lows[0]
    candles_since_low = 0
    last_confirmed_high = highs[0]

    for i in range(1, len(candles)):
        current_atr = atr[i] if atr[i] > 0 else closes[i] * 0.01
        min_move = current_atr * atr_mult

        # Check for new high
        if highs[i] > potential_high_price:
            potential_high_idx = i
            potential_high_price = highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            # Confirmed swing high if significant
            if candles_since_high == confirm_candles:
                if potential_high_price - last_confirmed_low > min_move:
                    swing_highs.append(SwingLevel(
                        idx=potential_high_idx,
                        price=potential_high_price,
                        type='high'
                    ))
                    last_confirmed_high = potential_high_price

        # Check for new low
        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            # Confirmed swing low if significant
            if candles_since_low == confirm_candles:
                if last_confirmed_high - potential_low_price > min_move:
                    swing_lows.append(SwingLevel(
                        idx=potential_low_idx,
                        price=potential_low_price,
                        type='low'
                    ))
                    last_confirmed_low = potential_low_price

        # Reset tracking
        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def collect_setups_test_a(candles: pd.DataFrame, quiet: bool = False) -> List[dict]:
    """Test A: ATR-based swing points."""

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    swing_highs, swing_lows = find_swing_points_atr(candles, atr_mult=1.5)

    print(f"  ATR Swing Highs: {len(swing_highs)}")
    print(f"  ATR Swing Lows: {len(swing_lows)}")

    setups = []
    sl_buffer_pct = 0.001
    touch_threshold = 0.003

    active_resistances = []
    active_supports = []
    used_pairs = set()

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Test A: ATR swings")

    for i in iterator:
        # Add new swing levels
        for sh in swing_highs:
            if sh.idx + 3 == i:
                active_resistances.append({'idx': sh.idx, 'price': sh.price})

        for sl in swing_lows:
            if sl.idx + 3 == i:
                active_supports.append({'idx': sl.idx, 'price': sl.price})

        # Clean invalidated levels
        active_resistances = [r for r in active_resistances
                            if closes[i] < r['price'] * 1.015 and i - r['idx'] < 200]
        active_supports = [s for s in active_supports
                         if closes[i] > s['price'] * 0.985 and i - s['idx'] < 200]

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Find channel setups
        for res in active_resistances:
            for sup in active_supports:
                if res['price'] <= sup['price']:
                    continue

                pair_key = (res['idx'], sup['idx'])
                if pair_key in used_pairs:
                    continue

                width_pct = (res['price'] - sup['price']) / sup['price']
                if width_pct > 0.05 or width_pct < 0.008:
                    continue

                # Support touch (LONG)
                if lows[i] <= sup['price'] * (1 + touch_threshold) and closes[i] > sup['price']:
                    entry_price = closes[i]
                    sl_price = sup['price'] * (1 - sl_buffer_pct)
                    tp_price = res['price'] * 0.998

                    risk = entry_price - sl_price
                    reward = tp_price - entry_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if lows[j] <= sl_price:
                                break
                            if highs[j] >= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i, 'type': 'LONG', 'setup_type': 'BOUNCE',
                            'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                            'rr_ratio': rr_ratio, 'channel_width': width_pct,
                            'level_age': i - sup['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0, 'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        used_pairs.add(pair_key)
                        break

                # Resistance touch (SHORT)
                elif highs[i] >= res['price'] * (1 - touch_threshold) and closes[i] < res['price']:
                    entry_price = closes[i]
                    sl_price = res['price'] * (1 + sl_buffer_pct)
                    tp_price = sup['price'] * 1.002

                    risk = sl_price - entry_price
                    reward = entry_price - tp_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if highs[j] >= sl_price:
                                break
                            if lows[j] <= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i, 'type': 'SHORT', 'setup_type': 'BOUNCE',
                            'entry': entry_price, 'sl': sl_price, 'tp': tp_price,
                            'rr_ratio': rr_ratio, 'channel_width': width_pct,
                            'level_age': i - res['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0, 'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        used_pairs.add(pair_key)
                        break

    return setups


def collect_setups_test_b(candles: pd.DataFrame, quiet: bool = False) -> List[dict]:
    """
    Test B: Partial TP at mid-channel + move SL to breakeven.

    - Enter at support/resistance
    - First TP: 50% position at mid-channel
    - After first TP: Move SL to breakeven (entry price)
    - Final TP: Remaining 50% at opposite channel edge
    """
    from ml_channel import find_swing_points

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    swing_highs, swing_lows = find_swing_points(candles, confirm_candles=3)

    setups = []
    sl_buffer_pct = 0.001
    touch_threshold = 0.003

    active_resistances = []
    active_supports = []
    used_pairs = set()

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Test B: Partial TP + BE")

    for i in iterator:
        for sh in swing_highs:
            if sh.idx + 3 == i:
                active_resistances.append({'idx': sh.idx, 'price': sh.price})

        for sl in swing_lows:
            if sl.idx + 3 == i:
                active_supports.append({'idx': sl.idx, 'price': sl.price})

        active_resistances = [r for r in active_resistances
                            if closes[i] < r['price'] * 1.015 and i - r['idx'] < 200]
        active_supports = [s for s in active_supports
                         if closes[i] > s['price'] * 0.985 and i - s['idx'] < 200]

        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        for res in active_resistances:
            for sup in active_supports:
                if res['price'] <= sup['price']:
                    continue

                pair_key = (res['idx'], sup['idx'])
                if pair_key in used_pairs:
                    continue

                width_pct = (res['price'] - sup['price']) / sup['price']
                if width_pct > 0.04 or width_pct < 0.005:
                    continue

                mid_price = (res['price'] + sup['price']) / 2

                # Support touch (LONG)
                if lows[i] <= sup['price'] * (1 + touch_threshold) and closes[i] > sup['price']:
                    entry_price = closes[i]
                    sl_price = sup['price'] * (1 - sl_buffer_pct)
                    tp1_price = mid_price  # First TP at mid
                    tp2_price = res['price'] * 0.998  # Final TP

                    risk = entry_price - sl_price
                    reward1 = tp1_price - entry_price
                    reward2 = tp2_price - entry_price

                    if risk > 0 and reward1 > 0:
                        # Simulate with partial TP + breakeven
                        outcome = 0
                        pnl_pct = 0
                        hit_tp1 = False
                        current_sl = sl_price

                        for j in range(i + 1, min(i + 100, len(candles))):
                            if not hit_tp1:
                                # Before first TP
                                if lows[j] <= current_sl:
                                    # Full loss
                                    pnl_pct = -risk / entry_price
                                    break
                                if highs[j] >= tp1_price:
                                    # First TP hit - take 50% profit
                                    pnl_pct += 0.5 * (reward1 / entry_price)
                                    hit_tp1 = True
                                    current_sl = entry_price  # Move to breakeven
                            else:
                                # After first TP, SL is at breakeven
                                if lows[j] <= current_sl:
                                    # Remaining 50% exits at breakeven (0 profit)
                                    break
                                if highs[j] >= tp2_price:
                                    # Final TP hit
                                    pnl_pct += 0.5 * (reward2 / entry_price)
                                    outcome = 1
                                    break

                        # If we hit TP1, consider it partial win
                        if hit_tp1 and outcome == 0:
                            outcome = 0.5  # Partial win

                        setups.append({
                            'idx': i, 'type': 'LONG', 'setup_type': 'PARTIAL_TP',
                            'entry': entry_price, 'sl': sl_price,
                            'tp1': tp1_price, 'tp2': tp2_price,
                            'rr_ratio': reward2 / risk,
                            'pnl_pct': pnl_pct,
                            'channel_width': width_pct,
                            'level_age': i - sup['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0, 'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        used_pairs.add(pair_key)
                        break

                # Resistance touch (SHORT)
                elif highs[i] >= res['price'] * (1 - touch_threshold) and closes[i] < res['price']:
                    entry_price = closes[i]
                    sl_price = res['price'] * (1 + sl_buffer_pct)
                    tp1_price = mid_price
                    tp2_price = sup['price'] * 1.002

                    risk = sl_price - entry_price
                    reward1 = entry_price - tp1_price
                    reward2 = entry_price - tp2_price

                    if risk > 0 and reward1 > 0:
                        outcome = 0
                        pnl_pct = 0
                        hit_tp1 = False
                        current_sl = sl_price

                        for j in range(i + 1, min(i + 100, len(candles))):
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
                                    break
                                if lows[j] <= tp2_price:
                                    pnl_pct += 0.5 * (reward2 / entry_price)
                                    outcome = 1
                                    break

                        if hit_tp1 and outcome == 0:
                            outcome = 0.5

                        setups.append({
                            'idx': i, 'type': 'SHORT', 'setup_type': 'PARTIAL_TP',
                            'entry': entry_price, 'sl': sl_price,
                            'tp1': tp1_price, 'tp2': tp2_price,
                            'rr_ratio': reward2 / risk,
                            'pnl_pct': pnl_pct,
                            'channel_width': width_pct,
                            'level_age': i - res['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0, 'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        used_pairs.add(pair_key)
                        break

    return setups


def analyze_results(setups: List[dict], test_name: str):
    """Analyze and print results."""
    df = pd.DataFrame(setups)

    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")

    print(f"\n  Total setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    if 'pnl_pct' in df.columns:
        # Test B with partial TP
        full_wins = len(df[df['outcome'] == 1])
        partial_wins = len(df[df['outcome'] == 0.5])
        losses = len(df[df['outcome'] == 0])

        print(f"\n  Full wins: {full_wins} ({full_wins/len(df)*100:.1f}%)")
        print(f"  Partial wins (TP1 only): {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
        print(f"  Losses: {losses} ({losses/len(df)*100:.1f}%)")

        avg_pnl = df['pnl_pct'].mean() * 100
        print(f"\n  Avg PnL per trade: {avg_pnl:+.3f}%")

        # Calculate EV
        total_pnl = df['pnl_pct'].sum()
        print(f"  Total cumulative PnL: {total_pnl*100:+.1f}%")

    else:
        # Test A standard
        win_rate = df['outcome'].mean()
        avg_rr = df['rr_ratio'].mean()
        ev = win_rate * avg_rr - (1 - win_rate)

        print(f"\n  Win rate: {win_rate*100:.1f}%")
        print(f"  Avg R:R: {avg_rr:.2f}")
        print(f"  EV: {ev:+.2f}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     Channel Strategy V2 - Two Tests                       ║
╚═══════════════════════════════════════════════════════════╝
""")

    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"

    print(f"Loading {timeframe} data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles")

    # Test A: ATR-based swings
    print("\n" + "="*60)
    print("  Running Test A: ATR-based Swing Detection")
    print("="*60)
    setups_a = collect_setups_test_a(candles)
    analyze_results(setups_a, "Test A: ATR-based Swings")

    # Test B: Partial TP + Breakeven
    print("\n" + "="*60)
    print("  Running Test B: Partial TP + Breakeven SL")
    print("="*60)
    setups_b = collect_setups_test_b(candles)
    analyze_results(setups_b, "Test B: Partial TP + Breakeven")


if __name__ == "__main__":
    main()
