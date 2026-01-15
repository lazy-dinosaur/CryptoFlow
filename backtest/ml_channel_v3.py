#!/usr/bin/env python3
"""
Channel Strategy V3 - Confirmed Channel Formation

Entry only AFTER channel is confirmed:
1. Find swing highs/lows (3 candle confirmation)
2. Support touched 2+ times, Resistance touched 2+ times = Channel formed
3. Enter on NEXT touch after formation
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class Level:
    """A support or resistance level."""
    idx: int
    price: float
    type: str  # 'support' or 'resistance'
    touches: List[int] = field(default_factory=list)  # List of candle indices that touched this level

    def touch_count(self) -> int:
        return len(self.touches)


@dataclass
class Channel:
    """A confirmed channel with support and resistance."""
    support: Level
    resistance: Level
    formed_at: int  # Candle index when channel was confirmed


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3):
    """
    Find swing highs and lows using physical price action.

    Swing high: Price rising, then 3 candles fail to make new high
    Swing low: Price falling, then 3 candles fail to make new low
    """
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []  # [(idx, price), ...]
    swing_lows = []

    potential_high_idx = 0
    potential_high_price = highs[0]
    candles_since_high = 0

    potential_low_idx = 0
    potential_low_price = lows[0]
    candles_since_low = 0

    for i in range(1, len(candles)):
        # Track potential swing high
        if highs[i] > potential_high_price:
            potential_high_idx = i
            potential_high_price = highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            if candles_since_high == confirm_candles:
                swing_highs.append((potential_high_idx, potential_high_price))

        # Track potential swing low
        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append((potential_low_idx, potential_low_price))

        # Reset after confirmation
        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def collect_channel_setups(candles: pd.DataFrame,
                           touch_threshold: float = 0.003,
                           min_touches: int = 2,
                           max_channel_width: float = 0.04,
                           min_channel_width: float = 0.005,
                           sl_buffer_pct: float = 0.001,
                           quiet: bool = False) -> List[dict]:
    """
    Collect channel setups with confirmed channel formation.

    Only enter AFTER channel is formed (2+ touches on each level).
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Find all swing points (2 candles to confirm = entry at 3rd candle after swing)
    swing_highs, swing_lows = find_swing_points(candles, confirm_candles=2)

    print(f"  Swing Highs found: {len(swing_highs)}")
    print(f"  Swing Lows found: {len(swing_lows)}")

    setups = []

    # Track active levels
    resistance_levels: List[Level] = []
    support_levels: List[Level] = []

    # Track confirmed channels
    confirmed_channels: List[Channel] = []

    # Track traded channels to avoid duplicate entries
    traded_channel_ids = set()

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Finding confirmed channels")

    for i in iterator:
        current_price = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        # Add new resistance levels from swing highs
        for sh_idx, sh_price in swing_highs:
            if sh_idx + 3 == i:  # Swing confirmed
                resistance_levels.append(Level(
                    idx=sh_idx,
                    price=sh_price,
                    type='resistance',
                    touches=[sh_idx]  # Initial swing is first touch
                ))

        # Add new support levels from swing lows
        for sl_idx, sl_price in swing_lows:
            if sl_idx + 3 == i:  # Swing confirmed
                support_levels.append(Level(
                    idx=sl_idx,
                    price=sl_price,
                    type='support',
                    touches=[sl_idx]  # Initial swing is first touch
                ))

        # Check for touches on existing levels and count them
        for res in resistance_levels:
            # Touch if high comes within threshold of resistance
            if current_high >= res.price * (1 - touch_threshold) and current_high <= res.price * (1 + touch_threshold):
                if len(res.touches) == 0 or i - res.touches[-1] >= 3:  # Avoid counting same touch twice
                    res.touches.append(i)

        for sup in support_levels:
            # Touch if low comes within threshold of support
            if current_low <= sup.price * (1 + touch_threshold) and current_low >= sup.price * (1 - touch_threshold):
                if len(sup.touches) == 0 or i - sup.touches[-1] >= 3:
                    sup.touches.append(i)

        # Clean invalidated levels (price broke through significantly)
        resistance_levels = [r for r in resistance_levels
                           if current_price < r.price * 1.02 and i - r.idx < 300]
        support_levels = [s for s in support_levels
                        if current_price > s.price * 0.98 and i - s.idx < 300]

        # Check for newly formed channels
        for res in resistance_levels:
            for sup in support_levels:
                if res.price <= sup.price:
                    continue

                channel_id = (res.idx, sup.idx)

                # Skip if already confirmed or traded
                already_confirmed = any(
                    c.resistance.idx == res.idx and c.support.idx == sup.idx
                    for c in confirmed_channels
                )
                if already_confirmed:
                    continue

                # Check channel width
                width_pct = (res.price - sup.price) / sup.price
                if width_pct > max_channel_width or width_pct < min_channel_width:
                    continue

                # Check if channel is now formed (2+ touches on each)
                if res.touch_count() >= min_touches and sup.touch_count() >= min_touches:
                    confirmed_channels.append(Channel(
                        support=sup,
                        resistance=res,
                        formed_at=i
                    ))

        # Remove old confirmed channels
        confirmed_channels = [c for c in confirmed_channels
                            if i - c.formed_at < 200
                            and current_price > c.support.price * 0.98
                            and current_price < c.resistance.price * 1.02]

        # Historical features for ML
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check for entry signals on confirmed channels
        for channel in confirmed_channels:
            channel_id = (channel.resistance.idx, channel.support.idx)

            if channel_id in traded_channel_ids:
                continue

            # Only trade AFTER channel formed (not on the formation candle)
            if i <= channel.formed_at:
                continue

            width_pct = (channel.resistance.price - channel.support.price) / channel.support.price

            # LONG: Touch support
            if current_low <= channel.support.price * (1 + touch_threshold) and closes[i] > channel.support.price:
                entry_price = closes[i]
                sl_price = channel.support.price * (1 - sl_buffer_pct)
                tp_price = channel.resistance.price * 0.998

                risk = entry_price - sl_price
                reward = tp_price - entry_price

                if risk > 0 and reward > 0:
                    rr_ratio = reward / risk

                    # Simulate outcome
                    outcome = 0
                    for j in range(i + 1, min(i + 100, len(candles))):
                        if lows[j] <= sl_price:
                            outcome = 0
                            break
                        if highs[j] >= tp_price:
                            outcome = 1
                            break

                    setups.append({
                        'idx': i,
                        'type': 'LONG',
                        'entry': entry_price,
                        'sl': sl_price,
                        'tp': tp_price,
                        'rr_ratio': rr_ratio,
                        'channel_width': width_pct,
                        'support_touches': channel.support.touch_count(),
                        'resistance_touches': channel.resistance.touch_count(),
                        'total_touches': channel.support.touch_count() + channel.resistance.touch_count(),
                        'candles_since_formation': i - channel.formed_at,
                        'volume_at_entry': volumes[i],
                        'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                        'delta_at_entry': deltas[i],
                        'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'body_bullish': 1 if closes[i] > opens[i] else 0,
                        'outcome': outcome
                    })

                    traded_channel_ids.add(channel_id)
                    break

            # SHORT: Touch resistance
            elif current_high >= channel.resistance.price * (1 - touch_threshold) and closes[i] < channel.resistance.price:
                entry_price = closes[i]
                sl_price = channel.resistance.price * (1 + sl_buffer_pct)
                tp_price = channel.support.price * 1.002

                risk = sl_price - entry_price
                reward = entry_price - tp_price

                if risk > 0 and reward > 0:
                    rr_ratio = reward / risk

                    outcome = 0
                    for j in range(i + 1, min(i + 100, len(candles))):
                        if highs[j] >= sl_price:
                            outcome = 0
                            break
                        if lows[j] <= tp_price:
                            outcome = 1
                            break

                    setups.append({
                        'idx': i,
                        'type': 'SHORT',
                        'entry': entry_price,
                        'sl': sl_price,
                        'tp': tp_price,
                        'rr_ratio': rr_ratio,
                        'channel_width': width_pct,
                        'support_touches': channel.support.touch_count(),
                        'resistance_touches': channel.resistance.touch_count(),
                        'total_touches': channel.support.touch_count() + channel.resistance.touch_count(),
                        'candles_since_formation': i - channel.formed_at,
                        'volume_at_entry': volumes[i],
                        'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                        'delta_at_entry': deltas[i],
                        'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'body_bullish': 1 if closes[i] > opens[i] else 0,
                        'outcome': outcome
                    })

                    traded_channel_ids.add(channel_id)
                    break

    return setups


def analyze_results(setups: List[dict]):
    """Analyze and print results."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    print(f"\n  Total setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    win_rate = df['outcome'].mean()
    avg_rr = df['rr_ratio'].mean()
    ev = win_rate * avg_rr - (1 - win_rate)

    print(f"\n  Win rate: {win_rate*100:.1f}%")
    print(f"  Avg R:R: {avg_rr:.2f}")
    print(f"  EV: {ev:+.3f}")

    # By touch count
    print("\n  Results by Total Touches:")
    for touches in sorted(df['total_touches'].unique()):
        subset = df[df['total_touches'] == touches]
        if len(subset) >= 10:
            wr = subset['outcome'].mean()
            rr = subset['rr_ratio'].mean()
            ev_sub = wr * rr - (1 - wr)
            print(f"    {touches} touches: {len(subset):4} setups, WR {wr*100:5.1f}%, R:R {rr:.2f}, EV {ev_sub:+.3f}")

    # Simulate simple backtest
    print("\n  Simple Backtest (1% risk per trade):")
    capital = 10000
    risk_pct = 0.01

    for _, trade in df.iterrows():
        risk_amount = capital * risk_pct
        if trade['outcome'] == 1:
            capital += risk_amount * trade['rr_ratio']
        else:
            capital -= risk_amount

    total_return = (capital - 10000) / 10000 * 100
    print(f"    Starting: $10,000")
    print(f"    Final:    ${capital:,.0f}")
    print(f"    Return:   {total_return:+.1f}%")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   Channel Strategy V3 - Confirmed Channel Formation       ║
║   Entry only AFTER 2+ touches on each level               ║
╚═══════════════════════════════════════════════════════════╝
""")

    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"

    print(f"Loading {timeframe} data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles\n")

    setups = collect_channel_setups(candles)

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    analyze_results(setups)


if __name__ == "__main__":
    main()
