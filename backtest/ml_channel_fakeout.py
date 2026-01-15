#!/usr/bin/env python3
"""
Channel Fakeout Strategy

1. Channel forms with 3+ total touches (H-L-H or L-H-L pattern)
2. Only trade FAKEOUT: Price breaks out, then returns into channel
3. Very tight SL at fakeout extreme, TP at opposite channel edge
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class Level:
    """A support or resistance level."""
    idx: int
    price: float
    type: str  # 'support' or 'resistance'
    touches: int = 1


@dataclass
class Channel:
    """A channel defined by support and resistance."""
    support_price: float
    support_idx: int
    resistance_price: float
    resistance_idx: int
    total_touches: int
    formed_at: int


@dataclass
class PendingFakeout:
    """A breakout waiting to become fakeout."""
    type: str  # 'bull' (broke resistance) or 'bear' (broke support)
    break_idx: int
    channel: Channel
    extreme: float  # The extreme price of the breakout


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 2):
    """Find swing highs and lows."""
    highs = candles['high'].values
    lows = candles['low'].values

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
                swing_highs.append((potential_high_idx, potential_high_price))

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append((potential_low_idx, potential_low_price))

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def collect_fakeout_setups(candles: pd.DataFrame,
                           touch_threshold: float = 0.003,
                           min_total_touches: int = 3,
                           max_channel_width: float = 0.04,
                           min_channel_width: float = 0.005,
                           sl_buffer_pct: float = 0.0005,
                           max_fakeout_wait: int = 5,
                           quiet: bool = False) -> List[dict]:
    """
    Collect fakeout-only setups.

    Channel forms with 3+ total touches, then trade fakeouts only.
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    swing_highs, swing_lows = find_swing_points(candles, confirm_candles=3)

    print(f"  Swing Highs: {len(swing_highs)}")
    print(f"  Swing Lows: {len(swing_lows)}")

    setups = []

    # Track levels with touch counts
    resistance_levels: List[Level] = []
    support_levels: List[Level] = []

    # Track confirmed channels
    confirmed_channels: List[Channel] = []

    # Track pending fakeouts
    pending_fakeouts: List[PendingFakeout] = []

    # Track traded channels
    traded_channels = set()

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Fakeout hunting")

    for i in iterator:
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        # Add new levels from swing points
        for sh_idx, sh_price in swing_highs:
            if sh_idx + 3 == i:
                resistance_levels.append(Level(idx=sh_idx, price=sh_price, type='resistance', touches=1))

        for sl_idx, sl_price in swing_lows:
            if sl_idx + 3 == i:
                support_levels.append(Level(idx=sl_idx, price=sl_price, type='support', touches=1))

        # Count new touches on existing levels
        for res in resistance_levels:
            if current_high >= res.price * (1 - touch_threshold) and current_high <= res.price * (1 + touch_threshold):
                if i - res.idx >= 3:  # Don't count initial swing as double touch
                    res.touches += 1
                    res.idx = i  # Update for spacing check

        for sup in support_levels:
            if current_low <= sup.price * (1 + touch_threshold) and current_low >= sup.price * (1 - touch_threshold):
                if i - sup.idx >= 3:
                    sup.touches += 1
                    sup.idx = i

        # Clean invalidated levels
        resistance_levels = [r for r in resistance_levels
                           if current_close < r.price * 1.02 and i - r.idx < 300]
        support_levels = [s for s in support_levels
                        if current_close > s.price * 0.98 and i - s.idx < 300]

        # Check for channel formation (3+ total touches)
        for res in resistance_levels:
            for sup in support_levels:
                if res.price <= sup.price:
                    continue

                width_pct = (res.price - sup.price) / sup.price
                if width_pct > max_channel_width or width_pct < min_channel_width:
                    continue

                total_touches = res.touches + sup.touches
                if total_touches < min_total_touches:
                    continue

                # Check if this channel already exists
                existing = False
                for ch in confirmed_channels:
                    if (abs(ch.support_price - sup.price) / sup.price < 0.002 and
                        abs(ch.resistance_price - res.price) / res.price < 0.002):
                        existing = True
                        ch.total_touches = max(ch.total_touches, total_touches)
                        break

                if not existing:
                    confirmed_channels.append(Channel(
                        support_price=sup.price,
                        support_idx=sup.idx,
                        resistance_price=res.price,
                        resistance_idx=res.idx,
                        total_touches=total_touches,
                        formed_at=i
                    ))

        # Clean old channels
        confirmed_channels = [c for c in confirmed_channels
                            if i - c.formed_at < 200
                            and current_close > c.support_price * 0.97
                            and current_close < c.resistance_price * 1.03]

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check for FAKEOUT entries (breakout returned)
        for fakeout in pending_fakeouts[:]:
            candles_since = i - fakeout.break_idx

            # Timeout
            if candles_since > max_fakeout_wait:
                pending_fakeouts.remove(fakeout)
                continue

            # Update extreme
            if fakeout.type == 'bear':
                fakeout.extreme = min(fakeout.extreme, current_low)
            else:
                fakeout.extreme = max(fakeout.extreme, current_high)

            channel = fakeout.channel
            channel_id = (round(channel.support_price, 2), round(channel.resistance_price, 2))

            # Mid price for partial TP
            mid_price = (channel.resistance_price + channel.support_price) / 2

            # Check for fakeout confirmation (price returned into channel)
            if fakeout.type == 'bear':
                # Broke below support, check if returned above
                if current_close > channel.support_price:
                    # LONG entry on bear fakeout
                    entry_price = current_close
                    sl_price = fakeout.extreme * (1 - sl_buffer_pct)
                    tp1_price = mid_price  # 50% at mid
                    tp2_price = channel.resistance_price * 0.998  # 50% at resistance

                    risk = entry_price - sl_price
                    reward1 = tp1_price - entry_price
                    reward2 = tp2_price - entry_price

                    if risk > 0 and reward1 > 0 and channel_id not in traded_channels:
                        rr_ratio = reward2 / risk

                        # Simulate with partial TP + breakeven SL
                        outcome = 0
                        pnl_pct = 0
                        hit_tp1 = False
                        current_sl = sl_price

                        for j in range(i + 1, min(i + 100, len(candles))):
                            if not hit_tp1:
                                if lows[j] <= current_sl:
                                    pnl_pct = -risk / entry_price
                                    outcome = 0
                                    break
                                if highs[j] >= tp1_price:
                                    pnl_pct += 0.5 * (reward1 / entry_price)
                                    hit_tp1 = True
                                    current_sl = entry_price  # Move to breakeven
                            else:
                                if lows[j] <= current_sl:
                                    outcome = 0.5  # Partial win
                                    break
                                if highs[j] >= tp2_price:
                                    pnl_pct += 0.5 * (reward2 / entry_price)
                                    outcome = 1
                                    break

                        setups.append({
                            'idx': i,
                            'type': 'LONG',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'rr_ratio': rr_ratio,
                            'pnl_pct': pnl_pct,
                            'channel_width': (channel.resistance_price - channel.support_price) / channel.support_price,
                            'total_touches': channel.total_touches,
                            'fakeout_depth_pct': (channel.support_price - fakeout.extreme) / channel.support_price * 100,
                            'candles_to_reclaim': candles_since,
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        traded_channels.add(channel_id)

                    pending_fakeouts.remove(fakeout)

            elif fakeout.type == 'bull':
                # Broke above resistance, check if returned below
                if current_close < channel.resistance_price:
                    # SHORT entry on bull fakeout
                    entry_price = current_close
                    sl_price = fakeout.extreme * (1 + sl_buffer_pct)
                    tp1_price = mid_price  # 50% at mid
                    tp2_price = channel.support_price * 1.002  # 50% at support

                    risk = sl_price - entry_price
                    reward1 = entry_price - tp1_price
                    reward2 = entry_price - tp2_price

                    if risk > 0 and reward1 > 0 and channel_id not in traded_channels:
                        rr_ratio = reward2 / risk

                        # Simulate with partial TP + breakeven SL
                        outcome = 0
                        pnl_pct = 0
                        hit_tp1 = False
                        current_sl = sl_price

                        for j in range(i + 1, min(i + 100, len(candles))):
                            if not hit_tp1:
                                if highs[j] >= current_sl:
                                    pnl_pct = -risk / entry_price
                                    outcome = 0
                                    break
                                if lows[j] <= tp1_price:
                                    pnl_pct += 0.5 * (reward1 / entry_price)
                                    hit_tp1 = True
                                    current_sl = entry_price  # Move to breakeven
                            else:
                                if highs[j] >= current_sl:
                                    outcome = 0.5  # Partial win
                                    break
                                if lows[j] <= tp2_price:
                                    pnl_pct += 0.5 * (reward2 / entry_price)
                                    outcome = 1
                                    break

                        setups.append({
                            'idx': i,
                            'type': 'SHORT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'rr_ratio': rr_ratio,
                            'pnl_pct': pnl_pct,
                            'channel_width': (channel.resistance_price - channel.support_price) / channel.support_price,
                            'total_touches': channel.total_touches,
                            'fakeout_depth_pct': (fakeout.extreme - channel.resistance_price) / channel.resistance_price * 100,
                            'candles_to_reclaim': candles_since,
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })
                        traded_channels.add(channel_id)

                    pending_fakeouts.remove(fakeout)

        # Check for new breakouts on confirmed channels
        for channel in confirmed_channels:
            channel_id = (round(channel.support_price, 2), round(channel.resistance_price, 2))

            if channel_id in traded_channels:
                continue

            # Already has a pending fakeout?
            has_pending = any(
                abs(f.channel.support_price - channel.support_price) < 1 and
                abs(f.channel.resistance_price - channel.resistance_price) < 1
                for f in pending_fakeouts
            )
            if has_pending:
                continue

            # Bear breakout (close below support)
            if current_close < channel.support_price * 0.998:
                pending_fakeouts.append(PendingFakeout(
                    type='bear',
                    break_idx=i,
                    channel=channel,
                    extreme=current_low
                ))

            # Bull breakout (close above resistance)
            elif current_close > channel.resistance_price * 1.002:
                pending_fakeouts.append(PendingFakeout(
                    type='bull',
                    break_idx=i,
                    channel=channel,
                    extreme=current_high
                ))

    return setups


def analyze_results(setups: List[dict]):
    """Analyze results."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    print(f"\n  Total Fakeout Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # Outcome breakdown
    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    # PnL stats
    avg_pnl = df['pnl_pct'].mean() * 100
    total_pnl = df['pnl_pct'].sum() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.3f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    # By fakeout depth
    print("\n  By Fakeout Depth:")
    df['depth_bucket'] = pd.cut(df['fakeout_depth_pct'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0, float('inf')])
    for bucket, group in df.groupby('depth_bucket'):
        if len(group) >= 5:
            wr = group['outcome'].mean()
            rr = group['rr_ratio'].mean()
            print(f"    {bucket}: {len(group):4} setups, WR {wr*100:5.1f}%, R:R {rr:.2f}")

    # By total touches
    print("\n  By Total Touches:")
    for touches in sorted(df['total_touches'].unique()):
        subset = df[df['total_touches'] == touches]
        if len(subset) >= 5:
            wr = subset['outcome'].mean()
            rr = subset['rr_ratio'].mean()
            ev_sub = wr * rr - (1 - wr)
            print(f"    {touches} touches: {len(subset):4} setups, WR {wr*100:5.1f}%, EV {ev_sub:+.3f}")

    # Simple backtest using actual PnL
    print("\n  Simple Backtest (1.5% risk per trade):")
    capital = 10000
    risk_pct = 0.015
    max_leverage = 20

    equity_curve = [capital]
    peak = capital
    max_dd = 0

    for _, trade in df.iterrows():
        # Calculate leverage based on SL distance
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        leverage = min(risk_pct / sl_dist, max_leverage) if sl_dist > 0 else 1

        # Apply PnL with leverage
        pnl = capital * trade['pnl_pct'] * leverage
        capital += pnl
        capital = max(capital, 0)  # Can't go negative

        equity_curve.append(capital)
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    print(f"    Starting: $10,000")
    print(f"    Final:    ${capital:,.0f}")
    print(f"    Return:   {total_return:+.1f}%")
    print(f"    Max DD:   {max_dd*100:.1f}%")
    print(f"    Trades:   {len(df)}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║       Channel FAKEOUT Strategy                            ║
║       3+ touches = channel, then trade fakeouts only      ║
╚═══════════════════════════════════════════════════════════╝
""")

    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"

    print(f"Loading {timeframe} data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles\n")

    setups = collect_fakeout_setups(candles)

    print("\n" + "="*60)
    print("  FAKEOUT RESULTS")
    print("="*60)

    analyze_results(setups)


if __name__ == "__main__":
    main()
