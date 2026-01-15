#!/usr/bin/env python3
"""
Proper Channel Strategy with Evolving Support/Resistance

Channel Logic:
1. Support = lowest swing low that contains subsequent lows
2. If new low < current support → support updates to new low
3. If new low > lowest low but < previous low → new low becomes support (channel tightens)
4. Same for resistance (inverse)

Trade Types:
- BOUNCE: Price touches support/resistance and bounces
- FAKEOUT: Price breaks through, comes back → tight SL at fakeout extreme

Exit Strategy:
- 50% at mid-channel
- Move SL to breakeven
- 50% at opposite channel edge
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class Channel:
    """Evolving channel with support and resistance."""
    support: float
    support_idx: int
    resistance: float
    resistance_idx: int
    lowest_low: float  # Track the absolute lowest for reference
    highest_high: float  # Track the absolute highest for reference
    support_touches: int = 1
    resistance_touches: int = 1
    confirmed: bool = False  # True when both S/R have been touched 2+ times


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
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
                swing_highs.append(SwingPoint(idx=potential_high_idx, price=potential_high_price, type='high'))

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append(SwingPoint(idx=potential_low_idx, price=potential_low_price, type='low'))

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def collect_proper_setups(candles: pd.DataFrame,
                          touch_threshold: float = 0.003,
                          max_channel_width: float = 0.04,
                          min_channel_width: float = 0.005,
                          sl_buffer_pct: float = 0.0008,
                          max_fakeout_wait: int = 5,
                          quiet: bool = False) -> List[dict]:
    """
    Collect setups with proper evolving channel logic.
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

    # Track evolving channels
    # Key: (initial_high_idx, initial_low_idx), Value: Channel
    active_channels: dict = {}

    # Track pending fakeouts
    pending_fakeouts = []

    # Track traded channels to avoid duplicate entries on same channel
    traded_channel_entries = set()  # (channel_key, entry_idx range)

    # Collect swing points by index for quick lookup
    swing_high_at = {sh.idx: sh for sh in swing_highs}
    swing_low_at = {sl.idx: sl for sl in swing_lows}

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Proper channel detection")

    for i in iterator:
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        # Check if new swing points are confirmed at this candle
        # Swing is confirmed 3 candles after the actual high/low
        new_high = None
        new_low = None

        for sh in swing_highs:
            if sh.idx + 3 == i:
                new_high = sh
                break

        for sl in swing_lows:
            if sl.idx + 3 == i:
                new_low = sl
                break

        # Create new channels from new swing point pairs
        if new_high and new_low:
            # We have both a new high and low confirmed at the same time
            if new_high.price > new_low.price:
                width_pct = (new_high.price - new_low.price) / new_low.price
                if min_channel_width <= width_pct <= max_channel_width:
                    key = (new_high.idx, new_low.idx)
                    active_channels[key] = Channel(
                        support=new_low.price,
                        support_idx=new_low.idx,
                        resistance=new_high.price,
                        resistance_idx=new_high.idx,
                        lowest_low=new_low.price,
                        highest_high=new_high.price,
                        support_touches=1,
                        resistance_touches=1,
                        confirmed=False
                    )

        elif new_high:
            # New high - try to pair with recent lows
            for sl in swing_lows[-20:]:  # Look at recent lows
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (new_high.idx, sl.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=sl.price,
                                support_idx=sl.idx,
                                resistance=new_high.price,
                                resistance_idx=new_high.idx,
                                lowest_low=sl.price,
                                highest_high=new_high.price,
                                support_touches=1,
                                resistance_touches=1,
                                confirmed=False
                            )

        elif new_low:
            # New low - try to pair with recent highs
            for sh in swing_highs[-20:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (sh.idx, new_low.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=new_low.price,
                                support_idx=new_low.idx,
                                resistance=sh.price,
                                resistance_idx=sh.idx,
                                lowest_low=new_low.price,
                                highest_high=sh.price,
                                support_touches=1,
                                resistance_touches=1,
                                confirmed=False
                            )

        # Update existing channels with new swing points (evolving logic)
        keys_to_remove = []
        for key, channel in active_channels.items():
            # Check if price broke through channel significantly
            if current_close < channel.lowest_low * 0.97 or current_close > channel.highest_high * 1.03:
                keys_to_remove.append(key)
                continue

            # Update support with new lows
            if new_low and new_low.price < channel.resistance:
                if new_low.price < channel.lowest_low:
                    # New lower low - update lowest and support
                    channel.lowest_low = new_low.price
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches = 1
                elif new_low.price > channel.lowest_low and new_low.price < channel.support:
                    # Higher low but still below old support - tighten support
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches += 1
                elif abs(new_low.price - channel.support) / channel.support < touch_threshold:
                    # Touch of existing support
                    channel.support_touches += 1

            # Update resistance with new highs
            if new_high and new_high.price > channel.support:
                if new_high.price > channel.highest_high:
                    # New higher high - update highest and resistance
                    channel.highest_high = new_high.price
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches = 1
                elif new_high.price < channel.highest_high and new_high.price > channel.resistance:
                    # Lower high but still above old resistance - tighten resistance
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches += 1
                elif abs(new_high.price - channel.resistance) / channel.resistance < touch_threshold:
                    # Touch of existing resistance
                    channel.resistance_touches += 1

            # Check if channel is confirmed (2+ touches on each level)
            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            # Check channel width is still valid
            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Process pending fakeouts
        for fakeout in pending_fakeouts[:]:
            candles_since = i - fakeout['break_idx']

            if candles_since > max_fakeout_wait:
                pending_fakeouts.remove(fakeout)
                continue

            # Update extreme
            if fakeout['type'] == 'bear':
                fakeout['extreme'] = min(fakeout['extreme'], current_low)
            else:
                fakeout['extreme'] = max(fakeout['extreme'], current_high)

            channel = fakeout['channel']
            channel_key = fakeout['channel_key']

            trade_key = (channel_key, 'fakeout', i // 10)  # Group by ~10 candle blocks
            if trade_key in traded_channel_entries:
                pending_fakeouts.remove(fakeout)
                continue

            mid_price = (channel.resistance + channel.support) / 2

            # Check for fakeout entry
            if fakeout['type'] == 'bear' and current_close > channel.support:
                # LONG on bear fakeout
                entry_price = current_close
                sl_price = fakeout['extreme'] * (1 - sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.resistance * 0.998

                risk = entry_price - sl_price
                reward1 = tp1_price - entry_price
                reward2 = tp2_price - entry_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(
                        candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                        channel, fakeout['extreme'], candles_since, 'FAKEOUT',
                        volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes
                    )
                    if setup:
                        setups.append(setup)
                        traded_channel_entries.add(trade_key)

                pending_fakeouts.remove(fakeout)

            elif fakeout['type'] == 'bull' and current_close < channel.resistance:
                # SHORT on bull fakeout
                entry_price = current_close
                sl_price = fakeout['extreme'] * (1 + sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.support * 1.002

                risk = sl_price - entry_price
                reward1 = entry_price - tp1_price
                reward2 = entry_price - tp2_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(
                        candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                        channel, fakeout['extreme'], candles_since, 'FAKEOUT',
                        volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes
                    )
                    if setup:
                        setups.append(setup)
                        traded_channel_entries.add(trade_key)

                pending_fakeouts.remove(fakeout)

        # Check for entries on confirmed channels
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue

            trade_key = (key, 'bounce', i // 10)
            if trade_key in traded_channel_entries:
                continue

            mid_price = (channel.resistance + channel.support) / 2
            width_pct = (channel.resistance - channel.support) / channel.support

            # Check for breakout (potential fakeout)
            if current_close < channel.support * 0.998:
                # Bear breakout
                pending_fakeouts.append({
                    'type': 'bear',
                    'break_idx': i,
                    'channel': channel,
                    'channel_key': key,
                    'extreme': current_low
                })
                continue

            if current_close > channel.resistance * 1.002:
                # Bull breakout
                pending_fakeouts.append({
                    'type': 'bull',
                    'break_idx': i,
                    'channel': channel,
                    'channel_key': key,
                    'extreme': current_high
                })
                continue

            # BOUNCE: Support touch
            if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
                entry_price = current_close
                sl_price = channel.support * (1 - sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.resistance * 0.998

                risk = entry_price - sl_price
                reward1 = tp1_price - entry_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(
                        candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                        channel, None, 0, 'BOUNCE',
                        volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes
                    )
                    if setup:
                        setups.append(setup)
                        traded_channel_entries.add(trade_key)

            # BOUNCE: Resistance touch
            elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
                entry_price = current_close
                sl_price = channel.resistance * (1 + sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = channel.support * 1.002

                risk = sl_price - entry_price
                reward1 = entry_price - tp1_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(
                        candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                        channel, None, 0, 'BOUNCE',
                        volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes
                    )
                    if setup:
                        setups.append(setup)
                        traded_channel_entries.add(trade_key)

    return setups


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price,
                   channel, fakeout_extreme, candles_to_reclaim, setup_type,
                   volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes):
    """Simulate a trade with partial TP and breakeven SL."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)
    rr_ratio = reward2 / risk if risk > 0 else 0

    outcome = 0
    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price

    for j in range(idx + 1, min(idx + 100, len(candles))):
        if trade_type == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    outcome = 0
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
                    outcome = 0
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

    width_pct = (channel.resistance - channel.support) / channel.support
    fakeout_depth = 0
    if fakeout_extreme:
        if trade_type == 'LONG':
            fakeout_depth = (channel.support - fakeout_extreme) / channel.support * 100
        else:
            fakeout_depth = (fakeout_extreme - channel.resistance) / channel.resistance * 100

    return {
        'idx': idx,
        'type': trade_type,
        'setup_type': setup_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'rr_ratio': rr_ratio,
        'pnl_pct': pnl_pct,
        'channel_width': width_pct,
        'support_touches': channel.support_touches,
        'resistance_touches': channel.resistance_touches,
        'total_touches': channel.support_touches + channel.resistance_touches,
        'fakeout_depth_pct': fakeout_depth,
        'candles_to_reclaim': candles_to_reclaim,
        'volume_at_entry': volumes[idx],
        'volume_ratio': volumes[idx] / avg_volume if avg_volume > 0 else 1,
        'delta_at_entry': deltas[idx],
        'delta_ratio': deltas[idx] / (abs(avg_delta) + 1),
        'cvd_recent': cvd_recent,
        'body_bullish': 1 if closes[idx] > opens[idx] else 0,
        'outcome': outcome
    }


def analyze_results(setups: List[dict]):
    """Analyze results."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # By setup type
    bounce_df = df[df['setup_type'] == 'BOUNCE']
    fakeout_df = df[df['setup_type'] == 'FAKEOUT']
    print(f"\n  BOUNCE:  {len(bounce_df)} setups")
    print(f"  FAKEOUT: {len(fakeout_df)} setups")

    # Outcome breakdown
    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    avg_pnl = df['pnl_pct'].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    # By setup type
    for stype in ['BOUNCE', 'FAKEOUT']:
        subset = df[df['setup_type'] == stype]
        if len(subset) > 0:
            avg = subset['pnl_pct'].mean() * 100
            wr = (subset['outcome'] >= 0.5).mean() * 100
            print(f"\n  {stype}:")
            print(f"    Trades: {len(subset)}, Avg PnL: {avg:+.4f}%, WR (incl partial): {wr:.1f}%")

    # Backtest
    print("\n  Backtest (1% risk per trade, with fees):")
    capital = 10000
    risk_pct = 0.01  # Risk 1% of capital per trade
    max_leverage = 15
    fee_pct = 0.0004  # 0.04% taker fee (entry + exit = 0.08% total)

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        # Position sizing: risk 1% of capital
        risk_amount = capital * risk_pct
        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        # Calculate PnL: trade's pnl_pct is already the price % move
        gross_pnl = position_value * trade['pnl_pct']

        # Deduct fees (entry + exit on full position)
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            print("    *** BANKRUPT ***")
            break

    total_return = (capital - 10000) / 10000 * 100
    actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"    Starting: $10,000")
    print(f"    Final:    ${capital:,.2f}")
    print(f"    Return:   {total_return:+.1f}%")
    print(f"    Max DD:   {max_dd*100:.1f}%")
    print(f"    Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
    print(f"    Trades:   {len(df)}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   Proper Channel Strategy with Evolving S/R               ║
║   BOUNCE + FAKEOUT | Partial TP + Breakeven SL            ║
╚═══════════════════════════════════════════════════════════╝
""")

    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"

    print(f"Loading {timeframe} data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles\n")

    setups = collect_proper_setups(candles)

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)

    analyze_results(setups)


if __name__ == "__main__":
    main()
