#!/usr/bin/env python3
"""
Multi-Timeframe Channel Strategy

Higher TF (1H/4H): Identify swing highs/lows and channels
Lower TF (15m): Execute entries with precision

Benefits:
- Higher TF swings are more significant
- Only need H-L (2 swings) to form channel on higher TF
- Precise entries on lower TF
- Tighter stops = better R:R
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class HTFChannel:
    """Channel identified on higher timeframe."""
    resistance: float
    resistance_idx: int
    support: float
    support_idx: int
    width_pct: float


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3):
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


def get_htf_channels(htf_candles: pd.DataFrame,
                     max_width_pct: float = 0.05,
                     min_width_pct: float = 0.008) -> List[Tuple[int, HTFChannel]]:
    """
    Get channels from higher timeframe.
    Only need H-L pair to form a channel.
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    channels = []

    # For each swing high, find nearby swing low to form channel
    for h_idx, h_price in swing_highs:
        for l_idx, l_price in swing_lows:
            if h_price <= l_price:
                continue

            # Both swings should be relatively close in time
            time_diff = abs(h_idx - l_idx)
            if time_diff > 50:  # Within 50 HTF candles
                continue

            width_pct = (h_price - l_price) / l_price
            if width_pct > max_width_pct or width_pct < min_width_pct:
                continue

            # Channel is valid from the later swing onwards
            valid_from = max(h_idx, l_idx) + 3

            channels.append((valid_from, HTFChannel(
                resistance=h_price,
                resistance_idx=h_idx,
                support=l_price,
                support_idx=l_idx,
                width_pct=width_pct
            )))

    print(f"  HTF Channels found: {len(channels)}")
    return channels


def htf_idx_to_ltf_idx(htf_idx: int, htf_timeframe: str, ltf_timeframe: str) -> int:
    """Convert HTF candle index to approximate LTF candle index."""
    # Ratio of candles
    tf_minutes = {
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240
    }

    htf_mins = tf_minutes.get(htf_timeframe, 60)
    ltf_mins = tf_minutes.get(ltf_timeframe, 15)

    ratio = htf_mins / ltf_mins
    return int(htf_idx * ratio)


def collect_mtf_setups(htf_candles: pd.DataFrame,
                       ltf_candles: pd.DataFrame,
                       htf_timeframe: str = "1h",
                       ltf_timeframe: str = "15m",
                       touch_threshold: float = 0.003,
                       sl_buffer_pct: float = 0.001,
                       max_fakeout_wait: int = 8,
                       quiet: bool = False) -> List[dict]:
    """
    Collect setups using multi-timeframe analysis.

    HTF: Channel identification
    LTF: Entry execution
    """
    # Get HTF channels
    htf_channels = get_htf_channels(htf_candles)

    # LTF data
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    setups = []

    # Track active channels on LTF
    active_channels: List[HTFChannel] = []

    # Track pending fakeouts
    pending_fakeouts = []

    # Track traded channels
    traded_channels = set()

    # Convert HTF channel valid times to LTF indices
    channel_activations = []
    for htf_idx, channel in htf_channels:
        ltf_idx = htf_idx_to_ltf_idx(htf_idx, htf_timeframe, ltf_timeframe)
        channel_activations.append((ltf_idx, channel))

    channel_activations.sort(key=lambda x: x[0])

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"MTF: {htf_timeframe}→{ltf_timeframe}")

    activation_ptr = 0

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        # Activate new channels
        while activation_ptr < len(channel_activations) and channel_activations[activation_ptr][0] <= i:
            _, channel = channel_activations[activation_ptr]
            active_channels.append(channel)
            activation_ptr += 1

        # Clean invalidated channels
        active_channels = [c for c in active_channels
                         if current_close > c.support * 0.97
                         and current_close < c.resistance * 1.03]

        # Historical features
        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check pending fakeouts
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
            channel_id = (round(channel.support, 2), round(channel.resistance, 2))

            if channel_id in traded_channels:
                pending_fakeouts.remove(fakeout)
                continue

            # Mid price for partial TP
            mid_price = (channel.resistance + channel.support) / 2

            # Check for fakeout confirmation
            if fakeout['type'] == 'bear' and current_close > channel.support:
                # LONG entry
                entry_price = current_close
                sl_price = fakeout['extreme'] * (1 - sl_buffer_pct)
                tp1_price = mid_price  # 50% at mid
                tp2_price = channel.resistance * 0.998  # 50% at resistance

                risk = entry_price - sl_price
                reward1 = tp1_price - entry_price
                reward2 = tp2_price - entry_price

                if risk > 0 and reward1 > 0:
                    rr_ratio = reward2 / risk

                    # Simulate with partial TP + breakeven
                    outcome = 0
                    pnl_pct = 0
                    hit_tp1 = False
                    current_sl = sl_price

                    for j in range(i + 1, min(i + 150, len(ltf_candles))):
                        if not hit_tp1:
                            if ltf_lows[j] <= current_sl:
                                pnl_pct = -risk / entry_price
                                outcome = 0
                                break
                            if ltf_highs[j] >= tp1_price:
                                pnl_pct += 0.5 * (reward1 / entry_price)
                                hit_tp1 = True
                                current_sl = entry_price
                        else:
                            if ltf_lows[j] <= current_sl:
                                outcome = 0.5
                                break
                            if ltf_highs[j] >= tp2_price:
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
                        'channel_width': channel.width_pct,
                        'fakeout_depth_pct': (channel.support - fakeout['extreme']) / channel.support * 100,
                        'candles_to_reclaim': candles_since,
                        'volume_at_entry': ltf_volumes[i],
                        'volume_ratio': ltf_volumes[i] / avg_volume if avg_volume > 0 else 1,
                        'delta_at_entry': ltf_deltas[i],
                        'delta_ratio': ltf_deltas[i] / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'body_bullish': 1 if ltf_closes[i] > ltf_opens[i] else 0,
                        'outcome': outcome
                    })
                    traded_channels.add(channel_id)

                pending_fakeouts.remove(fakeout)

            elif fakeout['type'] == 'bull' and current_close < channel.resistance:
                # SHORT entry
                entry_price = current_close
                sl_price = fakeout['extreme'] * (1 + sl_buffer_pct)
                tp1_price = mid_price  # 50% at mid
                tp2_price = channel.support * 1.002  # 50% at support

                risk = sl_price - entry_price
                reward1 = entry_price - tp1_price
                reward2 = entry_price - tp2_price

                if risk > 0 and reward1 > 0:
                    rr_ratio = reward2 / risk

                    # Simulate with partial TP + breakeven
                    outcome = 0
                    pnl_pct = 0
                    hit_tp1 = False
                    current_sl = sl_price

                    for j in range(i + 1, min(i + 150, len(ltf_candles))):
                        if not hit_tp1:
                            if ltf_highs[j] >= current_sl:
                                pnl_pct = -risk / entry_price
                                outcome = 0
                                break
                            if ltf_lows[j] <= tp1_price:
                                pnl_pct += 0.5 * (reward1 / entry_price)
                                hit_tp1 = True
                                current_sl = entry_price
                        else:
                            if ltf_highs[j] >= current_sl:
                                outcome = 0.5
                                break
                            if ltf_lows[j] <= tp2_price:
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
                        'channel_width': channel.width_pct,
                        'fakeout_depth_pct': (fakeout['extreme'] - channel.resistance) / channel.resistance * 100,
                        'candles_to_reclaim': candles_since,
                        'volume_at_entry': ltf_volumes[i],
                        'volume_ratio': ltf_volumes[i] / avg_volume if avg_volume > 0 else 1,
                        'delta_at_entry': ltf_deltas[i],
                        'delta_ratio': ltf_deltas[i] / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'body_bullish': 1 if ltf_closes[i] > ltf_opens[i] else 0,
                        'outcome': outcome
                    })
                    traded_channels.add(channel_id)

                pending_fakeouts.remove(fakeout)

        # Check for new breakouts
        for channel in active_channels:
            channel_id = (round(channel.support, 2), round(channel.resistance, 2))

            if channel_id in traded_channels:
                continue

            has_pending = any(
                abs(f['channel'].support - channel.support) < 1
                for f in pending_fakeouts
            )
            if has_pending:
                continue

            # Bear breakout
            if current_close < channel.support * 0.998:
                pending_fakeouts.append({
                    'type': 'bear',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_low
                })

            # Bull breakout
            elif current_close > channel.resistance * 1.002:
                pending_fakeouts.append({
                    'type': 'bull',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_high
                })

    return setups


def analyze_results(setups: List[dict]):
    """Analyze results."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # Outcome breakdown
    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    avg_pnl = df['pnl_pct'].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.3f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    # Backtest using actual PnL
    print("\n  Backtest (1.5% risk per trade):")
    capital = 10000
    risk_pct = 0.015
    max_leverage = 20

    peak = capital
    max_dd = 0

    for _, trade in df.iterrows():
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        leverage = min(risk_pct / sl_dist, max_leverage) if sl_dist > 0 else 1

        pnl = capital * trade['pnl_pct'] * leverage
        capital += pnl
        capital = max(capital, 0)

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

    # By type
    for trade_type in ['LONG', 'SHORT']:
        subset = df[df['type'] == trade_type]
        if len(subset) > 0:
            wr = subset['outcome'].mean()
            print(f"\n  {trade_type}: {len(subset)} trades, WR {wr*100:.1f}%")


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║       Multi-Timeframe Channel Fakeout Strategy            ║
║       HTF (1H) channels + LTF (15m) entries               ║
╚═══════════════════════════════════════════════════════════╝
""")

    htf = "1h"
    ltf = "15m"

    print(f"Loading {htf} data (channel detection)...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(htf_candles):,} candles\n")

    print(f"Loading {ltf} data (entry execution)...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(ltf_candles):,} candles\n")

    setups = collect_mtf_setups(htf_candles, ltf_candles, htf, ltf)

    print("\n" + "="*60)
    print("  MTF FAKEOUT RESULTS")
    print("="*60)

    analyze_results(setups)


if __name__ == "__main__":
    main()
