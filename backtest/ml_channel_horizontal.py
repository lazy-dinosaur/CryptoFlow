#!/usr/bin/env python3
"""
Horizontal Channel Strategy with Pivot-based Swing Detection

새로운 채널 감지 로직:
1. 스윙 감지: pivothigh 스타일 (양쪽 N개 캔들 비교)
2. 저장: 최근 3개 스윙 하이/로우만
3. 채널 조건: 2개가 수평(tolerance 이내)이면 채널 형성
4. 채널 높이: 0.5% 이상

Trade Types:
- BOUNCE: Price touches S/R and bounces
- FAKEOUT: Price breaks S/R, comes back

Exit Strategy:
- 50% at mid-channel
- Move SL to breakeven
- 50% at opposite channel edge
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
    support: float
    support_idx: int
    resistance: float
    resistance_idx: int
    start_idx: int


def find_pivot_swing_points(candles: pd.DataFrame, swing_len: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find swing highs and lows using pivot method (both sides comparison).
    Similar to TradingView's ta.pivothigh/ta.pivotlow.

    A swing high at index i is confirmed when:
    - high[i] > high[i-swing_len:i] AND high[i] > high[i+1:i+swing_len+1]
    - Confirmed swing_len candles AFTER the pivot point
    """
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    # Need at least swing_len candles on each side
    for i in range(swing_len, len(candles) - swing_len):
        # Check swing high
        is_swing_high = True
        for j in range(1, swing_len + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break

        if is_swing_high:
            # Swing is confirmed swing_len candles after the pivot
            confirm_idx = i + swing_len
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        # Check swing low
        is_swing_low = True
        for j in range(1, swing_len + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break

        if is_swing_low:
            confirm_idx = i + swing_len
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def build_horizontal_channels(htf_candles: pd.DataFrame,
                               swing_len: int = 3,
                               tolerance: float = 0.005,  # 0.5%
                               min_channel_height: float = 0.005) -> Dict[int, Channel]:
    """
    Build horizontal channels using pivot swing points.

    Channel conditions:
    1. 2 lows within tolerance -> support, 1 high -> resistance
    2. OR 2 highs within tolerance -> resistance, 1 low -> support
    3. Channel height > min_channel_height
    """
    swing_highs, swing_lows = find_pivot_swing_points(htf_candles, swing_len)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values

    # Result: HTF index -> active channel
    htf_channel_map: Dict[int, Channel] = {}

    # Track recent swing points (max 3 each)
    recent_highs: List[SwingPoint] = []
    recent_lows: List[SwingPoint] = []

    # Build map of when swings are confirmed
    # A swing at idx is confirmed at idx + swing_len
    high_confirm_map = {sh.idx + swing_len: sh for sh in swing_highs}
    low_confirm_map = {sl.idx + swing_len: sl for sl in swing_lows}

    channels_found = 0

    for i in range(len(htf_candles)):
        current_close = closes[i]

        # Check if new swing point confirmed at this index
        if i in high_confirm_map:
            sh = high_confirm_map[i]
            recent_highs.insert(0, sh)
            if len(recent_highs) > 3:
                recent_highs.pop()

        if i in low_confirm_map:
            sl = low_confirm_map[i]
            recent_lows.insert(0, sl)
            if len(recent_lows) > 3:
                recent_lows.pop()

        # Try to form horizontal channel
        channel = None

        # Case 1: 2 lows are horizontal + 1 high
        if len(recent_lows) >= 2 and len(recent_highs) >= 1:
            l1 = recent_lows[0]
            l2 = recent_lows[1]
            h1 = recent_highs[0]

            low_avg = (l1.price + l2.price) / 2
            low_diff = abs(l1.price - l2.price) / low_avg

            if low_diff <= tolerance:
                support_price = low_avg
                resistance_price = h1.price
                channel_height = (resistance_price - support_price) / support_price

                if channel_height > min_channel_height:
                    start_bar = min(l2.idx, h1.idx)
                    channel = Channel(
                        support=support_price,
                        support_idx=l1.idx,
                        resistance=resistance_price,
                        resistance_idx=h1.idx,
                        start_idx=start_bar
                    )

        # Case 2: 2 highs are horizontal + 1 low
        if channel is None and len(recent_highs) >= 2 and len(recent_lows) >= 1:
            h1 = recent_highs[0]
            h2 = recent_highs[1]
            l1 = recent_lows[0]

            high_avg = (h1.price + h2.price) / 2
            high_diff = abs(h1.price - h2.price) / high_avg

            if high_diff <= tolerance:
                resistance_price = high_avg
                support_price = l1.price
                channel_height = (resistance_price - support_price) / support_price

                if channel_height > min_channel_height:
                    start_bar = min(h2.idx, l1.idx)
                    channel = Channel(
                        support=support_price,
                        support_idx=l1.idx,
                        resistance=resistance_price,
                        resistance_idx=h1.idx,
                        start_idx=start_bar
                    )

        # Check if price is within channel range
        if channel:
            if channel.support * 0.98 <= current_close <= channel.resistance * 1.02:
                htf_channel_map[i] = channel
                channels_found += 1

    print(f"  HTF Candles with Channel: {channels_found}")

    return htf_channel_map


def collect_horizontal_setups(htf_candles: pd.DataFrame,
                               ltf_candles: pd.DataFrame,
                               htf_tf: str = "1h",
                               ltf_tf: str = "15m",
                               swing_len: int = 3,
                               tolerance: float = 0.005,
                               touch_threshold: float = 0.003,
                               sl_buffer_pct: float = 0.002,
                               quiet: bool = False) -> List[dict]:
    """Collect setups using horizontal channel analysis."""

    # Build HTF horizontal channels
    htf_channel_map = build_horizontal_channels(htf_candles, swing_len, tolerance)

    # LTF data
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    setups = []
    traded_entries = set()

    # Build LTF index to HTF mapping
    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    # Track pending fakeouts
    pending_breaks: List[dict] = []
    max_fakeout_wait = 5 * tf_ratio  # 5 HTF candles in LTF terms

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"Horizontal: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        # Get HTF channel for current LTF candle
        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias

        if not channel:
            # Check pending fakeouts even without active channel
            for pb in pending_breaks[:]:
                candles_since = i - pb['break_idx']
                if candles_since > max_fakeout_wait:
                    pending_breaks.remove(pb)
                    continue

                # Update extreme
                if pb['type'] == 'bear':
                    pb['extreme'] = min(pb['extreme'], current_low)
                    if current_close > pb['channel'].support:
                        # Fakeout confirmed - LONG entry
                        entry_price = current_close
                        sl_price = pb['extreme'] * (1 - sl_buffer_pct)
                        mid_price = (pb['channel'].resistance + pb['channel'].support) / 2
                        tp1_price = mid_price
                        tp2_price = pb['channel'].resistance * 0.998

                        trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                        if trade_key not in traded_entries and entry_price > sl_price and tp1_price > entry_price:
                            setup = simulate_trade(
                                ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                                pb['channel'], pb['extreme'], 'FAKEOUT',
                                ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                            )
                            if setup:
                                setups.append(setup)
                                traded_entries.add(trade_key)
                        pending_breaks.remove(pb)
                else:  # bull fakeout
                    pb['extreme'] = max(pb['extreme'], current_high)
                    if current_close < pb['channel'].resistance:
                        # Fakeout confirmed - SHORT entry
                        entry_price = current_close
                        sl_price = pb['extreme'] * (1 + sl_buffer_pct)
                        mid_price = (pb['channel'].resistance + pb['channel'].support) / 2
                        tp1_price = mid_price
                        tp2_price = pb['channel'].support * 1.002

                        trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                        if trade_key not in traded_entries and sl_price > entry_price and entry_price > tp1_price:
                            setup = simulate_trade(
                                ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                                pb['channel'], pb['extreme'], 'FAKEOUT',
                                ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                            )
                            if setup:
                                setups.append(setup)
                                traded_entries.add(trade_key)
                        pending_breaks.remove(pb)
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Check for breakouts (potential fakeouts)
        if current_close < channel.support * 0.997:
            already_tracking = any(
                pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                for pb in pending_breaks
            )
            if not already_tracking:
                pending_breaks.append({
                    'type': 'bear',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_low
                })
        elif current_close > channel.resistance * 1.003:
            already_tracking = any(
                pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                for pb in pending_breaks
            )
            if not already_tracking:
                pending_breaks.append({
                    'type': 'bull',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_high
                })

        # Process pending fakeouts
        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_fakeout_wait:
                pending_breaks.remove(pb)
                continue

            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], current_low)
                if current_close > pb['channel'].support:
                    entry_price = current_close
                    sl_price = pb['extreme'] * (1 - sl_buffer_pct)
                    tp1_price = (pb['channel'].resistance + pb['channel'].support) / 2
                    tp2_price = pb['channel'].resistance * 0.998

                    trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                    if trade_key not in traded_entries and entry_price > sl_price and tp1_price > entry_price:
                        setup = simulate_trade(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                            pb['channel'], pb['extreme'], 'FAKEOUT',
                            ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                        )
                        if setup:
                            setups.append(setup)
                            traded_entries.add(trade_key)
                    pending_breaks.remove(pb)
            else:
                pb['extreme'] = max(pb['extreme'], current_high)
                if current_close < pb['channel'].resistance:
                    entry_price = current_close
                    sl_price = pb['extreme'] * (1 + sl_buffer_pct)
                    tp1_price = (pb['channel'].resistance + pb['channel'].support) / 2
                    tp2_price = pb['channel'].support * 1.002

                    trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                    if trade_key not in traded_entries and sl_price > entry_price and entry_price > tp1_price:
                        setup = simulate_trade(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                            pb['channel'], pb['extreme'], 'FAKEOUT',
                            ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                        )
                        if setup:
                            setups.append(setup)
                            traded_entries.add(trade_key)
                    pending_breaks.remove(pb)

        # Check for bounce entries
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            if entry_price > sl_price and tp1_price > entry_price:
                setup = simulate_trade(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE',
                    ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

        # BOUNCE: Resistance touch
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            if sl_price > entry_price and entry_price > tp1_price:
                setup = simulate_trade(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE',
                    ltf_volumes, ltf_deltas, ltf_opens, ltf_closes
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

    return setups


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price,
                   channel, fakeout_extreme, setup_type,
                   volumes, deltas, opens, closes):
    """Simulate trade with partial TP + breakeven."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)
    rr_ratio = reward2 / risk if risk > 0 else 0

    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price
    outcome = 0

    # Historical features
    hist_start = max(0, idx - 20)
    avg_volume = volumes[hist_start:idx].mean() if idx > hist_start else volumes[idx]
    avg_delta = deltas[hist_start:idx].mean() if idx > hist_start else 0
    cvd_recent = deltas[hist_start:idx].sum() if idx > hist_start else 0

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == 'LONG':
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
        'fakeout_depth_pct': fakeout_depth,
        'volume_at_entry': volumes[idx],
        'volume_ratio': volumes[idx] / avg_volume if avg_volume > 0 else 1,
        'delta_at_entry': deltas[idx],
        'delta_ratio': deltas[idx] / (abs(avg_delta) + 1),
        'cvd_recent': cvd_recent,
        'body_bullish': 1 if closes[idx] > opens[idx] else 0,
        'outcome': outcome
    }


def run_backtest(df: pd.DataFrame, label: str = ""):
    """Run backtest on given dataframe."""
    if len(df) == 0:
        print(f"  {label}: No trades")
        return

    capital = 10000
    risk_pct = 0.015  # 1.5% risk per trade
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

        if net_pnl > 0:
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
    actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_pnl = df['pnl_pct'].mean() * 100

    print(f"\n  {label}:")
    print(f"    Trades: {len(df)}, Avg PnL: {avg_pnl:+.4f}%")
    print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"    Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
    print(f"    Final: ${capital:,.2f}")


def analyze_results(setups: List[dict], ltf_candles: pd.DataFrame):
    """Analyze results with IS/OOS split."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    # Get timestamps for each setup
    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    bounce_df = df[df['setup_type'] == 'BOUNCE']
    fakeout_df = df[df['setup_type'] == 'FAKEOUT']
    print(f"\n  BOUNCE:  {len(bounce_df)}")
    print(f"  FAKEOUT: {len(fakeout_df)}")

    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    avg_pnl = df['pnl_pct'].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    for stype in ['BOUNCE', 'FAKEOUT']:
        subset = df[df['setup_type'] == stype]
        if len(subset) > 0:
            avg = subset['pnl_pct'].mean() * 100
            wr = (subset['outcome'] >= 0.5).mean() * 100
            print(f"\n  {stype}: {len(subset)} trades, Avg PnL: {avg:+.4f}%, WR: {wr:.1f}%")

    # Split by year for IS/OOS
    years = sorted(df['year'].unique())
    print(f"\n  Years in data: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # IS = 2024, OOS = 2025
    is_df = df[df['year'] == 2024]
    oos_df = df[df['year'] == 2025]

    print("\n" + "="*60)
    print("  BACKTEST RESULTS (1.5% risk, with fees)")
    print("="*60)

    # Full backtest
    run_backtest(df, "FULL (All Data)")

    # IS/OOS split
    run_backtest(is_df, "IN-SAMPLE (2024)")
    run_backtest(oos_df, "OUT-OF-SAMPLE (2025) ⭐")


def main():
    # Parse arguments: python script.py [htf] [ltf] [swing_len] [tolerance]
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    swing_len = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tolerance = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Horizontal Channel Strategy (Pivot-based)               ║
║   HTF ({htf}) channels + LTF ({ltf}) entries                   ║
║   Swing Length: {swing_len}, Tolerance: {tolerance}%                     ║
║   BOUNCE + FAKEOUT | Partial TP + Breakeven               ║
╚═══════════════════════════════════════════════════════════╝
""")

    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(htf_candles):,} candles\n")

    print(f"Loading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(ltf_candles):,} candles")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}\n")

    setups = collect_horizontal_setups(
        htf_candles, ltf_candles, htf, ltf,
        swing_len=swing_len,
        tolerance=tolerance / 100  # Convert to decimal
    )

    print("\n" + "="*60)
    print("  HORIZONTAL CHANNEL RESULTS")
    print("="*60)

    analyze_results(setups, ltf_candles)


if __name__ == "__main__":
    main()
