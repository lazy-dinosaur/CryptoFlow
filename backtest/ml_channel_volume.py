#!/usr/bin/env python3
"""
Channel + Volume Squeeze Strategy

Volume Pattern Detection:
1. Volume contraction during channel formation (squeeze)
2. Volume expansion at channel edges → imminent move
3. Trade based on outcome:
   - FAKEOUT: Price breaks and returns → trade reversal
   - BREAKOUT: Price breaks and continues → trend follow

Entry Conditions:
1. Channel confirmed (3+ touches)
2. Volume squeeze detected (decreasing volume)
3. Volume expansion at edge (sudden increase)
4. Direction determined by price action after expansion
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str


@dataclass
class Channel:
    support: float
    support_idx: int
    resistance: float
    resistance_idx: int
    lowest_low: float
    highest_high: float
    support_touches: int = 1
    resistance_touches: int = 1
    confirmed: bool = False
    # Volume tracking
    avg_volume_at_formation: float = 0
    volume_contracting: bool = False


@dataclass
class FakeoutSignal:
    htf_idx: int
    type: str  # 'bull' or 'bear'
    channel: Channel
    extreme: float
    volume_at_break: float


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


def detect_volume_squeeze(volumes: np.ndarray, idx: int, lookback: int = 20) -> Tuple[bool, float]:
    """
    Detect volume squeeze (contraction).

    Returns:
        - is_squeeze: True if volume is contracting
        - squeeze_ratio: Current volume / avg volume (lower = more squeeze)
    """
    if idx < lookback:
        return False, 1.0

    recent_volumes = volumes[idx-lookback:idx]
    avg_volume = recent_volumes.mean()

    # Check if volume is decreasing (squeeze)
    first_half = recent_volumes[:lookback//2].mean()
    second_half = recent_volumes[lookback//2:].mean()

    is_contracting = second_half < first_half * 0.8  # 20% decrease

    current_volume = volumes[idx]
    squeeze_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    return is_contracting, squeeze_ratio


def detect_volume_expansion(volumes: np.ndarray, idx: int, lookback: int = 10) -> Tuple[bool, float]:
    """
    Detect volume expansion (sudden increase).

    Returns:
        - is_expansion: True if volume is expanding
        - expansion_ratio: Current volume / recent avg (higher = more expansion)
    """
    if idx < lookback:
        return False, 1.0

    recent_volumes = volumes[idx-lookback:idx]
    avg_volume = recent_volumes.mean()

    current_volume = volumes[idx]
    expansion_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

    is_expansion = expansion_ratio > 1.5  # 50% above average

    return is_expansion, expansion_ratio


def build_htf_channels_with_volume(htf_candles: pd.DataFrame,
                                    max_channel_width: float = 0.05,
                                    min_channel_width: float = 0.008,
                                    touch_threshold: float = 0.004) -> Tuple[Dict[int, Channel], List[FakeoutSignal], List[dict]]:
    """
    Build channels with volume analysis.

    Returns:
        - htf_channel_map: HTF index -> Channel
        - fakeout_signals: Fakeout signals with volume data
        - breakout_signals: Breakout signals with volume data
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    highs = htf_candles['high'].values
    lows = htf_candles['low'].values
    closes = htf_candles['close'].values
    volumes = htf_candles['volume'].values

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}
    fakeout_signals: List[FakeoutSignal] = []
    breakout_signals: List[dict] = []
    pending_breaks: List[dict] = []
    max_fakeout_wait_htf = 5

    for i in range(len(htf_candles)):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        current_volume = volumes[i]

        # Detect volume patterns
        is_squeeze, squeeze_ratio = detect_volume_squeeze(volumes, i)
        is_expansion, expansion_ratio = detect_volume_expansion(volumes, i)

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

        # Create new channels
        if new_high:
            for sl in swing_lows[-30:]:
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (new_high.idx, sl.idx)
                        if key not in active_channels:
                            # Calculate avg volume at formation
                            start_idx = max(0, sl.idx)
                            avg_vol = volumes[start_idx:i].mean() if i > start_idx else current_volume

                            active_channels[key] = Channel(
                                support=sl.price,
                                support_idx=sl.idx,
                                resistance=new_high.price,
                                resistance_idx=new_high.idx,
                                lowest_low=sl.price,
                                highest_high=new_high.price,
                                avg_volume_at_formation=avg_vol
                            )

        if new_low:
            for sh in swing_highs[-30:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (sh.idx, new_low.idx)
                        if key not in active_channels:
                            start_idx = max(0, new_low.idx)
                            avg_vol = volumes[start_idx:i].mean() if i > start_idx else current_volume

                            active_channels[key] = Channel(
                                support=new_low.price,
                                support_idx=new_low.idx,
                                resistance=sh.price,
                                resistance_idx=sh.idx,
                                lowest_low=new_low.price,
                                highest_high=sh.price,
                                avg_volume_at_formation=avg_vol
                            )

        # Process pending breakouts for fakeout/breakout detection
        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_fakeout_wait_htf:
                # No return = confirmed breakout
                breakout_signals.append({
                    'htf_idx': i,
                    'type': pb['type'],
                    'channel': pb['channel'],
                    'break_price': pb['break_price'],
                    'volume_at_break': pb['volume_at_break'],
                    'had_squeeze': pb['had_squeeze']
                })
                pending_breaks.remove(pb)
                continue

            # Update extreme
            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], current_low)
                # Check if price returned inside channel
                if current_close > pb['channel'].support:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bear',
                        channel=pb['channel'],
                        extreme=pb['extreme'],
                        volume_at_break=pb['volume_at_break']
                    ))
                    pending_breaks.remove(pb)
            else:  # bull
                pb['extreme'] = max(pb['extreme'], current_high)
                if current_close < pb['channel'].resistance:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bull',
                        channel=pb['channel'],
                        extreme=pb['extreme'],
                        volume_at_break=pb['volume_at_break']
                    ))
                    pending_breaks.remove(pb)

        # Check for new breakouts on confirmed channels (with volume expansion)
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue

            # Update volume squeeze status
            if is_squeeze:
                channel.volume_contracting = True

            # Bear breakout (with volume expansion preferred)
            if current_close < channel.support * 0.997:
                already_tracking = any(
                    pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append({
                        'type': 'bear',
                        'break_idx': i,
                        'channel': Channel(
                            support=channel.support,
                            support_idx=channel.support_idx,
                            resistance=channel.resistance,
                            resistance_idx=channel.resistance_idx,
                            lowest_low=channel.lowest_low,
                            highest_high=channel.highest_high,
                            support_touches=channel.support_touches,
                            resistance_touches=channel.resistance_touches,
                            confirmed=True,
                            avg_volume_at_formation=channel.avg_volume_at_formation,
                            volume_contracting=channel.volume_contracting
                        ),
                        'extreme': current_low,
                        'break_price': current_close,
                        'volume_at_break': current_volume,
                        'had_squeeze': channel.volume_contracting,
                        'expansion_ratio': expansion_ratio
                    })

            # Bull breakout
            elif current_close > channel.resistance * 1.003:
                already_tracking = any(
                    pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append({
                        'type': 'bull',
                        'break_idx': i,
                        'channel': Channel(
                            support=channel.support,
                            support_idx=channel.support_idx,
                            resistance=channel.resistance,
                            resistance_idx=channel.resistance_idx,
                            lowest_low=channel.lowest_low,
                            highest_high=channel.highest_high,
                            support_touches=channel.support_touches,
                            resistance_touches=channel.resistance_touches,
                            confirmed=True,
                            avg_volume_at_formation=channel.avg_volume_at_formation,
                            volume_contracting=channel.volume_contracting
                        ),
                        'extreme': current_high,
                        'break_price': current_close,
                        'volume_at_break': current_volume,
                        'had_squeeze': channel.volume_contracting,
                        'expansion_ratio': expansion_ratio
                    })

        # Update existing channels
        keys_to_remove = []
        for key, channel in active_channels.items():
            if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                keys_to_remove.append(key)
                continue

            if new_low and new_low.price < channel.resistance:
                if new_low.price < channel.lowest_low:
                    channel.lowest_low = new_low.price
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches = 1
                elif new_low.price > channel.lowest_low and new_low.price < channel.support:
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches += 1
                elif abs(new_low.price - channel.support) / channel.support < touch_threshold:
                    channel.support_touches += 1

            if new_high and new_high.price > channel.support:
                if new_high.price > channel.highest_high:
                    channel.highest_high = new_high.price
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches = 1
                elif new_high.price < channel.highest_high and new_high.price > channel.resistance:
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches += 1
                elif abs(new_high.price - channel.resistance) / channel.resistance < touch_threshold:
                    channel.resistance_touches += 1

            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Find best channel
        best_channel = None
        best_score = -1

        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            if score > best_score:
                best_score = score
                best_channel = channel

        if best_channel:
            htf_channel_map[i] = best_channel

    confirmed_count = len(set(id(c) for c in htf_channel_map.values()))
    print(f"  HTF Confirmed Channels: {confirmed_count}")
    print(f"  HTF Fakeout Signals: {len(fakeout_signals)}")
    print(f"  HTF Breakout Signals: {len(breakout_signals)}")

    return htf_channel_map, fakeout_signals, breakout_signals


def collect_setups_with_volume(htf_candles: pd.DataFrame,
                                ltf_candles: pd.DataFrame,
                                htf_tf: str = "1h",
                                ltf_tf: str = "15m",
                                touch_threshold: float = 0.003,
                                sl_buffer_pct: float = 0.0008,
                                require_squeeze: bool = False) -> List[dict]:
    """
    Collect setups with volume squeeze/expansion filtering.
    """
    htf_channel_map, htf_fakeout_signals, htf_breakout_signals = build_htf_channels_with_volume(htf_candles)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}
    htf_breakout_map = {bs['htf_idx']: bs for bs in htf_breakout_signals}

    setups = []
    traded_entries = set()

    iterator = tqdm(range(len(ltf_candles)), desc=f"Volume: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias

        if not channel:
            continue

        # Volume analysis on LTF
        is_squeeze, squeeze_ratio = detect_volume_squeeze(ltf_volumes, i)
        is_expansion, expansion_ratio = detect_volume_expansion(ltf_volumes, i)

        # Historical features
        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        mid_price = (channel.resistance + channel.support) / 2

        # =====================================================================
        # FAKEOUT Trades (HTF signal)
        # =====================================================================
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)  # Fix lookahead bias
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2

            trade_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)
            if trade_key not in traded_entries:
                if fakeout_signal.type == 'bear':
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 - sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.resistance * 0.998

                    risk = entry_price - sl_price
                    reward1 = tp1_price - entry_price

                    if risk > 0 and reward1 > 0:
                        setup = simulate_trade(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )
                        if setup:
                            setup['setup_type'] = 'FAKEOUT'
                            setup['volume_at_break'] = fakeout_signal.volume_at_break
                            setup['squeeze_ratio'] = squeeze_ratio
                            setup['expansion_ratio'] = expansion_ratio
                            setups.append(setup)
                            traded_entries.add(trade_key)

                else:  # bull fakeout
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.support * 1.002

                    risk = sl_price - entry_price
                    reward1 = entry_price - tp1_price

                    if risk > 0 and reward1 > 0:
                        setup = simulate_trade(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                        )
                        if setup:
                            setup['setup_type'] = 'FAKEOUT'
                            setup['volume_at_break'] = fakeout_signal.volume_at_break
                            setup['squeeze_ratio'] = squeeze_ratio
                            setup['expansion_ratio'] = expansion_ratio
                            setups.append(setup)
                            traded_entries.add(trade_key)

        # =====================================================================
        # BREAKOUT Trades (HTF signal) - Trend Following
        # =====================================================================
        breakout_signal = htf_breakout_map.get(htf_idx)
        if breakout_signal and i % tf_ratio == 0:
            b_channel = breakout_signal['channel']
            channel_height = b_channel.resistance - b_channel.support

            trade_key = (round(b_channel.support), round(b_channel.resistance), 'breakout', htf_idx)
            if trade_key not in traded_entries:
                # Trade breakouts (volume expansion preferred but not required)
                expansion = breakout_signal.get('expansion_ratio', 1)
                if True:  # Trade all breakouts for now, filter by volume later
                    if breakout_signal['type'] == 'bull':
                        # Bullish breakout → LONG
                        entry_price = current_close
                        sl_price = b_channel.resistance * 0.998  # Just below old resistance
                        tp1_price = entry_price + channel_height * 0.5  # 50% of channel
                        tp2_price = entry_price + channel_height  # Full channel height

                        risk = entry_price - sl_price
                        if risk > 0:
                            setup = simulate_trade(
                                ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                            )
                            if setup:
                                setup['setup_type'] = 'BREAKOUT'
                                setup['had_squeeze'] = breakout_signal.get('had_squeeze', False)
                                setup['expansion_ratio'] = expansion
                                setup['break_type'] = 'bull'
                                setups.append(setup)
                                traded_entries.add(trade_key)

                    else:  # bear breakout
                        entry_price = current_close
                        sl_price = b_channel.support * 1.002
                        tp1_price = entry_price - channel_height * 0.5
                        tp2_price = entry_price - channel_height

                        risk = sl_price - entry_price
                        if risk > 0:
                            setup = simulate_trade(
                                ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                            )
                            if setup:
                                setup['setup_type'] = 'BREAKOUT'
                                setup['had_squeeze'] = breakout_signal.get('had_squeeze', False)
                                setup['expansion_ratio'] = expansion
                                setup['break_type'] = 'bear'
                                setups.append(setup)
                                traded_entries.add(trade_key)

        # =====================================================================
        # BOUNCE Trades (Channel touches with volume)
        # =====================================================================
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        # Optional: require squeeze before bounce entry
        if require_squeeze and not is_squeeze:
            continue

        # Support touch → LONG
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            # Prefer entries with volume expansion at support
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price

            if risk > 0 and reward1 > 0:
                setup = simulate_trade(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                )
                if setup:
                    setup['setup_type'] = 'BOUNCE'
                    setup['squeeze_ratio'] = squeeze_ratio
                    setup['expansion_ratio'] = expansion_ratio
                    setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                    setups.append(setup)
                    traded_entries.add(trade_key)

        # Resistance touch → SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

            if risk > 0 and reward1 > 0:
                setup = simulate_trade(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                )
                if setup:
                    setup['setup_type'] = 'BOUNCE'
                    setup['squeeze_ratio'] = squeeze_ratio
                    setup['expansion_ratio'] = expansion_ratio
                    setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                    setups.append(setup)
                    traded_entries.add(trade_key)

    return setups


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price):
    """Simulate a trade with partial TP."""
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

    return {
        'idx': idx,
        'type': trade_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'rr_ratio': rr_ratio,
        'pnl_pct': pnl_pct,
        'outcome': outcome
    }


def analyze_results(setups: List[dict], ltf_candles: pd.DataFrame):
    """Analyze results with volume metrics."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # By setup type
    for stype in ['BOUNCE', 'FAKEOUT', 'BREAKOUT']:
        subset = df[df['setup_type'] == stype]
        if len(subset) > 0:
            avg = subset['pnl_pct'].mean() * 100
            wr = (subset['outcome'] >= 0.5).mean() * 100
            print(f"\n  {stype}: {len(subset)} trades, Avg PnL: {avg:+.4f}%, WR: {wr:.1f}%")

    # Overall stats
    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    avg_pnl = df['pnl_pct'].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}%")

    years = sorted(df['year'].unique())
    print(f"\n  Years: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # Backtest
    def run_backtest(subset: pd.DataFrame, label: str):
        if len(subset) == 0:
            print(f"\n  {label}: No trades")
            return

        subset = subset.sort_values('idx')

        capital = 10000
        risk_pct = 0.01
        max_leverage = 15
        fee_pct = 0.0004

        peak = capital
        max_dd = 0
        wins = 0
        losses = 0

        for _, trade in subset.iterrows():
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
                print(f"    *** BANKRUPT ***")
                break

        total_return = (capital - 10000) / 10000 * 100
        actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        avg_pnl = subset['pnl_pct'].mean() * 100

        print(f"\n  {label}:")
        print(f"    Trades: {len(subset)}, Avg PnL: {avg_pnl:+.4f}%")
        print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
        print(f"    Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
        print(f"    Final: ${capital:,.2f}")

    print("\n" + "="*60)
    print("  BACKTEST (1% risk, with fees)")
    print("="*60)

    run_backtest(df, "FULL (All Data)")

    is_df = df[df['year'] == 2024]
    oos_df = df[df['year'] == 2025]

    run_backtest(is_df, "IN-SAMPLE (2024)")
    run_backtest(oos_df, "OUT-OF-SAMPLE (2025)")

    # By setup type
    print("\n" + "="*60)
    print("  BY SETUP TYPE")
    print("="*60)

    for stype in ['BOUNCE', 'FAKEOUT', 'BREAKOUT']:
        subset = df[df['setup_type'] == stype]
        run_backtest(subset, stype)


def main():
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Channel + Volume Squeeze Strategy                       ║
║   HTF ({htf}) channels + LTF ({ltf}) entries                   ║
║                                                           ║
║   BOUNCE: Channel touches                                 ║
║   FAKEOUT: Break and return (reversal)                    ║
║   BREAKOUT: Break and continue (trend follow)             ║
║                                                           ║
║   Volume Filter: Squeeze → Expansion → Entry              ║
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

    setups = collect_setups_with_volume(htf_candles, ltf_candles, htf, ltf)

    print("\n" + "="*60)
    print("  VOLUME STRATEGY RESULTS")
    print("="*60)

    analyze_results(setups, ltf_candles)


if __name__ == "__main__":
    main()
