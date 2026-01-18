#!/usr/bin/env python3
"""
두 가지 채널 전략 조합 테스트

1. 기존 채널 감지 + BOUNCE only (ml_channel_tiebreaker_proper.py)
2. Oscillation 채널 감지 + FAKEOUT only (ml_pivot_oscillation.py)

두 전략을 동시에 사용하여 시너지 효과 확인
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


# ============================================================
# 기존 채널 감지 (ml_channel_tiebreaker_proper.py에서 가져옴)
# ============================================================

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


def find_swing_points_original(candles: pd.DataFrame, confirm_candles: int = 3):
    """Original swing point detection."""
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


def build_original_channels(htf_candles: pd.DataFrame,
                            max_channel_width: float = 0.05,
                            min_channel_width: float = 0.008,
                            touch_threshold: float = 0.004) -> Dict[int, Channel]:
    """Build channels using original method."""
    swing_highs, swing_lows = find_swing_points_original(htf_candles, confirm_candles=3)

    highs = htf_candles['high'].values
    lows = htf_candles['low'].values
    closes = htf_candles['close'].values

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

    for i in range(len(htf_candles)):
        current_close = closes[i]

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

        valid_swing_lows = [sl for sl in swing_lows if sl.idx + 3 <= i]
        valid_swing_highs = [sh for sh in swing_highs if sh.idx + 3 <= i]

        if new_high:
            for sl in valid_swing_lows[-30:]:
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
                                highest_high=new_high.price
                            )

        if new_low:
            for sh in valid_swing_highs[-30:]:
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
                                highest_high=sh.price
                            )

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

        candidates = []
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            candidates.append((score, channel))

        if candidates:
            best_channel = max(candidates, key=lambda c: c[0])[1]
            htf_channel_map[i] = best_channel

    return htf_channel_map


# ============================================================
# Oscillation 채널 감지 (ml_pivot_oscillation.py에서 가져옴)
# ============================================================

@dataclass
class OscillationChannel:
    support: float
    resistance: float
    oscillations: int
    start_idx: int

    @property
    def mid_price(self):
        return (self.support + self.resistance) / 2


@dataclass
class FakeoutSignal:
    htf_idx: int
    type: str  # 'bull' or 'bear'
    channel: OscillationChannel
    extreme: float


def find_pivot_points(df: pd.DataFrame, swing_len: int = 3):
    """Pivot-style swing point detection."""
    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(swing_len, len(df) - swing_len):
        is_pivot_high = True
        for j in range(1, swing_len + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_pivot_high = False
                break
        if is_pivot_high:
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        is_pivot_low = True
        for j in range(1, swing_len + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_pivot_low = False
                break
        if is_pivot_low:
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def detect_oscillation_channel(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_idx: int,
    tolerance_pct: float = 0.4,
    min_oscillations: int = 2,
    max_lookback: int = 100
) -> Optional[OscillationChannel]:
    """Detect oscillation-based channel."""
    recent_highs = [s for s in swing_highs if 0 < current_idx - s.idx <= max_lookback]
    recent_lows = [s for s in swing_lows if 0 < current_idx - s.idx <= max_lookback]

    if len(recent_highs) < 1 or len(recent_lows) < 1:
        return None

    best_channel = None
    best_oscillations = 0

    tolerance = tolerance_pct / 100

    for sl in recent_lows:
        support = sl.price
        support_touches = [s for s in recent_lows if abs(s.price - support) / support < tolerance]

        if len(support_touches) < 2:
            continue

        for sh in recent_highs:
            resistance = sh.price
            if resistance <= support:
                continue

            width_pct = (resistance - support) / support
            if width_pct < 0.008 or width_pct > 0.05:
                continue

            resistance_touches = [s for s in recent_highs if abs(s.price - resistance) / resistance < tolerance]

            if len(resistance_touches) < 1:
                continue

            all_touches = []
            for s in support_touches:
                all_touches.append(('S', s.idx))
            for r in resistance_touches:
                all_touches.append(('R', r.idx))

            all_touches.sort(key=lambda x: x[1])

            oscillations = 0
            last_type = None
            for touch_type, _ in all_touches:
                if last_type is None:
                    last_type = touch_type
                elif last_type != touch_type:
                    oscillations += 0.5
                    last_type = touch_type

            oscillations = int(oscillations)

            if oscillations >= min_oscillations and oscillations > best_oscillations:
                best_oscillations = oscillations
                start_idx = min(s.idx for s in support_touches + resistance_touches)
                best_channel = OscillationChannel(
                    support=support,
                    resistance=resistance,
                    oscillations=oscillations,
                    start_idx=start_idx
                )

    return best_channel


def build_oscillation_channels(htf_candles: pd.DataFrame) -> Dict[int, OscillationChannel]:
    """Build oscillation channel map."""
    swing_highs, swing_lows = find_pivot_points(htf_candles, swing_len=3)

    closes = htf_candles['close'].values
    channel_map = {}

    for i in range(100, len(htf_candles)):
        current_close = closes[i]
        channel = detect_oscillation_channel(swing_highs, swing_lows, i)

        if channel:
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            channel_map[i] = channel

    return channel_map


def detect_fakeouts(htf_candles: pd.DataFrame, channel_map: Dict[int, OscillationChannel]) -> List[FakeoutSignal]:
    """Detect fakeout signals."""
    closes = htf_candles['close'].values
    highs = htf_candles['high'].values
    lows = htf_candles['low'].values

    fakeout_signals = []
    pending_breaks = []
    max_wait = 5

    for i in range(len(htf_candles)):
        channel = channel_map.get(i)

        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_wait:
                pending_breaks.remove(pb)
                continue

            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], lows[i])
                if closes[i] > pb['channel'].support:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bear',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)
            else:
                pb['extreme'] = max(pb['extreme'], highs[i])
                if closes[i] < pb['channel'].resistance:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bull',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)

        if not channel:
            continue

        if closes[i] < channel.support * 0.997:
            already = any(pb['channel'].support == channel.support for pb in pending_breaks)
            if not already:
                pending_breaks.append({
                    'type': 'bear',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': lows[i]
                })
        elif closes[i] > channel.resistance * 1.003:
            already = any(pb['channel'].resistance == channel.resistance for pb in pending_breaks)
            if not already:
                pending_breaks.append({
                    'type': 'bull',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': highs[i]
                })

    return fakeout_signals


# ============================================================
# 매매 수집 및 시뮬레이션
# ============================================================

def simulate_trade(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """Simulate trade with partial TP."""
    risk = abs(entry - sl)
    reward1 = abs(tp1 - entry)
    reward2 = abs(tp2 - entry)

    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl

    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry
                    return 'loss', pnl_pct
                if highs[j] >= tp1:
                    pnl_pct += 0.5 * (reward1 / entry)
                    hit_tp1 = True
                    current_sl = entry
            else:
                if lows[j] <= current_sl:
                    return 'partial', pnl_pct
                if highs[j] >= tp2:
                    pnl_pct += 0.5 * (reward2 / entry)
                    return 'full', pnl_pct
        else:
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry
                    return 'loss', pnl_pct
                if lows[j] <= tp1:
                    pnl_pct += 0.5 * (reward1 / entry)
                    hit_tp1 = True
                    current_sl = entry
            else:
                if highs[j] >= current_sl:
                    return 'partial', pnl_pct
                if lows[j] <= tp2:
                    pnl_pct += 0.5 * (reward2 / entry)
                    return 'full', pnl_pct

    return None, 0


def collect_combined_trades(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    original_channel_map: Dict[int, Channel],
    oscillation_channel_map: Dict[int, OscillationChannel],
    fakeout_signals: List[FakeoutSignal],
    tf_ratio: int = 4
) -> List[dict]:
    """Collect trades from both strategies."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    touch_threshold = 0.003
    sl_buffer = 0.0008

    for i in range(50, len(ltf_candles) - 150):
        htf_idx = i // tf_ratio
        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]

        # ============ Strategy 1: Original Channel + BOUNCE ============
        orig_channel = original_channel_map.get(htf_idx - 1)  # Avoid lookahead

        if orig_channel:
            mid = (orig_channel.resistance + orig_channel.support) / 2
            bounce_key = ('orig', round(orig_channel.support), round(orig_channel.resistance), i // 20)

            if bounce_key not in traded_keys:
                # Support touch -> LONG
                if low <= orig_channel.support * (1 + touch_threshold) and close > orig_channel.support:
                    entry = close
                    sl = orig_channel.support * (1 - sl_buffer)
                    tp1 = mid
                    tp2 = orig_channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'strategy': 'ORIG_BOUNCE', 'direction': 'LONG',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl
                            })
                            traded_keys.add(bounce_key)

                # Resistance touch -> SHORT
                elif high >= orig_channel.resistance * (1 - touch_threshold) and close < orig_channel.resistance:
                    entry = close
                    sl = orig_channel.resistance * (1 + sl_buffer)
                    tp1 = mid
                    tp2 = orig_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'strategy': 'ORIG_BOUNCE', 'direction': 'SHORT',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl
                            })
                            traded_keys.add(bounce_key)

        # ============ Strategy 2: Oscillation Channel + FAKEOUT ============
        # NOTE: Using htf_idx has lookahead bias (trading at start of hour with close-confirmed signal)
        # Using htf_idx - 1 is correct but gives worse results, indicating the standalone is biased
        fakeout = fakeout_map.get(htf_idx - 1)  # No lookahead (conservative)

        if fakeout and i % tf_ratio == 0:
            f_channel = fakeout.channel
            f_mid = f_channel.mid_price
            fakeout_key = ('osc', round(f_channel.support), round(f_channel.resistance), htf_idx)

            if fakeout_key not in traded_keys:
                if fakeout.type == 'bear':
                    # Bear fakeout -> LONG
                    entry = close
                    sl = fakeout.extreme * (1 - sl_buffer)
                    tp1 = f_mid
                    tp2 = f_channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'strategy': 'OSC_FAKEOUT', 'direction': 'LONG',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl
                            })
                            traded_keys.add(fakeout_key)

                else:  # bull fakeout -> SHORT
                    entry = close
                    sl = fakeout.extreme * (1 + sl_buffer)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'strategy': 'OSC_FAKEOUT', 'direction': 'SHORT',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl
                            })
                            traded_keys.add(fakeout_key)

    return trades


def backtest(trades: List[dict], label: str, fixed_sizing: bool = True) -> dict:
    """Run backtest.

    Args:
        trades: List of trade dicts
        label: Label for printing
        fixed_sizing: If True, use fixed 1.5% risk per trade (no compounding)
                     If False, compound capital (unrealistic for comparison)
    """
    if not trades:
        print(f"\n{label}: 매매 없음")
        return {'trades': 0, 'wr': 0, 'dd': 0, 'ret': 0, 'final': 10000, 'avg_pnl': 0}

    initial_capital = 10000
    capital = initial_capital
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    wins, losses = 0, 0
    peak = capital
    max_dd = 0
    pnl_list = []

    for t in trades:
        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        if sl_dist <= 0:
            continue

        lev = min(risk_pct / sl_dist, max_lev)

        # Fixed sizing uses initial capital, compounding uses current capital
        base_capital = initial_capital if fixed_sizing else capital
        position = base_capital * lev

        pnl = position * t['pnl']
        fees = position * fee_pct * 2
        net = pnl - fees
        pnl_list.append(net / initial_capital * 100)  # Normalize to % of initial capital

        capital += net
        capital = max(capital, 0)

        if t['result'] != 'loss':
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / initial_capital - 1) * 100
    avg_pnl = np.mean([t['pnl'] for t in trades]) * 100

    print(f"\n{label}")
    print(f"  매매: {total}건, 승률: {wr:.1f}% ({wins}W/{losses}L)")
    print(f"  Avg PnL: {avg_pnl:+.4f}%")
    print(f"  수익률: {ret:+.1f}%, 최대 DD: {max_dd*100:.1f}%")
    print(f"  최종: ${capital:,.2f}")

    return {'trades': total, 'wr': wr, 'dd': max_dd * 100, 'ret': ret, 'final': capital, 'avg_pnl': avg_pnl}


def run_test(htf_candles, ltf_candles, label):
    """Run test for a period."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Build original channels
    print("  Building original channels...")
    orig_channel_map = build_original_channels(htf_candles)
    print(f"    Candles with channel: {len(orig_channel_map)}")

    # Build oscillation channels
    print("  Building oscillation channels...")
    osc_channel_map = build_oscillation_channels(htf_candles)
    print(f"    Candles with channel: {len(osc_channel_map)}")

    # Detect fakeouts
    print("  Detecting fakeouts...")
    fakeout_signals = detect_fakeouts(htf_candles, osc_channel_map)
    print(f"    Fakeout signals: {len(fakeout_signals)}")

    # Collect trades
    print("  Collecting trades...")
    all_trades = collect_combined_trades(
        htf_candles, ltf_candles,
        orig_channel_map, osc_channel_map, fakeout_signals
    )

    # Split by strategy
    orig_bounce = [t for t in all_trades if t['strategy'] == 'ORIG_BOUNCE']
    osc_fakeout = [t for t in all_trades if t['strategy'] == 'OSC_FAKEOUT']

    print(f"\n  수집된 매매:")
    print(f"    ORIG_BOUNCE: {len(orig_bounce)}")
    print(f"    OSC_FAKEOUT: {len(osc_fakeout)}")
    print(f"    Total: {len(all_trades)}")

    # Backtest each
    r1 = backtest(orig_bounce, "Strategy 1: Original + BOUNCE")
    r2 = backtest(osc_fakeout, "Strategy 2: Oscillation + FAKEOUT")
    r3 = backtest(all_trades, "Combined (Both Strategies)")

    return r1, r2, r3


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║   두 가지 채널 전략 조합 테스트                                          ║
║   1. Original Channel + BOUNCE                                         ║
║   2. Oscillation Channel + FAKEOUT                                      ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    # Split by year
    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]

    htf_2025 = htf_all[htf_all.index.year == 2025]
    ltf_2025 = ltf_all[ltf_all.index.year == 2025]

    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")
    print(f"  2025: HTF={len(htf_2025)}, LTF={len(ltf_2025)}")

    # Run tests
    r1_2024, r2_2024, r3_2024 = run_test(htf_2024, ltf_2024, "2024 (In-Sample)")
    r1_2025, r2_2025, r3_2025 = run_test(htf_2025, ltf_2025, "2025 (Out-of-Sample) ⭐")

    # Summary
    print("\n" + "="*100)
    print("  SUMMARY: 2024 vs 2025")
    print("="*100)

    print(f"\n{'전략':<20} {'2024':>40} | {'2025':>40}")
    print(f"{'':20} {'매매':>8} {'WR':>8} {'AvgPnL':>12} {'수익':>12} | {'매매':>8} {'WR':>8} {'AvgPnL':>12} {'수익':>12}")
    print("-"*105)

    print(f"{'ORIG_BOUNCE':<20} {r1_2024['trades']:>8} {r1_2024['wr']:>7.1f}% {r1_2024.get('avg_pnl',0):>+11.4f}% {r1_2024['ret']:>+11.1f}% | {r1_2025['trades']:>8} {r1_2025['wr']:>7.1f}% {r1_2025.get('avg_pnl',0):>+11.4f}% {r1_2025['ret']:>+11.1f}%")
    print(f"{'OSC_FAKEOUT':<20} {r2_2024['trades']:>8} {r2_2024['wr']:>7.1f}% {r2_2024.get('avg_pnl',0):>+11.4f}% {r2_2024['ret']:>+11.1f}% | {r2_2025['trades']:>8} {r2_2025['wr']:>7.1f}% {r2_2025.get('avg_pnl',0):>+11.4f}% {r2_2025['ret']:>+11.1f}%")
    print(f"{'COMBINED':<20} {r3_2024['trades']:>8} {r3_2024['wr']:>7.1f}% {r3_2024.get('avg_pnl',0):>+11.4f}% {r3_2024['ret']:>+11.1f}% | {r3_2025['trades']:>8} {r3_2025['wr']:>7.1f}% {r3_2025.get('avg_pnl',0):>+11.4f}% {r3_2025['ret']:>+11.1f}%")

    print("\n" + "="*100)
    print("  결론")
    print("="*100)

    # Check if combined is better
    combined_better = (r3_2025['ret'] > r1_2025['ret'] and r3_2025['ret'] > r2_2025['ret'])

    if combined_better:
        print(f"\n  ✅ Combined 전략이 가장 좋음!")
    else:
        best = 'ORIG_BOUNCE' if r1_2025['ret'] > r2_2025['ret'] else 'OSC_FAKEOUT'
        print(f"\n  ⚠️ 단일 전략({best})이 더 좋음")

    print(f"\n  2025 OOS 결과:")
    print(f"    ORIG_BOUNCE:  {r1_2025['trades']}건, {r1_2025['wr']:.1f}% WR, {r1_2025['ret']:+.1f}%")
    print(f"    OSC_FAKEOUT:  {r2_2025['trades']}건, {r2_2025['wr']:.1f}% WR, {r2_2025['ret']:+.1f}%")
    print(f"    COMBINED:     {r3_2025['trades']}건, {r3_2025['wr']:.1f}% WR, {r3_2025['ret']:+.1f}%")


if __name__ == "__main__":
    main()
