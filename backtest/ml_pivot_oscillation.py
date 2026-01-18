#!/usr/bin/env python3
"""
피봇 방식 + Oscillation 채널 감지 테스트

채널 감지:
- 스윙 감지: Pivot 방식 (양쪽 N개씩 비교)
- 채널 확정: Oscillation (S→R→S 왔다갔다 확인)

매매 유형:
- BOUNCE: S/R 터치 후 반등
- FAKEOUT: S/R 돌파 후 복귀

테스트:
1. BOUNCE만
2. FAKEOUT만
3. 둘 다

Volume/Delta 분석:
- 채널 내 평균 볼륨/델타
- 터치 시점의 ratio
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
    resistance: float
    start_idx: int
    oscillation_count: int  # S→R→S 왔다갔다 횟수
    avg_volume: float = 0
    avg_delta: float = 0

    @property
    def width_pct(self):
        return (self.resistance - self.support) / self.support * 100

    @property
    def mid_price(self):
        return (self.support + self.resistance) / 2


@dataclass
class FakeoutSignal:
    htf_idx: int
    type: str  # 'bull' or 'bear'
    channel: Channel
    extreme: float


def find_pivot_points(df: pd.DataFrame, swing_len: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    피봇 방식 스윙 포인트 찾기 (양쪽 N개씩 비교)
    """
    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(swing_len, len(df) - swing_len):
        # Pivot High: 양쪽 N개보다 모두 높아야 함
        is_pivot_high = True
        for j in range(1, swing_len + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_pivot_high = False
                break
        if is_pivot_high:
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        # Pivot Low: 양쪽 N개보다 모두 낮아야 함
        is_pivot_low = True
        for j in range(1, swing_len + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_pivot_low = False
                break
        if is_pivot_low:
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def detect_oscillation_channel(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_idx: int,
    tolerance_pct: float = 0.4,
    min_oscillations: int = 2,
    max_lookback: int = 100
) -> Optional[Channel]:
    """
    Oscillation 기반 채널 감지

    S→R→S 왔다갔다 패턴 확인:
    1. Support 터치 → Resistance 터치 → Support 터치 = 1 oscillation
    """
    # 최근 스윙 포인트만 사용
    recent_highs = [s for s in swing_highs if 0 < current_idx - s.idx <= max_lookback]
    recent_lows = [s for s in swing_lows if 0 < current_idx - s.idx <= max_lookback]

    if len(recent_highs) < 1 or len(recent_lows) < 1:
        return None

    # 가능한 S/R 레벨 찾기
    best_channel = None
    best_oscillations = 0

    tolerance = tolerance_pct / 100

    # 모든 low를 support 후보로
    for sl in recent_lows:
        support = sl.price

        # 같은 레벨의 다른 low들 찾기
        support_touches = [s for s in recent_lows
                         if abs(s.price - support) / support < tolerance]

        if len(support_touches) < 2:
            continue

        # 이 support와 호환되는 resistance 찾기
        for sh in recent_highs:
            resistance = sh.price

            if resistance <= support:
                continue

            width_pct = (resistance - support) / support
            if width_pct < 0.008 or width_pct > 0.05:
                continue

            resistance_touches = [s for s in recent_highs
                                 if abs(s.price - resistance) / resistance < tolerance]

            if len(resistance_touches) < 1:
                continue

            # Oscillation 카운트: 시간순으로 S→R→S 확인
            all_touches = []
            for s in support_touches:
                all_touches.append(('S', s.idx))
            for r in resistance_touches:
                all_touches.append(('R', r.idx))

            all_touches.sort(key=lambda x: x[1])

            # S→R→S 패턴 카운트
            oscillations = 0
            last_type = None
            for touch_type, _ in all_touches:
                if last_type is None:
                    last_type = touch_type
                elif last_type != touch_type:
                    if last_type == 'S' and touch_type == 'R':
                        oscillations += 0.5  # S→R
                    elif last_type == 'R' and touch_type == 'S':
                        oscillations += 0.5  # R→S
                    last_type = touch_type

            oscillations = int(oscillations)

            if oscillations >= min_oscillations and oscillations > best_oscillations:
                best_oscillations = oscillations
                start_idx = min(s.idx for s in support_touches + resistance_touches)
                best_channel = Channel(
                    support=support,
                    resistance=resistance,
                    start_idx=start_idx,
                    oscillation_count=oscillations
                )

    return best_channel


def build_channel_map(
    htf_candles: pd.DataFrame,
    swing_len: int = 3,
    min_oscillations: int = 2
) -> Dict[int, Channel]:
    """채널 맵 생성"""
    swing_highs, swing_lows = find_pivot_points(htf_candles, swing_len)

    print(f"  Pivot Highs: {len(swing_highs)}")
    print(f"  Pivot Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values
    volumes = htf_candles['volume'].values
    deltas = htf_candles['delta'].values

    channel_map = {}

    for i in tqdm(range(100, len(htf_candles)), desc="Building channels"):
        current_close = closes[i]

        channel = detect_oscillation_channel(
            htf_candles, swing_highs, swing_lows, i,
            tolerance_pct=0.4,
            min_oscillations=min_oscillations
        )

        if channel:
            # 가격이 채널 범위 내에 있는지 확인
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue

            # 채널 내 평균 볼륨/델타 계산
            start = max(0, channel.start_idx)
            end = i
            if end > start:
                channel.avg_volume = np.mean(volumes[start:end])
                channel.avg_delta = np.mean(np.abs(deltas[start:end]))

            channel_map[i] = channel

    print(f"  Candles with channel: {len(channel_map)}")
    return channel_map


def detect_fakeouts(
    htf_candles: pd.DataFrame,
    channel_map: Dict[int, Channel],
    max_wait: int = 5
) -> List[FakeoutSignal]:
    """Fakeout 신호 감지"""
    closes = htf_candles['close'].values
    highs = htf_candles['high'].values
    lows = htf_candles['low'].values

    fakeout_signals = []
    pending_breaks = []

    for i in range(len(htf_candles)):
        channel = channel_map.get(i)

        # Process pending breakouts
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

        # Check for new breakouts
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

    print(f"  Fakeout Signals: {len(fakeout_signals)}")
    return fakeout_signals


def collect_trades(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    channel_map: Dict[int, Channel],
    fakeout_signals: List[FakeoutSignal],
    tf_ratio: int = 4,
    include_bounce: bool = True,
    include_fakeout: bool = True
) -> List[dict]:
    """매매 수집"""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    touch_threshold = 0.003
    sl_buffer = 0.0008

    for i in range(50, len(ltf_candles) - 150):
        htf_idx = i // tf_ratio
        channel = channel_map.get(htf_idx)

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]

        # Volume/Delta ratio
        start = max(0, i - 20)
        avg_vol = np.mean(ltf_volumes[start:i]) if i > start else ltf_volumes[i]
        avg_delta = np.mean(np.abs(ltf_deltas[start:i])) if i > start else abs(ltf_deltas[i])
        vol_ratio = ltf_volumes[i] / avg_vol if avg_vol > 0 else 1
        delta_ratio = abs(ltf_deltas[i]) / avg_delta if avg_delta > 0 else 1

        # Channel-level volume/delta ratio
        channel_vol_ratio = 1
        channel_delta_ratio = 1
        if channel and channel.avg_volume > 0:
            channel_vol_ratio = ltf_volumes[i] / channel.avg_volume
        if channel and channel.avg_delta > 0:
            channel_delta_ratio = abs(ltf_deltas[i]) / channel.avg_delta

        # FAKEOUT trades
        if include_fakeout:
            fakeout = fakeout_map.get(htf_idx)
            if fakeout and i % tf_ratio == 0:
                f_channel = fakeout.channel
                fakeout_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)

                if fakeout_key not in traded_keys:
                    if fakeout.type == 'bear':
                        entry = close
                        sl = fakeout.extreme * (1 - sl_buffer)
                        tp1 = f_channel.mid_price
                        tp2 = f_channel.resistance * 0.998

                        if entry > sl and tp1 > entry:
                            result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)
                            if result:
                                trades.append({
                                    'idx': i, 'htf_idx': htf_idx,
                                    'direction': 'LONG', 'setup_type': 'FAKEOUT',
                                    'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                    'result': result, 'pnl': pnl,
                                    'vol_ratio': vol_ratio, 'delta_ratio': delta_ratio,
                                    'channel_vol_ratio': channel_vol_ratio,
                                    'channel_delta_ratio': channel_delta_ratio,
                                    'oscillations': f_channel.oscillation_count
                                })
                                traded_keys.add(fakeout_key)

                    else:  # bull fakeout → SHORT
                        entry = close
                        sl = fakeout.extreme * (1 + sl_buffer)
                        tp1 = f_channel.mid_price
                        tp2 = f_channel.support * 1.002

                        if sl > entry and entry > tp1:
                            result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)
                            if result:
                                trades.append({
                                    'idx': i, 'htf_idx': htf_idx,
                                    'direction': 'SHORT', 'setup_type': 'FAKEOUT',
                                    'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                    'result': result, 'pnl': pnl,
                                    'vol_ratio': vol_ratio, 'delta_ratio': delta_ratio,
                                    'channel_vol_ratio': channel_vol_ratio,
                                    'channel_delta_ratio': channel_delta_ratio,
                                    'oscillations': f_channel.oscillation_count
                                })
                                traded_keys.add(fakeout_key)

        # BOUNCE trades
        if include_bounce and channel:
            mid = channel.mid_price
            bounce_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)

            if bounce_key not in traded_keys:
                # Support touch → LONG
                if low <= channel.support * (1 + touch_threshold) and close > channel.support:
                    entry = close
                    sl = channel.support * (1 - sl_buffer)
                    tp1 = mid
                    tp2 = channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'htf_idx': htf_idx,
                                'direction': 'LONG', 'setup_type': 'BOUNCE',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl,
                                'vol_ratio': vol_ratio, 'delta_ratio': delta_ratio,
                                'channel_vol_ratio': channel_vol_ratio,
                                'channel_delta_ratio': channel_delta_ratio,
                                'oscillations': channel.oscillation_count
                            })
                            traded_keys.add(bounce_key)

                # Resistance touch → SHORT
                elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
                    entry = close
                    sl = channel.resistance * (1 + sl_buffer)
                    tp1 = mid
                    tp2 = channel.support * 1.002

                    if sl > entry and entry > tp1:
                        result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)
                        if result:
                            trades.append({
                                'idx': i, 'htf_idx': htf_idx,
                                'direction': 'SHORT', 'setup_type': 'BOUNCE',
                                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
                                'result': result, 'pnl': pnl,
                                'vol_ratio': vol_ratio, 'delta_ratio': delta_ratio,
                                'channel_vol_ratio': channel_vol_ratio,
                                'channel_delta_ratio': channel_delta_ratio,
                                'oscillations': channel.oscillation_count
                            })
                            traded_keys.add(bounce_key)

    return trades


def simulate_trade(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """매매 결과 시뮬레이션"""
    future_highs = highs[idx+1:min(idx+151, len(highs))]
    future_lows = lows[idx+1:min(idx+151, len(lows))]

    if len(future_highs) == 0:
        return None, 0

    if direction == 'LONG':
        hit_sl = np.any(future_lows <= sl)
        hit_tp1 = np.any(future_highs >= tp1)

        sl_idx = np.argmax(future_lows <= sl) if hit_sl else 999
        tp1_idx = np.argmax(future_highs >= tp1) if hit_tp1 else 999

        if sl_idx < tp1_idx:
            return 'loss', (sl - entry) / entry
        elif hit_tp1:
            future_h = future_highs[tp1_idx:] if tp1_idx < len(future_highs) else []
            future_l = future_lows[tp1_idx:] if tp1_idx < len(future_lows) else []

            hit_tp2 = np.any(future_h >= tp2) if len(future_h) > 0 else False
            hit_be = np.any(future_l <= entry) if len(future_l) > 0 else False

            tp2_idx = np.argmax(future_h >= tp2) if hit_tp2 else 999
            be_idx = np.argmax(future_l <= entry) if hit_be else 999

            if hit_tp2 and tp2_idx < be_idx:
                return 'full_win', (tp1 - entry) / entry * 0.5 + (tp2 - entry) / entry * 0.5
            else:
                return 'partial', (tp1 - entry) / entry * 0.5
        return None, 0

    else:  # SHORT
        hit_sl = np.any(future_highs >= sl)
        hit_tp1 = np.any(future_lows <= tp1)

        sl_idx = np.argmax(future_highs >= sl) if hit_sl else 999
        tp1_idx = np.argmax(future_lows <= tp1) if hit_tp1 else 999

        if sl_idx < tp1_idx:
            return 'loss', (entry - sl) / entry
        elif hit_tp1:
            future_l = future_lows[tp1_idx:] if tp1_idx < len(future_lows) else []
            future_h = future_highs[tp1_idx:] if tp1_idx < len(future_highs) else []

            hit_tp2 = np.any(future_l <= tp2) if len(future_l) > 0 else False
            hit_be = np.any(future_h >= entry) if len(future_h) > 0 else False

            tp2_idx = np.argmax(future_l <= tp2) if hit_tp2 else 999
            be_idx = np.argmax(future_h >= entry) if hit_be else 999

            if hit_tp2 and tp2_idx < be_idx:
                return 'full_win', (entry - tp1) / entry * 0.5 + (entry - tp2) / entry * 0.5
            else:
                return 'partial', (entry - tp1) / entry * 0.5
        return None, 0


def backtest(trades: List[dict], label: str) -> dict:
    """백테스트 실행"""
    if not trades:
        print(f"\n{label}: 매매 없음")
        return {'trades': 0, 'wr': 0, 'dd': 0, 'ret': 0, 'final': 10000}

    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    wins, losses = 0, 0
    peak = capital
    max_dd = 0

    for t in trades:
        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        if sl_dist <= 0:
            continue

        lev = min(risk_pct / sl_dist, max_lev)
        position = capital * lev

        pnl = position * t['pnl']
        fees = position * fee_pct * 2
        net = pnl - fees

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
    ret = (capital / 10000 - 1) * 100

    print(f"\n{label}")
    print(f"  매매: {total}건, 승률: {wr:.1f}% ({wins}W/{losses}L)")
    print(f"  수익률: {ret:+.1f}%, 최대 DD: {max_dd*100:.1f}%")
    print(f"  최종 자본: ${capital:,.2f}")

    return {
        'trades': total,
        'wr': wr,
        'dd': max_dd * 100,
        'ret': ret,
        'final': capital
    }


def analyze_volume_delta(trades: List[dict]):
    """Volume/Delta 분석"""
    if not trades:
        return

    df = pd.DataFrame(trades)

    print("\n" + "="*70)
    print("  Volume/Delta 분석")
    print("="*70)

    # Win vs Loss 비교
    wins = df[df['result'] != 'loss']
    losses = df[df['result'] == 'loss']

    print(f"\n  승리 매매 ({len(wins)}건):")
    print(f"    평균 vol_ratio: {wins['vol_ratio'].mean():.2f}")
    print(f"    평균 delta_ratio: {wins['delta_ratio'].mean():.2f}")
    print(f"    평균 channel_vol_ratio: {wins['channel_vol_ratio'].mean():.2f}")
    print(f"    평균 channel_delta_ratio: {wins['channel_delta_ratio'].mean():.2f}")

    print(f"\n  패배 매매 ({len(losses)}건):")
    print(f"    평균 vol_ratio: {losses['vol_ratio'].mean():.2f}")
    print(f"    평균 delta_ratio: {losses['delta_ratio'].mean():.2f}")
    print(f"    평균 channel_vol_ratio: {losses['channel_vol_ratio'].mean():.2f}")
    print(f"    평균 channel_delta_ratio: {losses['channel_delta_ratio'].mean():.2f}")

    # 필터 테스트
    print("\n  필터 테스트:")

    filters = [
        ("vol_ratio >= 1.0", df['vol_ratio'] >= 1.0),
        ("vol_ratio >= 1.5", df['vol_ratio'] >= 1.5),
        ("delta_ratio >= 1.0", df['delta_ratio'] >= 1.0),
        ("delta_ratio >= 1.5", df['delta_ratio'] >= 1.5),
        ("channel_vol_ratio >= 1.0", df['channel_vol_ratio'] >= 1.0),
        ("channel_delta_ratio >= 1.0", df['channel_delta_ratio'] >= 1.0),
    ]

    for name, mask in filters:
        subset = df[mask]
        if len(subset) > 0:
            wr = (subset['result'] != 'loss').mean() * 100
            print(f"    {name}: {len(subset)}건, WR: {wr:.1f}%")


def run_test(htf_candles, ltf_candles, label):
    """Run test for a specific period."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  HTF (1h): {len(htf_candles)} candles")
    print(f"  LTF (15m): {len(ltf_candles)} candles")

    # Build channel map
    print("\nBuilding oscillation channels...")
    channel_map = build_channel_map(htf_candles, swing_len=3, min_oscillations=2)

    # Detect fakeouts
    print("Detecting fakeouts...")
    fakeout_signals = detect_fakeouts(htf_candles, channel_map)

    # Test BOUNCE only
    trades_bounce = collect_trades(htf_candles, ltf_candles, channel_map, fakeout_signals,
                                   include_bounce=True, include_fakeout=False)
    r1 = backtest(trades_bounce, "BOUNCE only")

    # Test FAKEOUT only
    trades_fakeout = collect_trades(htf_candles, ltf_candles, channel_map, fakeout_signals,
                                    include_bounce=False, include_fakeout=True)
    r2 = backtest(trades_fakeout, "FAKEOUT only")

    # Test Both
    trades_both = collect_trades(htf_candles, ltf_candles, channel_map, fakeout_signals,
                                 include_bounce=True, include_fakeout=True)
    r3 = backtest(trades_both, "BOUNCE + FAKEOUT")

    return r1, r2, r3


def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   피봇 방식 + Oscillation 채널 감지 테스트                              ║
║   BOUNCE / FAKEOUT / 둘 다                                             ║
║   2024 (IS) vs 2025 (OOS) 비교                                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Load all data
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

    # Run 2024 test (In-Sample)
    r1_2024, r2_2024, r3_2024 = run_test(htf_2024, ltf_2024, "2024 (In-Sample)")

    # Run 2025 test (Out-of-Sample)
    r1_2025, r2_2025, r3_2025 = run_test(htf_2025, ltf_2025, "2025 (Out-of-Sample) ⭐")

    # 비교 요약
    print("\n" + "="*70)
    print("  2024 vs 2025 비교")
    print("="*70)

    print(f"\n{'전략':<20} {'2024 매매':>10} {'2024 WR':>10} {'2024 수익':>12} | {'2025 매매':>10} {'2025 WR':>10} {'2025 수익':>12}")
    print("-"*100)
    print(f"{'BOUNCE':<20} {r1_2024['trades']:>10} {r1_2024['wr']:>9.1f}% {r1_2024['ret']:>+11.1f}% | {r1_2025['trades']:>10} {r1_2025['wr']:>9.1f}% {r1_2025['ret']:>+11.1f}%")
    print(f"{'FAKEOUT':<20} {r2_2024['trades']:>10} {r2_2024['wr']:>9.1f}% {r2_2024['ret']:>+11.1f}% | {r2_2025['trades']:>10} {r2_2025['wr']:>9.1f}% {r2_2025['ret']:>+11.1f}%")
    print(f"{'BOUNCE + FAKEOUT':<20} {r3_2024['trades']:>10} {r3_2024['wr']:>9.1f}% {r3_2024['ret']:>+11.1f}% | {r3_2025['trades']:>10} {r3_2025['wr']:>9.1f}% {r3_2025['ret']:>+11.1f}%")

    print("\n" + "="*70)
    print("  결론")
    print("="*70)
    if r2_2025['ret'] > 0 and r2_2025['wr'] > 60:
        print(f"\n  ✅ FAKEOUT only 전략이 2025 OOS에서도 유효!")
        print(f"     2024: {r2_2024['trades']}건, {r2_2024['wr']:.1f}% WR, {r2_2024['ret']:+.1f}% 수익")
        print(f"     2025: {r2_2025['trades']}건, {r2_2025['wr']:.1f}% WR, {r2_2025['ret']:+.1f}% 수익")
    else:
        print(f"\n  ⚠️ FAKEOUT 전략 2025 성과:")
        print(f"     {r2_2025['trades']}건, {r2_2025['wr']:.1f}% WR, {r2_2025['ret']:+.1f}% 수익")


if __name__ == "__main__":
    main()
