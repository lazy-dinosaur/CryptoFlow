#!/usr/bin/env python3
"""
Channel Tiebreaker 비교 테스트

비교 대상:
1. FIRST - 기존 방식 (먼저 발견된 채널)
2. NARROW - 새 방식 (더 좁은 채널 우선)

테스트 항목:
- 승률
- 레버리지 적용 자본
- 최대 DD
- 매매당 수익
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys
import os

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
    resistance: float
    support_touches: int
    resistance_touches: int
    width_pct: float

    @property
    def mid_price(self):
        return (self.support + self.resistance) / 2


def find_swing_points(df: pd.DataFrame, confirm_candles: int = 3):
    """스윙 포인트 찾기"""
    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(confirm_candles, len(df) - confirm_candles):
        # Swing High
        is_high = True
        for j in range(1, confirm_candles + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        # Swing Low
        is_low = True
        for j in range(1, confirm_candles + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def build_channel_map(
    htf_candles: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    tiebreaker: str = 'first',
    touch_threshold: float = 0.004,
    min_width: float = 0.008,
    max_width: float = 0.05
) -> Dict[int, Channel]:
    """
    채널 맵 생성

    tiebreaker: 'first' or 'narrow'
    """
    closes = htf_candles['close'].values
    channel_map = {}

    for i in range(len(htf_candles)):
        current_close = closes[i]
        candidates = []

        # 유효한 스윙 포인트 (확정된 것만, 100캔들 이내)
        valid_highs = [s for s in swing_highs if s.idx + 3 <= i and i - s.idx <= 100]
        valid_lows = [s for s in swing_lows if s.idx + 3 <= i and i - s.idx <= 100]

        for sh in valid_highs[-30:]:
            for sl in valid_lows[-30:]:
                if sh.price <= sl.price:
                    continue

                width_pct = (sh.price - sl.price) / sl.price
                if width_pct < min_width or width_pct > max_width:
                    continue

                # 가격이 채널 범위 ±2% 내
                if current_close < sl.price * 0.98 or current_close > sh.price * 1.02:
                    continue

                # 터치 횟수 계산
                s_touches = sum(1 for s in valid_lows
                               if abs(s.price - sl.price) / sl.price < touch_threshold)
                r_touches = sum(1 for s in valid_highs
                               if abs(s.price - sh.price) / sh.price < touch_threshold)

                # 확정 조건: 각각 2회 이상
                if s_touches >= 2 and r_touches >= 2:
                    candidates.append(Channel(
                        support=sl.price,
                        resistance=sh.price,
                        support_touches=s_touches,
                        resistance_touches=r_touches,
                        width_pct=width_pct
                    ))

        if not candidates:
            continue

        # 점수 기준 1차 필터
        max_score = max(c.support_touches + c.resistance_touches for c in candidates)
        top_candidates = [c for c in candidates
                         if c.support_touches + c.resistance_touches == max_score]

        # Tiebreaker 적용
        if len(top_candidates) == 1:
            best = top_candidates[0]
        elif tiebreaker == 'narrow':
            best = min(top_candidates, key=lambda c: c.width_pct)
        else:  # first
            best = top_candidates[0]

        channel_map[i] = best

    return channel_map


@dataclass
class FakeoutSignal:
    """Fakeout 신호"""
    htf_idx: int
    type: str  # 'bull' or 'bear'
    channel: Channel
    extreme: float  # Fakeout extreme price


def detect_fakeouts(
    htf_candles: pd.DataFrame,
    channel_map: Dict[int, Channel],
    max_wait: int = 5
) -> List[FakeoutSignal]:
    """HTF에서 Fakeout 신호 감지"""
    closes = htf_candles['close'].values
    highs = htf_candles['high'].values
    lows = htf_candles['low'].values

    fakeout_signals = []
    pending_breaks = []  # {'type', 'break_idx', 'channel', 'extreme'}

    for i in range(len(htf_candles)):
        channel = channel_map.get(i)
        if not channel:
            continue

        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        # Process pending breakouts
        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_wait:
                pending_breaks.remove(pb)
                continue

            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], current_low)
                # Check if price returned inside channel
                if current_close > pb['channel'].support:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bear',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)
            else:  # bull
                pb['extreme'] = max(pb['extreme'], current_high)
                if current_close < pb['channel'].resistance:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bull',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)

        # Check for new breakouts
        # Bear breakout (price closes below support)
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
        # Bull breakout (price closes above resistance)
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

    return fakeout_signals


def collect_trades(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    channel_map: Dict[int, Channel],
    tf_ratio: int = 4,
    touch_threshold: float = 0.003,
    sl_buffer: float = 0.0008
) -> List[dict]:
    """매매 수집 (BOUNCE + FAKEOUT)"""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    # Fakeout 신호 감지
    fakeout_signals = detect_fakeouts(htf_candles, channel_map)
    fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    for i in range(50, len(ltf_candles) - 100):
        htf_idx = i // tf_ratio
        channel = channel_map.get(htf_idx)

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        volume = ltf_volumes[i]
        delta = ltf_deltas[i]

        # 볼륨/델타 ratio
        start = max(0, i - 20)
        avg_vol = np.mean(ltf_volumes[start:i]) if i > start else volume
        avg_delta = np.mean(np.abs(ltf_deltas[start:i])) if i > start else abs(delta)
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1
        delta_ratio = abs(delta) / avg_delta if avg_delta > 0 else 1

        # ============================================================
        # FAKEOUT 매매 (HTF 캔들의 첫 LTF 캔들에서만)
        # ============================================================
        fakeout = fakeout_map.get(htf_idx)
        if fakeout and i % tf_ratio == 0:
            f_channel = fakeout.channel
            f_mid = f_channel.mid_price

            fakeout_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)
            if fakeout_key not in traded_keys:
                if fakeout.type == 'bear':
                    # LONG on bear fakeout
                    entry = close
                    sl = fakeout.extreme * (1 - sl_buffer)
                    tp1 = f_mid
                    tp2 = f_channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        result, pnl = simulate_trade_result(
                            ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2
                        )
                        if result:
                            trades.append({
                                'idx': i,
                                'timestamp': ltf_candles.index[i],
                                'direction': 'LONG',
                                'setup_type': 'FAKEOUT',
                                'entry': entry,
                                'sl': sl,
                                'tp1': tp1,
                                'tp2': tp2,
                                'result': result,
                                'pnl': pnl,
                                'vol_ratio': vol_ratio,
                                'delta_ratio': delta_ratio,
                                'channel_width': f_channel.width_pct
                            })
                            traded_keys.add(fakeout_key)

                else:  # bull fakeout
                    # SHORT on bull fakeout
                    entry = close
                    sl = fakeout.extreme * (1 + sl_buffer)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        result, pnl = simulate_trade_result(
                            ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2
                        )
                        if result:
                            trades.append({
                                'idx': i,
                                'timestamp': ltf_candles.index[i],
                                'direction': 'SHORT',
                                'setup_type': 'FAKEOUT',
                                'entry': entry,
                                'sl': sl,
                                'tp1': tp1,
                                'tp2': tp2,
                                'result': result,
                                'pnl': pnl,
                                'vol_ratio': vol_ratio,
                                'delta_ratio': delta_ratio,
                                'channel_width': f_channel.width_pct
                            })
                            traded_keys.add(fakeout_key)

        # ============================================================
        # BOUNCE 매매
        # ============================================================
        if not channel:
            continue

        mid = channel.mid_price

        bounce_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if bounce_key in traded_keys:
            continue

        # LONG (Support touch)
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                result, pnl = simulate_trade_result(
                    ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2
                )
                if result:
                    trades.append({
                        'idx': i,
                        'timestamp': ltf_candles.index[i],
                        'direction': 'LONG',
                        'setup_type': 'BOUNCE',
                        'entry': entry,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'result': result,
                        'pnl': pnl,
                        'vol_ratio': vol_ratio,
                        'delta_ratio': delta_ratio,
                        'channel_width': channel.width_pct
                    })
                    traded_keys.add(bounce_key)

        # SHORT (Resistance touch)
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                result, pnl = simulate_trade_result(
                    ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2
                )
                if result:
                    trades.append({
                        'idx': i,
                        'timestamp': ltf_candles.index[i],
                        'direction': 'SHORT',
                        'setup_type': 'BOUNCE',
                        'entry': entry,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'result': result,
                        'pnl': pnl,
                        'vol_ratio': vol_ratio,
                        'delta_ratio': delta_ratio,
                        'channel_width': channel.width_pct
                    })
                    traded_keys.add(bounce_key)

    return trades


def simulate_trade_result(highs, lows, idx, direction, entry, sl, tp1, tp2):
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
            future_after_h = future_highs[tp1_idx:] if tp1_idx < len(future_highs) else []
            future_after_l = future_lows[tp1_idx:] if tp1_idx < len(future_lows) else []

            hit_tp2 = np.any(future_after_h >= tp2) if len(future_after_h) > 0 else False
            hit_be = np.any(future_after_l <= entry) if len(future_after_l) > 0 else False

            tp2_idx = np.argmax(future_after_h >= tp2) if hit_tp2 else 999
            be_idx = np.argmax(future_after_l <= entry) if hit_be else 999

            if hit_tp2 and tp2_idx < be_idx:
                return 'full_win', (tp1 - entry) / entry * 0.5 + (tp2 - entry) / entry * 0.5
            else:
                return 'partial', (tp1 - entry) / entry * 0.5
        else:
            return None, 0

    else:  # SHORT
        hit_sl = np.any(future_highs >= sl)
        hit_tp1 = np.any(future_lows <= tp1)

        sl_idx = np.argmax(future_highs >= sl) if hit_sl else 999
        tp1_idx = np.argmax(future_lows <= tp1) if hit_tp1 else 999

        if sl_idx < tp1_idx:
            return 'loss', (entry - sl) / entry
        elif hit_tp1:
            future_after_l = future_lows[tp1_idx:] if tp1_idx < len(future_lows) else []
            future_after_h = future_highs[tp1_idx:] if tp1_idx < len(future_highs) else []

            hit_tp2 = np.any(future_after_l <= tp2) if len(future_after_l) > 0 else False
            hit_be = np.any(future_after_h >= entry) if len(future_after_h) > 0 else False

            tp2_idx = np.argmax(future_after_l <= tp2) if hit_tp2 else 999
            be_idx = np.argmax(future_after_h >= entry) if hit_be else 999

            if hit_tp2 and tp2_idx < be_idx:
                return 'full_win', (entry - tp1) / entry * 0.5 + (entry - tp2) / entry * 0.5
            else:
                return 'partial', (entry - tp1) / entry * 0.5
        else:
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
    trade_returns = []

    for t in trades:
        entry = t['entry']
        sl = t['sl']
        pnl_pct = t['pnl']
        result = t['result']

        sl_dist = abs(entry - sl) / entry
        if sl_dist <= 0:
            continue

        lev = min(risk_pct / sl_dist, max_lev)
        position = capital * lev

        pnl = position * pnl_pct
        fees = position * fee_pct * 2
        net = pnl - fees

        trade_returns.append(net / capital * 100)
        capital += net
        capital = max(capital, 0)

        if result != 'loss':
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
    print(f"  매매: {total}건")
    print(f"  승률: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  최대 DD: {max_dd*100:.1f}%")
    print(f"  최종 자본: ${capital:,.2f}")
    print(f"  수익률: {ret:+.1f}%")
    if trade_returns:
        print(f"  매매당: 평균 {np.mean(trade_returns):+.2f}%, 중앙값 {np.median(trade_returns):+.2f}%")

    # BOUNCE/FAKEOUT 분리
    bounce_trades = [t for t in trades if t.get('setup_type') == 'BOUNCE']
    fakeout_trades = [t for t in trades if t.get('setup_type') == 'FAKEOUT']

    if bounce_trades:
        b_wins = sum(1 for t in bounce_trades if t['result'] != 'loss')
        b_total = len(bounce_trades)
        b_wr = b_wins / b_total * 100 if b_total > 0 else 0
        print(f"    BOUNCE:  {b_total}건, {b_wr:.1f}% WR")

    if fakeout_trades:
        f_wins = sum(1 for t in fakeout_trades if t['result'] != 'loss')
        f_total = len(fakeout_trades)
        f_wr = f_wins / f_total * 100 if f_total > 0 else 0
        print(f"    FAKEOUT: {f_total}건, {f_wr:.1f}% WR")

    return {
        'trades': total,
        'wr': wr,
        'dd': max_dd * 100,
        'ret': ret,
        'final': capital,
        'avg_per_trade': np.mean(trade_returns) if trade_returns else 0,
        'bounce_count': len(bounce_trades),
        'fakeout_count': len(fakeout_trades)
    }


def main():
    print("="*70)
    print("  Channel Tiebreaker 비교 테스트")
    print("  FIRST (기존) vs NARROW (새로운)")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')

    # 2024년만
    df_1h_2024 = df_1h[df_1h.index.year == 2024]
    df_15m_2024 = df_15m[df_15m.index.year == 2024]

    print(f"  HTF (1h): {len(df_1h_2024)}")
    print(f"  LTF (15m): {len(df_15m_2024)}")

    # 스윙 포인트 찾기
    print("\nFinding swing points...")
    swing_highs, swing_lows = find_swing_points(df_1h_2024, confirm_candles=3)
    print(f"  Swing Highs: {len(swing_highs)}")
    print(f"  Swing Lows: {len(swing_lows)}")

    # ============================================================
    # 1. FIRST (기존 방식)
    # ============================================================
    print("\n" + "="*70)
    print("  1. FIRST (기존 방식)")
    print("="*70)

    channel_map_first = build_channel_map(
        df_1h_2024, swing_highs, swing_lows, tiebreaker='first'
    )
    print(f"  채널 감지 캔들: {len(channel_map_first)}")

    trades_first = collect_trades(df_1h_2024, df_15m_2024, channel_map_first)
    print(f"  수집 매매: {len(trades_first)}")

    r1_base = backtest(trades_first, "FIRST - No ML")

    # 볼륨/델타 필터
    trades_first_vd = [t for t in trades_first
                       if t['vol_ratio'] >= 1.5 or t['delta_ratio'] >= 1.5]
    r1_vd = backtest(trades_first_vd, "FIRST - Vol/Delta Filter")

    # ============================================================
    # 2. NARROW (새 방식)
    # ============================================================
    print("\n" + "="*70)
    print("  2. NARROW (좁은 채널 우선)")
    print("="*70)

    channel_map_narrow = build_channel_map(
        df_1h_2024, swing_highs, swing_lows, tiebreaker='narrow'
    )
    print(f"  채널 감지 캔들: {len(channel_map_narrow)}")

    trades_narrow = collect_trades(df_1h_2024, df_15m_2024, channel_map_narrow)
    print(f"  수집 매매: {len(trades_narrow)}")

    r2_base = backtest(trades_narrow, "NARROW - No ML")

    # 볼륨/델타 필터
    trades_narrow_vd = [t for t in trades_narrow
                        if t['vol_ratio'] >= 1.5 or t['delta_ratio'] >= 1.5]
    r2_vd = backtest(trades_narrow_vd, "NARROW - Vol/Delta Filter")

    # ============================================================
    # 비교 요약
    # ============================================================
    print("\n" + "="*70)
    print("  비교 요약")
    print("="*70)

    print(f"\n{'전략':<30} {'매매':>6} {'승률':>8} {'DD':>8} {'수익률':>15} {'최종자본':>15}")
    print("-"*90)
    print(f"{'FIRST - No ML':<30} {r1_base['trades']:>6} {r1_base['wr']:>7.1f}% {r1_base['dd']:>7.1f}% {r1_base['ret']:>+14.1f}% ${r1_base['final']:>13,.0f}")
    print(f"{'FIRST - Vol/Delta':<30} {r1_vd['trades']:>6} {r1_vd['wr']:>7.1f}% {r1_vd['dd']:>7.1f}% {r1_vd['ret']:>+14.1f}% ${r1_vd['final']:>13,.0f}")
    print(f"{'NARROW - No ML':<30} {r2_base['trades']:>6} {r2_base['wr']:>7.1f}% {r2_base['dd']:>7.1f}% {r2_base['ret']:>+14.1f}% ${r2_base['final']:>13,.0f}")
    print(f"{'NARROW - Vol/Delta':<30} {r2_vd['trades']:>6} {r2_vd['wr']:>7.1f}% {r2_vd['dd']:>7.1f}% {r2_vd['ret']:>+14.1f}% ${r2_vd['final']:>13,.0f}")

    # 승자 판정
    print("\n" + "="*70)
    print("  결론")
    print("="*70)

    if r2_base['ret'] > r1_base['ret']:
        print(f"\n  ✅ NARROW 방식이 더 좋음!")
        print(f"     수익률: {r2_base['ret']:+.1f}% vs {r1_base['ret']:+.1f}%")
        print(f"     승률: {r2_base['wr']:.1f}% vs {r1_base['wr']:.1f}%")
    else:
        print(f"\n  ✅ FIRST 방식이 더 좋음!")
        print(f"     수익률: {r1_base['ret']:+.1f}% vs {r2_base['ret']:+.1f}%")
        print(f"     승률: {r1_base['wr']:.1f}% vs {r2_base['wr']:.1f}%")


if __name__ == "__main__":
    main()
