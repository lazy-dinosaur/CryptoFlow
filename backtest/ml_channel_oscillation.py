#!/usr/bin/env python3
"""
Oscillation-based Channel Detection

채널 감지 방식:
- Support → Resistance → Support 왕복을 확인
- 단순 터치 카운트가 아닌 연결된 움직임 확인
- 진짜 레인지 바운드 구간만 채널로 인식

전략 테스트:
1. NO_ML - 모든 신호
2. ML_ENTRY - ML 진입 필터
3. ML_COMBINED - ML 진입 + 동적 Exit
4. VOL_DELTA - 볼륨/델타 필터
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class OscillationChannel:
    support: float
    resistance: float
    oscillations: int  # 왕복 횟수
    start_idx: int
    last_update_idx: int
    swing_sequence: List[SwingPoint] = field(default_factory=list)

    @property
    def width_pct(self) -> float:
        return (self.resistance - self.support) / self.support * 100

    @property
    def mid_price(self) -> float:
        return (self.resistance + self.support) / 2


def find_swing_points(df: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    스윙 하이/로우 찾기
    confirm_candles 동안 더 높은/낮은 가격이 없으면 확정
    """
    highs = df['high'].values
    lows = df['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(confirm_candles, len(df) - confirm_candles):
        # Swing High: 양쪽으로 confirm_candles 동안 더 높은 고가 없음
        is_swing_high = True
        for j in range(1, confirm_candles + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        # Swing Low: 양쪽으로 confirm_candles 동안 더 낮은 저가 없음
        is_swing_low = True
        for j in range(1, confirm_candles + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def detect_oscillation_channel(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    current_idx: int,
    tolerance_pct: float = 0.004,  # 0.4%
    min_width_pct: float = 0.008,  # 0.8%
    max_width_pct: float = 0.05,   # 5%
    min_oscillations: int = 2,     # 최소 왕복 횟수
    max_lookback: int = 100
) -> Optional[OscillationChannel]:
    """
    왕복(oscillation) 기반 채널 감지

    로직:
    1. 최근 스윙 포인트들을 시간순으로 정렬
    2. Low → High → Low → High 패턴 찾기
    3. Low들이 비슷한 가격대 (support zone)
    4. High들이 비슷한 가격대 (resistance zone)
    5. 왕복 횟수 >= min_oscillations 이면 채널 확정
    """
    # 최근 스윙 포인트만 (lookback 이내, 현재 이전에 확정된 것만)
    # 스윙은 confirm_candles 후에 확정되므로, idx + 3 <= current_idx
    confirm_delay = 3

    recent_highs = [s for s in swing_highs
                   if s.idx + confirm_delay <= current_idx
                   and current_idx - s.idx <= max_lookback]
    recent_lows = [s for s in swing_lows
                  if s.idx + confirm_delay <= current_idx
                  and current_idx - s.idx <= max_lookback]

    if len(recent_highs) < 1 or len(recent_lows) < 1:
        return None

    # 모든 스윙 포인트를 시간순 정렬
    all_swings = sorted(recent_highs + recent_lows, key=lambda s: s.idx)

    if len(all_swings) < 3:
        return None

    # 왕복 패턴 찾기
    # Low → High → Low 또는 High → Low → High 순서로 번갈아가는지 확인

    best_channel = None
    best_oscillations = 0

    # 다양한 support/resistance 레벨 후보 시도
    for low1 in recent_lows[-10:]:  # 최근 10개 저점
        for high1 in recent_highs[-10:]:  # 최근 10개 고점
            if high1.price <= low1.price:
                continue

            support_level = low1.price
            resistance_level = high1.price

            width_pct = (resistance_level - support_level) / support_level
            if width_pct < min_width_pct or width_pct > max_width_pct:
                continue

            # 이 레벨로 왕복 횟수 계산
            oscillations = 0
            last_touch = None  # 'support' or 'resistance'
            sequence = []

            for swing in all_swings:
                # Support 터치 확인
                if swing.type == 'low':
                    if abs(swing.price - support_level) / support_level <= tolerance_pct:
                        if last_touch != 'support':
                            sequence.append(swing)
                            if last_touch == 'resistance':
                                oscillations += 0.5  # 반 왕복
                            last_touch = 'support'

                # Resistance 터치 확인
                elif swing.type == 'high':
                    if abs(swing.price - resistance_level) / resistance_level <= tolerance_pct:
                        if last_touch != 'resistance':
                            sequence.append(swing)
                            if last_touch == 'support':
                                oscillations += 0.5  # 반 왕복
                            last_touch = 'resistance'

            oscillations = int(oscillations)  # 완전한 왕복만

            if oscillations >= min_oscillations and oscillations > best_oscillations:
                best_oscillations = oscillations
                best_channel = OscillationChannel(
                    support=support_level,
                    resistance=resistance_level,
                    oscillations=oscillations,
                    start_idx=sequence[0].idx if sequence else current_idx,
                    last_update_idx=current_idx,
                    swing_sequence=sequence
                )

    return best_channel


def build_oscillation_channels(
    htf_candles: pd.DataFrame,
    confirm_candles: int = 3,
    tolerance_pct: float = 0.004,
    min_width_pct: float = 0.008,
    max_width_pct: float = 0.05,
    min_oscillations: int = 2
) -> Dict[int, OscillationChannel]:
    """
    HTF 캔들에서 oscillation 채널 맵 생성
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles)

    print(f"  Swing Highs: {len(swing_highs)}")
    print(f"  Swing Lows: {len(swing_lows)}")

    channel_map: Dict[int, OscillationChannel] = {}

    for i in range(len(htf_candles)):
        channel = detect_oscillation_channel(
            swing_highs, swing_lows, i,
            tolerance_pct=tolerance_pct,
            min_width_pct=min_width_pct,
            max_width_pct=max_width_pct,
            min_oscillations=min_oscillations
        )
        if channel:
            channel_map[i] = channel

    unique_channels = len(set((round(c.support), round(c.resistance))
                              for c in channel_map.values()))
    print(f"  Candles with Channel: {len(channel_map)}")
    print(f"  Unique Channels: {unique_channels}")

    return channel_map


def collect_trades(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    channel_map: Dict[int, OscillationChannel],
    tf_ratio: int = 4,
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.002
) -> List[dict]:
    """
    LTF에서 매매 신호 수집
    """
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    for i in tqdm(range(50, len(ltf_candles) - 100), desc="Collecting"):
        htf_idx = i // tf_ratio
        channel = channel_map.get(htf_idx)

        if not channel:
            continue

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        volume = ltf_volumes[i]
        delta = ltf_deltas[i]

        # 볼륨/델타 ratio
        lookback = 20
        start = max(0, i - lookback)
        avg_vol = np.mean(ltf_volumes[start:i]) if i > start else volume
        avg_delta = np.mean(np.abs(ltf_deltas[start:i])) if i > start else abs(delta)
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1
        delta_ratio = abs(delta) / avg_delta if avg_delta > 0 else 1

        mid = channel.mid_price

        # 중복 방지
        trade_key = (round(channel.support), round(channel.resistance), i // 10)
        if trade_key in traded_keys:
            continue

        # LONG - Support 터치
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer_pct)
            tp1 = mid
            tp2 = channel.resistance * (1 - sl_buffer_pct)

            if entry > sl and tp1 > entry:
                # 결과 시뮬레이션
                future_highs = ltf_highs[i+1:min(i+51, len(ltf_candles))]
                future_lows = ltf_lows[i+1:min(i+51, len(ltf_candles))]

                if len(future_highs) == 0:
                    continue

                hit_sl = np.any(future_lows < sl)
                hit_tp1 = np.any(future_highs >= tp1)

                sl_idx = np.argmax(future_lows < sl) if hit_sl else 999
                tp1_idx = np.argmax(future_highs >= tp1) if hit_tp1 else 999

                if sl_idx < tp1_idx:
                    result = 'loss'
                    pnl = (sl - entry) / entry
                elif hit_tp1:
                    # TP1 도달 후 TP2 체크
                    future_after = future_highs[tp1_idx:]
                    future_lows_after = future_lows[tp1_idx:]
                    hit_tp2 = np.any(future_after >= tp2) if len(future_after) > 0 else False
                    hit_be = np.any(future_lows_after < entry) if len(future_lows_after) > 0 else False

                    if hit_tp2 and not hit_be:
                        result = 'full_win'
                        pnl = (tp1 - entry) / entry * 0.5 + (tp2 - entry) / entry * 0.5
                    else:
                        result = 'partial'
                        pnl = (tp1 - entry) / entry * 0.5
                else:
                    continue

                trades.append({
                    'idx': i,
                    'timestamp': ltf_candles.index[i],
                    'direction': 'LONG',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'result': result,
                    'pnl': pnl,
                    'vol_ratio': vol_ratio,
                    'delta_ratio': delta_ratio,
                    'channel_width': channel.width_pct,
                    'oscillations': channel.oscillations
                })
                traded_keys.add(trade_key)

        # SHORT - Resistance 터치
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid
            tp2 = channel.support * (1 + sl_buffer_pct)

            if sl > entry and entry > tp1:
                future_highs = ltf_highs[i+1:min(i+51, len(ltf_candles))]
                future_lows = ltf_lows[i+1:min(i+51, len(ltf_candles))]

                if len(future_lows) == 0:
                    continue

                hit_sl = np.any(future_highs > sl)
                hit_tp1 = np.any(future_lows <= tp1)

                sl_idx = np.argmax(future_highs > sl) if hit_sl else 999
                tp1_idx = np.argmax(future_lows <= tp1) if hit_tp1 else 999

                if sl_idx < tp1_idx:
                    result = 'loss'
                    pnl = (entry - sl) / entry
                elif hit_tp1:
                    future_after = future_lows[tp1_idx:]
                    future_highs_after = future_highs[tp1_idx:]
                    hit_tp2 = np.any(future_after <= tp2) if len(future_after) > 0 else False
                    hit_be = np.any(future_highs_after > entry) if len(future_highs_after) > 0 else False

                    if hit_tp2 and not hit_be:
                        result = 'full_win'
                        pnl = (entry - tp1) / entry * 0.5 + (entry - tp2) / entry * 0.5
                    else:
                        result = 'partial'
                        pnl = (entry - tp1) / entry * 0.5
                else:
                    continue

                trades.append({
                    'idx': i,
                    'timestamp': ltf_candles.index[i],
                    'direction': 'SHORT',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'result': result,
                    'pnl': pnl,
                    'vol_ratio': vol_ratio,
                    'delta_ratio': delta_ratio,
                    'channel_width': channel.width_pct,
                    'oscillations': channel.oscillations
                })
                traded_keys.add(trade_key)

    return trades


def backtest(trades_df: pd.DataFrame, label: str) -> dict:
    """백테스트 실행"""
    if len(trades_df) == 0:
        print(f"\n{label}: 매매 없음")
        return {'trades': 0, 'wr': 0, 'dd': 0, 'ret': 0}

    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0005

    wins, losses = 0, 0
    peak = capital
    max_dd = 0
    trade_returns = []

    for _, row in trades_df.iterrows():
        entry = row['entry']
        sl = row['sl']
        pnl_pct = row['pnl']
        result = row['result']

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
    print(f"  최종: ${capital:,.2f}")
    print(f"  수익률: {ret:+.1f}%")
    if len(trade_returns) > 0:
        print(f"  매매당: {np.mean(trade_returns):+.2f}%")

    return {'trades': total, 'wr': wr, 'dd': max_dd*100, 'ret': ret}


def main():
    print("="*60)
    print("  Oscillation Channel Detection Test")
    print("  (왕복 기반 채널 감지)")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')

    # 2024년만
    df_1h_2024 = df_1h[df_1h.index.year == 2024]
    df_15m_2024 = df_15m[df_15m.index.year == 2024]

    print(f"  HTF (1h): {len(df_1h_2024)}")
    print(f"  LTF (15m): {len(df_15m_2024)}")

    # Build channels
    print("\n" + "="*60)
    print("  Building Oscillation Channels")
    print("="*60)

    channel_map = build_oscillation_channels(
        df_1h_2024,
        confirm_candles=3,
        tolerance_pct=0.004,
        min_width_pct=0.008,
        max_width_pct=0.05,
        min_oscillations=2
    )

    # Collect trades
    print("\n" + "="*60)
    print("  Collecting Trades")
    print("="*60)

    trades = collect_trades(
        df_1h_2024,
        df_15m_2024,
        channel_map,
        tf_ratio=4
    )

    df = pd.DataFrame(trades)
    print(f"\n총 수집: {len(df)}건")

    if len(df) == 0:
        print("매매가 없습니다. 파라미터 조정 필요.")
        return

    # 백테스트
    print("\n" + "="*60)
    print("  BACKTEST RESULTS (2024)")
    print("="*60)

    # 1. NO_ML - 모든 신호
    r1 = backtest(df, "1. NO_ML (모든 신호)")

    # 2. VOL_DELTA - 볼륨/델타 필터
    df_vd = df[(df['vol_ratio'] >= 1.5) | (df['delta_ratio'] >= 1.5)]
    r2 = backtest(df_vd, "2. VOL_DELTA (vol>=1.5 OR delta>=1.5)")

    # 3. ML_ENTRY (모델이 있으면)
    try:
        entry_model = joblib.load('models/entry_model.joblib')

        # Feature extraction
        features = df[['vol_ratio', 'delta_ratio', 'channel_width', 'oscillations']].copy()
        features['direction_long'] = (df['direction'] == 'LONG').astype(int)

        entry_probs = entry_model.predict_proba(features)[:, 1]
        df_ml = df[entry_probs >= 0.7]
        r3 = backtest(df_ml, "3. ML_ENTRY (threshold=0.7)")
    except:
        print("\n3. ML_ENTRY: 모델 없음 (skip)")
        r3 = None

    # 비교 표
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n{'전략':<35} {'매매':>6} {'승률':>8} {'DD':>8} {'수익률':>12}")
    print("-"*75)
    print(f"{'NO_ML':<35} {r1['trades']:>6} {r1['wr']:>7.1f}% {r1['dd']:>7.1f}% {r1['ret']:>+11.1f}%")
    print(f"{'VOL_DELTA':<35} {r2['trades']:>6} {r2['wr']:>7.1f}% {r2['dd']:>7.1f}% {r2['ret']:>+11.1f}%")
    if r3:
        print(f"{'ML_ENTRY':<35} {r3['trades']:>6} {r3['wr']:>7.1f}% {r3['dd']:>7.1f}% {r3['ret']:>+11.1f}%")


if __name__ == "__main__":
    main()
