#!/usr/bin/env python3
"""
델타/볼륨 변화량 분석

CVD가 아니라 실제 델타와 볼륨의 변화량이 중요:
- 터치 전 캔들들의 델타/볼륨 흐름
- 터치 캔들에서의 급변
- 변화 속도 (가속도)
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def collect_bounce_with_change(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """터치 시점의 델타/볼륨 변화량 수집."""
    data = []

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

    sl_buffer = 0.0008
    touch_threshold = 0.003

    for i in range(50, len(ltf_candles) - 150):
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx - 1)

        if not channel:
            continue

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        volume = ltf_volumes[i]
        delta = ltf_deltas[i]

        mid = (channel.resistance + channel.support) / 2

        # === 변화량 계산 ===
        # 이전 N개 캔들
        n = 5
        if i < n:
            continue

        prev_volumes = ltf_volumes[i-n:i]
        prev_deltas = ltf_deltas[i-n:i]

        # 평균값
        avg_vol_prev = np.mean(prev_volumes)
        avg_delta_prev = np.mean(prev_deltas)
        avg_abs_delta_prev = np.mean(np.abs(prev_deltas))

        # 1. 볼륨 변화: 현재 볼륨 vs 이전 평균
        vol_change = (volume - avg_vol_prev) / avg_vol_prev if avg_vol_prev > 0 else 0

        # 2. 볼륨 급등: 현재 > 이전 max?
        max_vol_prev = np.max(prev_volumes)
        vol_spike = volume > max_vol_prev * 1.5  # 50% 이상 급등

        # 3. 델타 변화: 현재 vs 이전 평균
        delta_change = delta - avg_delta_prev

        # 4. 델타 반전: 이전 흐름과 반대 방향?
        delta_direction_changed = (delta > 0 and avg_delta_prev < 0) or (delta < 0 and avg_delta_prev > 0)

        # 5. 델타 강도: 절대값 대비
        delta_strength = abs(delta) / avg_abs_delta_prev if avg_abs_delta_prev > 0 else 1

        # 6. 델타 가속도: 마지막 3개 캔들의 변화 추세
        if i >= 3:
            recent_deltas = ltf_deltas[i-3:i+1]
            delta_diff = np.diff(recent_deltas)
            delta_acceleration = np.mean(delta_diff)  # 양수면 증가 추세
        else:
            delta_acceleration = 0

        # Support touch → LONG
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid

            if entry > sl and tp1 > entry:
                # Simulate outcome
                success = False
                for j in range(i+1, min(i+150, len(ltf_highs))):
                    if ltf_lows[j] <= sl:
                        break
                    if ltf_highs[j] >= tp1:
                        success = True
                        break

                data.append({
                    'direction': 'LONG',
                    'success': success,
                    'vol_change': vol_change,
                    'vol_spike': 1 if vol_spike else 0,
                    'delta': delta,
                    'delta_change': delta_change,
                    'delta_direction_changed': 1 if delta_direction_changed else 0,
                    'delta_strength': delta_strength,
                    'delta_acceleration': delta_acceleration,
                    # LONG 특화
                    'delta_positive': 1 if delta > 0 else 0,
                    'delta_improving': 1 if delta_acceleration > 0 else 0,  # 매수세 증가
                })

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid

            if sl > entry and entry > tp1:
                success = False
                for j in range(i+1, min(i+150, len(ltf_highs))):
                    if ltf_highs[j] >= sl:
                        break
                    if ltf_lows[j] <= tp1:
                        success = True
                        break

                data.append({
                    'direction': 'SHORT',
                    'success': success,
                    'vol_change': vol_change,
                    'vol_spike': 1 if vol_spike else 0,
                    'delta': delta,
                    'delta_change': delta_change,
                    'delta_direction_changed': 1 if delta_direction_changed else 0,
                    'delta_strength': delta_strength,
                    'delta_acceleration': delta_acceleration,
                    # SHORT 특화
                    'delta_negative': 1 if delta < 0 else 0,
                    'delta_weakening': 1 if delta_acceleration < 0 else 0,  # 매도세 증가
                })

    return pd.DataFrame(data)


def analyze_long(df):
    """LONG 바운스 분석."""
    long_df = df[df['direction'] == 'LONG'].copy()
    if len(long_df) == 0:
        return

    wins = long_df[long_df['success'] == True]
    losses = long_df[long_df['success'] == False]
    base_wr = len(wins) / len(long_df) * 100

    print(f"\n{'='*70}")
    print(f"  LONG 바운스 분석 ({len(long_df)}건, 기본 WR: {base_wr:.1f}%)")
    print(f"{'='*70}")

    # 1. 볼륨 변화
    print("\n  📊 볼륨 변화량")
    print(f"    성공 평균 볼륨 변화: {wins['vol_change'].mean()*100:+.1f}%")
    print(f"    실패 평균 볼륨 변화: {losses['vol_change'].mean()*100:+.1f}%")

    # 볼륨 급등
    spike = long_df[long_df['vol_spike'] == 1]
    no_spike = long_df[long_df['vol_spike'] == 0]
    if len(spike) > 10:
        print(f"    볼륨 급등 (+50%): {len(spike)}건, WR: {spike['success'].mean()*100:.1f}%")
    if len(no_spike) > 10:
        print(f"    볼륨 급등 없음: {len(no_spike)}건, WR: {no_spike['success'].mean()*100:.1f}%")

    # 2. 델타 변화
    print("\n  📈 델타 변화량")
    print(f"    성공 평균 델타 변화: {wins['delta_change'].mean():+.1f}")
    print(f"    실패 평균 델타 변화: {losses['delta_change'].mean():+.1f}")

    # 델타 방향 반전
    reversed_delta = long_df[long_df['delta_direction_changed'] == 1]
    same_delta = long_df[long_df['delta_direction_changed'] == 0]
    if len(reversed_delta) > 10:
        print(f"    델타 방향 반전: {len(reversed_delta)}건, WR: {reversed_delta['success'].mean()*100:.1f}%")
    if len(same_delta) > 10:
        print(f"    델타 방향 유지: {len(same_delta)}건, WR: {same_delta['success'].mean()*100:.1f}%")

    # 3. 델타 가속도
    print("\n  🚀 델타 가속도 (LONG = 매수세 증가가 좋음)")
    improving = long_df[long_df['delta_improving'] == 1]
    weakening = long_df[long_df['delta_improving'] == 0]
    if len(improving) > 10:
        print(f"    매수세 증가 추세: {len(improving)}건, WR: {improving['success'].mean()*100:.1f}%")
    if len(weakening) > 10:
        print(f"    매수세 감소 추세: {len(weakening)}건, WR: {weakening['success'].mean()*100:.1f}%")

    # 4. 조합 테스트
    print("\n  🎯 필터 조합")
    filters = [
        ("볼륨급등 & 델타양수", (long_df['vol_spike'] == 1) & (long_df['delta_positive'] == 1)),
        ("볼륨급등 & 매수세증가", (long_df['vol_spike'] == 1) & (long_df['delta_improving'] == 1)),
        ("델타반전 (음→양)", (long_df['delta_direction_changed'] == 1) & (long_df['delta_positive'] == 1)),
        ("델타양수 & 매수세증가", (long_df['delta_positive'] == 1) & (long_df['delta_improving'] == 1)),
        ("델타강도>1.5 & 델타양수", (long_df['delta_strength'] > 1.5) & (long_df['delta_positive'] == 1)),
    ]

    for name, mask in filters:
        subset = long_df[mask]
        if len(subset) >= 15:
            wr = subset['success'].mean() * 100
            diff = wr - base_wr
            print(f"    {name}: {len(subset):>4}건, WR: {wr:.1f}% ({diff:+.1f}%)")


def analyze_short(df):
    """SHORT 바운스 분석."""
    short_df = df[df['direction'] == 'SHORT'].copy()
    if len(short_df) == 0:
        return

    wins = short_df[short_df['success'] == True]
    losses = short_df[short_df['success'] == False]
    base_wr = len(wins) / len(short_df) * 100

    print(f"\n{'='*70}")
    print(f"  SHORT 바운스 분석 ({len(short_df)}건, 기본 WR: {base_wr:.1f}%)")
    print(f"{'='*70}")

    # 1. 볼륨 변화
    print("\n  📊 볼륨 변화량")
    print(f"    성공 평균 볼륨 변화: {wins['vol_change'].mean()*100:+.1f}%")
    print(f"    실패 평균 볼륨 변화: {losses['vol_change'].mean()*100:+.1f}%")

    spike = short_df[short_df['vol_spike'] == 1]
    no_spike = short_df[short_df['vol_spike'] == 0]
    if len(spike) > 10:
        print(f"    볼륨 급등 (+50%): {len(spike)}건, WR: {spike['success'].mean()*100:.1f}%")
    if len(no_spike) > 10:
        print(f"    볼륨 급등 없음: {len(no_spike)}건, WR: {no_spike['success'].mean()*100:.1f}%")

    # 2. 델타 변화
    print("\n  📈 델타 변화량")
    print(f"    성공 평균 델타 변화: {wins['delta_change'].mean():+.1f}")
    print(f"    실패 평균 델타 변화: {losses['delta_change'].mean():+.1f}")

    reversed_delta = short_df[short_df['delta_direction_changed'] == 1]
    same_delta = short_df[short_df['delta_direction_changed'] == 0]
    if len(reversed_delta) > 10:
        print(f"    델타 방향 반전: {len(reversed_delta)}건, WR: {reversed_delta['success'].mean()*100:.1f}%")
    if len(same_delta) > 10:
        print(f"    델타 방향 유지: {len(same_delta)}건, WR: {same_delta['success'].mean()*100:.1f}%")

    # 3. 델타 가속도 (SHORT = 매도세 증가가 좋음)
    print("\n  🚀 델타 가속도 (SHORT = 매도세 증가가 좋음)")
    weakening = short_df[short_df['delta_weakening'] == 1]
    improving = short_df[short_df['delta_weakening'] == 0]
    if len(weakening) > 10:
        print(f"    매도세 증가 추세: {len(weakening)}건, WR: {weakening['success'].mean()*100:.1f}%")
    if len(improving) > 10:
        print(f"    매도세 감소 추세: {len(improving)}건, WR: {improving['success'].mean()*100:.1f}%")

    # 4. 조합 테스트
    print("\n  🎯 필터 조합")
    filters = [
        ("볼륨급등 & 델타음수", (short_df['vol_spike'] == 1) & (short_df['delta_negative'] == 1)),
        ("볼륨급등 & 매도세증가", (short_df['vol_spike'] == 1) & (short_df['delta_weakening'] == 1)),
        ("델타반전 (양→음)", (short_df['delta_direction_changed'] == 1) & (short_df['delta_negative'] == 1)),
        ("델타음수 & 매도세증가", (short_df['delta_negative'] == 1) & (short_df['delta_weakening'] == 1)),
        ("델타강도>1.5 & 델타음수", (short_df['delta_strength'] > 1.5) & (short_df['delta_negative'] == 1)),
    ]

    for name, mask in filters:
        subset = short_df[mask]
        if len(subset) >= 15:
            wr = subset['success'].mean() * 100
            diff = wr - base_wr
            print(f"    {name}: {len(subset):>4}건, WR: {wr:.1f}% ({diff:+.1f}%)")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   델타/볼륨 변화량 분석                                            ║
║   터치 시점의 변화 패턴이 성공을 예측하는가?                          ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]

    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")

    # Build channels
    print("\nBuilding channels...")
    channels_dict, _ = build_htf_channels(htf_2024)

    # Collect data
    print("\nCollecting bounce data with change metrics...")
    df = collect_bounce_with_change(htf_2024, ltf_2024, channels_dict)
    print(f"  Total: {len(df)} bounces")

    if len(df) == 0:
        print("No data!")
        return

    # Analyze
    analyze_long(df)
    analyze_short(df)

    # Summary
    print("\n" + "="*70)
    print("  💡 요약")
    print("="*70)
    print("""
  핵심 관찰 포인트:

  1. 볼륨 급등 (터치 캔들이 이전 5개 평균 대비 50% 이상 증가)
     - 강한 거부의 신호? 아니면 브레이크아웃 시작?

  2. 델타 방향 반전
     - LONG: 이전 매도세 → 터치에서 매수세 전환 = 좋은 신호?
     - SHORT: 이전 매수세 → 터치에서 매도세 전환 = 좋은 신호?

  3. 델타 가속도
     - 최근 3-4캔들의 델타 변화 추세
     - 진입 방향으로 가속되면 좋을까?

  4. 델타 강도
     - 평소 대비 얼마나 강한 델타인지
""")


if __name__ == "__main__":
    main()
