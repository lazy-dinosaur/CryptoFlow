#!/usr/bin/env python3
"""
터치 시점의 볼륨/델타 패턴 분석

핵심 질문:
1. 성공하는 바운스 vs 실패하는 바운스의 볼륨/델타 차이?
2. 어떤 조건이 좋은 필터가 될 수 있나?
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def collect_bounce_patterns(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """터치 시점의 볼륨/델타 패턴 수집."""
    data = []

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
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
        open_price = ltf_opens[i]
        volume = ltf_volumes[i]
        delta = ltf_deltas[i]

        mid = (channel.resistance + channel.support) / 2
        channel_width = (channel.resistance - channel.support) / channel.support

        # Calculate averages for comparison
        lookback = 20
        start_idx = max(0, i - lookback)
        avg_volume = np.mean(ltf_volumes[start_idx:i]) if i > start_idx else volume
        avg_delta = np.mean(np.abs(ltf_deltas[start_idx:i])) if i > start_idx else abs(delta)
        cvd_20 = np.sum(ltf_deltas[start_idx:i])

        # Volume/Delta ratios
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        delta_ratio = abs(delta) / avg_delta if avg_delta > 0 else 1

        # Candle characteristics
        body = close - open_price
        candle_range = high - low
        body_ratio = abs(body) / candle_range if candle_range > 0 else 0
        is_bullish = 1 if close > open_price else 0

        # Lower wick (for support bounce)
        lower_wick = min(open_price, close) - low
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        # Upper wick (for resistance bounce)
        upper_wick = high - max(open_price, close)
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0

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
                    'volume_ratio': volume_ratio,
                    'delta_ratio': delta_ratio,
                    'delta': delta,
                    'delta_positive': 1 if delta > 0 else 0,  # LONG에서 델타 양수 = 매수세
                    'cvd_20': cvd_20,
                    'cvd_bullish': 1 if cvd_20 > 0 else 0,
                    'body_ratio': body_ratio,
                    'is_bullish': is_bullish,  # LONG에서 양봉 = 좋은 신호?
                    'lower_wick_ratio': lower_wick_ratio,  # 긴 꼬리 = 거부 강함?
                    'channel_width': channel_width,
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
                    'volume_ratio': volume_ratio,
                    'delta_ratio': delta_ratio,
                    'delta': delta,
                    'delta_negative': 1 if delta < 0 else 0,  # SHORT에서 델타 음수 = 매도세
                    'cvd_20': cvd_20,
                    'cvd_bearish': 1 if cvd_20 < 0 else 0,
                    'body_ratio': body_ratio,
                    'is_bearish': 1 if not is_bullish else 0,  # SHORT에서 음봉 = 좋은 신호?
                    'upper_wick_ratio': upper_wick_ratio,  # 긴 위꼬리 = 저항 거부?
                    'channel_width': channel_width,
                })

    return pd.DataFrame(data)


def analyze_patterns(df, direction):
    """특정 방향의 패턴 분석."""
    if len(df) == 0:
        return

    dir_df = df[df['direction'] == direction].copy()
    if len(dir_df) == 0:
        return

    wins = dir_df[dir_df['success'] == True]
    losses = dir_df[dir_df['success'] == False]

    print(f"\n{'='*60}")
    print(f"  {direction} 바운스 분석 ({len(dir_df)}건, WR: {len(wins)/len(dir_df)*100:.1f}%)")
    print(f"{'='*60}")

    # 1. 볼륨 분석
    print("\n  📊 볼륨 분석")
    print(f"    전체 평균 볼륨 비율: {dir_df['volume_ratio'].mean():.2f}")
    print(f"    성공 평균 볼륨 비율: {wins['volume_ratio'].mean():.2f}")
    print(f"    실패 평균 볼륨 비율: {losses['volume_ratio'].mean():.2f}")

    # 볼륨 조건별 승률
    print("\n    [볼륨 조건별 승률]")
    for threshold in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        low_vol = dir_df[dir_df['volume_ratio'] <= threshold]
        high_vol = dir_df[dir_df['volume_ratio'] > threshold]
        if len(low_vol) > 10:
            wr = low_vol['success'].mean() * 100
            print(f"    볼륨 <= {threshold}: {len(low_vol):>4}건, WR: {wr:.1f}%")

    # 2. 델타 분석
    print("\n  📈 델타 분석")

    if direction == 'LONG':
        # LONG: 델타 양수가 좋을까?
        delta_aligned = dir_df[dir_df['delta_positive'] == 1]
        delta_opposed = dir_df[dir_df['delta_positive'] == 0]

        print(f"    델타 양수 (매수세): {len(delta_aligned)}건, WR: {delta_aligned['success'].mean()*100:.1f}%" if len(delta_aligned) > 0 else "")
        print(f"    델타 음수 (매도세): {len(delta_opposed)}건, WR: {delta_opposed['success'].mean()*100:.1f}%" if len(delta_opposed) > 0 else "")

        # CVD 분석
        cvd_bull = dir_df[dir_df['cvd_bullish'] == 1]
        cvd_bear = dir_df[dir_df['cvd_bullish'] == 0]
        print(f"    CVD 양수: {len(cvd_bull)}건, WR: {cvd_bull['success'].mean()*100:.1f}%" if len(cvd_bull) > 0 else "")
        print(f"    CVD 음수: {len(cvd_bear)}건, WR: {cvd_bear['success'].mean()*100:.1f}%" if len(cvd_bear) > 0 else "")
    else:
        # SHORT: 델타 음수가 좋을까?
        delta_aligned = dir_df[dir_df['delta_negative'] == 1]
        delta_opposed = dir_df[dir_df['delta_negative'] == 0]

        print(f"    델타 음수 (매도세): {len(delta_aligned)}건, WR: {delta_aligned['success'].mean()*100:.1f}%" if len(delta_aligned) > 0 else "")
        print(f"    델타 양수 (매수세): {len(delta_opposed)}건, WR: {delta_opposed['success'].mean()*100:.1f}%" if len(delta_opposed) > 0 else "")

        # CVD 분석
        cvd_bear = dir_df[dir_df['cvd_bearish'] == 1]
        cvd_bull = dir_df[dir_df['cvd_bearish'] == 0]
        print(f"    CVD 음수: {len(cvd_bear)}건, WR: {cvd_bear['success'].mean()*100:.1f}%" if len(cvd_bear) > 0 else "")
        print(f"    CVD 양수: {len(cvd_bull)}건, WR: {cvd_bull['success'].mean()*100:.1f}%" if len(cvd_bull) > 0 else "")

    # 3. 캔들 패턴 분석
    print("\n  🕯️ 캔들 패턴 분석")

    if direction == 'LONG':
        bullish = dir_df[dir_df['is_bullish'] == 1]
        bearish = dir_df[dir_df['is_bullish'] == 0]
        print(f"    양봉에서 진입: {len(bullish)}건, WR: {bullish['success'].mean()*100:.1f}%" if len(bullish) > 0 else "")
        print(f"    음봉에서 진입: {len(bearish)}건, WR: {bearish['success'].mean()*100:.1f}%" if len(bearish) > 0 else "")

        # 하단 꼬리 분석
        long_wick = dir_df[dir_df['lower_wick_ratio'] >= 0.5]
        short_wick = dir_df[dir_df['lower_wick_ratio'] < 0.5]
        print(f"    긴 하단꼬리 (>=50%): {len(long_wick)}건, WR: {long_wick['success'].mean()*100:.1f}%" if len(long_wick) > 0 else "")
        print(f"    짧은 하단꼬리 (<50%): {len(short_wick)}건, WR: {short_wick['success'].mean()*100:.1f}%" if len(short_wick) > 0 else "")
    else:
        bearish = dir_df[dir_df['is_bearish'] == 1]
        bullish = dir_df[dir_df['is_bearish'] == 0]
        print(f"    음봉에서 진입: {len(bearish)}건, WR: {bearish['success'].mean()*100:.1f}%" if len(bearish) > 0 else "")
        print(f"    양봉에서 진입: {len(bullish)}건, WR: {bullish['success'].mean()*100:.1f}%" if len(bullish) > 0 else "")

        # 상단 꼬리 분석
        long_wick = dir_df[dir_df['upper_wick_ratio'] >= 0.5]
        short_wick = dir_df[dir_df['upper_wick_ratio'] < 0.5]
        print(f"    긴 상단꼬리 (>=50%): {len(long_wick)}건, WR: {long_wick['success'].mean()*100:.1f}%" if len(long_wick) > 0 else "")
        print(f"    짧은 상단꼬리 (<50%): {len(short_wick)}건, WR: {short_wick['success'].mean()*100:.1f}%" if len(short_wick) > 0 else "")


def find_best_filters(df):
    """최적 필터 조합 탐색."""
    print("\n" + "="*60)
    print("  🎯 최적 필터 조합 탐색")
    print("="*60)

    results = []

    # LONG 필터
    long_df = df[df['direction'] == 'LONG'].copy()
    if len(long_df) > 0:
        base_wr = long_df['success'].mean() * 100
        print(f"\n  [LONG] 기본: {len(long_df)}건, WR: {base_wr:.1f}%")

        # 다양한 필터 조합 테스트
        filters = [
            ("볼륨 <= 1.0", long_df['volume_ratio'] <= 1.0),
            ("볼륨 <= 0.8", long_df['volume_ratio'] <= 0.8),
            ("델타 양수", long_df['delta_positive'] == 1),
            ("CVD 양수", long_df['cvd_bullish'] == 1),
            ("양봉", long_df['is_bullish'] == 1),
            ("긴 하단꼬리", long_df['lower_wick_ratio'] >= 0.5),
            ("볼륨<=1.0 & 델타양수", (long_df['volume_ratio'] <= 1.0) & (long_df['delta_positive'] == 1)),
            ("볼륨<=1.0 & CVD양수", (long_df['volume_ratio'] <= 1.0) & (long_df['cvd_bullish'] == 1)),
            ("볼륨<=1.0 & 양봉", (long_df['volume_ratio'] <= 1.0) & (long_df['is_bullish'] == 1)),
            ("델타양수 & CVD양수", (long_df['delta_positive'] == 1) & (long_df['cvd_bullish'] == 1)),
            ("델타양수 & 양봉", (long_df['delta_positive'] == 1) & (long_df['is_bullish'] == 1)),
            ("볼륨<=1.0 & 델타양수 & CVD양수", (long_df['volume_ratio'] <= 1.0) & (long_df['delta_positive'] == 1) & (long_df['cvd_bullish'] == 1)),
        ]

        for name, mask in filters:
            subset = long_df[mask]
            if len(subset) >= 20:
                wr = subset['success'].mean() * 100
                improvement = wr - base_wr
                if improvement > 0:
                    results.append(('LONG', name, len(subset), wr, improvement))
                    print(f"    {name}: {len(subset):>4}건, WR: {wr:.1f}% (+{improvement:.1f}%)")

    # SHORT 필터
    short_df = df[df['direction'] == 'SHORT'].copy()
    if len(short_df) > 0:
        base_wr = short_df['success'].mean() * 100
        print(f"\n  [SHORT] 기본: {len(short_df)}건, WR: {base_wr:.1f}%")

        filters = [
            ("볼륨 <= 1.0", short_df['volume_ratio'] <= 1.0),
            ("볼륨 <= 0.8", short_df['volume_ratio'] <= 0.8),
            ("델타 음수", short_df['delta_negative'] == 1),
            ("CVD 음수", short_df['cvd_bearish'] == 1),
            ("음봉", short_df['is_bearish'] == 1),
            ("긴 상단꼬리", short_df['upper_wick_ratio'] >= 0.5),
            ("볼륨<=1.0 & 델타음수", (short_df['volume_ratio'] <= 1.0) & (short_df['delta_negative'] == 1)),
            ("볼륨<=1.0 & CVD음수", (short_df['volume_ratio'] <= 1.0) & (short_df['cvd_bearish'] == 1)),
            ("볼륨<=1.0 & 음봉", (short_df['volume_ratio'] <= 1.0) & (short_df['is_bearish'] == 1)),
            ("델타음수 & CVD음수", (short_df['delta_negative'] == 1) & (short_df['cvd_bearish'] == 1)),
            ("델타음수 & 음봉", (short_df['delta_negative'] == 1) & (short_df['is_bearish'] == 1)),
            ("볼륨<=1.0 & 델타음수 & CVD음수", (short_df['volume_ratio'] <= 1.0) & (short_df['delta_negative'] == 1) & (short_df['cvd_bearish'] == 1)),
        ]

        for name, mask in filters:
            subset = short_df[mask]
            if len(subset) >= 20:
                wr = subset['success'].mean() * 100
                improvement = wr - base_wr
                if improvement > 0:
                    results.append(('SHORT', name, len(subset), wr, improvement))
                    print(f"    {name}: {len(subset):>4}건, WR: {wr:.1f}% (+{improvement:.1f}%)")

    # 최적 필터 요약
    if results:
        print("\n" + "-"*60)
        print("  📌 TOP 5 필터 (승률 개선 기준)")
        print("-"*60)
        sorted_results = sorted(results, key=lambda x: x[4], reverse=True)[:5]
        for dir, name, count, wr, improvement in sorted_results:
            print(f"  [{dir}] {name}: {count}건, WR {wr:.1f}% (+{improvement:.1f}%)")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   터치 시점 볼륨/델타 패턴 분석                                     ║
║   목표: 성공하는 바운스의 특성 파악                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    # Use 2024 for analysis
    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]

    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")

    # Build channels
    print("\nBuilding channels...")
    channels_dict, _ = build_htf_channels(htf_2024)
    print(f"  Channels: {len(channels_dict)}")

    # Collect patterns
    print("\nCollecting bounce patterns...")
    df = collect_bounce_patterns(htf_2024, ltf_2024, channels_dict)
    print(f"  Total samples: {len(df)}")

    if len(df) == 0:
        print("No data collected!")
        return

    # Overall stats
    print("\n" + "="*60)
    print("  전체 통계")
    print("="*60)
    total_wr = df['success'].mean() * 100
    print(f"  총 바운스: {len(df)}건")
    print(f"  전체 승률: {total_wr:.1f}%")

    long_df = df[df['direction'] == 'LONG']
    short_df = df[df['direction'] == 'SHORT']
    print(f"  LONG: {len(long_df)}건, WR: {long_df['success'].mean()*100:.1f}%")
    print(f"  SHORT: {len(short_df)}건, WR: {short_df['success'].mean()*100:.1f}%")

    # Analyze each direction
    analyze_patterns(df, 'LONG')
    analyze_patterns(df, 'SHORT')

    # Find best filters
    find_best_filters(df)

    # Summary
    print("\n" + "="*60)
    print("  💡 요약 및 권장사항")
    print("="*60)
    print("""
  위 분석 결과를 바탕으로:

  1. 볼륨 필터:
     - 볼륨이 평균 이하일 때 진입하면 승률이 높을까?

  2. 델타 필터:
     - LONG: 델타가 양수일 때 (매수세가 있을 때)
     - SHORT: 델타가 음수일 때 (매도세가 있을 때)

  3. CVD 필터:
     - 최근 CVD 추세가 진입 방향과 일치할 때

  4. 캔들 패턴:
     - LONG: 양봉 + 긴 하단꼬리 (지지 거부)
     - SHORT: 음봉 + 긴 상단꼬리 (저항 거부)
""")


if __name__ == "__main__":
    main()
