#!/usr/bin/env python3
"""
동시 포지션 분석
- 백테스트에서 동시에 몇 개의 포지션이 열려있는지 확인
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def simulate_trade_duration(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """매매 시뮬레이션 - 진입/청산 인덱스 반환."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return idx, j, 'SL'
            if highs[j] >= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:
                        return idx, k, 'TP1_BE'
                    if highs[k] >= tp2:
                        return idx, k, 'TP1_TP2'
                return idx, j + 50, 'TP1_BE'
        else:
            if highs[j] >= sl:
                return idx, j, 'SL'
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return idx, k, 'TP1_BE'
                    if lows[k] <= tp2:
                        return idx, k, 'TP1_TP2'
                return idx, j + 50, 'TP1_BE'
    return idx, idx + 150, 'TIMEOUT'


def collect_trades(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """매매 수집."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    sl_buffer = 0.0008
    touch_threshold = 0.003

    for i in range(100, len(ltf_candles) - 200):
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx - 1)

        if not channel:
            continue

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        mid = (channel.resistance + channel.support) / 2

        bounce_key = (round(channel.support), round(channel.resistance), i // 20)
        if bounce_key in traded_keys:
            continue

        # Support touch → LONG
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                trades.append({
                    'idx': i,
                    'direction': 'LONG',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                })
                traded_keys.add(bounce_key)

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                trades.append({
                    'idx': i,
                    'direction': 'SHORT',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                })
                traded_keys.add(bounce_key)

    return trades


def analyze_concurrent_positions(trades, ltf_candles):
    """동시 포지션 분석."""
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    # 각 매매의 진입/청산 인덱스 계산
    trade_ranges = []
    for t in trades:
        entry_idx, exit_idx, result = simulate_trade_duration(
            ltf_highs, ltf_lows, t['idx'],
            t['direction'], t['entry'], t['sl'], t['tp1'], t['tp2']
        )
        trade_ranges.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'result': result,
            'direction': t['direction'],
        })

    # 각 시점별 열린 포지션 수 계산
    max_idx = max(t['exit_idx'] for t in trade_ranges)
    position_count = np.zeros(max_idx + 1)

    for t in trade_ranges:
        position_count[t['entry_idx']:t['exit_idx']] += 1

    # 통계
    max_concurrent = int(position_count.max())
    avg_concurrent = position_count[position_count > 0].mean()

    # 분포
    distribution = defaultdict(int)
    for count in position_count:
        if count > 0:
            distribution[int(count)] += 1

    return {
        'max_concurrent': max_concurrent,
        'avg_concurrent': avg_concurrent,
        'distribution': dict(distribution),
        'trade_ranges': trade_ranges,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   동시 포지션 분석                                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    print("Building channels...")
    channels_all, _ = build_htf_channels(htf_all)

    print("Collecting trades...")
    trades = collect_trades(htf_all, ltf_all, channels_all)
    print(f"  Total trades: {len(trades)}")

    print("\nAnalyzing concurrent positions...")
    result = analyze_concurrent_positions(trades, ltf_all)

    print("\n" + "="*60)
    print("  📊 동시 포지션 분석 결과")
    print("="*60)

    print(f"\n  최대 동시 포지션: {result['max_concurrent']}개")
    print(f"  평균 동시 포지션: {result['avg_concurrent']:.2f}개")

    print("\n  [포지션 수 분포]")
    for count in sorted(result['distribution'].keys()):
        bars = int(result['distribution'][count] / 100)
        pct = result['distribution'][count] / sum(result['distribution'].values()) * 100
        print(f"    {count}개 포지션: {'█' * min(bars, 50)} ({pct:.1f}%)")

    # 평균 매매 기간
    durations = [t['exit_idx'] - t['entry_idx'] for t in result['trade_ranges']]
    avg_duration = np.mean(durations)
    print(f"\n  평균 매매 기간: {avg_duration:.1f} 캔들 ({avg_duration * 15 / 60:.1f}시간)")

    # 동시에 3개 이상인 경우
    multi_count = sum(1 for c in result['distribution'].keys() if c >= 3)
    if multi_count > 0:
        print(f"\n  ⚠️ 3개 이상 동시 포지션 발생: {sum(result['distribution'].get(i, 0) for i in range(3, 20))} 캔들")


if __name__ == "__main__":
    main()
