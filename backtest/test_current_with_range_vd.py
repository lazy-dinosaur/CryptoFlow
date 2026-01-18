#!/usr/bin/env python3
"""
현재 전략 + 레인지 대비 볼륨/델타 필터 테스트

핵심: 채널 내 평균 볼륨/델타 대비 터치 시점의 변화
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def simulate_trade_current(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """현재 전략: TP1 50% + TP2 50%, BE 적용."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:
                        return 'TP1_BE', 0.5 * (tp1 - entry) / entry
                    if highs[k] >= tp2:
                        return 'TP1_TP2', 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                return 'TP1_BE', 0.5 * (tp1 - entry) / entry
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return 'TP1_BE', 0.5 * (entry - tp1) / entry
                    if lows[k] <= tp2:
                        return 'TP1_TP2', 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
                return 'TP1_BE', 0.5 * (entry - tp1) / entry
    return None, 0


def collect_trades_with_range_vd(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """채널 내 평균 대비 볼륨/델타 정보 수집."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

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
        volume = ltf_volumes[i]
        delta = ltf_deltas[i]
        mid = (channel.resistance + channel.support) / 2

        # === 채널 내 평균 볼륨/델타 계산 ===
        # 최근 20개 캔들 (채널 내부라고 가정)
        range_lookback = 20
        start_idx = max(0, i - range_lookback)

        range_volumes = ltf_volumes[start_idx:i]
        range_deltas = ltf_deltas[start_idx:i]

        range_avg_vol = np.mean(range_volumes) if len(range_volumes) > 0 else volume
        range_avg_delta = np.mean(range_deltas) if len(range_deltas) > 0 else delta
        range_avg_abs_delta = np.mean(np.abs(range_deltas)) if len(range_deltas) > 0 else abs(delta)

        # 터치 캔들 vs 레인지 평균
        vol_vs_range = volume / range_avg_vol if range_avg_vol > 0 else 1
        delta_vs_range = delta / range_avg_abs_delta if range_avg_abs_delta > 0 else 1

        # 볼륨 급등 (레인지 평균 대비)
        vol_spike_1_5x = vol_vs_range >= 1.5
        vol_spike_2x = vol_vs_range >= 2.0
        vol_spike_3x = vol_vs_range >= 3.0
        vol_low = vol_vs_range <= 0.8

        # 델타 강도 (레인지 평균 대비)
        delta_strong = abs(delta_vs_range) >= 1.5
        delta_very_strong = abs(delta_vs_range) >= 2.0

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
                    # 레인지 대비 비율
                    'vol_vs_range': vol_vs_range,
                    'delta_vs_range': delta_vs_range,
                    # 필터 조건
                    'vol_spike_1_5x': vol_spike_1_5x,
                    'vol_spike_2x': vol_spike_2x,
                    'vol_spike_3x': vol_spike_3x,
                    'vol_low': vol_low,
                    'delta_aligned': delta > 0,  # LONG = 델타 양수
                    'delta_strong': delta_strong and delta > 0,
                    'delta_very_strong': delta_very_strong and delta > 0,
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
                    # 레인지 대비 비율
                    'vol_vs_range': vol_vs_range,
                    'delta_vs_range': delta_vs_range,
                    # 필터 조건
                    'vol_spike_1_5x': vol_spike_1_5x,
                    'vol_spike_2x': vol_spike_2x,
                    'vol_spike_3x': vol_spike_3x,
                    'vol_low': vol_low,
                    'delta_aligned': delta < 0,  # SHORT = 델타 음수
                    'delta_strong': delta_strong and delta < 0,
                    'delta_very_strong': delta_very_strong and delta < 0,
                })
                traded_keys.add(bounce_key)

    return trades


def backtest(trades, ltf_candles, label, filter_fn=None):
    """백테스트 실행."""
    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    wins, losses = 0, 0
    trade_returns = []

    if filter_fn:
        filtered_trades = [t for t in trades if filter_fn(t)]
    else:
        filtered_trades = trades

    for t in filtered_trades:
        result, pnl = simulate_trade_current(
            ltf_highs, ltf_lows, t['idx'],
            t['direction'], t['entry'], t['sl'], t['tp1'], t['tp2']
        )

        if result is None:
            continue

        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1
        position = capital * lev

        net_pnl = position * pnl - position * fee_pct * 2
        trade_returns.append(net_pnl / capital * 100)
        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    avg_pnl = np.mean(trade_returns) if trade_returns else 0

    return {
        'label': label,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'avg_pnl': avg_pnl,
    }


def print_result(r):
    print(f"  {r['label']:<40} | {r['trades']:>4} | {r['wr']:>5.1f}% | {r['avg_pnl']:>+6.2f}% | W{r['wins']}/L{r['losses']}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   현재 전략 + 레인지 대비 볼륨/델타 필터 테스트                       ║
║   핵심: 채널 내 평균 볼륨/델타 대비 터치 시점 비교                     ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    # 연도별 분리
    years = [2022, 2023, 2024, 2025]
    data_by_year = {}

    for year in years:
        htf_year = htf_all[htf_all.index.year == year]
        ltf_year = ltf_all[ltf_all.index.year == year]
        if len(htf_year) > 100:
            data_by_year[year] = {'htf': htf_year, 'ltf': ltf_year}
            print(f"  {year}: HTF={len(htf_year)}, LTF={len(ltf_year)}")

    # 전체 데이터
    htf_all_years = pd.concat([data_by_year[y]['htf'] for y in years if y in data_by_year])
    ltf_all_years = pd.concat([data_by_year[y]['ltf'] for y in years if y in data_by_year])

    print(f"\n  전체 (2022-2025): HTF={len(htf_all_years)}, LTF={len(ltf_all_years)}")

    # Build channels
    print("\nBuilding channels...")
    channels_all, _ = build_htf_channels(htf_all_years)

    # Collect trades
    print("Collecting trades with range volume/delta info...")
    trades_all = collect_trades_with_range_vd(htf_all_years, ltf_all_years, channels_all)

    long_trades = [t for t in trades_all if t['direction'] == 'LONG']
    short_trades = [t for t in trades_all if t['direction'] == 'SHORT']

    print(f"  전체: {len(trades_all)} trades (LONG: {len(long_trades)}, SHORT: {len(short_trades)})")

    # 필터 조건별 매매 수
    print("\n  필터별 매매 수:")
    print(f"    볼륨 >= 1.5x: {len([t for t in trades_all if t['vol_spike_1_5x']])}건")
    print(f"    볼륨 >= 2.0x: {len([t for t in trades_all if t['vol_spike_2x']])}건")
    print(f"    볼륨 >= 3.0x: {len([t for t in trades_all if t['vol_spike_3x']])}건")
    print(f"    볼륨 <= 0.8x: {len([t for t in trades_all if t['vol_low']])}건")
    print(f"    델타 방향일치: {len([t for t in trades_all if t['delta_aligned']])}건")
    print(f"    델타 강함 (1.5x+방향): {len([t for t in trades_all if t['delta_strong']])}건")

    # 필터 정의
    filters = {
        '기본 (전체)': lambda t: True,
        '볼륨 >= 1.5x (레인지 대비)': lambda t: t['vol_spike_1_5x'],
        '볼륨 >= 2.0x': lambda t: t['vol_spike_2x'],
        '볼륨 >= 3.0x': lambda t: t['vol_spike_3x'],
        '볼륨 <= 0.8x (레인지 대비)': lambda t: t['vol_low'],
        '델타 방향일치': lambda t: t['delta_aligned'],
        '델타 강함 (1.5x + 방향)': lambda t: t['delta_strong'],
        '볼륨>=1.5x & 델타방향': lambda t: t['vol_spike_1_5x'] and t['delta_aligned'],
        '볼륨>=2x & 델타방향': lambda t: t['vol_spike_2x'] and t['delta_aligned'],
        '볼륨>=1.5x & 델타강함': lambda t: t['vol_spike_1_5x'] and t['delta_strong'],
    }

    # ===== 전체 결과 =====
    print("\n" + "="*85)
    print("  전체 결과 (2022-2025) - 현재 전략 (TP1 50% + TP2 50%, BE)")
    print("="*85)
    print(f"\n  {'필터':<40} | {'건수':>4} | {'WR':>5} | {'AvgPnL':>7} | {'W/L'}")
    print("-"*85)

    print("\n  [전체 LONG + SHORT]")
    for name, fn in filters.items():
        r = backtest(trades_all, ltf_all_years, name, fn)
        if r['trades'] >= 10:
            print_result(r)

    print("\n  [LONG만]")
    for name, fn in filters.items():
        r = backtest(long_trades, ltf_all_years, name, fn)
        if r['trades'] >= 10:
            print_result(r)

    print("\n  [SHORT만]")
    for name, fn in filters.items():
        r = backtest(short_trades, ltf_all_years, name, fn)
        if r['trades'] >= 10:
            print_result(r)

    # ===== 연도별 일관성 검증 =====
    print("\n" + "="*85)
    print("  연도별 일관성 검증 - 주요 필터")
    print("="*85)

    key_filters = ['기본 (전체)', '볼륨 >= 1.5x (레인지 대비)', '델타 방향일치', '볼륨>=1.5x & 델타방향']

    for year in years:
        if year not in data_by_year:
            continue

        htf_year = data_by_year[year]['htf']
        ltf_year = data_by_year[year]['ltf']
        channels_year, _ = build_htf_channels(htf_year)
        trades_year = collect_trades_with_range_vd(htf_year, ltf_year, channels_year)

        print(f"\n  [{year}] {len(trades_year)}건")
        for name in key_filters:
            fn = filters[name]
            r = backtest(trades_year, ltf_year, name, fn)
            if r['trades'] >= 5:
                print(f"    {name:<35} | {r['trades']:>4} | WR {r['wr']:>5.1f}% | Avg {r['avg_pnl']:>+5.2f}%")

    # Summary
    print("\n" + "="*85)
    print("  📊 요약")
    print("="*85)
    print("""
  핵심 비교:
  - 기본 vs 볼륨급등 (레인지 대비)
  - 기본 vs 델타 방향일치
  - 기본 vs 볼륨급등 & 델타방향
""")


if __name__ == "__main__":
    main()
