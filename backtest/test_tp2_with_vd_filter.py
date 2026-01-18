#!/usr/bin/env python3
"""
TP2 100% + 볼륨/델타 필터 조합 테스트

조합:
1. 기본 (현재 전략): TP1 50% + TP2 50%, BE 적용
2. TP2 100% + BE
3. TP2 100% + BE + 볼륨급등 필터
4. TP2 100% + BE + 델타 방향 필터
5. TP2 100% + BE + 볼륨급등 & 델타방향 필터
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def simulate_trade_tp2_be(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """TP2 100% 청산, TP1에서 BE 이동."""
    tp1_hit = False
    current_sl = sl

    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= current_sl:
                if tp1_hit:
                    return 'BE', 0
                else:
                    return 'SL', (sl - entry) / entry

            if not tp1_hit and highs[j] >= tp1:
                tp1_hit = True
                current_sl = entry

            if highs[j] >= tp2:
                return 'TP2', (tp2 - entry) / entry
        else:
            if highs[j] >= current_sl:
                if tp1_hit:
                    return 'BE', 0
                else:
                    return 'SL', (entry - sl) / entry

            if not tp1_hit and lows[j] <= tp1:
                tp1_hit = True
                current_sl = entry

            if lows[j] <= tp2:
                return 'TP2', (entry - tp2) / entry

    return None, 0


def collect_trades_with_filters(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """볼륨/델타 정보와 함께 매매 수집."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

    sl_buffer = 0.0008
    touch_threshold = 0.003

    for i in range(50, len(ltf_candles) - 200):
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

        # 볼륨/델타 변화량 계산
        n = 5
        if i < n:
            continue

        prev_volumes = ltf_volumes[i-n:i]
        prev_deltas = ltf_deltas[i-n:i]

        avg_vol_prev = np.mean(prev_volumes)
        avg_abs_delta_prev = np.mean(np.abs(prev_deltas))

        max_vol_prev = np.max(prev_volumes)
        vol_spike = volume > max_vol_prev * 1.5

        delta_strength = abs(delta) / avg_abs_delta_prev if avg_abs_delta_prev > 0 else 1

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
                    'vol_spike': vol_spike,
                    'delta_aligned': delta > 0,  # LONG에서 델타 양수
                    'delta_strong': delta_strength > 1.5 and delta > 0,
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
                    'vol_spike': vol_spike,
                    'delta_aligned': delta < 0,  # SHORT에서 델타 음수
                    'delta_strong': delta_strength > 1.5 and delta < 0,
                })
                traded_keys.add(bounce_key)

    return trades


def backtest(trades, ltf_candles, simulate_fn, label, filter_fn=None):
    """백테스트 실행."""
    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    wins, losses, be_count = 0, 0, 0
    trade_returns = []

    # 필터 적용
    if filter_fn:
        filtered_trades = [t for t in trades if filter_fn(t)]
    else:
        filtered_trades = trades

    for t in filtered_trades:
        result, pnl = simulate_fn(
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

        # 실제 수익 기준으로 판정
        if net_pnl > 0:
            wins += 1
        elif net_pnl < 0:
            losses += 1
        else:
            be_count += 1

    total = wins + losses + be_count
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / 10000 - 1) * 100
    avg_pnl = np.mean(trade_returns) if trade_returns else 0

    return {
        'label': label,
        'trades': total,
        'wins': wins,
        'losses': losses,
        'be': be_count,
        'wr': wr,
        'avg_pnl': avg_pnl,
        'return': ret,
    }


def print_result(r):
    print(f"  {r['label']:<45} | {r['trades']:>4} | {r['wr']:>5.1f}% | {r['avg_pnl']:>+7.2f}% | W{r['wins']}/L{r['losses']}/BE{r['be']}")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   TP2 100% + 볼륨/델타 필터 조합 테스트                             ║
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

    # Training: 2022-2023, Test: 2024-2025
    htf_train = pd.concat([data_by_year[y]['htf'] for y in [2022, 2023] if y in data_by_year])
    ltf_train = pd.concat([data_by_year[y]['ltf'] for y in [2022, 2023] if y in data_by_year])
    htf_test = pd.concat([data_by_year[y]['htf'] for y in [2024, 2025] if y in data_by_year])
    ltf_test = pd.concat([data_by_year[y]['ltf'] for y in [2024, 2025] if y in data_by_year])

    print(f"\n  Train (2022-2023): HTF={len(htf_train)}, LTF={len(ltf_train)}")
    print(f"  Test (2024-2025): HTF={len(htf_test)}, LTF={len(ltf_test)}")

    # Build channels
    print("\nBuilding channels...")
    channels_train, _ = build_htf_channels(htf_train)
    channels_test, _ = build_htf_channels(htf_test)

    # Collect trades
    print("Collecting trades with volume/delta info...")
    trades_train = collect_trades_with_filters(htf_train, ltf_train, channels_train)
    trades_test = collect_trades_with_filters(htf_test, ltf_test, channels_test)

    long_train = [t for t in trades_train if t['direction'] == 'LONG']
    short_train = [t for t in trades_train if t['direction'] == 'SHORT']
    long_test = [t for t in trades_test if t['direction'] == 'LONG']
    short_test = [t for t in trades_test if t['direction'] == 'SHORT']

    print(f"  Train: {len(trades_train)} trades (LONG: {len(long_train)}, SHORT: {len(short_train)})")
    print(f"  Test: {len(trades_test)} trades (LONG: {len(long_test)}, SHORT: {len(short_test)})")

    # 필터 조건 확인
    vol_spike_train = len([t for t in trades_train if t['vol_spike']])
    delta_aligned_train = len([t for t in trades_train if t['delta_aligned']])
    both_train = len([t for t in trades_train if t['vol_spike'] and t['delta_aligned']])

    print(f"\n  필터 적용 시 남는 매매 수 (Train 2022-2023):")
    print(f"    볼륨급등: {vol_spike_train}건 ({vol_spike_train/len(trades_train)*100:.1f}%)")
    print(f"    델타방향일치: {delta_aligned_train}건 ({delta_aligned_train/len(trades_train)*100:.1f}%)")
    print(f"    둘 다: {both_train}건 ({both_train/len(trades_train)*100:.1f}%)")

    # 필터 정의
    filters = {
        '기본 (전체)': lambda t: True,
        '볼륨급등': lambda t: t['vol_spike'],
        '델타방향일치': lambda t: t['delta_aligned'],
        '볼륨급등 & 델타방향': lambda t: t['vol_spike'] and t['delta_aligned'],
        '델타강함 (>1.5x)': lambda t: t['delta_strong'],
    }

    # ===== Train Results (2022-2023) =====
    print("\n" + "="*90)
    print("  Train Results (2022-2023)")
    print("="*90)

    print(f"\n  {'전략':<45} | {'건수':>4} | {'WR':>5} | {'AvgPnL':>8} | {'W/L/BE'}")
    print("-"*90)

    print("\n  [전체 LONG + SHORT]")
    for filter_name, filter_fn in filters.items():
        r1 = backtest(trades_train, ltf_train, simulate_trade_current, f"현재 | {filter_name}", filter_fn)
        print_result(r1)

    print()
    for filter_name, filter_fn in filters.items():
        r2 = backtest(trades_train, ltf_train, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        print_result(r2)

    # LONG만
    print("\n  [LONG만]")
    for filter_name, filter_fn in filters.items():
        r = backtest(long_train, ltf_train, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        if r['trades'] > 0:
            print_result(r)

    # SHORT만
    print("\n  [SHORT만]")
    for filter_name, filter_fn in filters.items():
        r = backtest(short_train, ltf_train, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        if r['trades'] > 0:
            print_result(r)

    # ===== Test Results (2024-2025) =====
    print("\n" + "="*90)
    print("  Test Results (2024-2025)")
    print("="*90)

    print(f"\n  {'전략':<45} | {'건수':>4} | {'WR':>5} | {'AvgPnL':>8} | {'W/L/BE'}")
    print("-"*90)

    print("\n  [전체 LONG + SHORT]")
    for filter_name, filter_fn in filters.items():
        r1 = backtest(trades_test, ltf_test, simulate_trade_current, f"현재 | {filter_name}", filter_fn)
        print_result(r1)

    print()
    for filter_name, filter_fn in filters.items():
        r2 = backtest(trades_test, ltf_test, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        print_result(r2)

    # LONG만
    print("\n  [LONG만]")
    for filter_name, filter_fn in filters.items():
        r = backtest(long_test, ltf_test, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        if r['trades'] > 0:
            print_result(r)

    # SHORT만
    print("\n  [SHORT만]")
    for filter_name, filter_fn in filters.items():
        r = backtest(short_test, ltf_test, simulate_trade_tp2_be, f"TP2+BE | {filter_name}", filter_fn)
        if r['trades'] > 0:
            print_result(r)

    # Summary
    print("\n" + "="*90)
    print("  📊 요약")
    print("="*90)
    print("""
  비교 포인트:
  1. 현재 전략 vs TP2 100% + BE
  2. 필터 없음 vs 볼륨/델타 필터
  3. 2024 (IS) vs 2025 (OOS) 일관성
""")


if __name__ == "__main__":
    main()
