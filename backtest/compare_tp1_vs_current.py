#!/usr/bin/env python3
"""
TP1 100% vs 현재 전략 (TP1 50% + TP2 50% + BE) 비교

정확한 Python 백테스트로 비교
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def simulate_tp1_only(highs, lows, idx, direction, entry, sl, tp1):
    """TP1 100% 청산 전략."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp1:
                return 'TP1', (tp1 - entry) / entry
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry
            if lows[j] <= tp1:
                return 'TP1', (entry - tp1) / entry
    return None, 0


def simulate_current_strategy(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """현재 전략: TP1 50% + TP2 50%, BE 적용."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp1:
                # TP1 hit, now wait for TP2 or BE
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


def backtest(trades, ltf_candles, strategy='tp1_only'):
    """백테스트 실행 - 고정 자본 기준."""
    initial_capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    wins, losses = 0, 0
    trade_pnls = []  # 각 매매의 % 수익
    results_detail = {'SL': 0, 'TP1': 0, 'TP1_BE': 0, 'TP1_TP2': 0}

    for t in trades:
        if strategy == 'tp1_only':
            result, pnl = simulate_tp1_only(
                ltf_highs, ltf_lows, t['idx'],
                t['direction'], t['entry'], t['sl'], t['tp1']
            )
        else:
            result, pnl = simulate_current_strategy(
                ltf_highs, ltf_lows, t['idx'],
                t['direction'], t['entry'], t['sl'], t['tp1'], t['tp2']
            )

        if result is None:
            continue

        if result in results_detail:
            results_detail[result] += 1

        # 고정 자본 기준 레버리지 계산
        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1

        # 수익률 계산 (레버리지 적용, 수수료 차감)
        gross_return = pnl * lev
        net_return = gross_return - (fee_pct * 2 * lev)  # 수수료
        trade_pnls.append(net_return * 100)

        if net_return > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    avg_pnl = np.mean(trade_pnls) if trade_pnls else 0
    total_return = sum(trade_pnls)  # 단순 합 (복리 X)
    final_capital = initial_capital * (1 + total_return / 100)

    return {
        'trades': total,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'avg_pnl': avg_pnl,
        'total_return': total_return,
        'final_capital': final_capital,
        'results': results_detail,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   TP1 100% vs 현재 전략 (TP1 50% + TP2 50% + BE) 비교            ║
║   Python 정밀 백테스트                                            ║
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
        htf = htf_all[htf_all.index.year == year]
        ltf = ltf_all[ltf_all.index.year == year]
        if len(htf) > 100:
            data_by_year[year] = {'htf': htf, 'ltf': ltf}
            print(f"  {year}: HTF={len(htf)}, LTF={len(ltf)}")

    # 전체 데이터
    htf_all_years = pd.concat([data_by_year[y]['htf'] for y in years if y in data_by_year])
    ltf_all_years = pd.concat([data_by_year[y]['ltf'] for y in years if y in data_by_year])

    print(f"\n  전체: HTF={len(htf_all_years)}, LTF={len(ltf_all_years)}")

    # Build channels
    print("\nBuilding channels...")
    channels_all, _ = build_htf_channels(htf_all_years)

    # Collect trades
    print("Collecting trades...")
    trades = collect_trades(htf_all_years, ltf_all_years, channels_all)
    print(f"  Total trades: {len(trades)}")

    # ===== 전체 비교 =====
    print("\n" + "="*80)
    print("  전체 비교 (2022-2025)")
    print("="*80)

    tp1_result = backtest(trades, ltf_all_years, 'tp1_only')
    current_result = backtest(trades, ltf_all_years, 'current')

    print(f"\n  {'전략':<25} | {'매매':>5} | {'WR':>7} | {'Avg PnL':>9} | {'총수익':>10} | {'최종자본':>12}")
    print("-"*80)
    print(f"  {'TP1 100%':<25} | {tp1_result['trades']:>5} | {tp1_result['wr']:>6.1f}% | {tp1_result['avg_pnl']:>+8.2f}% | {tp1_result['total_return']:>+9.1f}% | ${tp1_result['final_capital']:>10,.0f}")
    print(f"  {'현재 (TP1 50%+TP2 50%+BE)':<25} | {current_result['trades']:>5} | {current_result['wr']:>6.1f}% | {current_result['avg_pnl']:>+8.2f}% | {current_result['total_return']:>+9.1f}% | ${current_result['final_capital']:>10,.0f}")

    # 결과 상세
    print("\n  [결과 분포]")
    print(f"    TP1 100%:")
    print(f"      SL: {tp1_result['results']['SL']}, TP1: {tp1_result['results']['TP1']}")
    print(f"    현재 전략:")
    print(f"      SL: {current_result['results']['SL']}, TP1+BE: {current_result['results']['TP1_BE']}, TP1+TP2: {current_result['results']['TP1_TP2']}")

    # ===== 연도별 비교 =====
    print("\n" + "="*80)
    print("  연도별 비교")
    print("="*80)

    for year in years:
        if year not in data_by_year:
            continue

        htf_year = data_by_year[year]['htf']
        ltf_year = data_by_year[year]['ltf']
        channels_year, _ = build_htf_channels(htf_year)
        trades_year = collect_trades(htf_year, ltf_year, channels_year)

        if len(trades_year) < 5:
            continue

        tp1_y = backtest(trades_year, ltf_year, 'tp1_only')
        current_y = backtest(trades_year, ltf_year, 'current')

        print(f"\n  [{year}] {len(trades_year)} trades")
        print(f"    TP1 100%:    WR {tp1_y['wr']:>5.1f}% | Avg {tp1_y['avg_pnl']:>+6.2f}% | Total {tp1_y['total_return']:>+8.1f}%")
        print(f"    현재 전략:    WR {current_y['wr']:>5.1f}% | Avg {current_y['avg_pnl']:>+6.2f}% | Total {current_y['total_return']:>+8.1f}%")

    # ===== 방향별 비교 =====
    print("\n" + "="*80)
    print("  방향별 비교 (전체)")
    print("="*80)

    for direction in ['LONG', 'SHORT']:
        dir_trades = [t for t in trades if t['direction'] == direction]
        if len(dir_trades) < 5:
            continue

        tp1_d = backtest(dir_trades, ltf_all_years, 'tp1_only')
        current_d = backtest(dir_trades, ltf_all_years, 'current')

        print(f"\n  [{direction}] {len(dir_trades)} trades")
        print(f"    TP1 100%:    WR {tp1_d['wr']:>5.1f}% | Avg {tp1_d['avg_pnl']:>+6.2f}% | Total {tp1_d['total_return']:>+8.1f}%")
        print(f"    현재 전략:    WR {current_d['wr']:>5.1f}% | Avg {current_d['avg_pnl']:>+6.2f}% | Total {current_d['total_return']:>+8.1f}%")

    # Summary
    print("\n" + "="*80)
    print("  💡 요약")
    print("="*80)

    wr_diff = tp1_result['wr'] - current_result['wr']
    return_diff = tp1_result['total_return'] - current_result['total_return']

    print(f"""
  TP1 100% vs 현재 전략:
  - WR: {tp1_result['wr']:.1f}% vs {current_result['wr']:.1f}% ({wr_diff:+.1f}%)
  - 총수익: {tp1_result['total_return']:+.1f}% vs {current_result['total_return']:+.1f}% ({return_diff:+.1f}%)

  결론: {"TP1 100%가 더 좋음" if return_diff > 0 else "현재 전략이 더 좋음"}
""")


if __name__ == "__main__":
    main()
