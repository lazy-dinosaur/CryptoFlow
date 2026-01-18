#!/usr/bin/env python3
"""
R:R 및 청산 전략 테스트

비교:
1. 현재 전략: TP1에서 50% 청산, TP2에서 50% 청산 (BE 적용)
2. TP1 100% 청산: TP1에서 전량 청산
3. TP2 100% 청산: TP2에서 전량 청산 (TP1 BE 없음)
4. TP2 100% + R:R >= 1:2: TP2까지 R:R 1:2 이상만 진입
5. TP2 100% + R:R >= 1:3: TP2까지 R:R 1:3 이상만 진입
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


def simulate_trade_current(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """현재 전략: TP1 50% + TP2 50% (BE 적용)."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp1:
                # TP1 hit - check for TP2 or BE
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:  # BE hit
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


def simulate_trade_tp1_only(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """TP1 100% 청산."""
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


def simulate_trade_tp2_only(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """TP2 100% 청산 (BE 없음)."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp2:
                return 'TP2', (tp2 - entry) / entry
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry
            if lows[j] <= tp2:
                return 'TP2', (entry - tp2) / entry
    return None, 0


def simulate_trade_tp2_with_be(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """TP2 100% 청산 (TP1에서 BE 이동)."""
    tp1_hit = False
    current_sl = sl

    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            # Check SL
            if lows[j] <= current_sl:
                if tp1_hit:
                    return 'BE', 0  # BE hit after TP1
                else:
                    return 'SL', (sl - entry) / entry

            # Check TP1 (move to BE)
            if not tp1_hit and highs[j] >= tp1:
                tp1_hit = True
                current_sl = entry  # Move SL to BE

            # Check TP2
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


def collect_trades(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """Collect all potential trades with their properties."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

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
                sl_dist = (entry - sl) / entry
                tp1_dist = (tp1 - entry) / entry
                tp2_dist = (tp2 - entry) / entry
                rr_tp1 = tp1_dist / sl_dist if sl_dist > 0 else 0
                rr_tp2 = tp2_dist / sl_dist if sl_dist > 0 else 0

                trades.append({
                    'idx': i,
                    'direction': 'LONG',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'sl_dist': sl_dist,
                    'rr_tp1': rr_tp1,
                    'rr_tp2': rr_tp2,
                })
                traded_keys.add(bounce_key)

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                sl_dist = (sl - entry) / entry
                tp1_dist = (entry - tp1) / entry
                tp2_dist = (entry - tp2) / entry
                rr_tp1 = tp1_dist / sl_dist if sl_dist > 0 else 0
                rr_tp2 = tp2_dist / sl_dist if sl_dist > 0 else 0

                trades.append({
                    'idx': i,
                    'direction': 'SHORT',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'sl_dist': sl_dist,
                    'rr_tp1': rr_tp1,
                    'rr_tp2': rr_tp2,
                })
                traded_keys.add(bounce_key)

    return trades


def backtest_strategy(trades, ltf_candles, simulate_fn, label, min_rr=0, use_tp2_rr=False):
    """Run backtest with specific strategy."""
    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    wins, losses, be_count = 0, 0, 0
    trade_returns = []

    # Filter by R:R if specified
    if min_rr > 0:
        if use_tp2_rr:
            filtered_trades = [t for t in trades if t['rr_tp2'] >= min_rr]
        else:
            filtered_trades = [t for t in trades if t['rr_tp1'] >= min_rr]
    else:
        filtered_trades = trades

    for t in filtered_trades:
        result, pnl = simulate_fn(
            ltf_highs, ltf_lows, t['idx'],
            t['direction'], t['entry'], t['sl'], t['tp1'], t['tp2']
        )

        if result is None:
            continue

        sl_dist = t['sl_dist']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1
        position = capital * lev

        net_pnl = position * pnl - position * fee_pct * 2
        trade_returns.append(net_pnl / capital * 100)
        capital += net_pnl
        capital = max(capital, 0)

        if 'SL' in result:
            losses += 1
        elif 'BE' in result or pnl == 0:
            be_count += 1
        else:
            wins += 1

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
        'final_capital': capital
    }


def print_result(r):
    """Print formatted result."""
    print(f"  {r['label']:<35} | {r['trades']:>4} | {r['wr']:>5.1f}% | {r['avg_pnl']:>+6.2f}% | {r['return']:>+7.1f}%")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   R:R 및 청산 전략 테스트                                          ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    # 2024 for training
    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]

    # 2025 for OOS
    htf_2025 = htf_all[htf_all.index.year == 2025]
    ltf_2025 = ltf_all[ltf_all.index.year == 2025]

    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")
    print(f"  2025: HTF={len(htf_2025)}, LTF={len(ltf_2025)}")

    # Build channels
    print("\nBuilding channels...")
    channels_2024, _ = build_htf_channels(htf_2024)
    channels_2025, _ = build_htf_channels(htf_2025)

    # Collect trades
    print("Collecting trades...")
    trades_2024 = collect_trades(htf_2024, ltf_2024, channels_2024)
    trades_2025 = collect_trades(htf_2025, ltf_2025, channels_2025)
    print(f"  2024: {len(trades_2024)} trades")
    print(f"  2025: {len(trades_2025)} trades")

    # R:R 분포 확인
    print("\n" + "="*70)
    print("  R:R 분포 (TP2 기준)")
    print("="*70)

    rr_values = [t['rr_tp2'] for t in trades_2024]
    print(f"  Mean R:R: {np.mean(rr_values):.2f}")
    print(f"  Median R:R: {np.median(rr_values):.2f}")
    print(f"  Min R:R: {np.min(rr_values):.2f}")
    print(f"  Max R:R: {np.max(rr_values):.2f}")

    # R:R 구간별 분포
    for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        count = len([t for t in trades_2024 if t['rr_tp2'] >= rr])
        pct = count / len(trades_2024) * 100
        print(f"  R:R >= {rr}: {count}건 ({pct:.1f}%)")

    # Test strategies
    print("\n" + "="*70)
    print("  2024 Results (In-Sample)")
    print("="*70)
    print(f"  {'전략':<35} | {'건수':>4} | {'WR':>5} | {'Avg':>7} | {'Return':>8}")
    print("-"*70)

    strategies_2024 = [
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_current, "1. 현재 (TP1 50% + TP2 50%, BE)"),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp1_only, "2. TP1 100% 청산"),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_only, "3. TP2 100% (BE 없음)"),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_with_be, "4. TP2 100% (TP1에서 BE 이동)"),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_with_be, "5. TP2 100% + R:R >= 1.5", min_rr=1.5, use_tp2_rr=True),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_with_be, "6. TP2 100% + R:R >= 2.0", min_rr=2.0, use_tp2_rr=True),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_with_be, "7. TP2 100% + R:R >= 2.5", min_rr=2.5, use_tp2_rr=True),
        backtest_strategy(trades_2024, ltf_2024, simulate_trade_tp2_with_be, "8. TP2 100% + R:R >= 3.0", min_rr=3.0, use_tp2_rr=True),
    ]

    for r in strategies_2024:
        print_result(r)

    # 2025 OOS
    print("\n" + "="*70)
    print("  2025 Results (Out-of-Sample)")
    print("="*70)
    print(f"  {'전략':<35} | {'건수':>4} | {'WR':>5} | {'Avg':>7} | {'Return':>8}")
    print("-"*70)

    strategies_2025 = [
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_current, "1. 현재 (TP1 50% + TP2 50%, BE)"),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp1_only, "2. TP1 100% 청산"),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_only, "3. TP2 100% (BE 없음)"),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_with_be, "4. TP2 100% (TP1에서 BE 이동)"),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_with_be, "5. TP2 100% + R:R >= 1.5", min_rr=1.5, use_tp2_rr=True),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_with_be, "6. TP2 100% + R:R >= 2.0", min_rr=2.0, use_tp2_rr=True),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_with_be, "7. TP2 100% + R:R >= 2.5", min_rr=2.5, use_tp2_rr=True),
        backtest_strategy(trades_2025, ltf_2025, simulate_trade_tp2_with_be, "8. TP2 100% + R:R >= 3.0", min_rr=3.0, use_tp2_rr=True),
    ]

    for r in strategies_2025:
        print_result(r)

    # Summary
    print("\n" + "="*70)
    print("  📊 요약")
    print("="*70)

    # Find best by avg_pnl in 2024
    best_2024 = max(strategies_2024, key=lambda x: x['avg_pnl'])
    print(f"\n  2024 최고 Avg PnL: {best_2024['label']}")
    print(f"    → {best_2024['trades']}건, WR {best_2024['wr']:.1f}%, Avg {best_2024['avg_pnl']:+.2f}%, Return {best_2024['return']:+.1f}%")

    # Compare current vs TP2 with BE
    curr_2024 = strategies_2024[0]
    tp2_be_2024 = strategies_2024[3]
    print(f"\n  현재 전략 vs TP2 100% (BE) 비교:")
    print(f"    현재:     {curr_2024['trades']}건, Avg {curr_2024['avg_pnl']:+.2f}%, Return {curr_2024['return']:+.1f}%")
    print(f"    TP2+BE:   {tp2_be_2024['trades']}건, Avg {tp2_be_2024['avg_pnl']:+.2f}%, Return {tp2_be_2024['return']:+.1f}%")


if __name__ == "__main__":
    main()
