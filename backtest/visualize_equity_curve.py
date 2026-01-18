#!/usr/bin/env python3
"""
현재 전략 Equity Curve 시각화
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def simulate_current_strategy(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """현재 전략: TP1 50% + TP2 50%, BE 적용."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry, j
            if highs[j] >= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:
                        return 'TP1_BE', 0.5 * (tp1 - entry) / entry, k
                    if highs[k] >= tp2:
                        return 'TP1_TP2', 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry, k
                return 'TP1_BE', 0.5 * (tp1 - entry) / entry, j + 50
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry, j
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return 'TP1_BE', 0.5 * (entry - tp1) / entry, k
                    if lows[k] <= tp2:
                        return 'TP1_TP2', 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry, k
                return 'TP1_BE', 0.5 * (entry - tp1) / entry, j + 50
    return None, 0, idx


def collect_trades(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """매매 수집."""
    trades = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_times = ltf_candles.index

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
                    'time': ltf_times[i],
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
                    'time': ltf_times[i],
                    'direction': 'SHORT',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                })
                traded_keys.add(bounce_key)

    return trades


def backtest_with_equity(trades, ltf_candles):
    """백테스트 + Equity Curve 생성."""
    initial_capital = 10000
    capital = initial_capital
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_times = ltf_candles.index

    equity_curve = []
    trade_results = []

    for t in trades:
        result, pnl, exit_idx = simulate_current_strategy(
            ltf_highs, ltf_lows, t['idx'],
            t['direction'], t['entry'], t['sl'], t['tp1'], t['tp2']
        )

        if result is None:
            continue

        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1
        position = capital * lev

        net_pnl = position * pnl - position * fee_pct * 2
        capital += net_pnl
        capital = max(capital, 100)  # 최소 자본

        exit_time = ltf_times[min(exit_idx, len(ltf_times) - 1)]

        equity_curve.append({
            'time': exit_time,
            'capital': capital,
            'pnl': net_pnl,
            'result': result,
            'direction': t['direction'],
        })

        trade_results.append({
            'entry_time': t['time'],
            'exit_time': exit_time,
            'direction': t['direction'],
            'result': result,
            'pnl_pct': pnl * 100,
            'capital': capital,
        })

    return equity_curve, trade_results


def main():
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    print("Building channels...")
    channels_all, _ = build_htf_channels(htf_all)

    print("Collecting trades...")
    trades = collect_trades(htf_all, ltf_all, channels_all)
    print(f"  Total trades: {len(trades)}")

    print("Running backtest...")
    equity_curve, trade_results = backtest_with_equity(trades, ltf_all)

    # Convert to DataFrame
    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_results)

    # ===== 시각화 =====
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Channel Bounce Strategy - Equity Curve (2022-2025)', fontsize=14, fontweight='bold')

    # 1. Equity Curve
    ax1 = axes[0]
    ax1.plot(df_equity['time'], df_equity['capital'], color='#2196F3', linewidth=1.5, label='Capital')
    ax1.fill_between(df_equity['time'], 10000, df_equity['capital'], alpha=0.3, color='#2196F3')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Capital ($)', fontsize=11)
    ax1.set_title('Equity Curve (Compounded)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 로그 스케일
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 2. Drawdown
    ax2 = axes[1]
    rolling_max = df_equity['capital'].expanding().max()
    drawdown = (df_equity['capital'] - rolling_max) / rolling_max * 100
    ax2.fill_between(df_equity['time'], 0, drawdown, color='#f44336', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_title('Drawdown', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # 3. Monthly Returns
    ax3 = axes[2]
    df_equity['month'] = df_equity['time'].dt.to_period('M')
    monthly_pnl = df_equity.groupby('month')['pnl'].sum()
    monthly_returns = monthly_pnl / 10000 * 100  # % of initial capital

    colors = ['#4CAF50' if x > 0 else '#f44336' for x in monthly_returns.values]
    ax3.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
    ax3.set_ylabel('Monthly Return (%)', fontsize=11)
    ax3.set_title('Monthly Returns (% of Initial Capital)', fontsize=12)
    ax3.set_xticks(range(0, len(monthly_returns), 3))
    ax3.set_xticklabels([str(monthly_returns.index[i]) for i in range(0, len(monthly_returns), 3)], rotation=45)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'results', 'equity_curve.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # ===== 통계 출력 =====
    print("\n" + "="*60)
    print("  📊 Performance Summary")
    print("="*60)

    final_capital = df_equity['capital'].iloc[-1]
    total_return = (final_capital / 10000 - 1) * 100
    max_dd = drawdown.min()

    wins = len(df_equity[df_equity['pnl'] > 0])
    losses = len(df_equity[df_equity['pnl'] <= 0])
    win_rate = wins / (wins + losses) * 100

    print(f"  Initial Capital:  $10,000")
    print(f"  Final Capital:    ${final_capital:,.0f}")
    print(f"  Total Return:     {total_return:+,.1f}%")
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  Win Rate:         {win_rate:.1f}%")
    print(f"  Total Trades:     {wins + losses}")

    # 연도별 수익
    print("\n  [연도별 수익]")
    df_equity['year'] = df_equity['time'].dt.year
    for year in sorted(df_equity['year'].unique()):
        year_data = df_equity[df_equity['year'] == year]
        year_pnl = year_data['pnl'].sum()
        year_trades = len(year_data)
        print(f"    {year}: {year_trades} trades, ${year_pnl:+,.0f}")

    plt.show()


if __name__ == "__main__":
    main()
