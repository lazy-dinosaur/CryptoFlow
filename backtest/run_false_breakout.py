#!/usr/bin/env python3
"""
False Breakout Strategy Backtest Runner
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from parse_data import load_candles
from engine import BacktestEngine, save_results
from strategies import FalseBreakoutStrategy


def main():
    symbol = "BTCUSDT"
    timeframe = "1h"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       False Breakout Strategy Backtest                    ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print(f"Loading {symbol} {timeframe} candles...")
    candles_pl = load_candles(symbol, timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles")
    print(f"  Period: {candles.index[0]} to {candles.index[-1]}")

    # Test 3 scenarios
    test_configs = [
        {'swing_lookback': 3, 'use_tp1_only': True, 'fixed_rr': False, 'name': '1. TP1 중간(50%) 전체종료'},
        {'swing_lookback': 3, 'use_tp1_only': False, 'fixed_rr': False, 'name': '2. 반대 고/저점(100%) 전체종료'},
        {'swing_lookback': 3, 'use_tp1_only': False, 'fixed_rr': True, 'sl_pct': 0.01, 'tp_pct': 0.03, 'name': '3. 고정 1:3 (SL1%, TP3%)'},
    ]

    print(f"\n{'='*90}")
    print(f"{'Config':^20} | {'Trades':^8} | {'Win Rate':^10} | {'Total PnL':^12} | {'Max DD':^10} | {'Sharpe':^8} | {'PF':^6}")
    print(f"{'='*90}")

    results = []
    for config in test_configs:
        name = config.pop('name')

        strategy = FalseBreakoutStrategy(params=config)
        # Pre-calculate all swings once (full scan)
        print(f"\n[{name}] Pre-calculating swings...")
        strategy.precalculate_all_swings(candles)

        engine = BacktestEngine(initial_capital=10000, commission=0.0004)
        result = engine.run(strategy, candles, symbol, timeframe)

        print(f"{name:^20} | {result.total_trades:^8} | {result.win_rate*100:^10.1f}% | {result.total_pnl_pct*100:^12.2f}% | {result.max_drawdown*100:^10.2f}% | {result.sharpe_ratio:^8.2f} | {result.profit_factor:^6.2f}")

        results.append((name, result))

    print(f"{'='*90}")

    # Detailed results for best config
    if not results:
        print("No results to show")
        return

    best_name, best_result = max(results, key=lambda x: x[1].total_pnl_pct)

    print(f"\n\nBest Config: {best_name}")
    print("-"*60)
    print(best_result.summary())

    if best_result.trades:
        # Show trade distribution
        long_trades = [t for t in best_result.trades if t.type == 'LONG']
        short_trades = [t for t in best_result.trades if t.type == 'SHORT']

        long_wins = len([t for t in long_trades if t.pnl_pct > 0])
        short_wins = len([t for t in short_trades if t.pnl_pct > 0])

        print(f"\nTrade Distribution:")
        print(f"  LONG:  {len(long_trades)} trades, {long_wins} wins ({long_wins/len(long_trades)*100:.1f}% win rate)" if long_trades else "  LONG:  0 trades")
        print(f"  SHORT: {len(short_trades)} trades, {short_wins} wins ({short_wins/len(short_trades)*100:.1f}% win rate)" if short_trades else "  SHORT: 0 trades")

        print("\n\nLast 15 Trades:")
        print("-"*100)
        for trade in best_result.trades[-15:]:
            pnl_str = f"+{trade.pnl_pct*100:.2f}%" if trade.pnl_pct > 0 else f"{trade.pnl_pct*100:.2f}%"
            print(f"  {trade.type:5} | {trade.entry_price:>10,.2f} → {trade.exit_price:>10,.2f} | {pnl_str:>8} | {trade.reason[:50]}")

    # Save best result
    save_results(best_result, os.path.join(os.path.dirname(__file__), "results"))


if __name__ == "__main__":
    main()
