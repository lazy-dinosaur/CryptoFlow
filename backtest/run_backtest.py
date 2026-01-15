#!/usr/bin/env python3
"""
Backtest Runner
Main CLI for running backtests
"""

import argparse
import sys
import os

# Add backtest directory to path
sys.path.insert(0, os.path.dirname(__file__))

from parse_data import load_candles, TIMEFRAMES
from engine import BacktestEngine, save_results
from strategies import SampleDeltaStrategy
import polars as pl


# Available strategies
STRATEGIES = {
    'sample': SampleDeltaStrategy,
}


def main():
    parser = argparse.ArgumentParser(description="Run backtests on historical data")
    parser.add_argument("--strategy", default="sample",
                        choices=list(STRATEGIES.keys()),
                        help="Strategy to test")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="15m",
                        choices=list(TIMEFRAMES.keys()),
                        help="Timeframe to test (default: 15m)")
    parser.add_argument("--capital", type=float, default=10000,
                        help="Initial capital (default: 10000)")
    parser.add_argument("--commission", type=float, default=0.0004,
                        help="Commission rate (default: 0.0004 = 0.04%%)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to file")

    # Strategy-specific parameters
    parser.add_argument("--sma-period", type=int, default=20,
                        help="SMA period for sample strategy")
    parser.add_argument("--stop-loss", type=float, default=0.01,
                        help="Stop loss %% (default: 0.01 = 1%%)")
    parser.add_argument("--take-profit", type=float, default=0.02,
                        help="Take profit %% (default: 0.02 = 2%%)")

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════╗
║           CryptoFlow Backtesting Engine           ║
╚═══════════════════════════════════════════════════╝
""")

    # Load data
    print(f"Loading {args.symbol} {args.timeframe} candles...")
    try:
        candles_pl = load_candles(args.symbol, args.timeframe)
        # Convert to pandas with time as index
        candles = candles_pl.to_pandas().set_index('time')
        print(f"  Loaded {len(candles):,} candles")
        print(f"  Date range: {candles.index[0]} to {candles.index[-1]}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run download_data.py and parse_data.py first:")
        print(f"  python download_data.py --symbol {args.symbol}")
        print(f"  python parse_data.py --symbol {args.symbol}")
        return

    # Initialize strategy
    strategy_params = {
        'sma_period': args.sma_period,
        'stop_loss_pct': args.stop_loss,
        'take_profit_pct': args.take_profit,
    }

    StrategyClass = STRATEGIES[args.strategy]
    strategy = StrategyClass(params=strategy_params)

    print(f"\nRunning {strategy.name} strategy...")
    print(f"  Parameters: {strategy_params}")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        commission=args.commission
    )

    result = engine.run(
        strategy=strategy,
        candles=candles,
        symbol=args.symbol,
        timeframe=args.timeframe
    )

    # Print results
    print(result.summary())

    # Show recent trades
    if result.trades:
        print("\nLast 10 Trades:")
        print("-" * 80)
        for trade in result.trades[-10:]:
            pnl_str = f"+{trade.pnl_pct*100:.2f}%" if trade.pnl_pct > 0 else f"{trade.pnl_pct*100:.2f}%"
            print(f"  {trade.type:5} | Entry: {trade.entry_price:,.2f} → Exit: {trade.exit_price:,.2f} | {pnl_str:>8} | {trade.reason}")

    # Save results
    if args.save:
        save_results(result)


if __name__ == "__main__":
    main()
