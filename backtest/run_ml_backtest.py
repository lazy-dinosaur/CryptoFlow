#!/usr/bin/env python3
"""
ML Backtest Runner
Tests ML strategy with different confidence thresholds
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from parse_data import load_candles
from engine import BacktestEngine
from strategies import MLStrategy

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def find_latest_model(symbol: str, timeframe: str):
    """Find the latest model files."""
    pattern = os.path.join(MODEL_DIR, f"xgb_{symbol}_{timeframe}_*.joblib")
    models = sorted(glob.glob(pattern))
    if not models:
        return None, None, None

    latest = models[-1]
    # Extract timestamp: xgb_BTCUSDT_1h_20260114_210620.joblib -> 20260114_210620
    basename = os.path.basename(latest).replace('.joblib', '')
    parts = basename.split('_')
    timestamp = '_'.join(parts[-2:])  # Last two parts are the timestamp

    model_path = latest
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{symbol}_{timeframe}_{timestamp}.joblib")
    features_path = os.path.join(MODEL_DIR, f"features_{symbol}_{timeframe}_{timestamp}.joblib")

    return model_path, scaler_path, features_path


def main():
    symbol = "BTCUSDT"
    timeframe = "1h"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           ML Strategy Backtest                            ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Find model
    model_path, scaler_path, features_path = find_latest_model(symbol, timeframe)
    if not model_path:
        print(f"No model found for {symbol} {timeframe}")
        print("Run ml_train.py first")
        return

    print(f"Model: {os.path.basename(model_path)}")

    # Load data
    print(f"\nLoading {symbol} {timeframe} candles...")
    candles_pl = load_candles(symbol, timeframe)
    candles = candles_pl.to_pandas().set_index('time')

    # Use only test period (last 20% of data, same as training)
    test_start = int(len(candles) * 0.8)
    test_candles = candles.iloc[test_start:]
    print(f"  Test period: {test_candles.index[0]} to {test_candles.index[-1]}")
    print(f"  Test candles: {len(test_candles):,}")

    # Test different confidence thresholds
    thresholds = [0.55, 0.60, 0.65, 0.70]

    print(f"\n{'='*80}")
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Total PnL':^12} | {'Max DD':^10} | {'Sharpe':^8}")
    print(f"{'='*80}")

    for threshold in thresholds:
        strategy = MLStrategy(params={
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path,
            'confidence_threshold': threshold,
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.03,  # 1:3 R:R
        })

        engine = BacktestEngine(initial_capital=10000, commission=0.0004)
        result = engine.run(strategy, test_candles, symbol, timeframe)

        print(f"{threshold:^12.0%} | {result.total_trades:^8} | {result.win_rate*100:^10.1f}% | {result.total_pnl_pct*100:^12.2f}% | {result.max_drawdown*100:^10.2f}% | {result.sharpe_ratio:^8.2f}")

    print(f"{'='*80}")

    # Detailed results for 60% threshold
    print(f"\n\nDetailed Results (60% Confidence Threshold):")
    print("-"*60)

    strategy = MLStrategy(params={
        'model_path': model_path,
        'scaler_path': scaler_path,
        'features_path': features_path,
        'confidence_threshold': 0.60,
        'stop_loss_pct': 0.01,
        'take_profit_pct': 0.03,  # 1:3 R:R
    })

    engine = BacktestEngine(initial_capital=10000, commission=0.0004)
    result = engine.run(strategy, test_candles, symbol, timeframe)

    print(result.summary())

    if result.trades:
        print("\nLast 10 Trades:")
        print("-"*80)
        for trade in result.trades[-10:]:
            pnl_str = f"+{trade.pnl_pct*100:.2f}%" if trade.pnl_pct > 0 else f"{trade.pnl_pct*100:.2f}%"
            print(f"  {trade.type:5} | {trade.entry_price:,.2f} → {trade.exit_price:,.2f} | {pnl_str:>8} | {trade.reason}")


if __name__ == "__main__":
    main()
