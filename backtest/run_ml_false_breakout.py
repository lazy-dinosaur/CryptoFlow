#!/usr/bin/env python3
"""
Backtest False Breakout Strategy with ML Filter
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from engine import BacktestEngine, save_results

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def find_swing_points(candles: pd.DataFrame, n: int = 3):
    """Find all swing highs and lows."""
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(n, len(candles) - n):
        before_max = highs[i-n:i].max()
        after_max = highs[i+1:i+n+1].max()
        if highs[i] > before_max and highs[i] > after_max:
            swing_highs.append({
                'index': i, 'price': highs[i],
                'volume': candles.iloc[i]['volume'],
                'delta': candles.iloc[i]['delta']
            })

        before_min = lows[i-n:i].min()
        after_min = lows[i+1:i+n+1].min()
        if lows[i] < before_min and lows[i] < after_min:
            swing_lows.append({
                'index': i, 'price': lows[i],
                'volume': candles.iloc[i]['volume'],
                'delta': candles.iloc[i]['delta']
            })

    return swing_highs, swing_lows


def backtest_with_ml_filter(candles: pd.DataFrame, confidence_threshold: float = 0.6,
                            risk_per_trade: float = 0.015, initial_capital: float = 10000):
    """
    Run backtest with ML filter and leverage.
    Uses state machine: breakout candle -> wait for return -> entry on next candle

    Args:
        risk_per_trade: Risk per trade as % of capital (0.015 = 1.5%)
        initial_capital: Starting capital
    """

    # Load ML model
    model = joblib.load(os.path.join(MODEL_DIR, "false_breakout_filter.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "false_breakout_scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "false_breakout_features.joblib"))

    # Find swings
    swing_highs, swing_lows = find_swing_points(candles, n=3)

    # Simulation with capital tracking
    trades = []
    position = None
    n = 3
    capital = initial_capital
    equity_curve = [capital]

    # State machine for entry
    pending_setup = None  # Stores breakout info waiting for return

    for i in range(50, len(candles) - 10):
        row = candles.iloc[i]
        price = row['close']
        high = row['high']
        low = row['low']

        # === State 1: Check if in position - handle exit ===
        if position:
            pnl_pct = 0
            exit_price = 0
            result = ""

            if position['type'] == 'LONG':
                if row['low'] <= position['sl']:
                    exit_price = position['sl']
                    pnl_pct = (exit_price - position['entry']) / position['entry']
                    result = 'SL'
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
                    pnl_pct = (exit_price - position['entry']) / position['entry']
                    result = 'TP'
            else:  # SHORT
                if row['high'] >= position['sl']:
                    exit_price = position['sl']
                    pnl_pct = (position['entry'] - exit_price) / position['entry']
                    result = 'SL'
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']
                    pnl_pct = (position['entry'] - exit_price) / position['entry']
                    result = 'TP'

            if result:
                # Calculate actual P&L with leverage
                leverage = position['leverage']
                capital_pnl = capital * pnl_pct * leverage
                capital += capital_pnl

                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'leverage': leverage,
                    'capital_pnl': capital_pnl,
                    'capital_after': capital,
                    'result': result
                })
                equity_curve.append(capital)
                position = None
            continue

        # === State 2: Check for entry from pending setup ===
        if pending_setup:
            setup = pending_setup
            entered = False

            if setup['direction'] == 'SHORT':
                # Wait for price to return below swing high
                if price < setup['swing_price']:
                    sl_price = setup['breakout_extreme'] * 1.001
                    sl_distance = (sl_price - price) / price

                    leverage = risk_per_trade / sl_distance if sl_distance > 0 else 1
                    leverage = min(leverage, 20)

                    position = {
                        'type': 'SHORT',
                        'entry': price,
                        'sl': sl_price,
                        'tp': setup['mid_price'],
                        'prob': setup['prob'],
                        'leverage': leverage
                    }
                    entered = True

                # Cancel if breakout continues (0.5% higher)
                elif high > setup['breakout_extreme'] * 1.005:
                    pending_setup = None

            elif setup['direction'] == 'LONG':
                # Wait for price to return above swing low
                if price > setup['swing_price']:
                    sl_price = setup['breakout_extreme'] * 0.999
                    sl_distance = (price - sl_price) / price

                    leverage = risk_per_trade / sl_distance if sl_distance > 0 else 1
                    leverage = min(leverage, 20)

                    position = {
                        'type': 'LONG',
                        'entry': price,
                        'sl': sl_price,
                        'tp': setup['mid_price'],
                        'prob': setup['prob'],
                        'leverage': leverage
                    }
                    entered = True

                # Cancel if breakout continues (0.5% lower)
                elif low < setup['breakout_extreme'] * 0.995:
                    pending_setup = None

            if entered:
                pending_setup = None
                continue

            # Timeout: cancel after 10 candles
            if i - setup['breakout_idx'] > 10:
                pending_setup = None

        # === State 3: Look for new breakout setup ===
        if pending_setup or position:
            continue

        # Get confirmed swings
        max_swing_idx = i - n
        recent_highs = [s for s in swing_highs if s['index'] <= max_swing_idx]
        recent_lows = [s for s in swing_lows if s['index'] <= max_swing_idx]

        if not recent_highs or not recent_lows:
            continue

        swing_high = recent_highs[-1]
        swing_low = recent_lows[-1]

        range_size = swing_high['price'] - swing_low['price']
        range_pct = range_size / swing_low['price']
        if range_pct < 0.005:
            continue

        mid_price = (swing_high['price'] + swing_low['price']) / 2

        # History features
        hist = candles.iloc[max(0, i-20):i]
        avg_volume = hist['volume'].mean()
        avg_delta = hist['delta'].mean()
        volatility = hist['close'].std() / hist['close'].mean()

        # Wick calculations
        body_top = max(row['open'], row['close'])
        body_bottom = min(row['open'], row['close'])
        upper_wick = row['high'] - body_top
        lower_wick = body_bottom - row['low']
        candle_range = row['high'] - row['low']
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        # Check SHORT setup (breakout above swing high)
        if high > swing_high['price']:
            breakout_distance = (high - swing_high['price']) / swing_high['price']
            volume_weak = int(row['volume'] < swing_high['volume'] * 0.8)
            delta_weak = int(row['delta'] < 0)
            wick_rejection = int(upper_wick_ratio > 0.3)

            if volume_weak or delta_weak or wick_rejection:
                # Build features
                features = pd.DataFrame([{
                    'range_pct': range_pct,
                    'breakout_distance': breakout_distance,
                    'volume_ratio': row['volume'] / avg_volume,
                    'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                    'upper_wick_ratio': upper_wick_ratio,
                    'lower_wick_ratio': lower_wick_ratio,
                    'volatility': volatility,
                    'volume_weak': volume_weak,
                    'delta_weak': delta_weak,
                    'wick_rejection': wick_rejection,
                    'swing_volume_ratio': row['volume'] / swing_high['volume'],
                    'price_vs_mid': (price - mid_price) / range_size,
                }])[feature_cols]

                # ML prediction
                prob = model.predict_proba(scaler.transform(features))[0][1]

                if prob >= confidence_threshold:
                    # Store pending setup - wait for return on NEXT candle
                    pending_setup = {
                        'direction': 'SHORT',
                        'breakout_idx': i,
                        'breakout_extreme': high,
                        'swing_price': swing_high['price'],
                        'mid_price': mid_price,
                        'prob': prob
                    }

        # Check LONG setup (breakout below swing low)
        if low < swing_low['price'] and not pending_setup:
            breakout_distance = (swing_low['price'] - low) / swing_low['price']
            volume_weak = int(row['volume'] < swing_low['volume'] * 0.8)
            delta_weak = int(row['delta'] > 0)
            wick_rejection = int(lower_wick_ratio > 0.3)

            if volume_weak or delta_weak or wick_rejection:
                features = pd.DataFrame([{
                    'range_pct': range_pct,
                    'breakout_distance': breakout_distance,
                    'volume_ratio': row['volume'] / avg_volume,
                    'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                    'upper_wick_ratio': upper_wick_ratio,
                    'lower_wick_ratio': lower_wick_ratio,
                    'volatility': volatility,
                    'volume_weak': volume_weak,
                    'delta_weak': delta_weak,
                    'wick_rejection': wick_rejection,
                    'swing_volume_ratio': row['volume'] / swing_low['volume'],
                    'price_vs_mid': (price - mid_price) / range_size,
                }])[feature_cols]

                prob = model.predict_proba(scaler.transform(features))[0][1]

                if prob >= confidence_threshold:
                    # Store pending setup - wait for return on NEXT candle
                    pending_setup = {
                        'direction': 'LONG',
                        'breakout_idx': i,
                        'breakout_extreme': low,
                        'swing_price': swing_low['price'],
                        'mid_price': mid_price,
                        'prob': prob
                    }

    return trades, equity_curve, capital


def precompute_setups(candles: pd.DataFrame, timeframe: str = "1h"):
    """Pre-compute all setups with ML probabilities (run once)."""
    from tqdm import tqdm

    suffix = f"_{timeframe}" if timeframe != "1h" else ""
    model = joblib.load(os.path.join(MODEL_DIR, f"false_breakout_filter{suffix}.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"false_breakout_scaler{suffix}.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, f"false_breakout_features{suffix}.joblib"))

    swing_highs, swing_lows = find_swing_points(candles, n=3)
    n = 3

    setups = []  # Store all potential setups with their probabilities

    for i in tqdm(range(50, len(candles) - 10), desc="Pre-computing setups"):
        row = candles.iloc[i]
        price = row['close']
        high = row['high']
        low = row['low']

        max_swing_idx = i - n
        recent_highs = [s for s in swing_highs if s['index'] <= max_swing_idx]
        recent_lows = [s for s in swing_lows if s['index'] <= max_swing_idx]

        if not recent_highs or not recent_lows:
            continue

        swing_high = recent_highs[-1]
        swing_low = recent_lows[-1]

        range_size = swing_high['price'] - swing_low['price']
        range_pct = range_size / swing_low['price']
        if range_pct < 0.005:
            continue

        mid_price = (swing_high['price'] + swing_low['price']) / 2

        hist = candles.iloc[max(0, i-20):i]
        avg_volume = hist['volume'].mean()
        avg_delta = hist['delta'].mean()
        volatility = hist['close'].std() / hist['close'].mean()

        body_top = max(row['open'], row['close'])
        body_bottom = min(row['open'], row['close'])
        upper_wick = row['high'] - body_top
        lower_wick = body_bottom - row['low']
        candle_range = row['high'] - row['low']
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        # Check SHORT setup
        if high > swing_high['price']:
            breakout_distance = (high - swing_high['price']) / swing_high['price']
            volume_weak = int(row['volume'] < swing_high['volume'] * 0.8)
            delta_weak = int(row['delta'] < 0)
            wick_rejection = int(upper_wick_ratio > 0.3)

            if volume_weak or delta_weak or wick_rejection:
                features = pd.DataFrame([{
                    'range_pct': range_pct,
                    'breakout_distance': breakout_distance,
                    'volume_ratio': row['volume'] / avg_volume,
                    'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                    'upper_wick_ratio': upper_wick_ratio,
                    'lower_wick_ratio': lower_wick_ratio,
                    'volatility': volatility,
                    'volume_weak': volume_weak,
                    'delta_weak': delta_weak,
                    'wick_rejection': wick_rejection,
                    'swing_volume_ratio': row['volume'] / swing_high['volume'],
                    'price_vs_mid': (price - mid_price) / range_size,
                }])[feature_cols]

                prob = model.predict_proba(scaler.transform(features))[0][1]

                setups.append({
                    'idx': i,
                    'direction': 'SHORT',
                    'breakout_extreme': high,
                    'swing_price': swing_high['price'],
                    'mid_price': mid_price,
                    'prob': prob
                })

        # Check LONG setup
        if low < swing_low['price']:
            breakout_distance = (swing_low['price'] - low) / swing_low['price']
            volume_weak = int(row['volume'] < swing_low['volume'] * 0.8)
            delta_weak = int(row['delta'] > 0)
            wick_rejection = int(lower_wick_ratio > 0.3)

            if volume_weak or delta_weak or wick_rejection:
                features = pd.DataFrame([{
                    'range_pct': range_pct,
                    'breakout_distance': breakout_distance,
                    'volume_ratio': row['volume'] / avg_volume,
                    'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                    'upper_wick_ratio': upper_wick_ratio,
                    'lower_wick_ratio': lower_wick_ratio,
                    'volatility': volatility,
                    'volume_weak': volume_weak,
                    'delta_weak': delta_weak,
                    'wick_rejection': wick_rejection,
                    'swing_volume_ratio': row['volume'] / swing_low['volume'],
                    'price_vs_mid': (price - mid_price) / range_size,
                }])[feature_cols]

                prob = model.predict_proba(scaler.transform(features))[0][1]

                setups.append({
                    'idx': i,
                    'direction': 'LONG',
                    'breakout_extreme': low,
                    'swing_price': swing_low['price'],
                    'mid_price': mid_price,
                    'prob': prob
                })

    return setups


def backtest_from_setups(candles: pd.DataFrame, setups: list, confidence_threshold: float,
                         risk_per_trade: float = 0.015, initial_capital: float = 10000,
                         commission: float = 0.0004):
    """Run backtest using pre-computed setups (fast with numpy).

    Args:
        commission: Per-trade commission rate (0.0004 = 0.04% taker fee)
    """
    trades = []
    position = None
    capital = initial_capital
    equity_curve = [capital]

    # Filter setups by threshold
    valid_setups = {s['idx']: s for s in setups if s['prob'] >= confidence_threshold}
    pending_setup = None

    # Pre-extract numpy arrays for speed (avoid pandas iloc in loop)
    closes = candles['close'].values
    highs = candles['high'].values
    lows = candles['low'].values

    for i in range(50, len(candles) - 10):
        price = closes[i]
        high = highs[i]
        low = lows[i]

        # Handle exit
        if position:
            pnl_pct = 0
            exit_price = 0
            result = ""

            if position['type'] == 'LONG':
                if low <= position['sl']:
                    exit_price = position['sl']
                    pnl_pct = (exit_price - position['entry']) / position['entry']
                    result = 'SL'
                elif high >= position['tp']:
                    exit_price = position['tp']
                    pnl_pct = (exit_price - position['entry']) / position['entry']
                    result = 'TP'
            else:
                if high >= position['sl']:
                    exit_price = position['sl']
                    pnl_pct = (position['entry'] - exit_price) / position['entry']
                    result = 'SL'
                elif low <= position['tp']:
                    exit_price = position['tp']
                    pnl_pct = (position['entry'] - exit_price) / position['entry']
                    result = 'TP'

            if result:
                leverage = position['leverage']
                # Deduct commission (entry + exit = 2x commission rate)
                commission_cost = commission * 2  # Roundtrip commission
                net_pnl_pct = pnl_pct - commission_cost
                capital_pnl = capital * net_pnl_pct * leverage
                capital += capital_pnl

                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'gross_pnl_pct': pnl_pct,
                    'net_pnl_pct': net_pnl_pct,
                    'leverage': leverage,
                    'commission_pct': commission_cost * leverage,
                    'capital_pnl': capital_pnl,
                    'capital_after': capital,
                    'result': result
                })
                equity_curve.append(capital)
                position = None
            continue

        # Check for entry from pending setup
        if pending_setup:
            setup = pending_setup
            entered = False

            if setup['direction'] == 'SHORT':
                if price < setup['swing_price']:
                    sl_price = setup['breakout_extreme'] * 1.001
                    sl_distance = (sl_price - price) / price
                    leverage = min(risk_per_trade / sl_distance if sl_distance > 0 else 1, 20)

                    position = {
                        'type': 'SHORT', 'entry': price, 'sl': sl_price,
                        'tp': setup['mid_price'], 'leverage': leverage
                    }
                    entered = True
                elif high > setup['breakout_extreme'] * 1.005:
                    pending_setup = None

            elif setup['direction'] == 'LONG':
                if price > setup['swing_price']:
                    sl_price = setup['breakout_extreme'] * 0.999
                    sl_distance = (price - sl_price) / price
                    leverage = min(risk_per_trade / sl_distance if sl_distance > 0 else 1, 20)

                    position = {
                        'type': 'LONG', 'entry': price, 'sl': sl_price,
                        'tp': setup['mid_price'], 'leverage': leverage
                    }
                    entered = True
                elif low < setup['breakout_extreme'] * 0.995:
                    pending_setup = None

            if entered:
                pending_setup = None
                continue

            if i - setup['idx'] > 10:
                pending_setup = None

        # Look for new setup
        if not pending_setup and not position and i in valid_setups:
            pending_setup = valid_setups[i]

    return trades, equity_curve, capital


def main(timeframe: str = "1h"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     ML-Enhanced False Breakout Backtest ({timeframe})             ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles ({timeframe})")

    initial_capital = 10000
    risk_per_trade = 0.015  # 1.5% risk per trade
    commission = 0.0005  # 0.05% taker fee (Binance futures, conservative)

    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  Risk per Trade: {risk_per_trade*100}%")
    print(f"  Commission: {commission*100:.2f}% per trade (taker)")

    # Load model for this timeframe
    suffix = f"_{timeframe}" if timeframe != "1h" else ""
    model_path = os.path.join(MODEL_DIR, f"false_breakout_filter{suffix}.joblib")

    if not os.path.exists(model_path):
        print(f"\n  ⚠️  No model found for {timeframe}. Train first with:")
        print(f"     python ml_false_breakout.py {timeframe}")
        return

    # Pre-compute all setups ONCE
    print("\nPre-computing ML setups...")
    setups = precompute_setups(candles, timeframe)
    print(f"  Found {len(setups)} potential setups")

    # Test different thresholds (FAST - no ML inference)
    print("\n" + "="*90)
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Final Cap':^12} | {'Total Return':^12} | {'Max DD':^10}")
    print("="*90)

    best_threshold = 0.6
    best_return = -999

    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        trades, equity_curve, final_capital = backtest_from_setups(
            candles, setups, confidence_threshold=threshold,
            risk_per_trade=risk_per_trade, initial_capital=initial_capital,
            commission=commission
        )

        if trades:
            wins = len([t for t in trades if t['net_pnl_pct'] > 0])
            win_rate = wins / len(trades)
            total_return = (final_capital - initial_capital) / initial_capital * 100

            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            print(f"{threshold:^12.0%} | {len(trades):^8} | {win_rate*100:^10.1f}% | ${final_capital:^11,.0f} | {total_return:^12.1f}% | {max_dd*100:^10.1f}%")

            if total_return > best_return:
                best_return = total_return
                best_threshold = threshold
        else:
            print(f"{threshold:^12.0%} | {'0':^8} | {'-':^10} | {'-':^12} | {'-':^12} | {'-':^10}")

    print("="*90)

    # Detailed results for best threshold
    print(f"\n\nDetailed Results ({best_threshold:.0%} Confidence):")
    print("-"*70)
    trades, equity_curve, final_capital = backtest_from_setups(
        candles, setups, confidence_threshold=best_threshold,
        risk_per_trade=risk_per_trade, initial_capital=initial_capital,
        commission=commission
    )

    if trades:
        wins = [t for t in trades if t['net_pnl_pct'] > 0]
        losses = [t for t in trades if t['net_pnl_pct'] <= 0]

        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Leverage stats
        leverages = [t['leverage'] for t in trades]
        avg_leverage = sum(leverages) / len(leverages)
        max_leverage = max(leverages)
        min_leverage = min(leverages)

        # Commission stats
        total_commission_pct = sum(t['commission_pct'] for t in trades)

        print(f"  Initial Capital: ${initial_capital:,}")
        print(f"  Final Capital:   ${final_capital:,.0f}")
        print(f"  Total Return:    {total_return:.1f}%")
        print(f"  Max Drawdown:    {max_dd*100:.1f}%")
        print(f"\n  Total Trades: {len(trades)}")
        print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")

        print(f"\n  Leverage Stats:")
        print(f"    Avg: {avg_leverage:.1f}x | Min: {min_leverage:.1f}x | Max: {max_leverage:.1f}x")
        print(f"  Total Commission Paid: {total_commission_pct*100:.1f}% of capital")

        long_trades = [t for t in trades if t['type'] == 'LONG']
        short_trades = [t for t in trades if t['type'] == 'SHORT']

        if long_trades:
            long_wins = len([t for t in long_trades if t['net_pnl_pct'] > 0])
            print(f"\n  LONG: {len(long_trades)} trades, {long_wins/len(long_trades)*100:.1f}% win rate")
        if short_trades:
            short_wins = len([t for t in short_trades if t['net_pnl_pct'] > 0])
            print(f"  SHORT: {len(short_trades)} trades, {short_wins/len(short_trades)*100:.1f}% win rate")

        print("\n  Last 10 Trades:")
        print("-"*100)
        print(f"  {'Type':5} | {'Entry':>10} → {'Exit':>10} | {'Gross':>8} | {'Net':>8} | {'Lev':>5} | {'P&L':>10} | {'Result'}")
        print("-"*100)
        for t in trades[-10:]:
            gross_str = f"+{t['gross_pnl_pct']*100:.2f}%" if t['gross_pnl_pct'] > 0 else f"{t['gross_pnl_pct']*100:.2f}%"
            net_str = f"+{t['net_pnl_pct']*100:.2f}%" if t['net_pnl_pct'] > 0 else f"{t['net_pnl_pct']*100:.2f}%"
            cap_pnl = f"+${t['capital_pnl']:,.0f}" if t['capital_pnl'] > 0 else f"-${abs(t['capital_pnl']):,.0f}"
            print(f"  {t['type']:5} | {t['entry']:>10,.2f} → {t['exit']:>10,.2f} | {gross_str:>8} | {net_str:>8} | {t['leverage']:>4.1f}x | {cap_pnl:>10} | {t['result']}")


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(timeframe)
