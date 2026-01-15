#!/usr/bin/env python3
"""
Order Block Strategy Backtest with ML Filter

Simulates trading with:
- Initial capital
- Leverage based on risk per trade
- Commission (taker fee)
- Equity curve tracking
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_orderblock import find_order_blocks, OrderBlock

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def backtest_orderblock(candles: pd.DataFrame,
                        timeframe: str = "15m",
                        confidence_threshold: float = 0.55,
                        risk_per_trade: float = 0.015,  # 1.5% risk
                        initial_capital: float = 10000,
                        commission: float = 0.0005,  # 0.05% taker
                        fixed_rr: float = 1.5,
                        sl_buffer_pct: float = 0.002):
    """Run backtest with ML filter and capital simulation."""

    # Load ML model
    suffix = f"_{timeframe}" if timeframe != "1h" else ""
    model = joblib.load(os.path.join(MODEL_DIR, f"orderblock_filter{suffix}.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"orderblock_scaler{suffix}.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, f"orderblock_features{suffix}.joblib"))

    # Find order blocks
    min_move = 0.02 if timeframe in ['1h', '4h'] else 0.01
    order_blocks = find_order_blocks(candles, min_move_pct=min_move, volume_mult=1.0)

    # Pre-extract numpy arrays
    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Track fresh zones
    fresh_demand = []
    fresh_supply = []

    # Trading state
    trades = []
    position = None
    capital = initial_capital
    equity_curve = [capital]

    for i in tqdm(range(len(candles)), desc="Backtesting"):
        row_high = highs[i]
        row_low = lows[i]
        row_close = closes[i]
        row_volume = volumes[i]
        row_delta = deltas[i]

        # Add new order blocks
        for ob in order_blocks:
            if ob.idx + 5 == i:
                if ob.type == 'demand':
                    fresh_demand.append(ob)
                else:
                    fresh_supply.append(ob)

        # Handle existing position
        if position:
            exit_price = None
            result = None

            if position['type'] == 'LONG':
                if row_low <= position['sl']:
                    exit_price = position['sl']
                    result = 'SL'
                elif row_high >= position['tp']:
                    exit_price = position['tp']
                    result = 'TP'
            else:  # SHORT
                if row_high >= position['sl']:
                    exit_price = position['sl']
                    result = 'SL'
                elif row_low <= position['tp']:
                    exit_price = position['tp']
                    result = 'TP'

            if result:
                # Calculate P&L
                if position['type'] == 'LONG':
                    gross_pnl_pct = (exit_price - position['entry']) / position['entry']
                else:
                    gross_pnl_pct = (position['entry'] - exit_price) / position['entry']

                # Deduct commission
                commission_cost = commission * 2
                net_pnl_pct = gross_pnl_pct - commission_cost
                leverage = position['leverage']
                capital_pnl = capital * net_pnl_pct * leverage
                capital += capital_pnl

                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'gross_pnl_pct': gross_pnl_pct,
                    'net_pnl_pct': net_pnl_pct,
                    'leverage': leverage,
                    'capital_pnl': capital_pnl,
                    'capital_after': capital,
                    'result': result,
                    'confidence': position['confidence']
                })
                equity_curve.append(capital)
                position = None
            continue

        # Skip if in position
        if position:
            continue

        # Calculate historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else row_volume
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check for LONG entries (demand zones)
        for ob in fresh_demand[:]:
            if row_low <= ob.top and row_close > ob.bottom:
                entry_price = row_close
                sl_price = ob.bottom * (1 - sl_buffer_pct)
                risk = entry_price - sl_price

                if risk <= 0:
                    fresh_demand.remove(ob)
                    continue

                tp_price = entry_price + (risk * fixed_rr)

                # Build features for ML
                features = pd.DataFrame([{
                    'ob_move_size': ob.move_size,
                    'zone_width_pct': (ob.top - ob.bottom) / ob.bottom,
                    'candles_since_ob': i - ob.idx,
                    'entry_volume_ratio': row_volume / avg_volume if avg_volume > 0 else 1,
                    'entry_delta_ratio': row_delta / (abs(avg_delta) + 1),
                    'cvd_recent': cvd_recent,
                    'delta_at_zone': row_delta,
                    'wick_into_zone': (ob.top - row_low) / (ob.top - ob.bottom) if ob.top > ob.bottom else 0,
                    'rr_ratio': fixed_rr
                }])[feature_cols]

                # ML prediction
                prob = model.predict_proba(scaler.transform(features))[0][1]

                if prob >= confidence_threshold:
                    # Calculate leverage
                    sl_distance = risk / entry_price
                    leverage = risk_per_trade / sl_distance if sl_distance > 0 else 1
                    leverage = min(leverage, 20)

                    position = {
                        'type': 'LONG',
                        'entry': entry_price,
                        'sl': sl_price,
                        'tp': tp_price,
                        'leverage': leverage,
                        'confidence': prob
                    }

                fresh_demand.remove(ob)
                break

        # Check for SHORT entries (supply zones)
        if not position:
            for ob in fresh_supply[:]:
                if row_high >= ob.bottom and row_close < ob.top:
                    entry_price = row_close
                    sl_price = ob.top * (1 + sl_buffer_pct)
                    risk = sl_price - entry_price

                    if risk <= 0:
                        fresh_supply.remove(ob)
                        continue

                    tp_price = entry_price - (risk * fixed_rr)

                    # Build features
                    features = pd.DataFrame([{
                        'ob_move_size': ob.move_size,
                        'zone_width_pct': (ob.top - ob.bottom) / ob.bottom,
                        'candles_since_ob': i - ob.idx,
                        'entry_volume_ratio': row_volume / avg_volume if avg_volume > 0 else 1,
                        'entry_delta_ratio': row_delta / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'delta_at_zone': row_delta,
                        'wick_into_zone': (row_high - ob.bottom) / (ob.top - ob.bottom) if ob.top > ob.bottom else 0,
                        'rr_ratio': fixed_rr
                    }])[feature_cols]

                    prob = model.predict_proba(scaler.transform(features))[0][1]

                    if prob >= confidence_threshold:
                        sl_distance = risk / entry_price
                        leverage = risk_per_trade / sl_distance if sl_distance > 0 else 1
                        leverage = min(leverage, 20)

                        position = {
                            'type': 'SHORT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'leverage': leverage,
                            'confidence': prob
                        }

                    fresh_supply.remove(ob)
                    break

        # Clean old zones
        fresh_demand = [ob for ob in fresh_demand if ob.top > row_close * 0.95]
        fresh_supply = [ob for ob in fresh_supply if ob.bottom < row_close * 1.05]

    return trades, equity_curve, capital


def main(timeframe: str = "15m"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     Order Block Backtest with ML Filter ({timeframe})            ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles ({timeframe})")

    # Settings
    initial_capital = 10000
    risk_per_trade = 0.015  # 1.5%
    commission = 0.0005  # 0.05% taker
    fixed_rr = 1.5

    print(f"  Initial Capital: ${initial_capital:,}")
    print(f"  Risk per Trade: {risk_per_trade*100}%")
    print(f"  Commission: {commission*100:.2f}% per trade")
    print(f"  R:R: 1:{fixed_rr}")

    # Test different confidence thresholds
    print("\n" + "="*100)
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Final Cap':^12} | {'Return':^10} | {'Max DD':^10} | {'Avg Lev':^8}")
    print("="*100)

    best_threshold = 0.55
    best_return = -999

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        trades, equity_curve, final_capital = backtest_orderblock(
            candles, timeframe=timeframe,
            confidence_threshold=threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            fixed_rr=fixed_rr
        )

        if trades:
            wins = len([t for t in trades if t['net_pnl_pct'] > 0])
            win_rate = wins / len(trades)
            total_return = (final_capital - initial_capital) / initial_capital * 100
            avg_leverage = sum(t['leverage'] for t in trades) / len(trades)

            # Max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            print(f"{threshold:^12.0%} | {len(trades):^8} | {win_rate*100:^10.1f}% | ${final_capital:^11,.0f} | {total_return:^10.1f}% | {max_dd*100:^10.1f}% | {avg_leverage:^8.1f}x")

            if total_return > best_return:
                best_return = total_return
                best_threshold = threshold
        else:
            print(f"{threshold:^12.0%} | {'0':^8} | {'-':^10} | {'-':^12} | {'-':^10} | {'-':^10} | {'-':^8}")

    print("="*100)

    # Detailed results for best threshold
    print(f"\n\nDetailed Results ({best_threshold:.0%} Confidence):")
    print("-"*80)

    trades, equity_curve, final_capital = backtest_orderblock(
        candles, timeframe=timeframe,
        confidence_threshold=best_threshold,
        risk_per_trade=risk_per_trade,
        initial_capital=initial_capital,
        commission=commission,
        fixed_rr=fixed_rr
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

        # Commission stats
        total_commission = sum(t['gross_pnl_pct'] - t['net_pnl_pct'] for t in trades) * avg_leverage

        print(f"  Initial Capital:   ${initial_capital:,}")
        print(f"  Final Capital:     ${final_capital:,.0f}")
        print(f"  Total Return:      {total_return:.1f}%")
        print(f"  Max Drawdown:      {max_dd*100:.1f}%")

        print(f"\n  Total Trades: {len(trades)}")
        print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")

        print(f"\n  Leverage Stats:")
        print(f"    Avg: {avg_leverage:.1f}x | Max: {max_leverage:.1f}x")
        print(f"  Total Commission: ~{total_commission*100:.1f}% of capital")

        # By type
        long_trades = [t for t in trades if t['type'] == 'LONG']
        short_trades = [t for t in trades if t['type'] == 'SHORT']

        if long_trades:
            long_wins = len([t for t in long_trades if t['net_pnl_pct'] > 0])
            print(f"\n  LONG:  {len(long_trades)} trades, {long_wins/len(long_trades)*100:.1f}% win rate")
        if short_trades:
            short_wins = len([t for t in short_trades if t['net_pnl_pct'] > 0])
            print(f"  SHORT: {len(short_trades)} trades, {short_wins/len(short_trades)*100:.1f}% win rate")

        # Calculate annualized stats
        total_candles = len(candles)
        if timeframe == "15m":
            hours = total_candles * 0.25
        elif timeframe == "1h":
            hours = total_candles
        elif timeframe == "4h":
            hours = total_candles * 4
        else:
            hours = total_candles

        days = hours / 24
        years = days / 365

        trades_per_year = len(trades) / years
        trades_per_month = trades_per_year / 12
        return_per_year = total_return / years

        print(f"\n  Annualized Stats ({years:.1f} years of data):")
        print(f"    Trades/Year:  {trades_per_year:.0f}")
        print(f"    Trades/Month: {trades_per_month:.1f}")
        print(f"    Return/Year:  {return_per_year:.1f}%")

        # Last 10 trades
        print("\n  Last 10 Trades:")
        print("-"*110)
        print(f"  {'Type':5} | {'Entry':>10} → {'Exit':>10} | {'Gross':>8} | {'Net':>8} | {'Lev':>5} | {'P&L':>10} | {'Conf':>6} | {'Result'}")
        print("-"*110)
        for t in trades[-10:]:
            gross = f"+{t['gross_pnl_pct']*100:.2f}%" if t['gross_pnl_pct'] > 0 else f"{t['gross_pnl_pct']*100:.2f}%"
            net = f"+{t['net_pnl_pct']*100:.2f}%" if t['net_pnl_pct'] > 0 else f"{t['net_pnl_pct']*100:.2f}%"
            pnl = f"+${t['capital_pnl']:,.0f}" if t['capital_pnl'] > 0 else f"-${abs(t['capital_pnl']):,.0f}"
            print(f"  {t['type']:5} | {t['entry']:>10,.2f} → {t['exit']:>10,.2f} | {gross:>8} | {net:>8} | {t['leverage']:>4.1f}x | {pnl:>10} | {t['confidence']*100:>5.1f}% | {t['result']}")


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"
    main(timeframe)
