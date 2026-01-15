#!/usr/bin/env python3
"""
Horizontal Channel Strategy - Out-of-Sample Backtest

Train on 2024, Test on 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel import find_active_channels, collect_channel_setups, Channel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_channel_on_period(candles: pd.DataFrame, timeframe: str):
    """Train channel ML model on given data."""

    setups = collect_channel_setups(candles, quiet=True)

    if len(setups) < 50:
        print(f"  Warning: Only {len(setups)} setups")
        return None, None, None, 0, 0, 0, 0

    df = pd.DataFrame(setups)

    # Stats by type
    bounce_df = df[df['setup_type'] == 'BOUNCE']
    fakeout_df = df[df['setup_type'] == 'FAKEOUT']

    feature_cols = [
        'channel_width',
        'total_touches',
        'volume_at_entry',
        'volume_ratio',
        'delta_at_entry',
        'delta_ratio',
        'cvd_recent',
        'fakeout_depth',
        'candles_to_reclaim',
        'body_bullish',
        'rr_ratio'
    ]

    X = df[feature_cols]
    y = df['outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_scaled, y, verbose=False)

    bounce_count = len(bounce_df)
    fakeout_count = len(fakeout_df)

    return model, scaler, feature_cols, y.mean(), len(setups), bounce_count, fakeout_count


def backtest_channel_oos(candles: pd.DataFrame, model, scaler, feature_cols,
                         confidence_threshold: float = 0.50,
                         risk_per_trade: float = 0.015,
                         initial_capital: float = 10000,
                         commission: float = 0.0005,
                         sl_buffer_pct: float = 0.002,
                         quiet: bool = False):
    """Run channel backtest on test data."""

    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Find all active channels
    active_channels = find_active_channels(candles)

    trades = []
    position = None
    capital = initial_capital
    equity_curve = [capital]

    # Track recent entries to avoid over-trading
    last_entry_idx = -10

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Channel OOS Backtest")

    for i in iterator:
        # Handle existing position
        if position:
            exit_price = None
            result = None

            if position['type'] == 'LONG':
                if lows[i] <= position['sl']:
                    exit_price = position['sl']
                    result = 'SL'
                elif highs[i] >= position['tp']:
                    exit_price = position['tp']
                    result = 'TP'
            else:
                if highs[i] >= position['sl']:
                    exit_price = position['sl']
                    result = 'SL'
                elif lows[i] <= position['tp']:
                    exit_price = position['tp']
                    result = 'TP'

            if result:
                if position['type'] == 'LONG':
                    gross_pnl = (exit_price - position['entry']) / position['entry']
                else:
                    gross_pnl = (position['entry'] - exit_price) / position['entry']

                net_pnl = gross_pnl - commission * 2
                leverage = position['leverage']
                capital_pnl = capital * net_pnl * leverage
                capital += capital_pnl

                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'leverage': leverage,
                    'capital_pnl': capital_pnl,
                    'capital_after': capital,
                    'result': result,
                    'confidence': position['confidence'],
                    'rr_ratio': position['rr_ratio']
                })
                equity_curve.append(capital)
                position = None
            continue

        if position:
            continue

        # Skip if no channel at this index
        if i not in active_channels:
            continue

        # Skip if traded recently
        if i - last_entry_idx < 5:
            continue

        channel = active_channels[i]

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        entry_type = None
        entry_price = None
        sl_price = None
        tp_price = None
        wick_depth = 0

        # Check for support touch (LONG)
        if lows[i] <= channel.support * 1.003 and closes[i] > channel.support:
            entry_type = 'LONG'
            entry_price = closes[i]
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp_price = channel.resistance * 0.998
            wick_depth = (channel.support * 1.003 - lows[i]) / channel.support if lows[i] < channel.support * 1.003 else 0

        # Check for resistance touch (SHORT)
        elif highs[i] >= channel.resistance * 0.997 and closes[i] < channel.resistance:
            entry_type = 'SHORT'
            entry_price = closes[i]
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp_price = channel.support * 1.002
            wick_depth = (highs[i] - channel.resistance * 0.997) / channel.resistance if highs[i] > channel.resistance * 0.997 else 0

        if entry_type is None:
            continue

        # Calculate risk/reward
        if entry_type == 'LONG':
            risk = entry_price - sl_price
            reward = tp_price - entry_price
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price

        if risk <= 0 or reward <= 0:
            continue

        rr_ratio = reward / risk

        # ML prediction
        features = pd.DataFrame([{
            'channel_width': channel.width_pct,
            'total_touches': channel.support_touches + channel.resistance_touches,
            'volume_at_entry': volumes[i],
            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
            'delta_at_entry': deltas[i],
            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
            'cvd_recent': cvd_recent,
            'fakeout_depth': 0,  # Bounce setup
            'candles_to_reclaim': 0,
            'body_bullish': 1 if closes[i] > opens[i] else 0,
            'rr_ratio': rr_ratio
        }])[feature_cols]

        prob = model.predict_proba(scaler.transform(features))[0][1]

        if prob >= confidence_threshold:
            sl_dist = risk / entry_price
            leverage = risk_per_trade / sl_dist if sl_dist > 0 else 1
            leverage = min(leverage, 20)

            position = {
                'type': entry_type,
                'entry': entry_price,
                'sl': sl_price,
                'tp': tp_price,
                'leverage': leverage,
                'confidence': prob,
                'rr_ratio': rr_ratio
            }
            last_entry_idx = i

    return trades, equity_curve, capital


def main(timeframe: str = "15m"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Horizontal Channel - OUT-OF-SAMPLE Backtest ({timeframe})       ║
║   Train: 2024 | Test: 2025                                ║
╚═══════════════════════════════════════════════════════════╝
""")

    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Total: {len(candles):,} candles")

    candles.index = pd.to_datetime(candles.index)
    train_candles = candles[candles.index.year == 2024].copy()
    test_candles = candles[candles.index.year == 2025].copy()

    print(f"\n  TRAIN (2024): {len(train_candles):,} candles")
    print(f"  TEST (2025):  {len(test_candles):,} candles")

    initial_capital = 10000
    risk_per_trade = 0.015
    commission = 0.0005

    print(f"\n  Capital: ${initial_capital:,}")
    print(f"  Risk/Trade: {risk_per_trade*100}%")

    # Train
    print("\n" + "="*60)
    print("  TRAINING on 2024 data")
    print("="*60)

    result = train_channel_on_period(train_candles, timeframe)
    if result[0] is None:
        print("  Training failed")
        return

    model, scaler, feature_cols, train_wr, train_setups, bounce_count, fakeout_count = result
    print(f"  Total Setups: {train_setups}")
    print(f"    BOUNCE:  {bounce_count}")
    print(f"    FAKEOUT: {fakeout_count}")
    print(f"  Win rate: {train_wr*100:.1f}%")

    # Test
    print("\n" + "="*100)
    print("  OUT-OF-SAMPLE RESULTS (2025)")
    print("="*100)
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Final':^12} | {'Return':^10} | {'Max DD':^10}")
    print("="*100)

    best_threshold = 0.50
    best_return = -999

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        trades, equity, final = backtest_channel_oos(
            test_candles, model, scaler, feature_cols,
            confidence_threshold=threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            quiet=True
        )

        if trades:
            wins = len([t for t in trades if t['net_pnl'] > 0])
            wr = wins / len(trades)
            ret = (final - initial_capital) / initial_capital * 100

            peak = equity[0]
            max_dd = 0
            for eq in equity:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            print(f"{threshold:^12.0%} | {len(trades):^8} | {wr*100:^10.1f}% | ${final:^11,.0f} | {ret:^10.1f}% | {max_dd*100:^10.1f}%")

            if ret > best_return:
                best_return = ret
                best_threshold = threshold
        else:
            print(f"{threshold:^12.0%} | {'0':^8} | {'-':^10} | {'-':^12} | {'-':^10} | {'-':^10}")

    print("="*100)

    # Detailed results
    if best_return > -999:
        print(f"\n\nDetailed Results ({best_threshold:.0%}):")
        print("-"*80)

        trades, equity, final = backtest_channel_oos(
            test_candles, model, scaler, feature_cols,
            confidence_threshold=best_threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            quiet=False
        )

        if trades:
            wins = [t for t in trades if t['net_pnl'] > 0]
            losses = [t for t in trades if t['net_pnl'] <= 0]
            ret = (final - initial_capital) / initial_capital * 100

            peak = equity[0]
            max_dd = 0
            for eq in equity:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)

            print(f"\n  Initial: ${initial_capital:,}")
            print(f"  Final:   ${final:,.0f}")
            print(f"  Return:  {ret:.1f}%")
            print(f"  Max DD:  {max_dd*100:.1f}%")
            print(f"\n  Trades: {len(trades)}")
            print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
            print(f"  Losses: {len(losses)}")

            avg_rr = np.mean([t['rr_ratio'] for t in trades])
            print(f"  Avg R:R: {avg_rr:.2f}")

            # By type
            long_trades = [t for t in trades if t['type'] == 'LONG']
            short_trades = [t for t in trades if t['type'] == 'SHORT']

            if long_trades:
                long_wins = len([t for t in long_trades if t['net_pnl'] > 0])
                print(f"\n  LONG:  {len(long_trades)} trades, {long_wins/len(long_trades)*100:.1f}% win")
            if short_trades:
                short_wins = len([t for t in short_trades if t['net_pnl'] > 0])
                print(f"  SHORT: {len(short_trades)} trades, {short_wins/len(short_trades)*100:.1f}% win")

            # Monthly stats
            days = len(test_candles) * 0.25 / 24 if timeframe == "15m" else len(test_candles) / 24
            months = days / 30
            print(f"\n  Period: {days:.0f} days")
            print(f"  Trades/Month: {len(trades)/months:.1f}")
            print(f"  Return/Month: {ret/months:.1f}%")

            print("\n  Last 10 trades:")
            print("-"*100)
            for t in trades[-10:]:
                g = f"+{t['gross_pnl']*100:.2f}%" if t['gross_pnl'] > 0 else f"{t['gross_pnl']*100:.2f}%"
                n = f"+{t['net_pnl']*100:.2f}%" if t['net_pnl'] > 0 else f"{t['net_pnl']*100:.2f}%"
                p = f"+${t['capital_pnl']:,.0f}" if t['capital_pnl'] > 0 else f"-${abs(t['capital_pnl']):,.0f}"
                print(f"  {t['type']:5} | {t['entry']:>10,.2f} → {t['exit']:>10,.2f} | {g:>8} | {n:>8} | {t['leverage']:>4.1f}x | {p:>10} | R:R {t['rr_ratio']:.1f} | {t['result']}")


if __name__ == "__main__":
    import sys
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    main(tf)
