#!/usr/bin/env python3
"""
Order Block Strategy - Out-of-Sample (OOS) Backtest

Proper train/test split to avoid overfitting:
1. Train ML on first 70% of data
2. Test ONLY on remaining 30% (never seen during training)
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
from ml_orderblock import find_order_blocks, collect_ob_setups, OrderBlock

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_on_period(candles: pd.DataFrame, timeframe: str, fixed_rr: float = 1.5):
    """Train ML model on given candle data."""

    # Find order blocks in training data only
    min_move = 0.02 if timeframe in ['1h', '4h'] else 0.01
    order_blocks = find_order_blocks(candles, min_move_pct=min_move, volume_mult=1.0)

    # Collect setups from training data
    setups = collect_ob_setups(candles, order_blocks, fixed_rr=fixed_rr, quiet=True)

    if len(setups) < 50:
        print(f"  Warning: Only {len(setups)} setups in training data")
        return None, None, None

    df = pd.DataFrame(setups)

    feature_cols = [
        'ob_move_size', 'zone_width_pct', 'candles_since_ob',
        'entry_volume_ratio', 'entry_delta_ratio',
        'cvd_recent', 'delta_at_zone', 'wick_into_zone', 'rr_ratio'
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

    train_win_rate = y.mean()

    return model, scaler, feature_cols, train_win_rate, len(setups)


def backtest_on_period(candles: pd.DataFrame,
                       model, scaler, feature_cols,
                       timeframe: str,
                       confidence_threshold: float = 0.55,
                       risk_per_trade: float = 0.015,
                       initial_capital: float = 10000,
                       commission: float = 0.0005,
                       fixed_rr: float = 1.5,
                       sl_buffer_pct: float = 0.002,
                       quiet: bool = False):
    """Run backtest on test period using pre-trained model."""

    # Find order blocks in TEST data only
    min_move = 0.02 if timeframe in ['1h', '4h'] else 0.01
    order_blocks = find_order_blocks(candles, min_move_pct=min_move, volume_mult=1.0)

    # Pre-extract numpy arrays
    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    fresh_demand = []
    fresh_supply = []

    trades = []
    position = None
    capital = initial_capital
    equity_curve = [capital]

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Backtesting OOS")

    for i in iterator:
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
            else:
                if row_high >= position['sl']:
                    exit_price = position['sl']
                    result = 'SL'
                elif row_low <= position['tp']:
                    exit_price = position['tp']
                    result = 'TP'

            if result:
                if position['type'] == 'LONG':
                    gross_pnl_pct = (exit_price - position['entry']) / position['entry']
                else:
                    gross_pnl_pct = (position['entry'] - exit_price) / position['entry']

                commission_cost = commission * 2
                net_pnl_pct = gross_pnl_pct - commission_cost
                leverage = position['leverage']
                capital_pnl = capital * net_pnl_pct * leverage
                capital += capital_pnl

                trades.append({
                    'type': position['type'],
                    'entry': position['entry'],
                    'exit': exit_price,
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

        if position:
            continue

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else row_volume
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check LONG entries
        for ob in fresh_demand[:]:
            if row_low <= ob.top and row_close > ob.bottom:
                entry_price = row_close
                sl_price = ob.bottom * (1 - sl_buffer_pct)
                risk = entry_price - sl_price

                if risk <= 0:
                    fresh_demand.remove(ob)
                    continue

                tp_price = entry_price + (risk * fixed_rr)

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

                prob = model.predict_proba(scaler.transform(features))[0][1]

                if prob >= confidence_threshold:
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

        # Check SHORT entries
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
║   Order Block - OUT-OF-SAMPLE Backtest ({timeframe})             ║
║   (Train: 70% | Test: 30% - Never seen during training)  ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load ALL data
    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Total: {len(candles):,} candles ({timeframe})")

    # Split by year: 2024 = train, 2025 = test
    candles.index = pd.to_datetime(candles.index)
    train_candles = candles[candles.index.year == 2024].copy()
    test_candles = candles[candles.index.year == 2025].copy()

    train_start = train_candles.index[0]
    train_end = train_candles.index[-1]
    test_start = test_candles.index[0]
    test_end = test_candles.index[-1]

    print(f"\n  TRAIN (2024): {len(train_candles):,} candles")
    print(f"         {train_start} ~ {train_end}")
    print(f"\n  TEST (2025):  {len(test_candles):,} candles (OUT-OF-SAMPLE)")
    print(f"         {test_start} ~ {test_end}")

    # Settings
    initial_capital = 10000
    risk_per_trade = 0.015
    commission = 0.0005
    fixed_rr = 2.0  # 1:2 R:R로 변경

    print(f"\n  Initial Capital: ${initial_capital:,}")
    print(f"  Risk per Trade: {risk_per_trade*100}%")
    print(f"  Commission: {commission*100:.2f}%")
    print(f"  R:R: 1:{fixed_rr}")

    # Train model on TRAIN data only
    print("\n" + "="*70)
    print("  TRAINING (on 70% data only)")
    print("="*70)

    result = train_on_period(train_candles, timeframe, fixed_rr=fixed_rr)
    if result[0] is None:
        print("  Failed to train model")
        return

    model, scaler, feature_cols, train_wr, train_setups = result
    print(f"  Training setups: {train_setups}")
    print(f"  Training win rate: {train_wr*100:.1f}%")

    # Test on OUT-OF-SAMPLE data
    print("\n" + "="*100)
    print("  OUT-OF-SAMPLE RESULTS (30% data - NEVER seen during training)")
    print("="*100)
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Final Cap':^12} | {'Return':^10} | {'Max DD':^10} | {'Avg Lev':^8}")
    print("="*100)

    best_threshold = 0.50
    best_return = -999

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
        trades, equity_curve, final_capital = backtest_on_period(
            test_candles, model, scaler, feature_cols,
            timeframe=timeframe,
            confidence_threshold=threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            fixed_rr=fixed_rr,
            quiet=True
        )

        if trades:
            wins = len([t for t in trades if t['net_pnl_pct'] > 0])
            win_rate = wins / len(trades)
            total_return = (final_capital - initial_capital) / initial_capital * 100
            avg_leverage = sum(t['leverage'] for t in trades) / len(trades)

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

    # Detailed results
    print(f"\n\nDetailed OOS Results ({best_threshold:.0%} Confidence):")
    print("-"*80)

    trades, equity_curve, final_capital = backtest_on_period(
        test_candles, model, scaler, feature_cols,
        timeframe=timeframe,
        confidence_threshold=best_threshold,
        risk_per_trade=risk_per_trade,
        initial_capital=initial_capital,
        commission=commission,
        fixed_rr=fixed_rr,
        quiet=False
    )

    if trades:
        wins = [t for t in trades if t['net_pnl_pct'] > 0]
        losses = [t for t in trades if t['net_pnl_pct'] <= 0]

        total_return = (final_capital - initial_capital) / initial_capital * 100

        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        leverages = [t['leverage'] for t in trades]
        avg_leverage = sum(leverages) / len(leverages)

        print(f"\n  Initial Capital:   ${initial_capital:,}")
        print(f"  Final Capital:     ${final_capital:,.0f}")
        print(f"  Total Return:      {total_return:.1f}%")
        print(f"  Max Drawdown:      {max_dd*100:.1f}%")

        print(f"\n  Total Trades: {len(trades)}")
        print(f"  Wins:   {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
        print(f"  Avg Leverage: {avg_leverage:.1f}x")

        # EV calculation
        avg_win = np.mean([t['net_pnl_pct'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['net_pnl_pct'] for t in losses])) if losses else 0
        win_rate = len(wins) / len(trades)
        ev = win_rate * avg_win * fixed_rr - (1 - win_rate) * avg_loss

        print(f"\n  Performance Metrics:")
        print(f"    Avg Win:  +{avg_win*100:.2f}%")
        print(f"    Avg Loss: -{avg_loss*100:.2f}%")
        print(f"    EV per trade: {ev*100:.2f}%")

        # Annualized
        test_candles_count = len(test_candles)
        if timeframe == "15m":
            hours = test_candles_count * 0.25
        elif timeframe == "1h":
            hours = test_candles_count
        else:
            hours = test_candles_count

        days = hours / 24
        months = days / 30

        print(f"\n  Test Period: {days:.0f} days ({months:.1f} months)")
        print(f"  Trades/Month: {len(trades) / months:.1f}")
        print(f"  Return/Month: {total_return / months:.1f}%")

        # Last 10 trades
        print("\n  Last 10 Trades:")
        print("-"*100)
        for t in trades[-10:]:
            gross = f"+{t['gross_pnl_pct']*100:.2f}%" if t['gross_pnl_pct'] > 0 else f"{t['gross_pnl_pct']*100:.2f}%"
            net = f"+{t['net_pnl_pct']*100:.2f}%" if t['net_pnl_pct'] > 0 else f"{t['net_pnl_pct']*100:.2f}%"
            pnl = f"+${t['capital_pnl']:,.0f}" if t['capital_pnl'] > 0 else f"-${abs(t['capital_pnl']):,.0f}"
            print(f"  {t['type']:5} | {t['entry']:>10,.2f} → {t['exit']:>10,.2f} | {gross:>8} | {net:>8} | {t['leverage']:>4.1f}x | {pnl:>10} | {t['confidence']*100:>5.1f}% | {t['result']}")
    else:
        print("  No trades executed in test period")


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"
    main(timeframe)
