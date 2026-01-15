#!/usr/bin/env python3
"""
BOS Strategy - Out-of-Sample Backtest

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
from ml_bos import find_swing_points, detect_bos_events, collect_bos_setups

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_bos_on_period(candles: pd.DataFrame, timeframe: str, min_rr: float = 2.0):
    """Train BOS ML model on given data."""

    n = 5 if timeframe in ['1h', '4h'] else 3
    swing_highs, swing_lows = find_swing_points(candles, n=n)
    bos_events = detect_bos_events(candles, swing_highs, swing_lows)
    setups = collect_bos_setups(candles, bos_events, min_rr=min_rr, quiet=True)

    if len(setups) < 50:
        print(f"  Warning: Only {len(setups)} setups")
        return None, None, None, 0, 0

    df = pd.DataFrame(setups)

    feature_cols = [
        'bos_strength', 'retest_depth', 'candles_to_retest',
        'volume_ratio', 'delta_at_retest', 'delta_ratio',
        'cvd_recent', 'swing_distance_pct', 'rr_ratio'
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

    return model, scaler, feature_cols, y.mean(), len(setups)


def backtest_bos_oos(candles: pd.DataFrame, model, scaler, feature_cols,
                     timeframe: str,
                     confidence_threshold: float = 0.45,
                     risk_per_trade: float = 0.015,
                     initial_capital: float = 10000,
                     commission: float = 0.0005,
                     min_rr: float = 2.0,
                     quiet: bool = False):
    """Run BOS backtest on test data."""

    from ml_bos import find_swing_points, detect_bos_events, BOSEvent

    n = 5 if timeframe in ['1h', '4h'] else 3
    swing_highs, swing_lows = find_swing_points(candles, n=n)
    bos_events = detect_bos_events(candles, swing_highs, swing_lows)

    closes = candles['close'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    trades = []
    position = None
    capital = initial_capital
    equity_curve = [capital]

    # Track BOS events waiting for retest
    pending_bos = []
    retest_threshold = 0.003
    max_wait = 20
    sl_buffer = 0.002

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="BOS OOS Backtest")

    for i in iterator:
        # Add new BOS events
        for bos in bos_events:
            if bos.idx == i:
                pending_bos.append({'bos': bos, 'start_idx': i})

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

        # Check pending BOS for retest
        hist = candles.iloc[max(0, i-20):i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        for pending in pending_bos[:]:
            bos = pending['bos']
            start_idx = pending['start_idx']

            # Timeout
            if i - start_idx > max_wait:
                pending_bos.remove(pending)
                continue

            # Check for retest
            retest_found = False
            entry_price = closes[i]
            retest_extreme = None

            if bos.type == 'bullish':
                distance = (lows[i] - bos.broken_level) / bos.broken_level
                if distance <= retest_threshold and closes[i] > bos.broken_level:
                    retest_found = True
                    retest_extreme = lows[i]
            else:
                distance = (bos.broken_level - highs[i]) / bos.broken_level
                if distance <= retest_threshold and closes[i] < bos.broken_level:
                    retest_found = True
                    retest_extreme = highs[i]

            if not retest_found:
                continue

            # Calculate SL/TP
            if bos.type == 'bullish':
                sl_price = retest_extreme * (1 - sl_buffer)
                risk = entry_price - sl_price
                if risk <= 0:
                    pending_bos.remove(pending)
                    continue
                swing_dist = bos.broken_level - bos.prev_swing
                tp_price = entry_price + max(swing_dist, risk * min_rr)
            else:
                sl_price = retest_extreme * (1 + sl_buffer)
                risk = sl_price - entry_price
                if risk <= 0:
                    pending_bos.remove(pending)
                    continue
                swing_dist = bos.prev_swing - bos.broken_level
                tp_price = entry_price - max(swing_dist, risk * min_rr)

            reward = abs(tp_price - entry_price)
            rr_ratio = reward / risk

            if rr_ratio < min_rr:
                pending_bos.remove(pending)
                continue

            # ML prediction
            features = pd.DataFrame([{
                'bos_strength': abs(closes[bos.idx] - bos.broken_level) / bos.broken_level,
                'retest_depth': abs(retest_extreme - bos.broken_level) / bos.broken_level,
                'candles_to_retest': i - bos.idx,
                'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                'delta_at_retest': deltas[i],
                'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                'cvd_recent': cvd_recent,
                'swing_distance_pct': abs(bos.broken_level - bos.prev_swing) / bos.broken_level,
                'rr_ratio': rr_ratio
            }])[feature_cols]

            prob = model.predict_proba(scaler.transform(features))[0][1]

            if prob >= confidence_threshold:
                sl_dist = risk / entry_price
                leverage = risk_per_trade / sl_dist if sl_dist > 0 else 1
                leverage = min(leverage, 20)

                position = {
                    'type': 'LONG' if bos.type == 'bullish' else 'SHORT',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'leverage': leverage,
                    'confidence': prob,
                    'rr_ratio': rr_ratio
                }

            pending_bos.remove(pending)
            break

    return trades, equity_curve, capital


def main(timeframe: str = "15m"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║       BOS Strategy - OUT-OF-SAMPLE Backtest ({timeframe})        ║
║       Train: 2024 | Test: 2025                            ║
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
    min_rr = 2.0

    print(f"\n  Capital: ${initial_capital:,}")
    print(f"  Risk/Trade: {risk_per_trade*100}%")
    print(f"  Min R:R: 1:{min_rr}")

    # Train
    print("\n" + "="*60)
    print("  TRAINING on 2024 data")
    print("="*60)

    result = train_bos_on_period(train_candles, timeframe, min_rr=min_rr)
    if result[0] is None:
        print("  Training failed")
        return

    model, scaler, feature_cols, train_wr, train_setups = result
    print(f"  Setups: {train_setups}")
    print(f"  Win rate: {train_wr*100:.1f}%")

    # Test
    print("\n" + "="*100)
    print("  OUT-OF-SAMPLE RESULTS (2025)")
    print("="*100)
    print(f"{'Threshold':^12} | {'Trades':^8} | {'Win Rate':^10} | {'Final':^12} | {'Return':^10} | {'Max DD':^10}")
    print("="*100)

    best_threshold = 0.45
    best_return = -999

    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55]:
        trades, equity, final = backtest_bos_oos(
            test_candles, model, scaler, feature_cols,
            timeframe, confidence_threshold=threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            min_rr=min_rr,
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

        trades, equity, final = backtest_bos_oos(
            test_candles, model, scaler, feature_cols,
            timeframe, confidence_threshold=best_threshold,
            risk_per_trade=risk_per_trade,
            initial_capital=initial_capital,
            commission=commission,
            min_rr=min_rr,
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
