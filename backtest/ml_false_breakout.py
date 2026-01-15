#!/usr/bin/env python3
"""
ML-Enhanced False Breakout Strategy
1. Collect all false breakout setups from historical data
2. Label them as win/loss based on actual outcome
3. Train ML to predict which setups will win
4. Use ML filter to only take high-probability setups
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def find_swing_points(candles: pd.DataFrame, n: int = 3):
    """Find all swing highs and lows."""
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(n, len(candles) - n):
        # Swing High
        before_max = highs[i-n:i].max()
        after_max = highs[i+1:i+n+1].max()
        if highs[i] > before_max and highs[i] > after_max:
            swing_highs.append({
                'index': i,
                'price': highs[i],
                'time': candles.index[i],
                'volume': candles.iloc[i]['volume'],
                'delta': candles.iloc[i]['delta']
            })

        # Swing Low
        before_min = lows[i-n:i].min()
        after_min = lows[i+1:i+n+1].min()
        if lows[i] < before_min and lows[i] < after_min:
            swing_lows.append({
                'index': i,
                'price': lows[i],
                'time': candles.index[i],
                'volume': candles.iloc[i]['volume'],
                'delta': candles.iloc[i]['delta']
            })

    return swing_highs, swing_lows


def collect_setups(candles: pd.DataFrame, swing_highs: list, swing_lows: list, n: int = 3):
    """
    Collect all false breakout setups and their outcomes.
    """
    setups = []

    for i in tqdm(range(50, len(candles) - 10), desc="Collecting setups"):
        row = candles.iloc[i]
        price = row['close']
        high = row['high']
        low = row['low']

        # Get confirmed swings up to this point
        max_swing_idx = i - n
        recent_highs = [s for s in swing_highs if s['index'] <= max_swing_idx]
        recent_lows = [s for s in swing_lows if s['index'] <= max_swing_idx]

        if not recent_highs or not recent_lows:
            continue

        # Get most recent swing high and low
        swing_high = recent_highs[-1]
        swing_low = recent_lows[-1]

        # Calculate range
        range_size = swing_high['price'] - swing_low['price']
        range_pct = range_size / swing_low['price']

        if range_pct < 0.005:  # Min 0.5% range
            continue

        mid_price = (swing_high['price'] + swing_low['price']) / 2

        # Calculate features from recent history
        hist = candles.iloc[i-20:i]
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

        # Check for breakout above swing high (potential SHORT setup)
        if high > swing_high['price']:
            breakout_distance = (high - swing_high['price']) / swing_high['price']

            # Weakness signals
            volume_weak = row['volume'] < swing_high['volume'] * 0.8
            delta_weak = row['delta'] < 0  # Long absorption
            wick_rejection = upper_wick_ratio > 0.3

            if volume_weak or delta_weak or wick_rejection:
                # Check if price returned below swing high (entry trigger)
                for j in range(i+1, min(i+10, len(candles))):
                    if candles.iloc[j]['close'] < swing_high['price']:
                        entry_price = candles.iloc[j]['close']
                        entry_idx = j

                        # Calculate outcome (did it reach TP at mid-range?)
                        tp_price = mid_price
                        sl_price = high * 1.001

                        outcome = None
                        exit_price = None
                        for k in range(entry_idx + 1, min(entry_idx + 50, len(candles))):
                            c = candles.iloc[k]
                            if c['high'] >= sl_price:
                                outcome = 0  # Loss
                                exit_price = sl_price
                                break
                            if c['low'] <= tp_price:
                                outcome = 1  # Win
                                exit_price = tp_price
                                break

                        if outcome is not None:
                            setups.append({
                                'type': 'SHORT',
                                'entry_idx': entry_idx,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'outcome': outcome,
                                # Features
                                'range_pct': range_pct,
                                'breakout_distance': breakout_distance,
                                'volume_ratio': row['volume'] / avg_volume,
                                'delta': row['delta'],
                                'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                                'upper_wick_ratio': upper_wick_ratio,
                                'lower_wick_ratio': lower_wick_ratio,
                                'volatility': volatility,
                                'volume_weak': int(volume_weak),
                                'delta_weak': int(delta_weak),
                                'wick_rejection': int(wick_rejection),
                                'swing_volume_ratio': row['volume'] / swing_high['volume'],
                                'price_vs_mid': (price - mid_price) / range_size,
                            })
                        break

        # Check for breakout below swing low (potential LONG setup)
        if low < swing_low['price']:
            breakout_distance = (swing_low['price'] - low) / swing_low['price']

            # Weakness signals
            volume_weak = row['volume'] < swing_low['volume'] * 0.8
            delta_weak = row['delta'] > 0  # Short absorption
            wick_rejection = lower_wick_ratio > 0.3

            if volume_weak or delta_weak or wick_rejection:
                # Check if price returned above swing low (entry trigger)
                for j in range(i+1, min(i+10, len(candles))):
                    if candles.iloc[j]['close'] > swing_low['price']:
                        entry_price = candles.iloc[j]['close']
                        entry_idx = j

                        # Calculate outcome
                        tp_price = mid_price
                        sl_price = low * 0.999

                        outcome = None
                        exit_price = None
                        for k in range(entry_idx + 1, min(entry_idx + 50, len(candles))):
                            c = candles.iloc[k]
                            if c['low'] <= sl_price:
                                outcome = 0  # Loss
                                exit_price = sl_price
                                break
                            if c['high'] >= tp_price:
                                outcome = 1  # Win
                                exit_price = tp_price
                                break

                        if outcome is not None:
                            setups.append({
                                'type': 'LONG',
                                'entry_idx': entry_idx,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'outcome': outcome,
                                # Features
                                'range_pct': range_pct,
                                'breakout_distance': breakout_distance,
                                'volume_ratio': row['volume'] / avg_volume,
                                'delta': row['delta'],
                                'delta_ratio': row['delta'] / (abs(avg_delta) + 1),
                                'upper_wick_ratio': upper_wick_ratio,
                                'lower_wick_ratio': lower_wick_ratio,
                                'volatility': volatility,
                                'volume_weak': int(volume_weak),
                                'delta_weak': int(delta_weak),
                                'wick_rejection': int(wick_rejection),
                                'swing_volume_ratio': row['volume'] / swing_low['volume'],
                                'price_vs_mid': (price - mid_price) / range_size,
                            })
                        break

    return setups


def train_ml_filter(setups: list):
    """Train ML model to predict setup success."""

    df = pd.DataFrame(setups)
    print(f"\nTotal setups collected: {len(df)}")
    print(f"  LONG:  {len(df[df['type']=='LONG'])}")
    print(f"  SHORT: {len(df[df['type']=='SHORT'])}")
    print(f"  Win rate: {df['outcome'].mean()*100:.1f}%")

    # Features
    feature_cols = [
        'range_pct', 'breakout_distance', 'volume_ratio', 'delta_ratio',
        'upper_wick_ratio', 'lower_wick_ratio', 'volatility',
        'volume_weak', 'delta_weak', 'wick_rejection',
        'swing_volume_ratio', 'price_vs_mid'
    ]

    X = df[feature_cols]
    y = df['outcome']

    # Time-based split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train win rate: {y_train.mean()*100:.1f}%")
    print(f"Test win rate: {y_test.mean()*100:.1f}%")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "="*60)
    print("  ML Filter Results (Test Set)")
    print("="*60)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(f"  Precision: {precision_score(y_test, y_pred)*100:.1f}%")
    print(f"  Recall: {recall_score(y_test, y_pred)*100:.1f}%")

    # Test with confidence threshold
    print("\n  Performance by Confidence Threshold:")
    print("-"*60)
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        filtered = y_prob >= threshold
        if filtered.sum() > 0:
            filtered_win_rate = y_test[filtered].mean()
            print(f"  >{threshold:.0%}: {filtered.sum():4} setups, {filtered_win_rate*100:.1f}% win rate")

    # Feature importance
    print("\n  Feature Importance:")
    print("-"*60)
    importance = model.feature_importances_
    for i, col in enumerate(feature_cols):
        print(f"  {col:25} {importance[i]:.4f}")

    # Save model with timeframe suffix
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Get timeframe from caller (default to 1h for backward compatibility)
    tf = getattr(train_ml_filter, '_timeframe', '1h')
    suffix = f"_{tf}" if tf != "1h" else ""

    joblib.dump(model, os.path.join(MODEL_DIR, f"false_breakout_filter{suffix}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"false_breakout_scaler{suffix}.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"false_breakout_features{suffix}.joblib"))

    print(f"\n  Model saved to {MODEL_DIR} (suffix: {suffix or 'none'})")

    return model, scaler, feature_cols


def main(timeframe: str = "1h"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     ML-Enhanced False Breakout Training ({timeframe})             ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles ({timeframe})")

    # Find swings
    print("\nFinding swing points...")
    swing_highs, swing_lows = find_swing_points(candles, n=3)
    print(f"  Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")

    # Collect setups
    print("\nCollecting false breakout setups...")
    setups = collect_setups(candles, swing_highs, swing_lows, n=3)

    if len(setups) < 100:
        print("Not enough setups collected")
        return

    # Train model (pass timeframe for model naming)
    train_ml_filter._timeframe = timeframe
    train_ml_filter(setups)


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(timeframe)
