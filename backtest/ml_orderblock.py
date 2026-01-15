#!/usr/bin/env python3
"""
Order Block Strategy with ML Filter

1. Find Order Blocks (Supply/Demand Zones)
   - Demand Zone: Last bearish candle before significant bullish move
   - Supply Zone: Last bullish candle before significant bearish move

2. Setup:
   - Entry when price returns to fresh (untested) order block
   - SL: Beyond the order block
   - TP: Next order block in opposite direction

3. ML Filter:
   - Delta reaction, volume, CVD divergence
   - Filter for high probability entries
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@dataclass
class OrderBlock:
    """Represents an order block zone."""
    idx: int              # Candle index where OB formed
    type: str             # 'demand' or 'supply'
    top: float            # Upper boundary
    bottom: float         # Lower boundary
    volume: float         # Volume at OB candle
    delta: float          # Delta at OB candle
    move_size: float      # Size of the move that created OB (%)
    tested: bool = False  # Has price returned to this zone?
    test_idx: Optional[int] = None  # When was it first tested?


def find_order_blocks(candles: pd.DataFrame,
                      min_move_pct: float = 0.02,  # 2% minimum move (강화)
                      lookback: int = 5,
                      volume_mult: float = 1.0) -> List[OrderBlock]:  # 볼륨 배수 조건
    """
    Find all order blocks in the data (STRICT version).

    Order Block = the last opposite-colored candle before a significant move

    강화된 조건:
    1. 최소 2% 이상 움직임 (기존 1%)
    2. OB 캔들 볼륨이 평균 이상
    3. OB 캔들이 의미있는 크기 (작은 도지 제외)
    """
    order_blocks = []

    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Pre-calculate rolling average volume
    avg_volumes = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i in range(lookback, len(candles) - lookback):
        # Check for significant bullish move (Demand Zone formation)
        future_high = highs[i+1:i+lookback+1].max()
        move_up = (future_high - closes[i]) / closes[i]

        if move_up >= min_move_pct:
            # Find the last bearish candle before this move
            for j in range(i, max(i-lookback, 0), -1):
                is_bearish = closes[j] < opens[j]
                body_size = abs(closes[j] - opens[j]) / closes[j]
                has_volume = volumes[j] >= avg_volumes[j] * volume_mult
                is_significant = body_size >= 0.002  # 최소 0.2% 바디 크기

                if is_bearish and has_volume and is_significant:
                    order_blocks.append(OrderBlock(
                        idx=j,
                        type='demand',
                        top=max(opens[j], closes[j]),
                        bottom=min(opens[j], closes[j]),
                        volume=volumes[j],
                        delta=deltas[j],
                        move_size=move_up
                    ))
                    break

        # Check for significant bearish move (Supply Zone formation)
        future_low = lows[i+1:i+lookback+1].min()
        move_down = (closes[i] - future_low) / closes[i]

        if move_down >= min_move_pct:
            # Find the last bullish candle before this move
            for j in range(i, max(i-lookback, 0), -1):
                is_bullish = closes[j] > opens[j]
                body_size = abs(closes[j] - opens[j]) / closes[j]
                has_volume = volumes[j] >= avg_volumes[j] * volume_mult
                is_significant = body_size >= 0.002

                if is_bullish and has_volume and is_significant:
                    order_blocks.append(OrderBlock(
                        idx=j,
                        type='supply',
                        top=max(opens[j], closes[j]),
                        bottom=min(opens[j], closes[j]),
                        volume=volumes[j],
                        delta=deltas[j],
                        move_size=move_down
                    ))
                    break

    # Remove duplicates (same idx)
    seen_idx = set()
    unique_blocks = []
    for ob in order_blocks:
        if ob.idx not in seen_idx:
            seen_idx.add(ob.idx)
            unique_blocks.append(ob)

    # Sort by index
    unique_blocks.sort(key=lambda x: x.idx)

    return unique_blocks


def collect_ob_setups(candles: pd.DataFrame, order_blocks: List[OrderBlock],
                      sl_buffer_pct: float = 0.002,  # 0.2% SL buffer
                      max_hold_candles: int = 50,
                      fixed_rr: float = 2.0,  # Fixed R:R ratio
                      quiet: bool = False) -> List[dict]:
    """
    Collect all order block setups and their outcomes.

    Setup triggers when price enters a fresh order block zone.
    """
    setups = []

    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Track which OBs are still fresh
    fresh_demand = []  # List of demand OBs not yet tested
    fresh_supply = []  # List of supply OBs not yet tested

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Collecting OB setups")

    for i in iterator:
        row_high = highs[i]
        row_low = lows[i]
        row_close = closes[i]
        row_volume = volumes[i]
        row_delta = deltas[i]

        # Add new order blocks that become visible
        for ob in order_blocks:
            # OB is "visible" after the move completes (5 candles after formation)
            if ob.idx + 5 == i:
                if ob.type == 'demand':
                    fresh_demand.append(ob)
                else:
                    fresh_supply.append(ob)

        # Check for demand zone entries (LONG)
        for ob in fresh_demand[:]:  # Copy list to allow modification
            # Price enters the zone
            if row_low <= ob.top and row_close > ob.bottom:
                # Calculate setup with fixed R:R
                entry_price = row_close
                sl_price = ob.bottom * (1 - sl_buffer_pct)

                # Fixed R:R approach - TP based on risk distance
                risk = entry_price - sl_price
                if risk <= 0:
                    fresh_demand.remove(ob)
                    continue

                tp_price = entry_price + (risk * fixed_rr)  # e.g., 1:2 R:R
                rr_ratio = fixed_rr

                # Calculate features
                hist = candles.iloc[max(0, i-20):i]
                avg_volume = hist['volume'].mean() if len(hist) > 0 else row_volume
                avg_delta = hist['delta'].mean() if len(hist) > 0 else 0

                # CVD (Cumulative Volume Delta) - simple version
                cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

                # Check outcome
                outcome = None
                exit_price = None
                for k in range(i + 1, min(i + max_hold_candles, len(candles))):
                    if lows[k] <= sl_price:
                        outcome = 0  # Loss
                        exit_price = sl_price
                        break
                    if highs[k] >= tp_price:
                        outcome = 1  # Win
                        exit_price = tp_price
                        break

                if outcome is not None:
                    setups.append({
                        'type': 'LONG',
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'exit_price': exit_price,
                        'rr_ratio': rr_ratio,
                        'outcome': outcome,
                        # Features
                        'ob_move_size': ob.move_size,
                        'ob_volume': ob.volume,
                        'ob_delta': ob.delta,
                        'zone_width_pct': (ob.top - ob.bottom) / ob.bottom,
                        'candles_since_ob': i - ob.idx,
                        'entry_volume_ratio': row_volume / avg_volume,
                        'entry_delta': row_delta,
                        'entry_delta_ratio': row_delta / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'delta_at_zone': row_delta,  # Positive = buyers stepping in
                        'wick_into_zone': (ob.top - row_low) / (ob.top - ob.bottom) if ob.top > ob.bottom else 0,
                    })

                # Mark as tested
                fresh_demand.remove(ob)

        # Check for supply zone entries (SHORT)
        for ob in fresh_supply[:]:
            # Price enters the zone
            if row_high >= ob.bottom and row_close < ob.top:
                # Calculate setup with fixed R:R
                entry_price = row_close
                sl_price = ob.top * (1 + sl_buffer_pct)

                # Fixed R:R approach
                risk = sl_price - entry_price
                if risk <= 0:
                    fresh_supply.remove(ob)
                    continue

                tp_price = entry_price - (risk * fixed_rr)  # e.g., 1:2 R:R
                rr_ratio = fixed_rr

                # Calculate features
                hist = candles.iloc[max(0, i-20):i]
                avg_volume = hist['volume'].mean() if len(hist) > 0 else row_volume
                avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
                cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

                # Check outcome
                outcome = None
                exit_price = None
                for k in range(i + 1, min(i + max_hold_candles, len(candles))):
                    if highs[k] >= sl_price:
                        outcome = 0  # Loss
                        exit_price = sl_price
                        break
                    if lows[k] <= tp_price:
                        outcome = 1  # Win
                        exit_price = tp_price
                        break

                if outcome is not None:
                    setups.append({
                        'type': 'SHORT',
                        'entry_idx': i,
                        'entry_price': entry_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'exit_price': exit_price,
                        'rr_ratio': rr_ratio,
                        'outcome': outcome,
                        # Features
                        'ob_move_size': ob.move_size,
                        'ob_volume': ob.volume,
                        'ob_delta': ob.delta,
                        'zone_width_pct': (ob.top - ob.bottom) / ob.bottom,
                        'candles_since_ob': i - ob.idx,
                        'entry_volume_ratio': row_volume / avg_volume,
                        'entry_delta': row_delta,
                        'entry_delta_ratio': row_delta / (abs(avg_delta) + 1),
                        'cvd_recent': cvd_recent,
                        'delta_at_zone': row_delta,  # Negative = sellers stepping in
                        'wick_into_zone': (row_high - ob.bottom) / (ob.top - ob.bottom) if ob.top > ob.bottom else 0,
                    })

                fresh_supply.remove(ob)

        # Clean up old zones (too far from current price)
        current_price = row_close
        fresh_demand = [ob for ob in fresh_demand if ob.top > current_price * 0.95]
        fresh_supply = [ob for ob in fresh_supply if ob.bottom < current_price * 1.05]

    return setups


def train_ob_ml_filter(setups: list, timeframe: str = "1h"):
    """Train ML model to predict setup success."""

    df = pd.DataFrame(setups)
    print(f"\nTotal setups collected: {len(df)}")
    print(f"  LONG:  {len(df[df['type']=='LONG'])}")
    print(f"  SHORT: {len(df[df['type']=='SHORT'])}")
    print(f"  Win rate: {df['outcome'].mean()*100:.1f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    # Features for ML
    feature_cols = [
        'ob_move_size', 'zone_width_pct', 'candles_since_ob',
        'entry_volume_ratio', 'entry_delta_ratio',
        'cvd_recent', 'delta_at_zone', 'wick_into_zone', 'rr_ratio'
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
    print("  Order Block ML Filter Results (Test Set)")
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
            avg_rr = df.iloc[split_idx:][filtered]['rr_ratio'].mean()
            expected_value = filtered_win_rate * avg_rr - (1 - filtered_win_rate) * 1
            print(f"  >{threshold:.0%}: {filtered.sum():4} setups, {filtered_win_rate*100:.1f}% win, R:R {avg_rr:.2f}, EV {expected_value:.2f}")

    # Feature importance
    print("\n  Feature Importance:")
    print("-"*60)
    importance = model.feature_importances_
    for i, col in enumerate(feature_cols):
        print(f"  {col:25} {importance[i]:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    suffix = f"_{timeframe}" if timeframe != "1h" else ""

    joblib.dump(model, os.path.join(MODEL_DIR, f"orderblock_filter{suffix}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"orderblock_scaler{suffix}.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"orderblock_features{suffix}.joblib"))

    print(f"\n  Model saved to {MODEL_DIR}")

    return model, scaler, feature_cols


def main(timeframe: str = "1h"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     Order Block Strategy - ML Training ({timeframe})              ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles ({timeframe})")

    # Find order blocks with STRICT criteria
    print("\nFinding order blocks (STRICT mode)...")
    # Stricter min_move based on timeframe
    min_move = 0.02 if timeframe in ['1h', '4h'] else 0.01  # 2% for higher TF, 1% for lower
    order_blocks = find_order_blocks(candles, min_move_pct=min_move, volume_mult=1.0)

    demand_count = len([ob for ob in order_blocks if ob.type == 'demand'])
    supply_count = len([ob for ob in order_blocks if ob.type == 'supply'])
    print(f"  Found {len(order_blocks)} order blocks")
    print(f"    Demand zones: {demand_count}")
    print(f"    Supply zones: {supply_count}")

    # Test different R:R ratios
    print("\n" + "="*70)
    print(f"{'R:R':^8} | {'Setups':^8} | {'Win Rate':^10} | {'EV':^8} | {'LONG Win':^10} | {'SHORT Win':^10}")
    print("="*70)

    best_rr = 2.0
    best_ev = -999

    for rr in [1.5, 2.0, 2.5, 3.0]:
        setups = collect_ob_setups(candles, order_blocks, fixed_rr=rr, quiet=True)
        if len(setups) < 20:
            continue

        df = pd.DataFrame(setups)
        win_rate = df['outcome'].mean()
        ev = win_rate * rr - (1 - win_rate) * 1

        long_df = df[df['type'] == 'LONG']
        short_df = df[df['type'] == 'SHORT']
        long_wr = long_df['outcome'].mean() if len(long_df) > 0 else 0
        short_wr = short_df['outcome'].mean() if len(short_df) > 0 else 0

        print(f"{rr:^8.1f} | {len(setups):^8} | {win_rate*100:^10.1f}% | {ev:^8.2f} | {long_wr*100:^10.1f}% | {short_wr*100:^10.1f}%")

        if ev > best_ev:
            best_ev = ev
            best_rr = rr

    print("="*70)
    print(f"\nBest R:R: 1:{best_rr:.1f} (EV: {best_ev:.2f})")

    # Collect setups with best R:R
    print(f"\nCollecting order block setups (R:R 1:{best_rr:.1f})...")
    setups = collect_ob_setups(candles, order_blocks, fixed_rr=best_rr)

    if len(setups) < 50:
        print("Not enough setups collected. Try adjusting parameters.")
        return

    # Train model
    train_ob_ml_filter(setups, timeframe)


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(timeframe)
