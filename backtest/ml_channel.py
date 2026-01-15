#!/usr/bin/env python3
"""
Horizontal Channel (Range) Trading Strategy with ML Filter

Two Entry Types:
1. BOUNCE: Trade at support/resistance
   - Long at support, Short at resistance
   - TP at opposite side of range

2. FAKEOUT: Trade failed breakouts
   - Price breaks out but returns into range
   - Very tight SL (fakeout extreme)
   - Excellent R:R (target opposite side)

Benefits:
- High win rate (range bounces are predictable)
- Many trading opportunities (multiple touches + fakeouts)
- Clear R:R based on range width
- Fakeouts have very tight stops
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@dataclass
class SwingLevel:
    """A swing high or low level."""
    idx: int
    price: float
    type: str  # 'high' or 'low'
    touched: bool = False


@dataclass
class Channel:
    """Horizontal price channel from swing highs/lows."""
    resistance_idx: int
    resistance: float
    support_idx: int
    support: float
    width_pct: float


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[SwingLevel], List[SwingLevel]]:
    """
    Find swing highs and lows.

    Swing high: Price was rising, then N candles fail to make new high
    Swing low: Price was falling, then N candles fail to make new low

    Args:
        confirm_candles: Number of candles needed to confirm swing (2-3)
    """
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    # Track potential swings
    potential_high_idx = 0
    potential_high_price = highs[0]
    candles_since_high = 0

    potential_low_idx = 0
    potential_low_price = lows[0]
    candles_since_low = 0

    for i in range(1, len(candles)):
        # Check for new high
        if highs[i] > potential_high_price:
            potential_high_idx = i
            potential_high_price = highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            # Confirmed swing high
            if candles_since_high == confirm_candles:
                swing_highs.append(SwingLevel(
                    idx=potential_high_idx,
                    price=potential_high_price,
                    type='high'
                ))

        # Check for new low
        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            # Confirmed swing low
            if candles_since_low == confirm_candles:
                swing_lows.append(SwingLevel(
                    idx=potential_low_idx,
                    price=potential_low_price,
                    type='low'
                ))

        # Reset tracking when swing is confirmed to find next swing
        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def find_active_channels(candles: pd.DataFrame,
                         lookback: int = 100,
                         max_width_pct: float = 0.04,
                         min_width_pct: float = 0.005,
                         touch_threshold: float = 0.003) -> dict:
    """
    Find active channels based on swing highs and lows.

    Method:
    1. Find swing high → draw horizontal resistance line
    2. Find swing low → draw horizontal support line
    3. Track first touch on each level for entry
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values

    # Find all swing points
    swing_highs, swing_lows = find_swing_points(candles, confirm_candles=3)

    active_channels = {}

    # Track active levels waiting for touch
    active_resistances = []  # List of SwingLevel
    active_supports = []  # List of SwingLevel

    for i in range(len(candles)):
        # Add new swing levels as they form (need n candles to confirm)
        for sh in swing_highs:
            if sh.idx + 5 == i:  # Swing confirmed
                active_resistances.append(SwingLevel(idx=sh.idx, price=sh.price, type='high', touched=False))

        for sl in swing_lows:
            if sl.idx + 5 == i:  # Swing confirmed
                active_supports.append(SwingLevel(idx=sl.idx, price=sl.price, type='low', touched=False))

        # Clean old levels (invalidated by price breaking through)
        active_resistances = [r for r in active_resistances
                             if closes[i] < r.price * 1.01 and i - r.idx < lookback]
        active_supports = [s for s in active_supports
                         if closes[i] > s.price * 0.99 and i - s.idx < lookback]

        # Find best channel (most recent untouched resistance + support pair)
        best_channel = None
        best_score = -1

        for res in active_resistances:
            for sup in active_supports:
                if res.price <= sup.price:
                    continue

                width_pct = (res.price - sup.price) / sup.price

                if width_pct > max_width_pct or width_pct < min_width_pct:
                    continue

                # Score by recency
                score = -max(i - res.idx, i - sup.idx)

                if score > best_score:
                    best_score = score
                    best_channel = Channel(
                        resistance_idx=res.idx,
                        resistance=res.price,
                        support_idx=sup.idx,
                        support=sup.price,
                        width_pct=width_pct
                    )

        if best_channel:
            # Check if price is touching support or resistance
            at_support = lows[i] <= best_channel.support * (1 + touch_threshold)
            at_resistance = highs[i] >= best_channel.resistance * (1 - touch_threshold)

            if at_support or at_resistance:
                active_channels[i] = best_channel

    return active_channels


def collect_channel_setups(candles: pd.DataFrame,
                           sl_buffer_pct: float = 0.001,
                           touch_threshold: float = 0.003,
                           quiet: bool = False) -> List[dict]:
    """
    Collect channel trading setups with outcomes.

    Two types:
    1. BOUNCE: First touch at support/resistance, TP at opposite side
    2. FAKEOUT: Price breaks out but returns, very tight SL at fakeout extreme
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # Find swing points
    swing_highs, swing_lows = find_swing_points(candles, confirm_candles=3)

    setups = []

    # Track active levels (untouched)
    active_resistances = []  # [(idx, price, touched)]
    active_supports = []  # [(idx, price, touched)]

    # Track pending fakeouts
    pending_fakeouts = []

    # Track used channels to avoid duplicate entries
    used_pairs = set()

    iterator = range(len(candles))
    if not quiet:
        iterator = tqdm(iterator, desc="Collecting channel setups")

    for i in iterator:
        # Add new swing levels as they confirm
        for sh in swing_highs:
            if sh.idx + 5 == i:
                active_resistances.append({'idx': sh.idx, 'price': sh.price, 'touched': False})

        for sl in swing_lows:
            if sl.idx + 5 == i:
                active_supports.append({'idx': sl.idx, 'price': sl.price, 'touched': False})

        # Clean invalidated levels (price broke through)
        active_resistances = [r for r in active_resistances
                             if closes[i] < r['price'] * 1.015 and i - r['idx'] < 200]
        active_supports = [s for s in active_supports
                         if closes[i] > s['price'] * 0.985 and i - s['idx'] < 200]

        # Historical features
        hist_start = max(0, i - 20)
        hist = candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check for FAKEOUT entries
        for fakeout in pending_fakeouts[:]:
            candles_since = i - fakeout['break_idx']

            if candles_since > 10:
                pending_fakeouts.remove(fakeout)
                continue

            if fakeout['type'] == 'bear_fakeout':
                if closes[i] > fakeout['support'] and lows[i] > fakeout['extreme'] * 0.998:
                    entry_price = closes[i]
                    sl_price = fakeout['extreme'] * (1 - sl_buffer_pct)
                    tp_price = fakeout['resistance'] * 0.998

                    risk = entry_price - sl_price
                    reward = tp_price - entry_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk
                        width_pct = (fakeout['resistance'] - fakeout['support']) / fakeout['support']

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if lows[j] <= sl_price:
                                outcome = 0
                                break
                            if highs[j] >= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i,
                            'type': 'LONG',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'rr_ratio': rr_ratio,
                            'channel_width': width_pct,
                            'level_age': i - fakeout['support_idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': abs(fakeout['extreme'] - fakeout['support']) / fakeout['support'],
                            'candles_to_reclaim': candles_since,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })

                    pending_fakeouts.remove(fakeout)
                    continue

            elif fakeout['type'] == 'bull_fakeout':
                if closes[i] < fakeout['resistance'] and highs[i] < fakeout['extreme'] * 1.002:
                    entry_price = closes[i]
                    sl_price = fakeout['extreme'] * (1 + sl_buffer_pct)
                    tp_price = fakeout['support'] * 1.002

                    risk = sl_price - entry_price
                    reward = entry_price - tp_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk
                        width_pct = (fakeout['resistance'] - fakeout['support']) / fakeout['support']

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if highs[j] >= sl_price:
                                outcome = 0
                                break
                            if lows[j] <= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i,
                            'type': 'SHORT',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'rr_ratio': rr_ratio,
                            'channel_width': width_pct,
                            'level_age': i - fakeout['resistance_idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': abs(fakeout['extreme'] - fakeout['resistance']) / fakeout['resistance'],
                            'candles_to_reclaim': candles_since,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })

                    pending_fakeouts.remove(fakeout)
                    continue

        # Find best channel for BOUNCE setup
        for res in active_resistances:
            for sup in active_supports:
                if res['price'] <= sup['price']:
                    continue

                pair_key = (res['idx'], sup['idx'])
                if pair_key in used_pairs:
                    continue

                width_pct = (res['price'] - sup['price']) / sup['price']
                if width_pct > 0.04 or width_pct < 0.005:
                    continue

                # Check for support touch (LONG)
                if lows[i] <= sup['price'] * (1 + touch_threshold) and closes[i] > sup['price']:
                    entry_price = closes[i]
                    sl_price = sup['price'] * (1 - sl_buffer_pct)
                    tp_price = res['price'] * 0.998

                    risk = entry_price - sl_price
                    reward = tp_price - entry_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if lows[j] <= sl_price:
                                outcome = 0
                                break
                            if highs[j] >= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i,
                            'type': 'LONG',
                            'setup_type': 'BOUNCE',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'rr_ratio': rr_ratio,
                            'channel_width': width_pct,
                            'level_age': i - sup['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0,
                            'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })

                        used_pairs.add(pair_key)
                        sup['touched'] = True
                        break

                # Check for resistance touch (SHORT)
                elif highs[i] >= res['price'] * (1 - touch_threshold) and closes[i] < res['price']:
                    entry_price = closes[i]
                    sl_price = res['price'] * (1 + sl_buffer_pct)
                    tp_price = sup['price'] * 1.002

                    risk = sl_price - entry_price
                    reward = entry_price - tp_price

                    if risk > 0 and reward > 0:
                        rr_ratio = reward / risk

                        outcome = 0
                        for j in range(i + 1, min(i + 100, len(candles))):
                            if highs[j] >= sl_price:
                                outcome = 0
                                break
                            if lows[j] <= tp_price:
                                outcome = 1
                                break

                        setups.append({
                            'idx': i,
                            'type': 'SHORT',
                            'setup_type': 'BOUNCE',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp': tp_price,
                            'rr_ratio': rr_ratio,
                            'channel_width': width_pct,
                            'level_age': i - res['idx'],
                            'volume_at_entry': volumes[i],
                            'volume_ratio': volumes[i] / avg_volume if avg_volume > 0 else 1,
                            'delta_at_entry': deltas[i],
                            'delta_ratio': deltas[i] / (abs(avg_delta) + 1),
                            'cvd_recent': cvd_recent,
                            'fakeout_depth': 0,
                            'candles_to_reclaim': 0,
                            'body_bullish': 1 if closes[i] > opens[i] else 0,
                            'outcome': outcome
                        })

                        used_pairs.add(pair_key)
                        res['touched'] = True
                        break

                # Check for breakout (potential fakeout)
                elif closes[i] < sup['price'] * 0.998:
                    pending_fakeouts.append({
                        'type': 'bear_fakeout',
                        'break_idx': i,
                        'support': sup['price'],
                        'support_idx': sup['idx'],
                        'resistance': res['price'],
                        'resistance_idx': res['idx'],
                        'extreme': lows[i]
                    })
                    used_pairs.add(pair_key)
                    break

                elif closes[i] > res['price'] * 1.002:
                    pending_fakeouts.append({
                        'type': 'bull_fakeout',
                        'break_idx': i,
                        'support': sup['price'],
                        'support_idx': sup['idx'],
                        'resistance': res['price'],
                        'resistance_idx': res['idx'],
                        'extreme': highs[i]
                    })
                    used_pairs.add(pair_key)
                    break

    return setups


def train_channel_model(candles: pd.DataFrame, timeframe: str = "15m"):
    """Train ML model to filter channel setups."""

    print("Collecting channel setups...")
    setups = collect_channel_setups(candles)

    if len(setups) < 100:
        print(f"Not enough setups: {len(setups)}")
        return None, None, None

    df = pd.DataFrame(setups)

    print(f"\nTotal setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # By setup type
    bounce_df = df[df['setup_type'] == 'BOUNCE']
    fakeout_df = df[df['setup_type'] == 'FAKEOUT']
    print(f"\n  BOUNCE setups:  {len(bounce_df)} (WR: {bounce_df['outcome'].mean()*100:.1f}%)")
    print(f"  FAKEOUT setups: {len(fakeout_df)} (WR: {fakeout_df['outcome'].mean()*100:.1f}%)")
    print(f"\n  Overall win rate: {df['outcome'].mean()*100:.1f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    # Features for ML
    feature_cols = [
        'channel_width',
        'level_age',
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

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost
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

    # Feature importance
    print("\nFeature Importance:")
    importance = dict(zip(feature_cols, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    suffix = f"_{timeframe}" if timeframe != "1h" else ""
    joblib.dump(model, os.path.join(MODEL_DIR, f"channel_filter{suffix}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"channel_scaler{suffix}.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"channel_features{suffix}.joblib"))

    print(f"\nModel saved to {MODEL_DIR}")

    # Test predictions
    probs = model.predict_proba(X_scaled)[:, 1]

    print("\nFiltered Results by Confidence:")
    print("="*70)
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        mask = probs >= thresh
        if mask.sum() > 0:
            filtered_wr = df.loc[mask, 'outcome'].mean()
            avg_rr = df.loc[mask, 'rr_ratio'].mean()
            ev = filtered_wr * avg_rr - (1 - filtered_wr)
            print(f"  {thresh:.0%}: {mask.sum():4} setups, WR {filtered_wr*100:5.1f}%, R:R {avg_rr:.2f}, EV {ev:+.2f}")

    return model, scaler, feature_cols


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     Horizontal Channel Strategy - ML Training             ║
╚═══════════════════════════════════════════════════════════╝
""")

    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"

    print(f"Loading {timeframe} data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles")

    train_channel_model(candles, timeframe)


if __name__ == "__main__":
    main()
