#!/usr/bin/env python3
"""
Break of Structure (BOS) Strategy with ML Filter

구조:
1. Swing High/Low 탐지
2. BOS 발생 (가격이 스윙 돌파)
3. Retest 대기 (돌파 레벨로 복귀)
4. Entry on Retest
5. SL: Retest 저점/고점 너머
6. TP: 다음 구조 레벨 or 이전 스윙 거리만큼

R:R: 구조 기반으로 자연스럽게 결정됨
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
class SwingPoint:
    idx: int
    price: float
    type: str  # 'high' or 'low'
    volume: float
    delta: float


@dataclass
class BOSEvent:
    idx: int              # BOS 발생 캔들
    type: str             # 'bullish' or 'bearish'
    broken_level: float   # 돌파된 레벨
    swing_idx: int        # 돌파된 스윙 인덱스
    prev_swing: float     # 이전 스윙 (TP 계산용)


def find_swing_points(candles: pd.DataFrame, n: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find swing highs and lows."""
    swing_highs = []
    swing_lows = []

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    for i in range(n, len(candles) - n):
        # Swing High: 좌우 n개보다 높음
        if highs[i] == max(highs[i-n:i+n+1]):
            swing_highs.append(SwingPoint(
                idx=i, price=highs[i], type='high',
                volume=volumes[i], delta=deltas[i]
            ))

        # Swing Low: 좌우 n개보다 낮음
        if lows[i] == min(lows[i-n:i+n+1]):
            swing_lows.append(SwingPoint(
                idx=i, price=lows[i], type='low',
                volume=volumes[i], delta=deltas[i]
            ))

    return swing_highs, swing_lows


def detect_bos_events(candles: pd.DataFrame, swing_highs: List[SwingPoint],
                      swing_lows: List[SwingPoint], confirm_candles: int = 2) -> List[BOSEvent]:
    """Detect Break of Structure events."""
    bos_events = []

    closes = candles['close'].values
    highs = candles['high'].values
    lows = candles['low'].values

    # Track recent swing points for BOS detection
    for i in range(50, len(candles)):
        # Get confirmed swings (at least confirm_candles bars ago)
        recent_highs = [s for s in swing_highs if s.idx <= i - confirm_candles and s.idx > i - 100]
        recent_lows = [s for s in swing_lows if s.idx <= i - confirm_candles and s.idx > i - 100]

        if not recent_highs or not recent_lows:
            continue

        # Most recent swing high and low
        last_high = max(recent_highs, key=lambda x: x.idx)
        last_low = max(recent_lows, key=lambda x: x.idx)

        # Previous swing for TP calculation
        prev_highs = [s for s in recent_highs if s.idx < last_high.idx]
        prev_lows = [s for s in recent_lows if s.idx < last_low.idx]

        # Bullish BOS: Close above recent swing high
        if closes[i] > last_high.price and closes[i-1] <= last_high.price:
            prev_low_price = prev_lows[-1].price if prev_lows else last_low.price
            bos_events.append(BOSEvent(
                idx=i,
                type='bullish',
                broken_level=last_high.price,
                swing_idx=last_high.idx,
                prev_swing=prev_low_price
            ))

        # Bearish BOS: Close below recent swing low
        if closes[i] < last_low.price and closes[i-1] >= last_low.price:
            prev_high_price = prev_highs[-1].price if prev_highs else last_high.price
            bos_events.append(BOSEvent(
                idx=i,
                type='bearish',
                broken_level=last_low.price,
                swing_idx=last_low.idx,
                prev_swing=prev_high_price
            ))

    return bos_events


def collect_bos_setups(candles: pd.DataFrame, bos_events: List[BOSEvent],
                       retest_threshold: float = 0.003,  # 0.3% 이내 복귀
                       max_wait_candles: int = 20,
                       sl_buffer: float = 0.002,
                       min_rr: float = 1.5,
                       quiet: bool = False) -> List[dict]:
    """Collect BOS retest setups and their outcomes."""
    setups = []

    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    iterator = bos_events
    if not quiet:
        iterator = tqdm(bos_events, desc="Collecting BOS setups")

    for bos in iterator:
        # Wait for retest
        retest_found = False
        entry_idx = None
        entry_price = None
        retest_extreme = None

        for i in range(bos.idx + 1, min(bos.idx + max_wait_candles, len(candles) - 50)):
            if bos.type == 'bullish':
                # Bullish: Wait for price to come back down to broken level
                distance_to_level = (lows[i] - bos.broken_level) / bos.broken_level

                if distance_to_level <= retest_threshold and closes[i] > bos.broken_level:
                    # Retest found! Price touched level and bounced
                    retest_found = True
                    entry_idx = i
                    entry_price = closes[i]
                    retest_extreme = lows[i]
                    break

            else:  # bearish
                # Bearish: Wait for price to come back up to broken level
                distance_to_level = (bos.broken_level - highs[i]) / bos.broken_level

                if distance_to_level <= retest_threshold and closes[i] < bos.broken_level:
                    retest_found = True
                    entry_idx = i
                    entry_price = closes[i]
                    retest_extreme = highs[i]
                    break

        if not retest_found:
            continue

        # Calculate SL and TP
        if bos.type == 'bullish':
            sl_price = retest_extreme * (1 - sl_buffer)
            risk = entry_price - sl_price

            # TP: 이전 스윙 거리만큼 위로, 또는 최소 1:2
            swing_distance = bos.broken_level - bos.prev_swing
            tp_price = entry_price + max(swing_distance, risk * 2)

        else:  # bearish
            sl_price = retest_extreme * (1 + sl_buffer)
            risk = sl_price - entry_price

            swing_distance = bos.prev_swing - bos.broken_level
            tp_price = entry_price - max(swing_distance, risk * 2)

        if risk <= 0:
            continue

        reward = abs(tp_price - entry_price)
        rr_ratio = reward / risk

        if rr_ratio < min_rr:
            continue

        # Calculate features
        hist = candles.iloc[max(0, entry_idx-20):entry_idx]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else volumes[entry_idx]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        # Check outcome
        outcome = None
        exit_price = None

        for k in range(entry_idx + 1, min(entry_idx + 100, len(candles))):
            if bos.type == 'bullish':
                if lows[k] <= sl_price:
                    outcome = 0
                    exit_price = sl_price
                    break
                if highs[k] >= tp_price:
                    outcome = 1
                    exit_price = tp_price
                    break
            else:
                if highs[k] >= sl_price:
                    outcome = 0
                    exit_price = sl_price
                    break
                if lows[k] <= tp_price:
                    outcome = 1
                    exit_price = tp_price
                    break

        if outcome is None:
            continue

        setups.append({
            'type': 'LONG' if bos.type == 'bullish' else 'SHORT',
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_price': exit_price,
            'rr_ratio': rr_ratio,
            'outcome': outcome,
            # Features
            'bos_strength': abs(closes[bos.idx] - bos.broken_level) / bos.broken_level,
            'retest_depth': abs(retest_extreme - bos.broken_level) / bos.broken_level,
            'candles_to_retest': entry_idx - bos.idx,
            'volume_ratio': volumes[entry_idx] / avg_volume if avg_volume > 0 else 1,
            'delta_at_retest': deltas[entry_idx],
            'delta_ratio': deltas[entry_idx] / (abs(avg_delta) + 1),
            'cvd_recent': cvd_recent,
            'swing_distance_pct': abs(bos.broken_level - bos.prev_swing) / bos.broken_level,
        })

    return setups


def train_bos_ml_filter(setups: list, timeframe: str = "1h"):
    """Train ML model for BOS setup filtering."""

    df = pd.DataFrame(setups)
    print(f"\nTotal setups collected: {len(df)}")
    print(f"  LONG:  {len(df[df['type']=='LONG'])}")
    print(f"  SHORT: {len(df[df['type']=='SHORT'])}")
    print(f"  Win rate: {df['outcome'].mean()*100:.1f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    feature_cols = [
        'bos_strength', 'retest_depth', 'candles_to_retest',
        'volume_ratio', 'delta_at_retest', 'delta_ratio',
        'cvd_recent', 'swing_distance_pct', 'rr_ratio'
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "="*60)
    print("  BOS ML Filter Results (Test Set)")
    print("="*60)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    print(f"  Recall: {recall_score(y_test, y_pred, zero_division=0)*100:.1f}%")

    print("\n  Performance by Confidence Threshold:")
    print("-"*60)
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
        filtered = y_prob >= threshold
        if filtered.sum() > 0:
            filtered_win_rate = y_test[filtered].mean()
            avg_rr = df.iloc[split_idx:][filtered]['rr_ratio'].mean()
            ev = filtered_win_rate * avg_rr - (1 - filtered_win_rate) * 1
            print(f"  >{threshold:.0%}: {filtered.sum():4} setups, {filtered_win_rate*100:.1f}% win, R:R {avg_rr:.2f}, EV {ev:.2f}")

    print("\n  Feature Importance:")
    print("-"*60)
    importance = model.feature_importances_
    for i, col in enumerate(feature_cols):
        print(f"  {col:25} {importance[i]:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    suffix = f"_{timeframe}" if timeframe != "1h" else ""

    joblib.dump(model, os.path.join(MODEL_DIR, f"bos_filter{suffix}.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"bos_scaler{suffix}.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, f"bos_features{suffix}.joblib"))

    print(f"\n  Model saved to {MODEL_DIR}")

    return model, scaler, feature_cols


def main(timeframe: str = "1h"):
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     Break of Structure (BOS) Strategy - ML Training ({timeframe}) ║
╚═══════════════════════════════════════════════════════════╝
""")

    print("Loading data...")
    candles_pl = load_candles("BTCUSDT", timeframe)
    candles = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(candles):,} candles ({timeframe})")

    # Find swings
    print("\nFinding swing points...")
    n = 5 if timeframe in ['1h', '4h'] else 3
    swing_highs, swing_lows = find_swing_points(candles, n=n)
    print(f"  Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")

    # Detect BOS events
    print("\nDetecting BOS events...")
    bos_events = detect_bos_events(candles, swing_highs, swing_lows)
    bullish_count = len([b for b in bos_events if b.type == 'bullish'])
    bearish_count = len([b for b in bos_events if b.type == 'bearish'])
    print(f"  Found {len(bos_events)} BOS events")
    print(f"    Bullish: {bullish_count}")
    print(f"    Bearish: {bearish_count}")

    # Test different R:R minimums
    print("\n" + "="*70)
    print(f"{'Min R:R':^10} | {'Setups':^8} | {'Win Rate':^10} | {'EV':^8} | {'Avg R:R':^10}")
    print("="*70)

    best_rr = 2.0
    best_ev = -999

    for min_rr in [1.5, 2.0, 2.5, 3.0]:
        setups = collect_bos_setups(candles, bos_events, min_rr=min_rr, quiet=True)
        if len(setups) < 20:
            continue

        df = pd.DataFrame(setups)
        win_rate = df['outcome'].mean()
        avg_rr = df['rr_ratio'].mean()
        ev = win_rate * avg_rr - (1 - win_rate) * 1

        print(f"{min_rr:^10.1f} | {len(setups):^8} | {win_rate*100:^10.1f}% | {ev:^8.2f} | {avg_rr:^10.2f}")

        if ev > best_ev:
            best_ev = ev
            best_rr = min_rr

    print("="*70)
    print(f"\nBest Min R:R: 1:{best_rr:.1f} (EV: {best_ev:.2f})")

    # Collect setups with best R:R
    print(f"\nCollecting BOS setups (min R:R 1:{best_rr:.1f})...")
    setups = collect_bos_setups(candles, bos_events, min_rr=best_rr)

    if len(setups) < 50:
        print("Not enough setups collected.")
        return

    train_bos_ml_filter(setups, timeframe)


if __name__ == "__main__":
    import sys
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(timeframe)
