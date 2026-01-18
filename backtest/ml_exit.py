#!/usr/bin/env python3
"""
ML Exit Management System

Uses ML to decide optimal exit strategy for each trade:
- EXIT_TP1: Exit 100% at TP1 (price likely to reverse)
- EXIT_TP2: Exit 50% at TP1, 50% at TP2 (current strategy)
- HOLD_TP3: Exit 33% at TP1, 33% at TP2, 34% at TP3 (strong continuation)

Features at entry time:
- Channel width
- Touch counts (S/R)
- Volume/Delta patterns
- ATR (volatility)
- Position in channel
- Recent momentum
- Setup type (BOUNCE vs FAKEOUT)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import (
    find_swing_points, build_htf_channels, Channel, FakeoutSignal
)

# Exit strategies
EXIT_TP1 = 0  # 100% at TP1
EXIT_TP2 = 1  # 50% TP1 + 50% TP2
HOLD_TP3 = 2  # 33% TP1 + 33% TP2 + 34% TP3

EXIT_NAMES = {
    EXIT_TP1: 'EXIT_TP1',
    EXIT_TP2: 'EXIT_TP2',
    HOLD_TP3: 'HOLD_TP3'
}


@dataclass
class TradeFeatures:
    """Features extracted at entry time for ML prediction."""
    # Channel features
    channel_width_pct: float
    support_touches: int
    resistance_touches: int
    total_touches: int

    # Price position
    price_in_channel_pct: float  # 0 = at support, 1 = at resistance

    # Volume/Delta
    volume_ratio: float
    delta_ratio: float
    cvd_recent: float
    volume_ma_20: float
    delta_ma_20: float

    # Volatility
    atr_14: float
    atr_ratio: float  # current ATR vs avg

    # Momentum
    momentum_5: float  # 5-candle momentum
    momentum_20: float  # 20-candle momentum
    rsi_14: float

    # Setup type
    is_bounce: int  # 1 = BOUNCE, 0 = FAKEOUT
    is_long: int  # 1 = LONG, 0 = SHORT

    # Candle pattern
    body_size_pct: float
    wick_ratio: float  # upper wick / body
    is_bullish: int

    # Time features
    hour: int
    day_of_week: int

    # Fakeout specific
    fakeout_depth_pct: float


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR."""
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        )
    )
    tr[0] = highs[0] - lows[0]

    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])

    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
    deltas = np.diff(closes)
    deltas = np.insert(deltas, 0, 0)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)

    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])

    for i in range(period+1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def extract_features(
    candles: pd.DataFrame,
    idx: int,
    channel: Channel,
    trade_type: str,
    setup_type: str,
    fakeout_extreme: Optional[float] = None
) -> TradeFeatures:
    """Extract features at entry time."""

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    current_close = closes[idx]
    current_high = highs[idx]
    current_low = lows[idx]
    current_open = opens[idx]

    # Calculate indicators
    atr = calculate_atr(highs, lows, closes, 14)
    rsi = calculate_rsi(closes, 14)

    # Historical window
    hist_start = max(0, idx - 20)

    # Channel features
    channel_width_pct = (channel.resistance - channel.support) / channel.support
    price_in_channel_pct = (current_close - channel.support) / (channel.resistance - channel.support)

    # Volume/Delta
    avg_volume_20 = np.mean(volumes[hist_start:idx]) if idx > hist_start else volumes[idx]
    avg_delta_20 = np.mean(deltas[hist_start:idx]) if idx > hist_start else 0
    volume_ratio = volumes[idx] / avg_volume_20 if avg_volume_20 > 0 else 1
    delta_ratio = deltas[idx] / (abs(avg_delta_20) + 1)
    cvd_recent = np.sum(deltas[hist_start:idx])

    # ATR
    avg_atr = np.mean(atr[max(0, idx-50):idx]) if idx > 50 else atr[idx]
    atr_ratio = atr[idx] / avg_atr if avg_atr > 0 else 1

    # Momentum
    momentum_5 = (closes[idx] - closes[max(0, idx-5)]) / closes[max(0, idx-5)] * 100
    momentum_20 = (closes[idx] - closes[max(0, idx-20)]) / closes[max(0, idx-20)] * 100

    # Candle pattern
    body = abs(current_close - current_open)
    body_size_pct = body / current_close * 100
    upper_wick = current_high - max(current_close, current_open)
    wick_ratio = upper_wick / body if body > 0 else 0

    # Time features
    timestamp = candles.index[idx]
    if isinstance(timestamp, pd.Timestamp):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
    else:
        hour = 12
        day_of_week = 0

    # Fakeout depth
    fakeout_depth_pct = 0
    if fakeout_extreme:
        if trade_type == 'LONG':
            fakeout_depth_pct = (channel.support - fakeout_extreme) / channel.support * 100
        else:
            fakeout_depth_pct = (fakeout_extreme - channel.resistance) / channel.resistance * 100

    return TradeFeatures(
        channel_width_pct=channel_width_pct,
        support_touches=channel.support_touches,
        resistance_touches=channel.resistance_touches,
        total_touches=channel.support_touches + channel.resistance_touches,
        price_in_channel_pct=price_in_channel_pct,
        volume_ratio=volume_ratio,
        delta_ratio=delta_ratio,
        cvd_recent=cvd_recent,
        volume_ma_20=avg_volume_20,
        delta_ma_20=avg_delta_20,
        atr_14=atr[idx],
        atr_ratio=atr_ratio,
        momentum_5=momentum_5,
        momentum_20=momentum_20,
        rsi_14=rsi[idx],
        is_bounce=1 if setup_type == 'BOUNCE' else 0,
        is_long=1 if trade_type == 'LONG' else 0,
        body_size_pct=body_size_pct,
        wick_ratio=wick_ratio,
        is_bullish=1 if current_close > current_open else 0,
        hour=hour,
        day_of_week=day_of_week,
        fakeout_depth_pct=fakeout_depth_pct
    )


def simulate_trade_with_optimal_exit(
    candles: pd.DataFrame,
    idx: int,
    trade_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    channel: Channel,
    fakeout_extreme: Optional[float],
    setup_type: str
) -> Tuple[dict, int]:
    """
    Simulate trade and determine optimal exit strategy.

    Returns:
        (trade_data, optimal_exit)
    """
    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    # TP3 = channel breakout (opposite edge + extension)
    if trade_type == 'LONG':
        tp3_price = channel.resistance * 1.01  # 1% beyond resistance
    else:
        tp3_price = channel.support * 0.99  # 1% below support

    reward3 = abs(tp3_price - entry_price)

    # Track what actually happened
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False
    hit_sl_before_tp1 = False
    hit_sl_after_tp1 = False  # SL at breakeven after TP1
    max_favorable = 0  # Maximum favorable excursion after entry

    current_sl = sl_price

    for j in range(idx + 1, min(idx + 200, len(candles))):
        if trade_type == 'LONG':
            # Track maximum favorable excursion
            excursion = (highs[j] - entry_price) / entry_price
            max_favorable = max(max_favorable, excursion)

            if not hit_tp1:
                if lows[j] <= sl_price:
                    hit_sl_before_tp1 = True
                    break
                if highs[j] >= tp1_price:
                    hit_tp1 = True
                    current_sl = entry_price
            elif not hit_tp2:
                if lows[j] <= current_sl:
                    hit_sl_after_tp1 = True
                    break
                if highs[j] >= tp2_price:
                    hit_tp2 = True
            elif not hit_tp3:
                if lows[j] <= entry_price:  # Still at BE
                    break
                if highs[j] >= tp3_price:
                    hit_tp3 = True
                    break
        else:  # SHORT
            excursion = (entry_price - lows[j]) / entry_price
            max_favorable = max(max_favorable, excursion)

            if not hit_tp1:
                if highs[j] >= sl_price:
                    hit_sl_before_tp1 = True
                    break
                if lows[j] <= tp1_price:
                    hit_tp1 = True
                    current_sl = entry_price
            elif not hit_tp2:
                if highs[j] >= current_sl:
                    hit_sl_after_tp1 = True
                    break
                if lows[j] <= tp2_price:
                    hit_tp2 = True
            elif not hit_tp3:
                if highs[j] >= entry_price:
                    break
                if lows[j] <= tp3_price:
                    hit_tp3 = True
                    break

    # Determine optimal exit
    if hit_sl_before_tp1:
        # Loss - any strategy would have lost
        optimal_exit = EXIT_TP2  # Default
        pnl_best = -risk / entry_price
    elif hit_tp1 and hit_sl_after_tp1 and not hit_tp2:
        # Should have exited 100% at TP1
        optimal_exit = EXIT_TP1
        pnl_best = reward1 / entry_price
    elif hit_tp2 and not hit_tp3:
        # Current strategy (TP1 + TP2) is good
        optimal_exit = EXIT_TP2
        pnl_best = 0.5 * (reward1 + reward2) / entry_price
    elif hit_tp3:
        # Should have held for TP3
        optimal_exit = HOLD_TP3
        pnl_best = 0.33 * reward1 + 0.33 * reward2 + 0.34 * reward3
        pnl_best = pnl_best / entry_price
    elif hit_tp1 and not hit_tp2:
        # Only TP1 hit, then BE - EXIT_TP1 would be best
        optimal_exit = EXIT_TP1
        pnl_best = reward1 / entry_price
    else:
        # No TP hit, stuck
        optimal_exit = EXIT_TP2
        pnl_best = 0

    # Calculate PnL for each strategy
    pnl_tp1 = 0  # 100% at TP1
    pnl_tp2 = 0  # 50% TP1 + 50% TP2
    pnl_tp3 = 0  # 33% each

    if hit_sl_before_tp1:
        pnl_tp1 = pnl_tp2 = pnl_tp3 = -risk / entry_price
    elif hit_tp1:
        pnl_tp1 = reward1 / entry_price

        if hit_tp2:
            pnl_tp2 = 0.5 * (reward1 + reward2) / entry_price
            if hit_tp3:
                pnl_tp3 = (0.33 * reward1 + 0.33 * reward2 + 0.34 * reward3) / entry_price
            else:
                pnl_tp3 = (0.33 * reward1 + 0.33 * reward2) / entry_price  # 34% still held
        else:
            # TP1 hit but TP2 not hit
            if hit_sl_after_tp1:
                pnl_tp2 = 0.5 * reward1 / entry_price  # 50% profit, 50% BE
                pnl_tp3 = 0.33 * reward1 / entry_price  # 33% profit, rest BE
            else:
                pnl_tp2 = 0.5 * reward1 / entry_price
                pnl_tp3 = 0.33 * reward1 / entry_price

    # Verify optimal based on actual PnL
    pnls = {EXIT_TP1: pnl_tp1, EXIT_TP2: pnl_tp2, HOLD_TP3: pnl_tp3}
    actual_optimal = max(pnls, key=pnls.get)

    trade_data = {
        'idx': idx,
        'type': trade_type,
        'setup_type': setup_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'tp3': tp3_price,
        'hit_tp1': hit_tp1,
        'hit_tp2': hit_tp2,
        'hit_tp3': hit_tp3,
        'hit_sl_before_tp1': hit_sl_before_tp1,
        'hit_sl_after_tp1': hit_sl_after_tp1,
        'pnl_tp1': pnl_tp1,
        'pnl_tp2': pnl_tp2,
        'pnl_tp3': pnl_tp3,
        'optimal_exit': actual_optimal,
        'max_favorable': max_favorable,
        'channel_width': (channel.resistance - channel.support) / channel.support
    }

    return trade_data, actual_optimal


def collect_training_data(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008
) -> Tuple[List[TradeFeatures], List[int], List[dict]]:
    """
    Collect training data with features and optimal exit labels.

    Returns:
        (features, labels, trade_data)
    """
    print("Building HTF channels...")
    htf_channel_map, htf_fakeout_signals = build_htf_channels(htf_candles)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    features_list = []
    labels = []
    trade_data_list = []
    traded_entries = set()

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}

    print(f"\nScanning {len(ltf_candles)} LTF candles for setups...")

    for i in tqdm(range(50, len(ltf_candles) - 200)):  # Need history and future
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Check for fakeout
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)  # Fix lookahead bias
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2

            trade_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)
            if trade_key not in traded_entries:
                if fakeout_signal.type == 'bear':
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 - sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.resistance * 0.998

                    if entry_price > sl_price and tp1_price > entry_price:
                        features = extract_features(
                            ltf_candles, i, f_channel, 'LONG', 'FAKEOUT', fakeout_signal.extreme
                        )
                        trade_data, optimal = simulate_trade_with_optimal_exit(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        features_list.append(features)
                        labels.append(optimal)
                        trade_data_list.append(trade_data)
                        traded_entries.add(trade_key)

                else:  # bull fakeout
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.support * 1.002

                    if sl_price > entry_price and entry_price > tp1_price:
                        features = extract_features(
                            ltf_candles, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme
                        )
                        trade_data, optimal = simulate_trade_with_optimal_exit(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        features_list.append(features)
                        labels.append(optimal)
                        trade_data_list.append(trade_data)
                        traded_entries.add(trade_key)

        # Check for bounce - use smaller window to capture more trades
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 10)
        if trade_key in traded_entries:
            continue

        # Support bounce → LONG
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            if entry_price > sl_price and tp1_price > entry_price:
                features = extract_features(ltf_candles, i, channel, 'LONG', 'BOUNCE', None)
                trade_data, optimal = simulate_trade_with_optimal_exit(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE'
                )

                features_list.append(features)
                labels.append(optimal)
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

        # Resistance bounce → SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            if sl_price > entry_price and entry_price > tp1_price:
                features = extract_features(ltf_candles, i, channel, 'SHORT', 'BOUNCE', None)
                trade_data, optimal = simulate_trade_with_optimal_exit(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE'
                )

                features_list.append(features)
                labels.append(optimal)
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

    return features_list, labels, trade_data_list


def features_to_array(features_list: List[TradeFeatures]) -> np.ndarray:
    """Convert list of TradeFeatures to numpy array."""
    if not features_list:
        return np.array([])

    return np.array([
        [
            f.channel_width_pct,
            f.support_touches,
            f.resistance_touches,
            f.total_touches,
            f.price_in_channel_pct,
            f.volume_ratio,
            f.delta_ratio,
            f.cvd_recent,
            f.atr_14,
            f.atr_ratio,
            f.momentum_5,
            f.momentum_20,
            f.rsi_14,
            f.is_bounce,
            f.is_long,
            f.body_size_pct,
            f.wick_ratio,
            f.is_bullish,
            f.hour,
            f.day_of_week,
            f.fakeout_depth_pct
        ]
        for f in features_list
    ])


FEATURE_NAMES = [
    'channel_width_pct', 'support_touches', 'resistance_touches', 'total_touches',
    'price_in_channel_pct', 'volume_ratio', 'delta_ratio', 'cvd_recent',
    'atr_14', 'atr_ratio', 'momentum_5', 'momentum_20', 'rsi_14',
    'is_bounce', 'is_long', 'body_size_pct', 'wick_ratio', 'is_bullish',
    'hour', 'day_of_week', 'fakeout_depth_pct'
]


def train_exit_model(X: np.ndarray, y: np.ndarray, model_path: str = None):
    """Train ML model for exit prediction."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\nTraining data: {len(X)} samples")
    print(f"Label distribution:")
    for label_id, name in EXIT_NAMES.items():
        count = np.sum(y == label_id)
        print(f"  {name}: {count} ({count/len(y)*100:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nRandom Forest Results:")
    y_pred = rf_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=list(EXIT_NAMES.values())))

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i in range(min(10, len(indices))):
        print(f"  {FEATURE_NAMES[indices[i]]}: {importance[indices[i]]:.4f}")

    # Save model
    if model_path:
        with open(model_path, 'wb') as f:
            pickle.dump({'model': rf_model, 'scaler': scaler}, f)
        print(f"\nModel saved to {model_path}")

    return rf_model, scaler


def backtest_with_ml_exit(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    model,
    scaler,
    label: str = ""
):
    """Backtest using ML exit predictions."""

    X = features_to_array(features_list)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    exit_stats = {EXIT_TP1: 0, EXIT_TP2: 0, HOLD_TP3: 0}

    for i, (trade, pred_exit) in enumerate(zip(trade_data_list, predictions)):
        exit_stats[pred_exit] += 1

        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        # Use PnL based on predicted exit strategy
        if pred_exit == EXIT_TP1:
            pnl_pct = trade['pnl_tp1']
        elif pred_exit == EXIT_TP2:
            pnl_pct = trade['pnl_tp2']
        else:  # HOLD_TP3
            pnl_pct = trade['pnl_tp3']

        gross_pnl = position_value * pnl_pct
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n{label}:")
    print(f"  Trades: {len(trade_data_list)}")
    print(f"  Exit Distribution: TP1={exit_stats[EXIT_TP1]}, TP2={exit_stats[EXIT_TP2]}, TP3={exit_stats[HOLD_TP3]}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital


def backtest_fixed_exit(trade_data_list: List[dict], exit_strategy: int, label: str = ""):
    """Backtest using fixed exit strategy."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    pnl_key = {EXIT_TP1: 'pnl_tp1', EXIT_TP2: 'pnl_tp2', HOLD_TP3: 'pnl_tp3'}[exit_strategy]

    for trade in trade_data_list:
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        pnl_pct = trade[pnl_key]
        gross_pnl = position_value * pnl_pct
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n{label}:")
    print(f"  Trades: {len(trade_data_list)}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   ML Exit Management System                               ║
║   Learn optimal exit strategy for each trade              ║
╚═══════════════════════════════════════════════════════════╝
""")

    htf = "1h"
    ltf = "15m"

    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(htf_candles):,} candles\n")

    print(f"Loading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(ltf_candles):,} candles")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}\n")

    # Collect training data
    print("="*60)
    print("  Collecting Training Data")
    print("="*60)

    features_list, labels, trade_data_list = collect_training_data(
        htf_candles, ltf_candles, htf, ltf
    )

    print(f"\nTotal samples: {len(features_list)}")

    # Analyze optimal exit distribution
    print("\n" + "="*60)
    print("  Optimal Exit Distribution")
    print("="*60)

    labels_arr = np.array(labels)
    for label_id, name in EXIT_NAMES.items():
        count = np.sum(labels_arr == label_id)
        print(f"  {name}: {count} ({count/len(labels_arr)*100:.1f}%)")

    # Convert features to array
    X = features_to_array(features_list)
    y = labels_arr

    # Train model
    print("\n" + "="*60)
    print("  Training ML Model")
    print("="*60)

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'exit_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model, scaler = train_exit_model(X, y, model_path)

    # Backtest comparison
    print("\n" + "="*60)
    print("  Backtest Comparison")
    print("="*60)

    # Split by year
    trade_df = pd.DataFrame(trade_data_list)
    trade_df['time'] = ltf_candles.index[trade_df['idx'].values]
    trade_df['year'] = pd.to_datetime(trade_df['time']).dt.year

    is_mask = trade_df['year'] == 2024
    oos_mask = trade_df['year'] == 2025

    print("\n--- Fixed Exit Strategies ---")

    # OOS only for comparison
    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask.iloc[i]]
    oos_features = [features_list[i] for i in range(len(features_list)) if oos_mask.iloc[i]]

    print("\nOUT-OF-SAMPLE (2025):")
    backtest_fixed_exit(oos_trades, EXIT_TP1, "  Fixed EXIT_TP1 (100% at TP1)")
    backtest_fixed_exit(oos_trades, EXIT_TP2, "  Fixed EXIT_TP2 (50%+50%) - Current")
    backtest_fixed_exit(oos_trades, HOLD_TP3, "  Fixed HOLD_TP3 (33%+33%+34%)")

    print("\n--- ML Exit Strategy ---")
    backtest_with_ml_exit(oos_trades, oos_features, model, scaler, "  ML Exit (OOS 2025) ⭐")

    # Optimal (if we knew the future)
    print("\n--- Optimal (Oracle) ---")
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    for trade in oos_trades:
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        # Use actual optimal
        pnl_key = {EXIT_TP1: 'pnl_tp1', EXIT_TP2: 'pnl_tp2', HOLD_TP3: 'pnl_tp3'}[trade['optimal_exit']]
        pnl_pct = trade[pnl_key]

        gross_pnl = position_value * pnl_pct
        fees = position_value * fee_pct * 2
        capital += gross_pnl - fees
        capital = max(capital, 0)

    print(f"  Optimal (Oracle): ${capital:,.2f} ({(capital-10000)/100:+.1f}%)")


if __name__ == "__main__":
    main()
