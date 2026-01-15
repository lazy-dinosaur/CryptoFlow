#!/usr/bin/env python3
"""
ML Entry Filter System

Uses ML to decide whether to take a signal:
- TAKE: High probability signal, take the trade
- SKIP: Low probability signal, skip it

Features at entry time:
- Same as ML Exit features
- Labels based on trade outcome (win/loss)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels, Channel
from ml_exit import (
    extract_features, TradeFeatures, features_to_array, FEATURE_NAMES,
    calculate_atr, calculate_rsi, EXIT_TP1, EXIT_TP2, HOLD_TP3
)

# Entry labels
TAKE = 1  # Take the trade
SKIP = 0  # Skip the trade


def simulate_trade_for_entry_label(
    candles: pd.DataFrame,
    idx: int,
    trade_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float
) -> Tuple[bool, float]:
    """
    Simulate trade and determine if it was profitable.

    Returns:
        (is_winner, pnl_pct)
    """
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    hit_tp1 = False
    pnl_pct = 0
    current_sl = sl_price

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    return False, pnl_pct
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    return True, pnl_pct  # TP1 hit, then BE
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    return True, pnl_pct
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    return False, pnl_pct
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    return True, pnl_pct
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    return True, pnl_pct

    # Timeout - count as loss if TP1 not hit
    return hit_tp1, pnl_pct


def collect_entry_training_data(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008
) -> Tuple[List[TradeFeatures], List[int], List[dict]]:
    """
    Collect training data for entry filter.

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

    for i in tqdm(range(50, len(ltf_candles) - 150)):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Check for fakeout
        fakeout_signal = htf_fakeout_map.get(htf_idx)
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
                        is_win, pnl_pct = simulate_trade_for_entry_label(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )

                        features_list.append(features)
                        labels.append(TAKE if is_win else SKIP)
                        trade_data_list.append({
                            'idx': i,
                            'type': 'LONG',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'is_win': is_win,
                            'pnl_pct': pnl_pct
                        })
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
                        is_win, pnl_pct = simulate_trade_for_entry_label(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                        )

                        features_list.append(features)
                        labels.append(TAKE if is_win else SKIP)
                        trade_data_list.append({
                            'idx': i,
                            'type': 'SHORT',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'is_win': is_win,
                            'pnl_pct': pnl_pct
                        })
                        traded_entries.add(trade_key)

        # Check for bounce
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
                is_win, pnl_pct = simulate_trade_for_entry_label(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                )

                features_list.append(features)
                labels.append(TAKE if is_win else SKIP)
                trade_data_list.append({
                    'idx': i,
                    'type': 'LONG',
                    'setup_type': 'BOUNCE',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': tp1_price,
                    'tp2': tp2_price,
                    'is_win': is_win,
                    'pnl_pct': pnl_pct
                })
                traded_entries.add(trade_key)

        # Resistance bounce → SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            if sl_price > entry_price and entry_price > tp1_price:
                features = extract_features(ltf_candles, i, channel, 'SHORT', 'BOUNCE', None)
                is_win, pnl_pct = simulate_trade_for_entry_label(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                )

                features_list.append(features)
                labels.append(TAKE if is_win else SKIP)
                trade_data_list.append({
                    'idx': i,
                    'type': 'SHORT',
                    'setup_type': 'BOUNCE',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': tp1_price,
                    'tp2': tp2_price,
                    'is_win': is_win,
                    'pnl_pct': pnl_pct
                })
                traded_entries.add(trade_key)

    return features_list, labels, trade_data_list


def train_entry_model(X: np.ndarray, y: np.ndarray, model_path: str = None):
    """Train ML model for entry prediction."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score

    print(f"\nTraining data: {len(X)} samples")
    print(f"Label distribution:")
    print(f"  TAKE (win): {np.sum(y == TAKE)} ({np.sum(y == TAKE)/len(y)*100:.1f}%)")
    print(f"  SKIP (loss): {np.sum(y == SKIP)} ({np.sum(y == SKIP)/len(y)*100:.1f}%)")

    # Scale features first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest on all data (small sample size)
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled, y)

    # Evaluate with cross-validation
    print("\nRandom Forest Results:")
    try:
        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, X_scaled, y, cv=cv)
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    except:
        print("CV not possible with current data distribution")

    # Training accuracy
    y_pred = rf_model.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred)
    print(f"Training Accuracy: {train_acc:.3f}")

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


def backtest_no_filter(trade_data_list: List[dict], label: str = ""):
    """Backtest without any filter (baseline)."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for trade in trade_data_list:
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade['pnl_pct']
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


def backtest_with_ml_entry(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    model,
    scaler,
    threshold: float = 0.5,
    label: str = ""
):
    """Backtest using ML entry filter."""

    X = features_to_array(features_list)
    X_scaled = scaler.transform(X)

    # Get probabilities
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of TAKE
    predictions = (probs >= threshold).astype(int)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0
    trades_taken = 0
    trades_skipped = 0

    for i, (trade, pred) in enumerate(zip(trade_data_list, predictions)):
        if pred == SKIP:
            trades_skipped += 1
            continue

        trades_taken += 1

        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade['pnl_pct']
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
    print(f"  Signals: {len(trade_data_list)}, Taken: {trades_taken}, Skipped: {trades_skipped}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital, trades_taken, wins, losses


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   ML Entry Filter System                                  ║
║   Learn to filter low-probability signals                 ║
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

    features_list, labels, trade_data_list = collect_entry_training_data(
        htf_candles, ltf_candles, htf, ltf
    )

    print(f"\nTotal samples: {len(features_list)}")

    # Convert to array
    X = features_to_array(features_list)
    y = np.array(labels)

    # Split by year
    trade_df = pd.DataFrame(trade_data_list)
    trade_df['time'] = ltf_candles.index[trade_df['idx'].values]
    trade_df['year'] = pd.to_datetime(trade_df['time']).dt.year

    is_mask = trade_df['year'] == 2024
    oos_mask = trade_df['year'] == 2025

    # Train on all data (small sample size)
    X_train = X
    y_train = y

    # Train model
    print("\n" + "="*60)
    print("  Training ML Model (All Data)")
    print("="*60)

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'entry_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model, scaler = train_entry_model(X_train, y_train, model_path)

    # Backtest comparison
    print("\n" + "="*60)
    print("  Backtest Comparison (OOS 2025)")
    print("="*60)

    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask.iloc[i]]
    oos_features = [features_list[i] for i in range(len(features_list)) if oos_mask.iloc[i]]

    print("\n--- Baseline (No Filter) ---")
    backtest_no_filter(oos_trades, "  No Filter")

    print("\n--- ML Entry Filter ---")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        backtest_with_ml_entry(oos_trades, oos_features, model, scaler, threshold, f"  ML Entry (threshold={threshold})")


if __name__ == "__main__":
    main()
