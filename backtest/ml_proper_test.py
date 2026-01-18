#!/usr/bin/env python3
"""
Proper IS/OOS Test - No Data Leakage

Train ONLY on 2024 data
Test ONLY on 2025 data (never seen during training)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import (
    extract_features, TradeFeatures, features_to_array, FEATURE_NAMES,
    simulate_trade_with_optimal_exit, EXIT_TP1, EXIT_TP2, HOLD_TP3, EXIT_NAMES
)
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP


def collect_all_data(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008
):
    """Collect all trade data with features and labels."""

    print("Building HTF channels...")
    htf_channel_map, htf_fakeout_signals = build_htf_channels(htf_candles)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    features_list = []
    trade_data_list = []
    traded_entries = set()

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}

    print(f"\nScanning {len(ltf_candles)} LTF candles...")

    for i in tqdm(range(50, len(ltf_candles) - 200)):
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

                        is_win, base_pnl = simulate_trade_for_entry_label(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )

                        exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        # Get timestamp for year split
                        timestamp = ltf_candles.index[i]

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'is_win': is_win,
                            'entry_label': TAKE if is_win else SKIP,
                            'exit_label': optimal_exit,
                            'timestamp': timestamp
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

                        is_win, base_pnl = simulate_trade_for_entry_label(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                        )

                        exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        timestamp = ltf_candles.index[i]

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'is_win': is_win,
                            'entry_label': TAKE if is_win else SKIP,
                            'exit_label': optimal_exit,
                            'timestamp': timestamp
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

                is_win, base_pnl = simulate_trade_for_entry_label(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                )

                exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE'
                )

                timestamp = ltf_candles.index[i]

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'is_win': is_win,
                    'entry_label': TAKE if is_win else SKIP,
                    'exit_label': optimal_exit,
                    'timestamp': timestamp
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

                is_win, base_pnl = simulate_trade_for_entry_label(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                )

                exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                    channel, None, 'BOUNCE'
                )

                timestamp = ltf_candles.index[i]

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'is_win': is_win,
                    'entry_label': TAKE if is_win else SKIP,
                    'exit_label': optimal_exit,
                    'timestamp': timestamp
                })
                traded_entries.add(trade_key)

    return features_list, trade_data_list


def train_models_on_is(X_is, entry_labels_is, exit_labels_is):
    """Train both models ONLY on IS (2024) data."""

    print("\n" + "="*60)
    print("  Training Models on IS (2024) ONLY")
    print("="*60)

    # Entry model
    print(f"\nEntry Model:")
    print(f"  Training samples: {len(X_is)}")
    print(f"  TAKE: {np.sum(entry_labels_is == TAKE)} ({np.sum(entry_labels_is == TAKE)/len(entry_labels_is)*100:.1f}%)")
    print(f"  SKIP: {np.sum(entry_labels_is == SKIP)} ({np.sum(entry_labels_is == SKIP)/len(entry_labels_is)*100:.1f}%)")

    entry_scaler = StandardScaler()
    X_is_scaled = entry_scaler.fit_transform(X_is)

    entry_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    entry_model.fit(X_is_scaled, entry_labels_is)

    entry_train_acc = accuracy_score(entry_labels_is, entry_model.predict(X_is_scaled))
    print(f"  Training Accuracy: {entry_train_acc:.3f}")

    # Exit model
    print(f"\nExit Model:")
    print(f"  Training samples: {len(X_is)}")
    for label_id, name in EXIT_NAMES.items():
        count = np.sum(exit_labels_is == label_id)
        print(f"  {name}: {count} ({count/len(exit_labels_is)*100:.1f}%)")

    exit_scaler = StandardScaler()
    X_is_exit_scaled = exit_scaler.fit_transform(X_is)

    exit_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    exit_model.fit(X_is_exit_scaled, exit_labels_is)

    exit_train_acc = accuracy_score(exit_labels_is, exit_model.predict(X_is_exit_scaled))
    print(f"  Training Accuracy: {exit_train_acc:.3f}")

    return entry_model, entry_scaler, exit_model, exit_scaler


def backtest(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    entry_model=None,
    entry_scaler=None,
    exit_model=None,
    exit_scaler=None,
    entry_threshold: float = 0.5,
    label: str = ""
):
    """Run backtest with optional ML models."""

    X = features_to_array(features_list)

    # Get predictions if models provided
    if entry_model and entry_scaler:
        X_entry_scaled = entry_scaler.transform(X)
        entry_probs = entry_model.predict_proba(X_entry_scaled)[:, 1]
        entry_preds = (entry_probs >= entry_threshold).astype(int)
    else:
        entry_preds = np.ones(len(X), dtype=int)  # Take all

    if exit_model and exit_scaler:
        X_exit_scaled = exit_scaler.transform(X)
        exit_preds = exit_model.predict(X_exit_scaled)
    else:
        exit_preds = np.full(len(X), EXIT_TP2)  # Fixed TP2

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0
    trades_taken = 0

    for i, (trade, take_trade, exit_pred) in enumerate(zip(trade_data_list, entry_preds, exit_preds)):
        if take_trade == SKIP:
            continue

        trades_taken += 1

        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        pnl_key = {EXIT_TP1: 'pnl_tp1', EXIT_TP2: 'pnl_tp2', HOLD_TP3: 'pnl_tp3'}[exit_pred]
        gross_pnl = position_value * trade[pnl_key]
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
    print(f"  Trades: {trades_taken}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital, trades_taken, wins, losses


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   PROPER IS/OOS TEST - NO DATA LEAKAGE                    ║
║   Train on 2024 ONLY, Test on 2025 ONLY                   ║
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

    # Collect all data
    print("="*60)
    print("  Collecting Trade Data")
    print("="*60)

    features_list, trade_data_list = collect_all_data(htf_candles, ltf_candles, htf, ltf)

    print(f"\nTotal samples: {len(features_list)}")

    # Split by year - STRICT
    years = [pd.to_datetime(t['timestamp']).year for t in trade_data_list]
    # IS: 2022-2023, OOS: 2024-2025
    is_mask = np.array([y in [2022, 2023] for y in years])
    oos_mask = np.array([y in [2024, 2025] for y in years])

    print(f"\nIS (2024): {np.sum(is_mask)} trades")
    print(f"OOS (2025): {np.sum(oos_mask)} trades")

    # Prepare data
    X = features_to_array(features_list)
    entry_labels = np.array([t['entry_label'] for t in trade_data_list])
    exit_labels = np.array([t['exit_label'] for t in trade_data_list])

    X_is = X[is_mask]
    entry_labels_is = entry_labels[is_mask]
    exit_labels_is = exit_labels[is_mask]

    X_oos = X[oos_mask]
    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask[i]]
    oos_features = [features_list[i] for i in range(len(features_list)) if oos_mask[i]]

    # Check if we have enough IS data
    if len(X_is) < 10:
        print(f"\n⚠️  WARNING: Only {len(X_is)} IS samples - too few for reliable ML!")
        print("   Results may not be meaningful.\n")

    # Train models on IS only
    entry_model, entry_scaler, exit_model, exit_scaler = train_models_on_is(
        X_is, entry_labels_is, exit_labels_is
    )

    # Test on OOS
    print("\n" + "="*60)
    print("  OOS RESULTS (2025) - NEVER SEEN DURING TRAINING")
    print("="*60)

    print("\n--- 1. Baseline (No ML) ---")
    backtest(oos_trades, oos_features, label="  Baseline")

    print("\n--- 2. ML Entry Only (trained on 2024) ---")
    backtest(
        oos_trades, oos_features,
        entry_model=entry_model,
        entry_scaler=entry_scaler,
        entry_threshold=0.5,
        label="  ML Entry"
    )

    print("\n--- 3. ML Exit Only (trained on 2024) ---")
    backtest(
        oos_trades, oos_features,
        exit_model=exit_model,
        exit_scaler=exit_scaler,
        label="  ML Exit"
    )

    print("\n--- 4. ML Combined (trained on 2024) ---")
    backtest(
        oos_trades, oos_features,
        entry_model=entry_model,
        entry_scaler=entry_scaler,
        exit_model=exit_model,
        exit_scaler=exit_scaler,
        entry_threshold=0.5,
        label="  ML Combined ⭐"
    )


if __name__ == "__main__":
    main()
