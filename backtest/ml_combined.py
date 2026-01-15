#!/usr/bin/env python3
"""
Combined ML Entry + Exit System

Tests:
1. ML Entry alone (filter signals)
2. ML Exit alone (optimize exit)
3. ML Entry + Exit combined
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import (
    extract_features, TradeFeatures, features_to_array, FEATURE_NAMES,
    simulate_trade_with_optimal_exit, EXIT_TP1, EXIT_TP2, HOLD_TP3, EXIT_NAMES
)
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP


def collect_combined_data(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008
):
    """Collect data for both entry and exit models."""

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

    print(f"\nScanning {len(ltf_candles)} LTF candles for setups...")

    for i in tqdm(range(50, len(ltf_candles) - 200)):
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

                        # Get entry label (win/loss)
                        is_win, base_pnl = simulate_trade_for_entry_label(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )

                        # Get optimal exit
                        exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'is_win': is_win,
                            'entry_label': TAKE if is_win else SKIP
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

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'is_win': is_win,
                            'entry_label': TAKE if is_win else SKIP
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

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'is_win': is_win,
                    'entry_label': TAKE if is_win else SKIP
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

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'is_win': is_win,
                    'entry_label': TAKE if is_win else SKIP
                })
                traded_entries.add(trade_key)

    return features_list, trade_data_list


def backtest_baseline(trade_data_list: List[dict], label: str = ""):
    """Baseline: No ML, fixed TP2 exit."""
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

        gross_pnl = position_value * trade['pnl_tp2']  # Fixed TP2 exit
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


def backtest_ml_entry_only(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    entry_model,
    entry_scaler,
    threshold: float = 0.5,
    label: str = ""
):
    """ML Entry filter + Fixed TP2 exit."""
    X = features_to_array(features_list)
    X_scaled = entry_scaler.transform(X)

    entry_probs = entry_model.predict_proba(X_scaled)[:, 1]
    entry_preds = (entry_probs >= threshold).astype(int)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0
    trades_taken = 0

    for i, (trade, take_trade) in enumerate(zip(trade_data_list, entry_preds)):
        if take_trade == SKIP:
            continue

        trades_taken += 1
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade['pnl_tp2']  # Fixed TP2
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
    print(f"  Signals: {len(trade_data_list)}, Taken: {trades_taken}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital


def backtest_ml_exit_only(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    exit_model,
    exit_scaler,
    label: str = ""
):
    """No entry filter + ML Exit."""
    X = features_to_array(features_list)
    X_scaled = exit_scaler.transform(X)

    exit_preds = exit_model.predict(X_scaled)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    exit_stats = {EXIT_TP1: 0, EXIT_TP2: 0, HOLD_TP3: 0}

    for i, (trade, exit_pred) in enumerate(zip(trade_data_list, exit_preds)):
        exit_stats[exit_pred] += 1

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
    print(f"  Trades: {len(trade_data_list)}")
    print(f"  Exit Distribution: TP1={exit_stats[EXIT_TP1]}, TP2={exit_stats[EXIT_TP2]}, TP3={exit_stats[HOLD_TP3]}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital


def backtest_ml_combined(
    trade_data_list: List[dict],
    features_list: List[TradeFeatures],
    entry_model,
    entry_scaler,
    exit_model,
    exit_scaler,
    entry_threshold: float = 0.5,
    label: str = ""
):
    """ML Entry + ML Exit combined."""
    X = features_to_array(features_list)

    # Entry predictions
    X_entry_scaled = entry_scaler.transform(X)
    entry_probs = entry_model.predict_proba(X_entry_scaled)[:, 1]
    entry_preds = (entry_probs >= entry_threshold).astype(int)

    # Exit predictions
    X_exit_scaled = exit_scaler.transform(X)
    exit_preds = exit_model.predict(X_exit_scaled)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0
    trades_taken = 0

    exit_stats = {EXIT_TP1: 0, EXIT_TP2: 0, HOLD_TP3: 0}

    for i, (trade, take_trade, exit_pred) in enumerate(zip(trade_data_list, entry_preds, exit_preds)):
        if take_trade == SKIP:
            continue

        trades_taken += 1
        exit_stats[exit_pred] += 1

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
    print(f"  Signals: {len(trade_data_list)}, Taken: {trades_taken}")
    print(f"  Exit Distribution: TP1={exit_stats[EXIT_TP1]}, TP2={exit_stats[EXIT_TP2]}, TP3={exit_stats[HOLD_TP3]}")
    print(f"  Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Final: ${capital:,.2f}")

    return capital


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║   Combined ML Entry + Exit Test                           ║
║   Compare: Baseline vs Entry vs Exit vs Combined          ║
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

    # Load models
    print("Loading ML models...")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    with open(os.path.join(models_dir, 'entry_model.pkl'), 'rb') as f:
        entry_data = pickle.load(f)
        entry_model = entry_data['model']
        entry_scaler = entry_data['scaler']

    with open(os.path.join(models_dir, 'exit_model.pkl'), 'rb') as f:
        exit_data = pickle.load(f)
        exit_model = exit_data['model']
        exit_scaler = exit_data['scaler']

    print("  Entry model loaded")
    print("  Exit model loaded")

    # Collect data
    print("\n" + "="*60)
    print("  Collecting Trade Data")
    print("="*60)

    features_list, trade_data_list = collect_combined_data(
        htf_candles, ltf_candles, htf, ltf
    )

    print(f"\nTotal samples: {len(features_list)}")

    # Split by year
    trade_df = pd.DataFrame(trade_data_list)
    trade_df['time'] = ltf_candles.index[trade_df['idx'].values]
    trade_df['year'] = pd.to_datetime(trade_df['time']).dt.year

    oos_mask = trade_df['year'] == 2025

    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask.iloc[i]]
    oos_features = [features_list[i] for i in range(len(features_list)) if oos_mask.iloc[i]]

    # Run backtests
    print("\n" + "="*60)
    print("  OUT-OF-SAMPLE RESULTS (2025)")
    print("="*60)

    print("\n--- 1. Baseline (No ML) ---")
    baseline_capital = backtest_baseline(oos_trades, "  Baseline (Fixed TP2)")

    print("\n--- 2. ML Entry Only ---")
    entry_capital = backtest_ml_entry_only(
        oos_trades, oos_features,
        entry_model, entry_scaler,
        threshold=0.5,
        label="  ML Entry (threshold=0.5)"
    )

    print("\n--- 3. ML Exit Only ---")
    exit_capital = backtest_ml_exit_only(
        oos_trades, oos_features,
        exit_model, exit_scaler,
        label="  ML Exit"
    )

    print("\n--- 4. ML Entry + Exit Combined ---")
    combined_capital = backtest_ml_combined(
        oos_trades, oos_features,
        entry_model, entry_scaler,
        exit_model, exit_scaler,
        entry_threshold=0.5,
        label="  ML Entry + Exit ⭐"
    )

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY (OOS 2025)")
    print("="*60)
    print(f"\n  Baseline:      ${baseline_capital:>12,.2f} ({(baseline_capital-10000)/100:+.0f}%)")
    print(f"  ML Entry:      ${entry_capital:>12,.2f} ({(entry_capital-10000)/100:+.0f}%)")
    print(f"  ML Exit:       ${exit_capital:>12,.2f} ({(exit_capital-10000)/100:+.0f}%)")
    print(f"  ML Combined:   ${combined_capital:>12,.2f} ({(combined_capital-10000)/100:+.0f}%) ⭐")


if __name__ == "__main__":
    main()
