#!/usr/bin/env python3
"""
ML Exit Management Test
- Train: 2022-2023
- Test: 2024-2025
- ML이 TP1/TP2/TP3 중 최적 출구 결정

실행: python ml_test_exit.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import (
    simulate_trade_with_optimal_exit, extract_features,
    EXIT_TP1, EXIT_TP2, HOLD_TP3, EXIT_NAMES
)
from ml_entry import simulate_trade_for_entry_label
from tqdm import tqdm


def collect_trades(df_1h, df_15m):
    """Collect all trade data with features."""
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()
    features_list = []
    trade_data_list = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    for i in tqdm(range(50, len(df_15m) - 200), desc='Collecting trades'):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Fakeout
        fakeout_signal = htf_fakeout_map.get(htf_idx)
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2
            trade_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)

            if trade_key not in traded_entries:
                if fakeout_signal.type == 'bear':
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 - sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        features = extract_features(df_15m, i, f_channel, 'LONG', 'FAKEOUT', fakeout_signal.extreme)
                        exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'timestamp': df_15m.index[i],
                            'exit_label': optimal_exit
                        })
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        features = extract_features(df_15m, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme)
                        exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )

                        features_list.append(features)
                        trade_data_list.append({
                            **exit_data,
                            'timestamp': df_15m.index[i],
                            'exit_label': optimal_exit
                        })
                        traded_entries.add(trade_key)

        # Bounce
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 10)
        if trade_key in traded_entries:
            continue

        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry = current_close
            sl = channel.support * (1 - sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                features = extract_features(df_15m, i, channel, 'LONG', 'BOUNCE', None)
                exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'timestamp': df_15m.index[i],
                    'exit_label': optimal_exit
                })
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                features = extract_features(df_15m, i, channel, 'SHORT', 'BOUNCE', None)
                exit_data, optimal_exit = simulate_trade_with_optimal_exit(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )

                features_list.append(features)
                trade_data_list.append({
                    **exit_data,
                    'timestamp': df_15m.index[i],
                    'exit_label': optimal_exit
                })
                traded_entries.add(trade_key)

    return features_list, trade_data_list


def features_to_array(features_list):
    """Convert features to numpy array."""
    return np.array([[
        f.channel_width_pct, f.support_touches, f.resistance_touches, f.total_touches,
        f.price_in_channel_pct, f.volume_ratio, f.delta_ratio, f.cvd_recent,
        f.volume_ma_20, f.delta_ma_20, f.atr_14, f.atr_ratio,
        f.momentum_5, f.momentum_20, f.rsi_14, f.is_bounce, f.is_long,
        f.body_size_pct, f.wick_ratio, f.is_bullish, f.hour, f.day_of_week,
        f.fakeout_depth_pct
    ] for f in features_list])


def backtest(trades, exit_preds, label):
    """Run backtest with exit strategy."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0
    trade_returns = []

    pnl_keys = {EXIT_TP1: 'pnl_tp1', EXIT_TP2: 'pnl_tp2', HOLD_TP3: 'pnl_tp3'}

    for trade, exit_pred in zip(trades, exit_preds):
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position = capital * leverage

        pnl_key = pnl_keys[exit_pred]
        pnl = position * trade[pnl_key]
        fees = position * fee_pct * 2
        net = pnl - fees

        trade_returns.append(net / capital * 100)
        capital += net
        capital = max(capital, 0)

        if net > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / 10000 - 1) * 100
    trade_returns = np.array(trade_returns)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {total}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")
    print(f"  Return: {ret:+,.1f}%")
    if len(trade_returns) > 0:
        print(f"\n  매매당 수익: 평균 {trade_returns.mean():+.3f}%, 중앙값 {np.median(trade_returns):+.3f}%")

    return capital, wr, total


def main():
    print("="*60)
    print("  ML EXIT MANAGEMENT TEST")
    print("  Train: 2022-2023 | Test: 2024-2025")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')
    print(f"  1H: {len(df_1h)}, 15m: {len(df_15m)}")
    print(f"  Range: {df_15m.index.min()} ~ {df_15m.index.max()}")

    # Collect trades
    print("\nCollecting trades...")
    features_list, trade_data_list = collect_trades(df_1h, df_15m)
    print(f"  Total trades: {len(trade_data_list)}")

    # Split IS/OOS
    years = np.array([t['timestamp'].year for t in trade_data_list])
    is_mask = np.isin(years, [2022, 2023])
    oos_mask = np.isin(years, [2024, 2025])

    print(f"\n  IS (2022-2023): {is_mask.sum()} trades")
    print(f"  OOS (2024-2025): {oos_mask.sum()} trades")

    # Prepare data
    X = features_to_array(features_list)
    exit_labels = np.array([t['exit_label'] for t in trade_data_list])

    X_is = X[is_mask]
    y_is = exit_labels[is_mask]

    X_oos = X[oos_mask]
    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask[i]]

    # Train Exit Model
    print("\n" + "="*60)
    print("  Training Exit Model on IS (2022-2023)")
    print("="*60)

    scaler = StandardScaler()
    X_is_scaled = scaler.fit_transform(X_is)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_is_scaled, y_is)

    print(f"  Training samples: {len(X_is)}")
    print(f"  EXIT_TP1: {(y_is == EXIT_TP1).sum()} ({(y_is == EXIT_TP1).mean()*100:.1f}%)")
    print(f"  EXIT_TP2: {(y_is == EXIT_TP2).sum()} ({(y_is == EXIT_TP2).mean()*100:.1f}%)")
    print(f"  HOLD_TP3: {(y_is == HOLD_TP3).sum()} ({(y_is == HOLD_TP3).mean()*100:.1f}%)")
    print(f"  Training Accuracy: {model.score(X_is_scaled, y_is):.3f}")

    # Predict on OOS
    X_oos_scaled = scaler.transform(X_oos)
    exit_preds = model.predict(X_oos_scaled)

    # Test Results
    print("\n" + "="*60)
    print("  OOS RESULTS (2024-2025)")
    print("="*60)

    # Baseline strategies
    baseline_tp1 = np.full(len(oos_trades), EXIT_TP1)
    baseline_tp2 = np.full(len(oos_trades), EXIT_TP2)
    baseline_tp3 = np.full(len(oos_trades), HOLD_TP3)

    backtest(oos_trades, baseline_tp1, "Baseline: Always TP1 (100% exit)")
    backtest(oos_trades, baseline_tp2, "Baseline: Always TP2 (50%+50%)")
    backtest(oos_trades, baseline_tp3, "Baseline: Always TP3 (33%+33%+34%)")

    # ML Exit
    backtest(oos_trades, exit_preds, "ML Exit (Dynamic TP selection)")

    # Show exit distribution
    print("\n" + "="*60)
    print("  ML Exit Predictions Distribution")
    print("="*60)
    print(f"  EXIT_TP1: {(exit_preds == EXIT_TP1).sum()} ({(exit_preds == EXIT_TP1).mean()*100:.1f}%)")
    print(f"  EXIT_TP2: {(exit_preds == EXIT_TP2).sum()} ({(exit_preds == EXIT_TP2).mean()*100:.1f}%)")
    print(f"  HOLD_TP3: {(exit_preds == HOLD_TP3).sum()} ({(exit_preds == HOLD_TP3).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
