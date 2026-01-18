#!/usr/bin/env python3
"""
ML Full Exit Test
- TP1: 100% exit at TP1
- TP2: 100% exit at TP2 (hold past TP1)
- TP3: 100% exit at TP3 (hold past TP1, TP2)

ML decides which single target to aim for.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import extract_features
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP
from tqdm import tqdm


# Full exit strategies
FULL_TP1 = 0  # 100% at TP1
FULL_TP2 = 1  # 100% at TP2
FULL_TP3 = 2  # 100% at TP3


def simulate_trade_full_exit(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price, channel):
    """Simulate trade with full exit at each target."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    # TP3 = channel edge
    if trade_type == 'LONG':
        tp3_price = channel.resistance * 0.998
    else:
        tp3_price = channel.support * 1.002
    reward3 = abs(tp3_price - entry_price)

    is_long = trade_type == 'LONG'

    hit_sl = False
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False

    # Check outcomes
    for j in range(idx + 1, min(idx + 200, len(candles))):
        if is_long:
            if lows[j] <= sl_price:
                hit_sl = True
                break
            if highs[j] >= tp1_price:
                hit_tp1 = True
            if highs[j] >= tp2_price:
                hit_tp2 = True
            if highs[j] >= tp3_price:
                hit_tp3 = True
                break
        else:
            if highs[j] >= sl_price:
                hit_sl = True
                break
            if lows[j] <= tp1_price:
                hit_tp1 = True
            if lows[j] <= tp2_price:
                hit_tp2 = True
            if lows[j] <= tp3_price:
                hit_tp3 = True
                break

    # Calculate PnL for each full exit strategy
    pnl_full_tp1 = 0
    pnl_full_tp2 = 0
    pnl_full_tp3 = 0

    if hit_sl:
        pnl_full_tp1 = pnl_full_tp2 = pnl_full_tp3 = -risk / entry_price
    else:
        # FULL_TP1: Exit 100% at TP1
        if hit_tp1:
            pnl_full_tp1 = reward1 / entry_price
        else:
            pnl_full_tp1 = -risk / entry_price  # Eventually hit SL

        # FULL_TP2: Hold past TP1, exit 100% at TP2
        if hit_tp2:
            pnl_full_tp2 = reward2 / entry_price
        elif hit_tp1:
            pnl_full_tp2 = -risk / entry_price  # Hit TP1 but not TP2, eventually SL
        else:
            pnl_full_tp2 = -risk / entry_price

        # FULL_TP3: Hold until TP3
        if hit_tp3:
            pnl_full_tp3 = reward3 / entry_price
        elif hit_tp2:
            pnl_full_tp3 = -risk / entry_price
        elif hit_tp1:
            pnl_full_tp3 = -risk / entry_price
        else:
            pnl_full_tp3 = -risk / entry_price

    # Determine optimal strategy
    pnls = {FULL_TP1: pnl_full_tp1, FULL_TP2: pnl_full_tp2, FULL_TP3: pnl_full_tp3}
    optimal = max(pnls, key=pnls.get)

    return {
        'idx': idx,
        'type': trade_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'tp3': tp3_price,
        'pnl_full_tp1': pnl_full_tp1,
        'pnl_full_tp2': pnl_full_tp2,
        'pnl_full_tp3': pnl_full_tp3,
        'optimal_full': optimal,
        'hit_tp1': hit_tp1,
        'hit_tp2': hit_tp2,
        'hit_tp3': hit_tp3,
        'hit_sl': hit_sl
    }, optimal


def collect_trades(df_1h, df_15m):
    """Collect trades with full exit PnL."""
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

    for i in tqdm(range(50, len(df_15m) - 200), desc='Collecting'):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Fakeout
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)  # Fix lookahead bias
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
                        trade_data, optimal = simulate_trade_full_exit(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2, f_channel
                        )
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'LONG', entry, sl, tp1, tp2)
                        trade_data['timestamp'] = df_15m.index[i]
                        trade_data['entry_label'] = TAKE if is_win else SKIP

                        features_list.append(features)
                        trade_data_list.append(trade_data)
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        features = extract_features(df_15m, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme)
                        trade_data, optimal = simulate_trade_full_exit(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2, f_channel
                        )
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)
                        trade_data['timestamp'] = df_15m.index[i]
                        trade_data['entry_label'] = TAKE if is_win else SKIP

                        features_list.append(features)
                        trade_data_list.append(trade_data)
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
                trade_data, optimal = simulate_trade_full_exit(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2, channel
                )
                is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'LONG', entry, sl, tp1, tp2)
                trade_data['timestamp'] = df_15m.index[i]
                trade_data['entry_label'] = TAKE if is_win else SKIP

                features_list.append(features)
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                features = extract_features(df_15m, i, channel, 'SHORT', 'BOUNCE', None)
                trade_data, optimal = simulate_trade_full_exit(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2, channel
                )
                is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)
                trade_data['timestamp'] = df_15m.index[i]
                trade_data['entry_label'] = TAKE if is_win else SKIP

                features_list.append(features)
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

    return features_list, trade_data_list


def features_to_array(features_list):
    return np.array([[
        f.channel_width_pct, f.support_touches, f.resistance_touches, f.total_touches,
        f.price_in_channel_pct, f.volume_ratio, f.delta_ratio, f.cvd_recent,
        f.volume_ma_20, f.delta_ma_20, f.atr_14, f.atr_ratio,
        f.momentum_5, f.momentum_20, f.rsi_14, f.is_bounce, f.is_long,
        f.body_size_pct, f.wick_ratio, f.is_bullish, f.hour, f.day_of_week,
        f.fakeout_depth_pct
    ] for f in features_list])


def backtest(trades, entry_preds, exit_preds, label):
    """Backtest with full exit strategy."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0005  # 0.05%

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0
    trades_taken = 0
    trade_returns = []

    pnl_keys = {FULL_TP1: 'pnl_full_tp1', FULL_TP2: 'pnl_full_tp2', FULL_TP3: 'pnl_full_tp3'}

    for trade, take_trade, exit_pred in zip(trades, entry_preds, exit_preds):
        if take_trade == SKIP:
            continue

        trades_taken += 1
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
    print(f"  Trades: {trades_taken}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")
    print(f"  Return: {ret:+,.1f}%")
    if len(trade_returns) > 0:
        print(f"  매매당: 평균 {trade_returns.mean():+.3f}%, 중앙값 {np.median(trade_returns):+.3f}%")

    return capital, wr, max_dd


def main():
    print("="*60)
    print("  FULL EXIT STRATEGY TEST")
    print("  TP1/TP2/TP3 각각 100% 종료")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')

    # Collect trades
    print("\nCollecting trades...")
    features_list, trade_data_list = collect_trades(df_1h, df_15m)
    print(f"  Total: {len(trade_data_list)}")

    # Split
    years = np.array([t['timestamp'].year for t in trade_data_list])
    is_mask = np.isin(years, [2022, 2023])
    oos_mask = np.isin(years, [2024])

    print(f"  IS (2022-2023): {is_mask.sum()}")
    print(f"  OOS (2024): {oos_mask.sum()}")

    X = features_to_array(features_list)
    entry_labels = np.array([t['entry_label'] for t in trade_data_list])
    exit_labels = np.array([t['optimal_full'] for t in trade_data_list])

    X_is = X[is_mask]
    entry_labels_is = entry_labels[is_mask]
    exit_labels_is = exit_labels[is_mask]

    X_oos = X[oos_mask]
    oos_trades = [trade_data_list[i] for i in range(len(trade_data_list)) if oos_mask[i]]

    # Train Entry Model
    print("\n" + "="*60)
    print("  Training Entry Model")
    print("="*60)
    entry_scaler = StandardScaler()
    X_is_scaled = entry_scaler.fit_transform(X_is)

    entry_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    entry_model.fit(X_is_scaled, entry_labels_is)
    print(f"  Accuracy: {entry_model.score(X_is_scaled, entry_labels_is):.3f}")

    # Train Full Exit Model
    print("\n" + "="*60)
    print("  Training Full Exit Model")
    print("="*60)
    exit_scaler = StandardScaler()
    X_is_exit_scaled = exit_scaler.fit_transform(X_is)

    exit_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    exit_model.fit(X_is_exit_scaled, exit_labels_is)

    tp1_cnt = (exit_labels_is == FULL_TP1).sum()
    tp2_cnt = (exit_labels_is == FULL_TP2).sum()
    tp3_cnt = (exit_labels_is == FULL_TP3).sum()
    print(f"  FULL_TP1: {tp1_cnt} ({tp1_cnt/len(exit_labels_is)*100:.1f}%)")
    print(f"  FULL_TP2: {tp2_cnt} ({tp2_cnt/len(exit_labels_is)*100:.1f}%)")
    print(f"  FULL_TP3: {tp3_cnt} ({tp3_cnt/len(exit_labels_is)*100:.1f}%)")
    print(f"  Accuracy: {exit_model.score(X_is_exit_scaled, exit_labels_is):.3f}")

    # Predict
    X_oos_scaled = entry_scaler.transform(X_oos)
    X_oos_exit_scaled = exit_scaler.transform(X_oos)

    entry_probs = entry_model.predict_proba(X_oos_scaled)[:, 1]
    entry_preds = (entry_probs >= 0.7).astype(int)
    exit_preds = exit_model.predict(X_oos_exit_scaled)

    # Test
    print("\n" + "="*60)
    print("  OOS RESULTS (2024)")
    print("="*60)

    baseline_entry = np.ones(len(oos_trades), dtype=int)

    # Fixed strategies
    backtest(oos_trades, baseline_entry, np.full(len(oos_trades), FULL_TP1), "No ML + TP1 (100%)")
    backtest(oos_trades, baseline_entry, np.full(len(oos_trades), FULL_TP2), "No ML + TP2 (100%)")
    backtest(oos_trades, baseline_entry, np.full(len(oos_trades), FULL_TP3), "No ML + TP3 (100%)")

    # ML Entry + Fixed exit
    backtest(oos_trades, entry_preds, np.full(len(oos_trades), FULL_TP1), "ML 0.7 + TP1 (100%)")
    backtest(oos_trades, entry_preds, np.full(len(oos_trades), FULL_TP2), "ML 0.7 + TP2 (100%)")
    backtest(oos_trades, entry_preds, np.full(len(oos_trades), FULL_TP3), "ML 0.7 + TP3 (100%)")

    # ML Entry + ML Exit (dynamic)
    backtest(oos_trades, entry_preds, exit_preds, "ML 0.7 + ML Exit (동적)")


if __name__ == "__main__":
    main()
