#!/usr/bin/env python3
"""
Volume + Delta 기반 필터 테스트 (ML 없이)

규칙:
- 볼륨 낮음 + 델타 낮음 → BOUNCE 진입 OK (레인지 지속)
- 볼륨 낮음 + 델타 높음 → 주의 (물량 쌓임, fakeout 가능)
- 볼륨 높음 → SKIP (브레이크아웃 가능성)
"""

import numpy as np
import pandas as pd
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import simulate_trade_with_optimal_exit, extract_features
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP
from tqdm import tqdm


def collect_trades_with_vd(df_1h, df_15m):
    """Collect trades with volume/delta info."""
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()
    trade_data_list = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values
    ltf_volumes = df_15m['volume'].values
    ltf_deltas = df_15m['delta'].values

    for i in tqdm(range(50, len(df_15m) - 200), desc='Collecting'):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        current_volume = ltf_volumes[i]
        current_delta = ltf_deltas[i]
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx)

        if not channel:
            continue

        # Calculate average volume/delta in recent range
        lookback = 20
        start_idx = max(0, i - lookback)
        avg_volume = np.mean(ltf_volumes[start_idx:i]) if i > start_idx else current_volume
        avg_delta = np.mean(np.abs(ltf_deltas[start_idx:i])) if i > start_idx else abs(current_delta)

        # Ratios
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        delta_ratio = abs(current_delta) / avg_delta if avg_delta > 0 else 1

        mid_price = (channel.resistance + channel.support) / 2

        # Fakeout signals
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
                        trade_data, _ = simulate_trade_with_optimal_exit(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )
                        trade_data['timestamp'] = df_15m.index[i]
                        trade_data['setup_type'] = 'FAKEOUT'
                        trade_data['volume_ratio'] = volume_ratio
                        trade_data['delta_ratio'] = delta_ratio
                        trade_data['volume_low'] = volume_ratio < 1.0
                        trade_data['delta_low'] = delta_ratio < 1.0
                        trade_data_list.append(trade_data)
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        trade_data, _ = simulate_trade_with_optimal_exit(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )
                        trade_data['timestamp'] = df_15m.index[i]
                        trade_data['setup_type'] = 'FAKEOUT'
                        trade_data['volume_ratio'] = volume_ratio
                        trade_data['delta_ratio'] = delta_ratio
                        trade_data['volume_low'] = volume_ratio < 1.0
                        trade_data['delta_low'] = delta_ratio < 1.0
                        trade_data_list.append(trade_data)
                        traded_entries.add(trade_key)

        # Bounce signals
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 10)
        if trade_key in traded_entries:
            continue

        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry = current_close
            sl = channel.support * (1 - sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                trade_data, _ = simulate_trade_with_optimal_exit(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )
                trade_data['timestamp'] = df_15m.index[i]
                trade_data['setup_type'] = 'BOUNCE'
                trade_data['volume_ratio'] = volume_ratio
                trade_data['delta_ratio'] = delta_ratio
                trade_data['volume_low'] = volume_ratio < 1.0
                trade_data['delta_low'] = delta_ratio < 1.0
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                trade_data, _ = simulate_trade_with_optimal_exit(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )
                trade_data['timestamp'] = df_15m.index[i]
                trade_data['setup_type'] = 'BOUNCE'
                trade_data['volume_ratio'] = volume_ratio
                trade_data['delta_ratio'] = delta_ratio
                trade_data['volume_low'] = volume_ratio < 1.0
                trade_data['delta_low'] = delta_ratio < 1.0
                trade_data_list.append(trade_data)
                traded_entries.add(trade_key)

    return trade_data_list


def backtest(trades, label):
    """Run backtest."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0005

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0
    trade_returns = []

    for trade in trades:
        sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position = capital * leverage

        pnl = position * trade['pnl_tp2']  # 반익반익
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
    trade_returns = np.array(trade_returns)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {total}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")
    if len(trade_returns) > 0:
        print(f"  매매당: 평균 {trade_returns.mean():+.3f}%, 중앙값 {np.median(trade_returns):+.3f}%")

    return total, wr, max_dd


def main():
    print("="*60)
    print("  VOLUME + DELTA 필터 테스트 (No ML)")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')

    # Collect
    print("\nCollecting trades...")
    all_trades = collect_trades_with_vd(df_1h, df_15m)
    print(f"  Total: {len(all_trades)}")

    # Filter for 2024
    trades_2024 = [t for t in all_trades if t['timestamp'].year == 2024]
    print(f"  2024: {len(trades_2024)}")

    # Split by setup type
    bounce_trades = [t for t in trades_2024 if t['setup_type'] == 'BOUNCE']
    fakeout_trades = [t for t in trades_2024 if t['setup_type'] == 'FAKEOUT']

    print(f"\n  BOUNCE: {len(bounce_trades)}")
    print(f"  FAKEOUT: {len(fakeout_trades)}")

    # Test different filters
    print("\n" + "="*60)
    print("  2024 RESULTS")
    print("="*60)

    # Baseline - all trades
    backtest(trades_2024, "1. Baseline (모든 신호)")

    # Volume low only
    vol_low = [t for t in trades_2024 if t['volume_low']]
    backtest(vol_low, "2. 볼륨 낮음만")

    # Delta low only
    delta_low = [t for t in trades_2024 if t['delta_low']]
    backtest(delta_low, "3. 델타 낮음만")

    # Volume low + Delta low (레인지 지속)
    vol_low_delta_low = [t for t in trades_2024 if t['volume_low'] and t['delta_low']]
    backtest(vol_low_delta_low, "4. 볼륨↓ + 델타↓ (레인지 지속)")

    # Volume low + Delta high (물량 쌓임)
    vol_low_delta_high = [t for t in trades_2024 if t['volume_low'] and not t['delta_low']]
    backtest(vol_low_delta_high, "5. 볼륨↓ + 델타↑ (덫 가능성)")

    # Volume high (브레이크아웃 가능)
    vol_high = [t for t in trades_2024 if not t['volume_low']]
    backtest(vol_high, "6. 볼륨↑ (브레이크아웃 주의)")

    # BOUNCE only with volume/delta filter
    print("\n" + "="*60)
    print("  BOUNCE ONLY")
    print("="*60)

    backtest(bounce_trades, "BOUNCE - 전체")
    bounce_filtered = [t for t in bounce_trades if t['volume_low'] and t['delta_low']]
    backtest(bounce_filtered, "BOUNCE - 볼륨↓ + 델타↓")

    # FAKEOUT only
    print("\n" + "="*60)
    print("  FAKEOUT ONLY")
    print("="*60)

    backtest(fakeout_trades, "FAKEOUT - 전체")
    fakeout_filtered = [t for t in fakeout_trades if t['volume_low']]
    backtest(fakeout_filtered, "FAKEOUT - 볼륨↓")


if __name__ == "__main__":
    main()
