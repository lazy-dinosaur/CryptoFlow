#!/usr/bin/env python3
"""
2024-2025 데이터만 테스트 (ML 없음)
실행: python test_2024_2025.py
"""

import numpy as np
import pandas as pd
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import simulate_trade_with_optimal_exit
from tqdm import tqdm

def backtest(trades, label, initial_capital=10000):
    """백테스트 실행"""
    capital = initial_capital
    risk_pct = 0.015  # 1.5% 리스크
    max_leverage = 15
    fee_pct = 0.0004  # 0.04% 수수료

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0

    for t in trades:
        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position = capital * leverage

        pnl = position * t['pnl_tp2']
        fees = position * fee_pct * 2
        net = pnl - fees

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

        if capital <= 0:
            print(f"  BANKRUPT!")
            break

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / initial_capital - 1) * 100

    print(f"\n{label}:")
    print(f"  Trades: {total}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Return: {ret:+,.1f}%")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")

    return capital, wr, total


def main():
    print("=" * 60)
    print("  2024-2025 데이터 테스트 (ML 없음)")
    print("=" * 60)

    # 데이터 로드
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')

    # 2024-2025만 필터링
    df_1h = df_1h[df_1h.index.year >= 2024]
    df_15m = df_15m[df_15m.index.year >= 2024]

    print(f"  1H candles: {len(df_1h)}")
    print(f"  15m candles: {len(df_15m)}")
    print(f"  Date range: {df_15m.index.min()} ~ {df_15m.index.max()}")

    # 채널 구축
    print("\nBuilding channels...")
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    # 파라미터
    sl_buffer_pct = 0.002  # 0.2%
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()
    trade_data_list = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    print("\nScanning for trades...")
    for i in tqdm(range(50, len(df_15m) - 50)):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx)
        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Fakeout 체크
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
                        exit_data, _ = simulate_trade_with_optimal_exit(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )
                        trade_data_list.append({
                            **exit_data,
                            'timestamp': df_15m.index[i],
                            'setup': 'FAKEOUT'
                        })
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        exit_data, _ = simulate_trade_with_optimal_exit(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                            f_channel, fakeout_signal.extreme, 'FAKEOUT'
                        )
                        trade_data_list.append({
                            **exit_data,
                            'timestamp': df_15m.index[i],
                            'setup': 'FAKEOUT'
                        })
                        traded_entries.add(trade_key)

        # Bounce 체크
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 10)
        if trade_key in traded_entries:
            continue

        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry = current_close
            sl = channel.support * (1 - sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                exit_data, _ = simulate_trade_with_optimal_exit(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )
                trade_data_list.append({
                    **exit_data,
                    'timestamp': df_15m.index[i],
                    'setup': 'BOUNCE'
                })
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                exit_data, _ = simulate_trade_with_optimal_exit(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2,
                    channel, None, 'BOUNCE'
                )
                trade_data_list.append({
                    **exit_data,
                    'timestamp': df_15m.index[i],
                    'setup': 'BOUNCE'
                })
                traded_entries.add(trade_key)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  TRADE SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal trades: {len(trade_data_list)}")

    bounce_trades = [t for t in trade_data_list if t['setup'] == 'BOUNCE']
    fakeout_trades = [t for t in trade_data_list if t['setup'] == 'FAKEOUT']
    print(f"  BOUNCE: {len(bounce_trades)}")
    print(f"  FAKEOUT: {len(fakeout_trades)}")

    by_year = {}
    for t in trade_data_list:
        y = t['timestamp'].year
        by_year[y] = by_year.get(y, 0) + 1
    print(f"  By year: {by_year}")

    # 백테스트
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS ($10,000 initial)")
    print(f"{'='*60}")

    backtest(trade_data_list, "ALL (BOUNCE + FAKEOUT)")
    backtest(bounce_trades, "BOUNCE only")
    backtest(fakeout_trades, "FAKEOUT only")

    # 연도별
    trades_2024 = [t for t in trade_data_list if t['timestamp'].year == 2024]
    trades_2025 = [t for t in trade_data_list if t['timestamp'].year == 2025]
    backtest(trades_2024, "2024 only")
    backtest(trades_2025, "2025 only")


if __name__ == "__main__":
    main()
