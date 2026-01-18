#!/usr/bin/env python3
"""
ML 필터 vs 비필터 비교 - 매매 수 고려

핵심: 매매 수 줄어드는 것 대비 WR/수익 개선 효과
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def extract_wave_features(ltf_candles, idx, channel, direction):
    """파동 패턴 피처 추출."""
    closes = ltf_candles['close'].values
    highs = ltf_candles['high'].values
    lows = ltf_candles['low'].values
    volumes = ltf_candles['volume'].values
    deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

    range_lookback = 20
    start = max(0, idx - range_lookback)
    range_volumes = volumes[start:idx]
    range_deltas = deltas[start:idx]

    range_avg_vol = np.mean(range_volumes) if len(range_volumes) > 0 else 1
    range_avg_delta = np.mean(range_deltas) if len(range_deltas) > 0 else 0
    range_avg_abs_delta = np.mean(np.abs(range_deltas)) if len(range_deltas) > 0 else 1
    range_vol_std = np.std(range_volumes) if len(range_volumes) > 1 else 1

    wave_lookback = 5
    wave_start = max(0, idx - wave_lookback)
    wave_volumes = volumes[wave_start:idx+1]
    wave_deltas = deltas[wave_start:idx+1]

    if len(wave_volumes) >= 2:
        vol_slope = np.polyfit(range(len(wave_volumes)), wave_volumes, 1)[0]
        vol_trend = vol_slope / (range_avg_vol + 1e-10)
    else:
        vol_trend = 0

    if len(wave_deltas) >= 2:
        delta_slope = np.polyfit(range(len(wave_deltas)), wave_deltas, 1)[0]
        delta_trend = delta_slope / (range_avg_abs_delta + 1e-10)
    else:
        delta_trend = 0

    if len(wave_volumes) >= 3:
        vol_diff = np.diff(wave_volumes)
        vol_accel = np.mean(np.diff(vol_diff)) if len(vol_diff) >= 2 else 0
    else:
        vol_accel = 0

    if len(wave_deltas) >= 3:
        delta_diff = np.diff(wave_deltas)
        delta_accel = np.mean(np.diff(delta_diff)) if len(delta_diff) >= 2 else 0
    else:
        delta_accel = 0

    touch_vol = volumes[idx]
    touch_delta = deltas[idx]

    vol_vs_range = touch_vol / (range_avg_vol + 1e-10)
    delta_vs_range = touch_delta / (range_avg_abs_delta + 1e-10)
    vol_zscore = (touch_vol - range_avg_vol) / (range_vol_std + 1e-10)

    wave_avg_vol = np.mean(wave_volumes[:-1]) if len(wave_volumes) > 1 else touch_vol
    wave_avg_delta = np.mean(wave_deltas[:-1]) if len(wave_deltas) > 1 else touch_delta

    vol_vs_wave = touch_vol / (wave_avg_vol + 1e-10)
    delta_vs_wave = touch_delta / (np.abs(wave_avg_delta) + 1e-10) if wave_avg_delta != 0 else 0

    if direction == 'LONG':
        delta_aligned = 1 if touch_delta > 0 else 0
        delta_trend_aligned = 1 if delta_trend > 0 else 0
    else:
        delta_aligned = 1 if touch_delta < 0 else 0
        delta_trend_aligned = 1 if delta_trend < 0 else 0

    vol_spike = 1 if vol_vs_range >= 2.0 else 0
    vol_low = 1 if vol_vs_range <= 0.5 else 0
    delta_reversal = 1 if (wave_avg_delta < 0 and touch_delta > 0) or (wave_avg_delta > 0 and touch_delta < 0) else 0

    cvd_wave = np.sum(wave_deltas)
    cvd_range = np.sum(range_deltas)

    open_price = ltf_candles['open'].values[idx]
    close_price = closes[idx]
    candle_body = close_price - open_price
    candle_range = highs[idx] - lows[idx]
    body_ratio = abs(candle_body) / (candle_range + 1e-10)
    is_bullish = 1 if close_price > open_price else 0

    if direction == 'LONG':
        wick_ratio = (min(open_price, close_price) - lows[idx]) / (candle_range + 1e-10)
    else:
        wick_ratio = (highs[idx] - max(open_price, close_price)) / (candle_range + 1e-10)

    return {
        'vol_vs_range': vol_vs_range, 'delta_vs_range': delta_vs_range, 'vol_zscore': vol_zscore,
        'vol_trend': vol_trend, 'delta_trend': delta_trend,
        'vol_accel': vol_accel / (range_avg_vol + 1e-10),
        'delta_accel': delta_accel / (range_avg_abs_delta + 1e-10),
        'vol_vs_wave': vol_vs_wave, 'delta_vs_wave': delta_vs_wave,
        'delta_aligned': delta_aligned, 'delta_trend_aligned': delta_trend_aligned,
        'vol_spike': vol_spike, 'vol_low': vol_low, 'delta_reversal': delta_reversal,
        'cvd_wave': cvd_wave / (range_avg_abs_delta * wave_lookback + 1e-10),
        'cvd_range': cvd_range / (range_avg_abs_delta * range_lookback + 1e-10),
        'body_ratio': body_ratio, 'is_bullish': is_bullish, 'wick_ratio': wick_ratio,
        'is_long': 1 if direction == 'LONG' else 0,
    }


def simulate_trade_full(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """TP1 50% + TP2 50%, BE 적용."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry
            if highs[j] >= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:
                        return 'TP1_BE', 0.5 * (tp1 - entry) / entry
                    if highs[k] >= tp2:
                        return 'TP1_TP2', 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                return 'TP1_BE', 0.5 * (tp1 - entry) / entry
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return 'TP1_BE', 0.5 * (entry - tp1) / entry
                    if lows[k] <= tp2:
                        return 'TP1_TP2', 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
                return 'TP1_BE', 0.5 * (entry - tp1) / entry
    return None, 0


def collect_data(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """데이터 수집 with full trade simulation."""
    data = []
    traded_keys = set()

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    sl_buffer = 0.0008
    touch_threshold = 0.003

    for i in range(100, len(ltf_candles) - 200):
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx - 1)

        if not channel:
            continue

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        mid = (channel.resistance + channel.support) / 2

        bounce_key = (round(channel.support), round(channel.resistance), i // 20)
        if bounce_key in traded_keys:
            continue

        # Support touch → LONG
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                features = extract_wave_features(ltf_candles, i, channel, 'LONG')
                result, pnl = simulate_trade_full(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)

                if result:
                    features['result'] = result
                    features['pnl'] = pnl
                    features['success'] = 1 if pnl > 0 else 0
                    features['direction'] = 'LONG'
                    features['idx'] = i
                    features['entry'] = entry
                    features['sl'] = sl
                    features['tp1'] = tp1
                    features['tp2'] = tp2
                    data.append(features)
                    traded_keys.add(bounce_key)

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                features = extract_wave_features(ltf_candles, i, channel, 'SHORT')
                result, pnl = simulate_trade_full(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)

                if result:
                    features['result'] = result
                    features['pnl'] = pnl
                    features['success'] = 1 if pnl > 0 else 0
                    features['direction'] = 'SHORT'
                    features['idx'] = i
                    features['entry'] = entry
                    features['sl'] = sl
                    features['tp1'] = tp1
                    features['tp2'] = tp2
                    data.append(features)
                    traded_keys.add(bounce_key)

    return pd.DataFrame(data)


def backtest_trades(df, label):
    """백테스트 - 실제 수익 계산."""
    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    total_trades = len(df)
    if total_trades == 0:
        return None

    wins = 0
    losses = 0
    total_pnl = 0
    trade_pnls = []

    for _, row in df.iterrows():
        sl_dist = abs(row['entry'] - row['sl']) / row['entry']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1
        position = capital * lev

        pnl = row['pnl']
        net_pnl = position * pnl - position * fee_pct * 2
        trade_pnls.append(net_pnl / capital * 100)
        total_pnl += net_pnl

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    wr = wins / total_trades * 100
    avg_pnl = np.mean(trade_pnls) if trade_pnls else 0
    total_return = total_pnl / 10000 * 100

    return {
        'label': label,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'avg_pnl': avg_pnl,
        'total_return': total_return,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   ML 필터 vs 비필터 비교                                          ║
║   핵심: 매매 수 감소 vs WR/수익 개선                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    data_by_year = {}
    for year in [2022, 2023, 2024, 2025]:
        htf = htf_all[htf_all.index.year == year]
        ltf = ltf_all[ltf_all.index.year == year]
        if len(htf) > 100:
            data_by_year[year] = {'htf': htf, 'ltf': ltf}

    # Train/Test
    htf_train = pd.concat([data_by_year[y]['htf'] for y in [2022, 2023] if y in data_by_year])
    ltf_train = pd.concat([data_by_year[y]['ltf'] for y in [2022, 2023] if y in data_by_year])
    htf_test = pd.concat([data_by_year[y]['htf'] for y in [2024, 2025] if y in data_by_year])
    ltf_test = pd.concat([data_by_year[y]['ltf'] for y in [2024, 2025] if y in data_by_year])

    print(f"  Train: 2022-2023, Test: 2024-2025")

    # Build channels
    print("Building channels...")
    channels_train, _ = build_htf_channels(htf_train)
    channels_test, _ = build_htf_channels(htf_test)

    # Collect data
    print("Collecting data...")
    train_df = collect_data(htf_train, ltf_train, channels_train)
    test_df = collect_data(htf_test, ltf_test, channels_test)

    print(f"  Train: {len(train_df)} trades")
    print(f"  Test: {len(test_df)} trades")

    # Train ML model
    feature_cols = [
        'vol_vs_range', 'delta_vs_range', 'vol_zscore',
        'vol_trend', 'delta_trend', 'vol_accel', 'delta_accel',
        'vol_vs_wave', 'delta_vs_wave',
        'delta_aligned', 'delta_trend_aligned',
        'vol_spike', 'vol_low', 'delta_reversal',
        'cvd_wave', 'cvd_range',
        'body_ratio', 'is_bullish', 'wick_ratio',
        'is_long',
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['success'].values
    X_test = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10,
        random_state=42, class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # ===== 비교 분석 =====
    print("\n" + "="*90)
    print("  전체 비교 (Test 2024-2025)")
    print("="*90)
    print(f"\n  {'필터':<20} | {'매매수':>6} | {'유지율':>7} | {'WR':>7} | {'Avg PnL':>8} | {'총수익':>10}")
    print("-"*90)

    baseline = backtest_trades(test_df, "ML 없음 (전체)")
    print(f"  {'ML 없음 (전체)':<20} | {baseline['trades']:>6} | {'100.0%':>7} | {baseline['wr']:>6.1f}% | {baseline['avg_pnl']:>+7.2f}% | {baseline['total_return']:>+9.1f}%")

    for threshold in [0.4, 0.5, 0.6, 0.7]:
        mask = test_proba >= threshold
        filtered_df = test_df[mask]
        if len(filtered_df) >= 10:
            result = backtest_trades(filtered_df, f"ML p>={threshold}")
            retention = len(filtered_df) / len(test_df) * 100
            print(f"  {f'ML p>={threshold}':<20} | {result['trades']:>6} | {retention:>6.1f}% | {result['wr']:>6.1f}% | {result['avg_pnl']:>+7.2f}% | {result['total_return']:>+9.1f}%")

    # ===== LONG만 =====
    print("\n" + "="*90)
    print("  LONG만 비교")
    print("="*90)
    print(f"\n  {'필터':<20} | {'매매수':>6} | {'유지율':>7} | {'WR':>7} | {'Avg PnL':>8} | {'총수익':>10}")
    print("-"*90)

    long_df = test_df[test_df['direction'] == 'LONG']
    long_proba = test_proba[test_df['direction'] == 'LONG']

    baseline_long = backtest_trades(long_df, "ML 없음 (LONG)")
    print(f"  {'ML 없음 (LONG)':<20} | {baseline_long['trades']:>6} | {'100.0%':>7} | {baseline_long['wr']:>6.1f}% | {baseline_long['avg_pnl']:>+7.2f}% | {baseline_long['total_return']:>+9.1f}%")

    for threshold in [0.4, 0.5, 0.6, 0.7]:
        mask = long_proba >= threshold
        filtered_df = long_df[mask]
        if len(filtered_df) >= 5:
            result = backtest_trades(filtered_df, f"ML p>={threshold}")
            retention = len(filtered_df) / len(long_df) * 100
            print(f"  {f'ML p>={threshold}':<20} | {result['trades']:>6} | {retention:>6.1f}% | {result['wr']:>6.1f}% | {result['avg_pnl']:>+7.2f}% | {result['total_return']:>+9.1f}%")

    # ===== SHORT만 =====
    print("\n" + "="*90)
    print("  SHORT만 비교")
    print("="*90)
    print(f"\n  {'필터':<20} | {'매매수':>6} | {'유지율':>7} | {'WR':>7} | {'Avg PnL':>8} | {'총수익':>10}")
    print("-"*90)

    short_df = test_df[test_df['direction'] == 'SHORT']
    short_proba = test_proba[test_df['direction'] == 'SHORT']

    baseline_short = backtest_trades(short_df, "ML 없음 (SHORT)")
    print(f"  {'ML 없음 (SHORT)':<20} | {baseline_short['trades']:>6} | {'100.0%':>7} | {baseline_short['wr']:>6.1f}% | {baseline_short['avg_pnl']:>+7.2f}% | {baseline_short['total_return']:>+9.1f}%")

    for threshold in [0.4, 0.5, 0.6, 0.7]:
        mask = short_proba >= threshold
        filtered_df = short_df[mask]
        if len(filtered_df) >= 5:
            result = backtest_trades(filtered_df, f"ML p>={threshold}")
            retention = len(filtered_df) / len(short_df) * 100
            print(f"  {f'ML p>={threshold}':<20} | {result['trades']:>6} | {retention:>6.1f}% | {result['wr']:>6.1f}% | {result['avg_pnl']:>+7.2f}% | {result['total_return']:>+9.1f}%")

    # ===== 연도별 =====
    print("\n" + "="*90)
    print("  연도별 비교 (2024 vs 2025)")
    print("="*90)

    for year in [2024, 2025]:
        if year not in data_by_year:
            continue

        htf_year = data_by_year[year]['htf']
        ltf_year = data_by_year[year]['ltf']
        channels_year, _ = build_htf_channels(htf_year)
        year_df = collect_data(htf_year, ltf_year, channels_year)

        if len(year_df) == 0:
            continue

        X_year = scaler.transform(year_df[feature_cols].values)
        year_proba = model.predict_proba(X_year)[:, 1]

        print(f"\n  [{year}]")
        print(f"  {'필터':<20} | {'매매수':>6} | {'유지율':>7} | {'WR':>7} | {'Avg PnL':>8} | {'총수익':>10}")
        print("-"*90)

        baseline_year = backtest_trades(year_df, "ML 없음")
        print(f"  {'ML 없음':<20} | {baseline_year['trades']:>6} | {'100.0%':>7} | {baseline_year['wr']:>6.1f}% | {baseline_year['avg_pnl']:>+7.2f}% | {baseline_year['total_return']:>+9.1f}%")

        for threshold in [0.5, 0.6]:
            mask = year_proba >= threshold
            filtered_df = year_df[mask]
            if len(filtered_df) >= 5:
                result = backtest_trades(filtered_df, f"ML p>={threshold}")
                retention = len(filtered_df) / len(year_df) * 100
                print(f"  {f'ML p>={threshold}':<20} | {result['trades']:>6} | {retention:>6.1f}% | {result['wr']:>6.1f}% | {result['avg_pnl']:>+7.2f}% | {result['total_return']:>+9.1f}%")

    # Summary
    print("\n" + "="*90)
    print("  💡 결론")
    print("="*90)
    print("""
  핵심 질문: 매매 수 줄어드는 것 대비 개선 효과가 있는가?

  비교 포인트:
  1. 유지율 (매매 수 감소율)
  2. WR 개선폭
  3. 총수익 변화 (가장 중요!)
""")


if __name__ == "__main__":
    main()
