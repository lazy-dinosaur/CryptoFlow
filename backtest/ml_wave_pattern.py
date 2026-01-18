#!/usr/bin/env python3
"""
ML 파동 패턴 분석 - 볼륨/델타 추세 학습

핵심 아이디어:
1. 레인지 내 평균 볼륨/델타 (기준선)
2. 터치로 오는 파동의 볼륨/델타 추세 (증가? 감소?)
3. 터치 캔들에서의 반응

학습: 2022-2023
테스트: 2024-2025
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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

    # === 1. 레인지 내 평균 (기준선) ===
    range_lookback = 20
    start = max(0, idx - range_lookback)

    range_volumes = volumes[start:idx]
    range_deltas = deltas[start:idx]

    range_avg_vol = np.mean(range_volumes) if len(range_volumes) > 0 else 1
    range_avg_delta = np.mean(range_deltas) if len(range_deltas) > 0 else 0
    range_avg_abs_delta = np.mean(np.abs(range_deltas)) if len(range_deltas) > 0 else 1
    range_vol_std = np.std(range_volumes) if len(range_volumes) > 1 else 1

    # === 2. 파동 추세 (터치 직전 5개 캔들) ===
    wave_lookback = 5
    wave_start = max(0, idx - wave_lookback)

    wave_volumes = volumes[wave_start:idx+1]
    wave_deltas = deltas[wave_start:idx+1]

    # 볼륨 추세 (기울기)
    if len(wave_volumes) >= 2:
        vol_slope = np.polyfit(range(len(wave_volumes)), wave_volumes, 1)[0]
        vol_trend = vol_slope / (range_avg_vol + 1e-10)  # 정규화
    else:
        vol_trend = 0

    # 델타 추세 (기울기)
    if len(wave_deltas) >= 2:
        delta_slope = np.polyfit(range(len(wave_deltas)), wave_deltas, 1)[0]
        delta_trend = delta_slope / (range_avg_abs_delta + 1e-10)
    else:
        delta_trend = 0

    # 볼륨/델타 가속도 (추세의 추세)
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

    # === 3. 터치 캔들 반응 ===
    touch_vol = volumes[idx]
    touch_delta = deltas[idx]

    # 레인지 대비
    vol_vs_range = touch_vol / (range_avg_vol + 1e-10)
    delta_vs_range = touch_delta / (range_avg_abs_delta + 1e-10)
    vol_zscore = (touch_vol - range_avg_vol) / (range_vol_std + 1e-10)

    # 파동 평균 대비
    wave_avg_vol = np.mean(wave_volumes[:-1]) if len(wave_volumes) > 1 else touch_vol
    wave_avg_delta = np.mean(wave_deltas[:-1]) if len(wave_deltas) > 1 else touch_delta

    vol_vs_wave = touch_vol / (wave_avg_vol + 1e-10)
    delta_vs_wave = touch_delta / (np.abs(wave_avg_delta) + 1e-10) if wave_avg_delta != 0 else 0

    # === 4. 방향 정렬 ===
    # LONG: 델타 양수가 좋음, SHORT: 델타 음수가 좋음
    if direction == 'LONG':
        delta_aligned = 1 if touch_delta > 0 else 0
        delta_trend_aligned = 1 if delta_trend > 0 else 0  # 델타 상승 추세
    else:
        delta_aligned = 1 if touch_delta < 0 else 0
        delta_trend_aligned = 1 if delta_trend < 0 else 0  # 델타 하락 추세

    # === 5. 추가 피처 ===
    # 볼륨 급증/급감
    vol_spike = 1 if vol_vs_range >= 2.0 else 0
    vol_low = 1 if vol_vs_range <= 0.5 else 0

    # 델타 반전 (파동과 반대 방향)
    delta_reversal = 1 if (wave_avg_delta < 0 and touch_delta > 0) or (wave_avg_delta > 0 and touch_delta < 0) else 0

    # CVD (누적 델타) 추세
    cvd_wave = np.sum(wave_deltas)
    cvd_range = np.sum(range_deltas)

    # 캔들 패턴
    open_price = ltf_candles['open'].values[idx]
    close_price = closes[idx]
    candle_body = close_price - open_price
    candle_range = highs[idx] - lows[idx]
    body_ratio = abs(candle_body) / (candle_range + 1e-10)

    is_bullish = 1 if close_price > open_price else 0

    # 꼬리 비율
    if direction == 'LONG':
        wick_ratio = (min(open_price, close_price) - lows[idx]) / (candle_range + 1e-10)  # 하단 꼬리
    else:
        wick_ratio = (highs[idx] - max(open_price, close_price)) / (candle_range + 1e-10)  # 상단 꼬리

    features = {
        # 레인지 대비
        'vol_vs_range': vol_vs_range,
        'delta_vs_range': delta_vs_range,
        'vol_zscore': vol_zscore,

        # 파동 추세
        'vol_trend': vol_trend,
        'delta_trend': delta_trend,
        'vol_accel': vol_accel / (range_avg_vol + 1e-10),
        'delta_accel': delta_accel / (range_avg_abs_delta + 1e-10),

        # 파동 대비
        'vol_vs_wave': vol_vs_wave,
        'delta_vs_wave': delta_vs_wave,

        # 방향 정렬
        'delta_aligned': delta_aligned,
        'delta_trend_aligned': delta_trend_aligned,

        # 볼륨/델타 상태
        'vol_spike': vol_spike,
        'vol_low': vol_low,
        'delta_reversal': delta_reversal,

        # CVD
        'cvd_wave': cvd_wave / (range_avg_abs_delta * wave_lookback + 1e-10),
        'cvd_range': cvd_range / (range_avg_abs_delta * range_lookback + 1e-10),

        # 캔들 패턴
        'body_ratio': body_ratio,
        'is_bullish': is_bullish,
        'wick_ratio': wick_ratio,

        # 방향
        'is_long': 1 if direction == 'LONG' else 0,
    }

    return features


def simulate_trade(highs, lows, idx, direction, entry, sl, tp1):
    """TP1 도달 여부로 성공 판정."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 0  # 실패
            if highs[j] >= tp1:
                return 1  # 성공
        else:
            if highs[j] >= sl:
                return 0
            if lows[j] <= tp1:
                return 1
    return 0  # 타임아웃 = 실패


def collect_data(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """학습 데이터 수집."""
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

            if entry > sl and tp1 > entry:
                features = extract_wave_features(ltf_candles, i, channel, 'LONG')
                success = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1)

                features['success'] = success
                features['direction'] = 'LONG'
                features['idx'] = i
                features['entry'] = entry
                features['sl'] = sl
                features['tp1'] = tp1

                data.append(features)
                traded_keys.add(bounce_key)

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid

            if sl > entry and entry > tp1:
                features = extract_wave_features(ltf_candles, i, channel, 'SHORT')
                success = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1)

                features['success'] = success
                features['direction'] = 'SHORT'
                features['idx'] = i
                features['entry'] = entry
                features['sl'] = sl
                features['tp1'] = tp1

                data.append(features)
                traded_keys.add(bounce_key)

    return pd.DataFrame(data)


def train_and_evaluate(train_df, test_df):
    """ML 모델 학습 및 평가."""

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
    y_test = test_df['success'].values

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 학습
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    # 예측
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return model, scaler, train_pred, test_pred, train_proba, test_proba, importance


def backtest_with_filter(df, proba, threshold, label):
    """필터 적용 백테스트."""
    mask = proba >= threshold
    filtered = df[mask]

    if len(filtered) == 0:
        return None

    total = len(filtered)
    wins = filtered['success'].sum()
    wr = wins / total * 100

    return {
        'label': label,
        'threshold': threshold,
        'trades': total,
        'wins': wins,
        'wr': wr,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   ML 파동 패턴 분석                                               ║
║   볼륨/델타 추세 학습                                              ║
║   Train: 2022-2023 | Test: 2024-2025                             ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    # 연도별 분리
    data_by_year = {}
    for year in [2022, 2023, 2024, 2025]:
        htf = htf_all[htf_all.index.year == year]
        ltf = ltf_all[ltf_all.index.year == year]
        if len(htf) > 100:
            data_by_year[year] = {'htf': htf, 'ltf': ltf}
            print(f"  {year}: HTF={len(htf)}, LTF={len(ltf)}")

    # Train/Test 분리
    htf_train = pd.concat([data_by_year[y]['htf'] for y in [2022, 2023] if y in data_by_year])
    ltf_train = pd.concat([data_by_year[y]['ltf'] for y in [2022, 2023] if y in data_by_year])
    htf_test = pd.concat([data_by_year[y]['htf'] for y in [2024, 2025] if y in data_by_year])
    ltf_test = pd.concat([data_by_year[y]['ltf'] for y in [2024, 2025] if y in data_by_year])

    print(f"\n  Train (2022-2023): HTF={len(htf_train)}, LTF={len(ltf_train)}")
    print(f"  Test (2024-2025): HTF={len(htf_test)}, LTF={len(ltf_test)}")

    # Build channels
    print("\nBuilding channels...")
    channels_train, _ = build_htf_channels(htf_train)
    channels_test, _ = build_htf_channels(htf_test)

    # Collect data
    print("\nCollecting training data...")
    train_df = collect_data(htf_train, ltf_train, channels_train)
    print(f"  Train samples: {len(train_df)} (Success: {train_df['success'].sum()}, {train_df['success'].mean()*100:.1f}%)")

    print("\nCollecting test data...")
    test_df = collect_data(htf_test, ltf_test, channels_test)
    print(f"  Test samples: {len(test_df)} (Success: {test_df['success'].sum()}, {test_df['success'].mean()*100:.1f}%)")

    # Train ML model
    print("\n" + "="*70)
    print("  ML 모델 학습")
    print("="*70)

    model, scaler, train_pred, test_pred, train_proba, test_proba, importance = train_and_evaluate(train_df, test_df)

    # Feature importance
    print("\n  📊 Feature Importance (Top 10)")
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        print(f"    {i+1}. {feat:<25}: {imp:.4f}")

    # 기본 성능
    print("\n  📈 기본 성능 (threshold=0.5)")
    train_baseline_wr = train_df['success'].mean() * 100
    test_baseline_wr = test_df['success'].mean() * 100

    train_filtered_wr = train_df[train_pred == 1]['success'].mean() * 100 if (train_pred == 1).sum() > 0 else 0
    test_filtered_wr = test_df[test_pred == 1]['success'].mean() * 100 if (test_pred == 1).sum() > 0 else 0

    print(f"\n    Train:")
    print(f"      기본 WR: {train_baseline_wr:.1f}% ({len(train_df)}건)")
    print(f"      ML 필터 WR: {train_filtered_wr:.1f}% ({(train_pred == 1).sum()}건)")

    print(f"\n    Test:")
    print(f"      기본 WR: {test_baseline_wr:.1f}% ({len(test_df)}건)")
    print(f"      ML 필터 WR: {test_filtered_wr:.1f}% ({(test_pred == 1).sum()}건)")

    # Threshold 별 성능
    print("\n" + "="*70)
    print("  Threshold별 성능 (Test)")
    print("="*70)
    print(f"\n  {'Threshold':<12} | {'건수':>6} | {'WR':>8} | {'vs 기본':>10}")
    print("-"*50)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        result = backtest_with_filter(test_df, test_proba, threshold, f"p>={threshold}")
        if result and result['trades'] >= 10:
            diff = result['wr'] - test_baseline_wr
            print(f"  {threshold:<12} | {result['trades']:>6} | {result['wr']:>7.1f}% | {diff:>+9.1f}%")

    # LONG/SHORT 별 분석
    print("\n" + "="*70)
    print("  방향별 성능 (Test)")
    print("="*70)

    for direction in ['LONG', 'SHORT']:
        dir_mask = test_df['direction'] == direction
        dir_df = test_df[dir_mask]
        dir_proba = test_proba[dir_mask]

        baseline = dir_df['success'].mean() * 100
        print(f"\n  [{direction}] 기본: {len(dir_df)}건, WR {baseline:.1f}%")

        for threshold in [0.5, 0.6, 0.7]:
            result = backtest_with_filter(dir_df, dir_proba, threshold, f"p>={threshold}")
            if result and result['trades'] >= 5:
                diff = result['wr'] - baseline
                print(f"    p>={threshold}: {result['trades']}건, WR {result['wr']:.1f}% ({diff:+.1f}%)")

    # 연도별 일관성
    print("\n" + "="*70)
    print("  연도별 일관성 (Test 데이터 내)")
    print("="*70)

    # 2024 vs 2025 분리
    for year in [2024, 2025]:
        if year not in data_by_year:
            continue

        htf_year = data_by_year[year]['htf']
        ltf_year = data_by_year[year]['ltf']
        channels_year, _ = build_htf_channels(htf_year)
        year_df = collect_data(htf_year, ltf_year, channels_year)

        if len(year_df) == 0:
            continue

        # 예측
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
        X_year = scaler.transform(year_df[feature_cols].values)
        year_proba = model.predict_proba(X_year)[:, 1]

        baseline = year_df['success'].mean() * 100
        print(f"\n  [{year}] 기본: {len(year_df)}건, WR {baseline:.1f}%")

        for threshold in [0.5, 0.6, 0.7]:
            result = backtest_with_filter(year_df, year_proba, threshold, f"p>={threshold}")
            if result and result['trades'] >= 5:
                diff = result['wr'] - baseline
                print(f"    p>={threshold}: {result['trades']}건, WR {result['wr']:.1f}% ({diff:+.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("  💡 요약")
    print("="*70)
    print("""
  핵심 피처:
  - 볼륨/델타 추세 (파동 방향)
  - 레인지 대비 터치 캔들 반응
  - 델타 방향 정렬

  다음 단계:
  - 최적 threshold 선택
  - Paper trading에 적용
""")


if __name__ == "__main__":
    main()
