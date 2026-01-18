#!/usr/bin/env python3
"""
ML Volume/Delta 기반 시그널 필터링

목표: BOUNCE 시그널의 승률(55%)을 ML 필터링으로 개선

핵심 아이디어:
- 채널 내부 평균 볼륨/델타 vs 바운스 시점 볼륨/델타 비교
- "강한 바운스" 패턴 학습
- 나쁜 시그널 필터링

Walk-forward Validation:
- 2024-01 ~ 2024-06: Training
- 2024-07 ~ 2024-12: Validation (IS)
- 2025-01 ~ 현재: Test (OOS)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels, Channel, FakeoutSignal


@dataclass
class BounceFeatures:
    """Features for ML filtering of bounce signals."""
    # 채널 내부 통계 (Baseline)
    channel_avg_volume: float
    channel_std_volume: float
    channel_avg_delta: float
    channel_cvd_trend: float  # CVD 추세 (기울기)

    # 바운스 시점 Feature
    bounce_volume: float
    bounce_delta: float
    bounce_body_ratio: float  # 몸통 비율 (양봉/음봉 강도)
    bounce_wick_ratio: float  # 꼬리 비율 (거부 강도)

    # 비교 Feature (핵심)
    volume_vs_avg: float  # bounce_volume / channel_avg_volume
    delta_vs_avg: float   # bounce_delta / channel_avg_delta
    volume_zscore: float  # (bounce_volume - avg) / std

    # 방향성 Feature
    cvd_direction: int    # CVD 방향 (1=bullish, -1=bearish)
    delta_direction: int  # 델타 방향 (1=positive, -1=negative)
    direction_aligned: int  # 방향 일치 여부

    # 채널 정보
    channel_width: float
    total_touches: int

    # Setup type
    is_fakeout: int
    fakeout_depth: float

    # 거래 방향
    is_long: int

    # 시간 정보
    hour: int
    day_of_week: int


def extract_bounce_features(
    candles: pd.DataFrame,
    idx: int,
    channel: Channel,
    trade_type: str,
    setup_type: str = 'BOUNCE',
    fakeout_extreme: float = None,
    lookback: int = 20
) -> BounceFeatures:
    """바운스 시점에서 feature 추출."""

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    # 채널 내부 통계 (최근 N개 캔들)
    hist_start = max(0, idx - lookback)
    hist_volumes = volumes[hist_start:idx]
    hist_deltas = deltas[hist_start:idx]

    channel_avg_volume = np.mean(hist_volumes) if len(hist_volumes) > 0 else volumes[idx]
    channel_std_volume = np.std(hist_volumes) if len(hist_volumes) > 1 else 1.0
    channel_avg_delta = np.mean(hist_deltas) if len(hist_deltas) > 0 else 0

    # CVD 추세 (기울기)
    if len(hist_deltas) >= 5:
        cvd = np.cumsum(hist_deltas)
        x = np.arange(len(cvd))
        cvd_trend = np.polyfit(x, cvd, 1)[0] if len(cvd) > 1 else 0
    else:
        cvd_trend = 0

    # 바운스 시점 Feature
    bounce_volume = volumes[idx]
    bounce_delta = deltas[idx]

    # 몸통 비율 = |close - open| / (high - low)
    candle_range = highs[idx] - lows[idx]
    body = abs(closes[idx] - opens[idx])
    bounce_body_ratio = body / candle_range if candle_range > 0 else 0

    # 꼬리 비율 (방향에 따라)
    if trade_type == 'LONG':
        # LONG: 하단 꼬리 (지지 거부)
        wick = min(opens[idx], closes[idx]) - lows[idx]
    else:
        # SHORT: 상단 꼬리 (저항 거부)
        wick = highs[idx] - max(opens[idx], closes[idx])
    bounce_wick_ratio = wick / candle_range if candle_range > 0 else 0

    # 비교 Feature
    volume_vs_avg = bounce_volume / channel_avg_volume if channel_avg_volume > 0 else 1
    delta_vs_avg = bounce_delta / (abs(channel_avg_delta) + 1)
    volume_zscore = (bounce_volume - channel_avg_volume) / channel_std_volume if channel_std_volume > 0 else 0

    # 방향성 Feature
    cvd_recent = np.sum(hist_deltas) if len(hist_deltas) > 0 else 0
    cvd_direction = 1 if cvd_recent > 0 else -1
    delta_direction = 1 if bounce_delta > 0 else -1

    # 방향 일치: LONG일 때 CVD/Delta가 양수면 일치
    if trade_type == 'LONG':
        direction_aligned = 1 if (cvd_direction == 1 or delta_direction == 1) else 0
    else:
        direction_aligned = 1 if (cvd_direction == -1 or delta_direction == -1) else 0

    # 채널 정보
    channel_width = (channel.resistance - channel.support) / channel.support
    total_touches = channel.support_touches + channel.resistance_touches

    # Fakeout 정보
    is_fakeout = 1 if setup_type == 'FAKEOUT' else 0
    fakeout_depth = 0.0
    if fakeout_extreme is not None:
        if trade_type == 'LONG':
            fakeout_depth = (channel.support - fakeout_extreme) / channel.support * 100
        else:
            fakeout_depth = (fakeout_extreme - channel.resistance) / channel.resistance * 100

    # 시간 정보
    timestamp = candles.index[idx]
    hour = timestamp.hour if hasattr(timestamp, 'hour') else 12
    day_of_week = timestamp.weekday() if hasattr(timestamp, 'weekday') else 0

    return BounceFeatures(
        channel_avg_volume=channel_avg_volume,
        channel_std_volume=channel_std_volume,
        channel_avg_delta=channel_avg_delta,
        channel_cvd_trend=cvd_trend,
        bounce_volume=bounce_volume,
        bounce_delta=bounce_delta,
        bounce_body_ratio=bounce_body_ratio,
        bounce_wick_ratio=bounce_wick_ratio,
        volume_vs_avg=volume_vs_avg,
        delta_vs_avg=delta_vs_avg,
        volume_zscore=volume_zscore,
        cvd_direction=cvd_direction,
        delta_direction=delta_direction,
        direction_aligned=direction_aligned,
        channel_width=channel_width,
        total_touches=total_touches,
        is_fakeout=is_fakeout,
        fakeout_depth=fakeout_depth,
        is_long=1 if trade_type == 'LONG' else 0,
        hour=hour,
        day_of_week=day_of_week
    )


def features_to_array(features: BounceFeatures) -> np.ndarray:
    """Convert BounceFeatures to numpy array for ML."""
    return np.array([
        features.volume_vs_avg,
        features.delta_vs_avg,
        features.volume_zscore,
        features.bounce_body_ratio,
        features.bounce_wick_ratio,
        features.channel_cvd_trend,
        features.cvd_direction,
        features.delta_direction,
        features.direction_aligned,
        features.channel_width,
        features.total_touches,
        features.is_fakeout,
        features.fakeout_depth,
        features.is_long,
        features.hour,
        features.day_of_week
    ])


FEATURE_NAMES = [
    'volume_vs_avg',
    'delta_vs_avg',
    'volume_zscore',
    'bounce_body_ratio',
    'bounce_wick_ratio',
    'channel_cvd_trend',
    'cvd_direction',
    'delta_direction',
    'direction_aligned',
    'channel_width',
    'total_touches',
    'is_fakeout',
    'fakeout_depth',
    'is_long',
    'hour',
    'day_of_week'
]


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price):
    """Simulate trade with partial TP + breakeven. Returns (pnl_pct, outcome)."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price
    outcome = 0  # 0=loss, 0.5=partial, 1=full win

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    outcome = 0
                    break
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    outcome = 0.5
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    outcome = 0
                    break
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    outcome = 0.5
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break

    return pnl_pct, outcome


def collect_all_signals(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008
) -> List[dict]:
    """Collect BOUNCE and FAKEOUT signals with features and outcomes."""

    # Build HTF channels and fakeout signals
    htf_channel_map, htf_fakeout_signals = build_htf_channels(htf_candles)

    # Build fakeout map
    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}

    # LTF data
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    signals = []
    traded_entries = set()

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    for i in tqdm(range(50, len(ltf_candles) - 200), desc="Collecting signals"):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio

        # ============ FAKEOUT SIGNALS ============
        # Check for fakeout at start of HTF candle (avoid lookahead with htf_idx - 1)
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2

            fakeout_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)
            if fakeout_key not in traded_entries:
                if fakeout_signal.type == 'bear':
                    # Bear fakeout -> LONG
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 - sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.resistance * 0.998

                    if entry_price > sl_price and tp1_price > entry_price:
                        features = extract_bounce_features(
                            ltf_candles, i, f_channel, 'LONG', 'FAKEOUT', fakeout_signal.extreme
                        )
                        pnl_pct, outcome = simulate_trade(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )
                        success = 1 if outcome >= 0.5 else 0

                        signals.append({
                            'idx': i,
                            'timestamp': ltf_candles.index[i],
                            'type': 'LONG',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'pnl_pct': pnl_pct,
                            'outcome': outcome,
                            'success': success,
                            'features': features
                        })
                        traded_entries.add(fakeout_key)

                else:  # bull fakeout -> SHORT
                    entry_price = current_close
                    sl_price = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1_price = f_mid
                    tp2_price = f_channel.support * 1.002

                    if sl_price > entry_price and entry_price > tp1_price:
                        features = extract_bounce_features(
                            ltf_candles, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme
                        )
                        pnl_pct, outcome = simulate_trade(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                        )
                        success = 1 if outcome >= 0.5 else 0

                        signals.append({
                            'idx': i,
                            'timestamp': ltf_candles.index[i],
                            'type': 'SHORT',
                            'setup_type': 'FAKEOUT',
                            'entry': entry_price,
                            'sl': sl_price,
                            'tp1': tp1_price,
                            'tp2': tp2_price,
                            'pnl_pct': pnl_pct,
                            'outcome': outcome,
                            'success': success,
                            'features': features
                        })
                        traded_entries.add(fakeout_key)

        # ============ BOUNCE SIGNALS ============
        # Get HTF channel (use htf_idx - 1 to avoid lookahead)
        channel = htf_channel_map.get(htf_idx - 1)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Avoid duplicate trades
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch (LONG)
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price

            if risk > 0 and reward1 > 0:
                # Extract features
                features = extract_bounce_features(ltf_candles, i, channel, 'LONG', 'BOUNCE')

                # Simulate trade
                pnl_pct, outcome = simulate_trade(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                )

                # Success = TP1 or TP2 hit
                success = 1 if outcome >= 0.5 else 0

                signals.append({
                    'idx': i,
                    'timestamp': ltf_candles.index[i],
                    'type': 'LONG',
                    'setup_type': 'BOUNCE',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': tp1_price,
                    'tp2': tp2_price,
                    'pnl_pct': pnl_pct,
                    'outcome': outcome,
                    'success': success,
                    'features': features
                })
                traded_entries.add(trade_key)

        # BOUNCE: Resistance touch (SHORT)
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

            if risk > 0 and reward1 > 0:
                # Extract features
                features = extract_bounce_features(ltf_candles, i, channel, 'SHORT', 'BOUNCE')

                # Simulate trade
                pnl_pct, outcome = simulate_trade(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                )

                # Success = TP1 or TP2 hit
                success = 1 if outcome >= 0.5 else 0

                signals.append({
                    'idx': i,
                    'timestamp': ltf_candles.index[i],
                    'type': 'SHORT',
                    'setup_type': 'BOUNCE',
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp1': tp1_price,
                    'tp2': tp2_price,
                    'pnl_pct': pnl_pct,
                    'outcome': outcome,
                    'success': success,
                    'features': features
                })
                traded_entries.add(trade_key)

    return signals


def prepare_ml_data(signals: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert signals to ML arrays."""
    X = np.array([features_to_array(s['features']) for s in signals])
    y = np.array([s['success'] for s in signals])
    return X, y


def run_backtest(trades: List[dict], label: str):
    """Run backtest on trades."""
    if len(trades) == 0:
        print(f"  {label}: No trades")
        return {}

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for trade in trades:
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
    avg_pnl = np.mean([t['pnl_pct'] for t in trades]) * 100

    print(f"\n  {label}:")
    print(f"    Trades: {len(trades)}, Avg PnL: {avg_pnl:+.4f}%")
    print(f"    Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
    print(f"    Final: ${capital:,.2f}")

    return {
        'trades': len(trades),
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'return': total_return,
        'max_dd': max_dd * 100,
        'final': capital
    }


def analyze_by_threshold(signals, probs, thresholds, label):
    """Analyze results by threshold."""
    print(f"\n  {label}:")
    print(f"  {'Thresh':<8} {'Trades':>8} {'WR':>8} {'Avg PnL':>12} {'Return':>12}")
    print(f"  {'-'*52}")

    results = []
    for thresh in thresholds:
        filtered = [s for s, p in zip(signals, probs) if p >= thresh]
        if len(filtered) < 3:
            continue

        wr = np.mean([s['success'] for s in filtered]) * 100
        avg_pnl = np.mean([s['pnl_pct'] for s in filtered]) * 100

        # Quick backtest for return
        capital = 10000
        for trade in filtered:
            sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
            if sl_dist > 0:
                leverage = min(0.015 / sl_dist, 15)
                position = capital * leverage
                net_pnl = position * trade['pnl_pct'] - position * 0.0004 * 2
                capital += net_pnl
                capital = max(capital, 0)

        total_return = (capital - 10000) / 10000 * 100

        print(f"  {thresh:<8.1f} {len(filtered):>8} {wr:>7.1f}% {avg_pnl:>+11.4f}% {total_return:>+11.1f}%")
        results.append({'thresh': thresh, 'trades': len(filtered), 'wr': wr, 'avg_pnl': avg_pnl, 'return': total_return})

    return results


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║   ML Volume/Delta 기반 시그널 필터링                                    ║
║   Original Channel + BOUNCE only 전략                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_candles = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_candles = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    print(f"  HTF (1H): {len(htf_candles):,} candles")
    print(f"  LTF (15m): {len(ltf_candles):,} candles")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}")

    # Collect signals (BOUNCE only - best strategy)
    print("\nCollecting BOUNCE signals only (best strategy)...")
    all_signals_raw = collect_all_signals(htf_candles, ltf_candles)

    # Filter BOUNCE only
    all_signals = [s for s in all_signals_raw if s['setup_type'] == 'BOUNCE']
    print(f"  BOUNCE signals: {len(all_signals)}")

    # Add year/month info
    for s in all_signals:
        s['year'] = s['timestamp'].year
        s['month'] = s['timestamp'].month

    # Split by period
    # Training: 2022-2023
    train_signals = [s for s in all_signals if s['year'] <= 2023]
    # Validation: 2024
    val_signals = [s for s in all_signals if s['year'] == 2024]
    # Test: 2025+
    test_signals = [s for s in all_signals if s['year'] >= 2025]

    print(f"\n  Training (2022-2023): {len(train_signals)} signals")
    print(f"  Validation (2024):    {len(val_signals)} signals")
    print(f"  Test (2025+):         {len(test_signals)} signals")

    # Breakdown by setup type
    for period_name, period_signals in [("Train", train_signals), ("Val", val_signals), ("Test", test_signals)]:
        bounce = len([s for s in period_signals if s['setup_type'] == 'BOUNCE'])
        fakeout = len([s for s in period_signals if s['setup_type'] == 'FAKEOUT'])
        print(f"    {period_name}: BOUNCE={bounce}, FAKEOUT={fakeout}")

    # Prepare ML data
    X_train, y_train = prepare_ml_data(train_signals)
    X_val, y_val = prepare_ml_data(val_signals)
    X_test, y_test = prepare_ml_data(test_signals)

    print(f"\n  Training set class balance: {y_train.mean()*100:.1f}% success")
    print(f"  Validation set class balance: {y_val.mean()*100:.1f}% success")
    if len(y_test) > 0:
        print(f"  Test set class balance: {y_test.mean()*100:.1f}% success")

    # Train model
    print("\n" + "="*70)
    print("  Training Random Forest Model")
    print("="*70)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Prevent overfitting
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Feature importance
    print("\n  Feature Importance:")
    importance = list(zip(FEATURE_NAMES, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importance[:10]:
        print(f"    {name}: {imp:.4f}")

    # Predictions
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else []

    # Threshold analysis
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n" + "="*70)
    print("  THRESHOLD ANALYSIS")
    print("="*70)

    val_results = analyze_by_threshold(val_signals, val_probs, thresholds, "Validation (2024)")

    if len(test_signals) > 0:
        test_results = analyze_by_threshold(test_signals, test_probs, thresholds, "Test (2025+ OOS)")

    # Find best threshold based on validation avg_pnl
    best_thresh = max(val_results, key=lambda x: x['avg_pnl'])['thresh'] if val_results else 0.5
    print(f"\n  Best Threshold (by Val Avg PnL): {best_thresh}")

    # Detailed results for best threshold
    print("\n" + "="*70)
    print(f"  DETAILED RESULTS (Threshold = {best_thresh})")
    print("="*70)

    # Validation
    print("\n  [Validation Period: 2024]")
    run_backtest(val_signals, "Baseline (All Signals)")

    filtered_val = [s for s, p in zip(val_signals, val_probs) if p >= best_thresh]
    run_backtest(filtered_val, f"ML Filtered (thresh={best_thresh})")

    # By setup type (validation)
    val_bounce = [s for s in val_signals if s['setup_type'] == 'BOUNCE']
    val_fakeout = [s for s in val_signals if s['setup_type'] == 'FAKEOUT']
    if len(val_bounce) > 0:
        bounce_wr = np.mean([s['success'] for s in val_bounce]) * 100
        bounce_pnl = np.mean([s['pnl_pct'] for s in val_bounce]) * 100
        print(f"\n    BOUNCE only: {len(val_bounce)} trades, WR: {bounce_wr:.1f}%, Avg PnL: {bounce_pnl:+.4f}%")
    if len(val_fakeout) > 0:
        fakeout_wr = np.mean([s['success'] for s in val_fakeout]) * 100
        fakeout_pnl = np.mean([s['pnl_pct'] for s in val_fakeout]) * 100
        print(f"    FAKEOUT only: {len(val_fakeout)} trades, WR: {fakeout_wr:.1f}%, Avg PnL: {fakeout_pnl:+.4f}%")

    # Test
    if len(test_signals) > 0:
        print("\n  [Test Period: 2025+ (Out-of-Sample)]")
        run_backtest(test_signals, "Baseline (All Signals)")

        filtered_test = [s for s, p in zip(test_signals, test_probs) if p >= best_thresh]
        run_backtest(filtered_test, f"ML Filtered (thresh={best_thresh})")

        # By setup type (test)
        test_bounce = [s for s in test_signals if s['setup_type'] == 'BOUNCE']
        test_fakeout = [s for s in test_signals if s['setup_type'] == 'FAKEOUT']
        if len(test_bounce) > 0:
            bounce_wr = np.mean([s['success'] for s in test_bounce]) * 100
            bounce_pnl = np.mean([s['pnl_pct'] for s in test_bounce]) * 100
            print(f"\n    BOUNCE only: {len(test_bounce)} trades, WR: {bounce_wr:.1f}%, Avg PnL: {bounce_pnl:+.4f}%")
        if len(test_fakeout) > 0:
            fakeout_wr = np.mean([s['success'] for s in test_fakeout]) * 100
            fakeout_pnl = np.mean([s['pnl_pct'] for s in test_fakeout]) * 100
            print(f"    FAKEOUT only: {len(test_fakeout)} trades, WR: {fakeout_wr:.1f}%, Avg PnL: {fakeout_pnl:+.4f}%")

        # Removed signals analysis
        print("\n  제거된 시그널 분석 (Test):")
        removed_test = [s for s, p in zip(test_signals, test_probs) if p < best_thresh]
        if len(removed_test) > 0:
            removed_wr = np.mean([s['success'] for s in removed_test]) * 100
            removed_pnl = np.mean([s['pnl_pct'] for s in removed_test]) * 100
            print(f"    제거된 매매: {len(removed_test)}")
            print(f"    제거된 매매 WR: {removed_wr:.1f}%")
            print(f"    제거된 매매 Avg PnL: {removed_pnl:+.4f}%")


if __name__ == "__main__":
    main()
