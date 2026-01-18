#!/usr/bin/env python3
"""
ML Dynamic Exit System
- TP1 도달 시점에 "청산 vs 홀딩" 결정
- 진입 시점이 아닌, TP1 도달 시점의 피처 사용
- Train: 2022-2023 | Test: 2024-2025

실행: python ml_dynamic_exit.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import extract_features
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple

# Labels for TP1 decision
EXIT_AT_TP1 = 0  # TP1에서 전량 청산이 최적
HOLD_FOR_TP2 = 1  # TP2까지 홀딩이 최적


@dataclass
class DynamicExitFeatures:
    """TP1 도달 시점에서 수집하는 피처"""
    # 트레이드 진행 정보
    candles_to_tp1: int          # TP1까지 걸린 캔들 수
    time_to_tp1_minutes: float   # TP1까지 걸린 시간 (분)

    # 트레이드 중 델타/볼륨
    delta_during_trade: float    # 진입~TP1 누적 델타
    volume_during_trade: float   # 진입~TP1 누적 볼륨
    delta_ratio_during: float    # 평균 대비 델타 비율
    volume_ratio_during: float   # 평균 대비 볼륨 비율

    # TP1 시점 시장 상태
    momentum_at_tp1: float       # TP1 시점 모멘텀 (5캔들)
    rsi_at_tp1: float            # TP1 시점 RSI
    atr_at_tp1: float            # TP1 시점 ATR

    # 가격 행동
    max_favorable_excursion: float   # 최대 유리 방향 이동 (%)
    price_vs_tp1: float              # 현재가 vs TP1 (%)

    # 채널 정보
    distance_to_tp2_pct: float   # TP2까지 남은 거리 (%)
    channel_width_pct: float     # 채널 폭 (%)

    # 캔들 패턴
    last_candle_body_pct: float  # 마지막 캔들 몸통 크기
    last_candle_is_bullish: int  # 마지막 캔들이 양봉인지

    # 트레이드 타입
    is_long: int
    is_fakeout: int


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Calculate RSI at current point."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def simulate_trade_dynamic(
    candles: pd.DataFrame,
    entry_idx: int,
    trade_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    channel_width_pct: float,
    is_fakeout: bool
) -> Tuple[Optional[DynamicExitFeatures], Optional[int], dict]:
    """
    캔들 하나씩 시뮬레이션하면서 TP1 도달 시 피처 수집.

    Returns:
        features: TP1 도달 시점의 피처 (TP1 미도달시 None)
        label: EXIT_AT_TP1 or HOLD_FOR_TP2 (최적 출구 기준)
        trade_result: 트레이드 결과 정보
    """
    is_long = trade_type == 'LONG'
    max_candles = 200  # 최대 200캔들 후 타임아웃

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values if 'delta' in candles.columns else np.zeros(len(candles))

    # 트래킹 변수
    hit_tp1 = False
    hit_tp2 = False
    hit_sl = False
    tp1_idx = None
    tp2_idx = None
    sl_idx = None

    cumulative_delta = 0
    cumulative_volume = 0
    max_favorable = 0

    # 평균 델타/볼륨 (진입 전 20캔들)
    lookback = 20
    if entry_idx >= lookback:
        avg_delta = np.mean(np.abs(deltas[entry_idx-lookback:entry_idx]))
        avg_volume = np.mean(volumes[entry_idx-lookback:entry_idx])
    else:
        avg_delta = 1
        avg_volume = 1

    # 캔들 하나씩 시뮬레이션
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(candles))):
        high = highs[i]
        low = lows[i]
        close = closes[i]

        cumulative_delta += deltas[i] if i < len(deltas) else 0
        cumulative_volume += volumes[i] if i < len(volumes) else 0

        # 최대 유리 방향 이동
        if is_long:
            favorable = (high - entry_price) / entry_price
        else:
            favorable = (entry_price - low) / entry_price
        max_favorable = max(max_favorable, favorable)

        # SL 체크 (TP1 전)
        if not hit_tp1:
            if is_long and low <= sl_price:
                hit_sl = True
                sl_idx = i
                break
            elif not is_long and high >= sl_price:
                hit_sl = True
                sl_idx = i
                break

        # TP1 체크
        if not hit_tp1:
            if is_long and high >= tp1_price:
                hit_tp1 = True
                tp1_idx = i
            elif not is_long and low <= tp1_price:
                hit_tp1 = True
                tp1_idx = i

        # TP2 체크 (TP1 후)
        if hit_tp1 and not hit_tp2:
            if is_long and high >= tp2_price:
                hit_tp2 = True
                tp2_idx = i
                break
            elif not is_long and low <= tp2_price:
                hit_tp2 = True
                tp2_idx = i
                break

            # TP1 후 SL (BE) 체크
            if is_long and low <= entry_price:
                break  # BE 청산
            elif not is_long and high >= entry_price:
                break  # BE 청산

    # 결과 정보
    trade_result = {
        'hit_tp1': hit_tp1,
        'hit_tp2': hit_tp2,
        'hit_sl': hit_sl,
        'entry_price': entry_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'sl_price': sl_price
    }

    # TP1 미도달 → 피처 없음
    if not hit_tp1:
        return None, None, trade_result

    # TP1 도달 시점 피처 수집
    candles_to_tp1 = tp1_idx - entry_idx
    time_to_tp1 = candles_to_tp1 * 15  # 15분봉 가정

    # TP1 시점 시장 상태
    tp1_closes = closes[:tp1_idx+1]
    momentum_at_tp1 = (closes[tp1_idx] - closes[tp1_idx-5]) / closes[tp1_idx-5] if tp1_idx >= 5 else 0
    rsi_at_tp1 = calculate_rsi(tp1_closes)

    # ATR at TP1
    if tp1_idx >= 14:
        tr = np.maximum(
            highs[tp1_idx-14:tp1_idx] - lows[tp1_idx-14:tp1_idx],
            np.abs(highs[tp1_idx-14:tp1_idx] - closes[tp1_idx-15:tp1_idx-1])
        )
        atr_at_tp1 = np.mean(tr)
    else:
        atr_at_tp1 = 0

    # 가격 vs TP1
    price_vs_tp1 = (closes[tp1_idx] - tp1_price) / tp1_price

    # TP2까지 거리
    if is_long:
        distance_to_tp2 = (tp2_price - tp1_price) / tp1_price
    else:
        distance_to_tp2 = (tp1_price - tp2_price) / tp1_price

    # 마지막 캔들 정보
    last_body = abs(closes[tp1_idx] - candles['open'].values[tp1_idx])
    last_body_pct = last_body / closes[tp1_idx]
    last_is_bullish = 1 if closes[tp1_idx] > candles['open'].values[tp1_idx] else 0

    features = DynamicExitFeatures(
        candles_to_tp1=candles_to_tp1,
        time_to_tp1_minutes=time_to_tp1,
        delta_during_trade=cumulative_delta,
        volume_during_trade=cumulative_volume,
        delta_ratio_during=cumulative_delta / (avg_delta * candles_to_tp1 + 1e-10),
        volume_ratio_during=cumulative_volume / (avg_volume * candles_to_tp1 + 1e-10),
        momentum_at_tp1=momentum_at_tp1,
        rsi_at_tp1=rsi_at_tp1,
        atr_at_tp1=atr_at_tp1,
        max_favorable_excursion=max_favorable,
        price_vs_tp1=price_vs_tp1,
        distance_to_tp2_pct=distance_to_tp2,
        channel_width_pct=channel_width_pct,
        last_candle_body_pct=last_body_pct,
        last_candle_is_bullish=last_is_bullish,
        is_long=1 if is_long else 0,
        is_fakeout=1 if is_fakeout else 0
    )

    # 라벨 결정: TP2 도달했으면 HOLD가 최적, 아니면 EXIT
    # (실제로는 PnL 비교가 더 정확하지만 단순화)
    label = HOLD_FOR_TP2 if hit_tp2 else EXIT_AT_TP1

    return features, label, trade_result


def collect_dynamic_data(df_1h, df_15m):
    """동적 출구 학습 데이터 수집."""
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()
    features_list = []
    labels_list = []
    timestamps = []
    trade_results = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    for i in tqdm(range(50, len(df_15m) - 250), desc='Collecting dynamic data'):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2
        channel_width = (channel.resistance - channel.support) / channel.support

        # Fakeout
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)  # Fix lookahead bias
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2
            f_width = (f_channel.resistance - f_channel.support) / f_channel.support
            trade_key = (round(f_channel.support), round(f_channel.resistance), 'fakeout', htf_idx)

            if trade_key not in traded_entries:
                if fakeout_signal.type == 'bear':
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 - sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.resistance * 0.998

                    if entry > sl and tp1 > entry:
                        features, label, result = simulate_trade_dynamic(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2, f_width, True
                        )
                        if features is not None:
                            features_list.append(features)
                            labels_list.append(label)
                            timestamps.append(df_15m.index[i])
                            trade_results.append(result)
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        features, label, result = simulate_trade_dynamic(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2, f_width, True
                        )
                        if features is not None:
                            features_list.append(features)
                            labels_list.append(label)
                            timestamps.append(df_15m.index[i])
                            trade_results.append(result)
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
                features, label, result = simulate_trade_dynamic(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2, channel_width, False
                )
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    timestamps.append(df_15m.index[i])
                    trade_results.append(result)
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                features, label, result = simulate_trade_dynamic(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2, channel_width, False
                )
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    timestamps.append(df_15m.index[i])
                    trade_results.append(result)
                traded_entries.add(trade_key)

    return features_list, labels_list, timestamps, trade_results


def features_to_array(features_list):
    """Convert features to numpy array."""
    return np.array([[
        f.candles_to_tp1,
        f.time_to_tp1_minutes,
        f.delta_during_trade,
        f.volume_during_trade,
        f.delta_ratio_during,
        f.volume_ratio_during,
        f.momentum_at_tp1,
        f.rsi_at_tp1,
        f.atr_at_tp1,
        f.max_favorable_excursion,
        f.price_vs_tp1,
        f.distance_to_tp2_pct,
        f.channel_width_pct,
        f.last_candle_body_pct,
        f.last_candle_is_bullish,
        f.is_long,
        f.is_fakeout
    ] for f in features_list])


def backtest_dynamic(trade_results, labels, predictions, label_name):
    """동적 출구 전략 백테스트."""
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0
    trade_returns = []

    for result, pred in zip(trade_results, predictions):
        entry = result['entry_price']
        sl = result['sl_price']
        tp1 = result['tp1_price']
        tp2 = result['tp2_price']
        hit_tp2 = result['hit_tp2']

        sl_dist = abs(entry - sl) / entry
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position = capital * leverage

        # PnL 계산
        is_long = entry < tp1  # LONG이면 tp1 > entry

        if pred == EXIT_AT_TP1:
            # TP1에서 전량 청산
            if is_long:
                pnl_pct = (tp1 - entry) / entry
            else:
                pnl_pct = (entry - tp1) / entry
        else:  # HOLD_FOR_TP2
            if hit_tp2:
                # TP2 도달: 50% at TP1, 50% at TP2
                if is_long:
                    pnl_pct = 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                else:
                    pnl_pct = 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
            else:
                # TP2 미도달: 50% at TP1, 50% at BE
                if is_long:
                    pnl_pct = 0.5 * (tp1 - entry) / entry
                else:
                    pnl_pct = 0.5 * (entry - tp1) / entry

        gross_pnl = position * pnl_pct
        fees = position * fee_pct * 2
        net_pnl = gross_pnl - fees

        trade_returns.append(net_pnl / capital * 100)
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

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / 10000 - 1) * 100
    trade_returns = np.array(trade_returns)

    print(f"\n{'='*60}")
    print(f"  {label_name}")
    print(f"{'='*60}")
    print(f"  Trades (TP1 도달): {total}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")
    print(f"  Return: {ret:+,.1f}%")
    if len(trade_returns) > 0:
        print(f"\n  매매당 수익: 평균 {trade_returns.mean():+.3f}%, 중앙값 {np.median(trade_returns):+.3f}%")

    return capital


def main():
    print("="*60)
    print("  ML DYNAMIC EXIT SYSTEM")
    print("  TP1 도달 시점에 '청산 vs 홀딩' 결정")
    print("  Train: 2022-2023 | Test: 2024-2025")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')
    print(f"  1H: {len(df_1h)}, 15m: {len(df_15m)}")

    # Collect data
    print("\nCollecting dynamic exit data...")
    features_list, labels_list, timestamps, trade_results = collect_dynamic_data(df_1h, df_15m)
    print(f"  Total TP1-reached trades: {len(features_list)}")

    # Split IS/OOS
    years = np.array([t.year for t in timestamps])
    is_mask = np.isin(years, [2022, 2023])
    oos_mask = np.isin(years, [2024, 2025])

    print(f"\n  IS (2022-2023): {is_mask.sum()} trades")
    print(f"  OOS (2024-2025): {oos_mask.sum()} trades")

    # Label distribution
    labels = np.array(labels_list)
    print(f"\n  Label distribution (전체):")
    print(f"    EXIT_AT_TP1: {(labels == EXIT_AT_TP1).sum()} ({(labels == EXIT_AT_TP1).mean()*100:.1f}%)")
    print(f"    HOLD_FOR_TP2: {(labels == HOLD_FOR_TP2).sum()} ({(labels == HOLD_FOR_TP2).mean()*100:.1f}%)")

    # Prepare data
    X = features_to_array(features_list)

    X_is = X[is_mask]
    y_is = labels[is_mask]

    X_oos = X[oos_mask]
    y_oos = labels[oos_mask]
    oos_results = [trade_results[i] for i in range(len(trade_results)) if oos_mask[i]]

    # Train model
    print("\n" + "="*60)
    print("  Training Dynamic Exit Model on IS (2022-2023)")
    print("="*60)

    scaler = StandardScaler()
    X_is_scaled = scaler.fit_transform(X_is)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_is_scaled, y_is)

    print(f"  Training samples: {len(X_is)}")
    print(f"  EXIT_AT_TP1: {(y_is == EXIT_AT_TP1).sum()} ({(y_is == EXIT_AT_TP1).mean()*100:.1f}%)")
    print(f"  HOLD_FOR_TP2: {(y_is == HOLD_FOR_TP2).sum()} ({(y_is == HOLD_FOR_TP2).mean()*100:.1f}%)")
    print(f"  Training Accuracy: {model.score(X_is_scaled, y_is):.3f}")

    # Feature importance
    feature_names = [
        'candles_to_tp1', 'time_to_tp1', 'delta_during', 'volume_during',
        'delta_ratio', 'volume_ratio', 'momentum_tp1', 'rsi_tp1', 'atr_tp1',
        'max_favorable', 'price_vs_tp1', 'dist_to_tp2', 'channel_width',
        'candle_body', 'is_bullish', 'is_long', 'is_fakeout'
    ]
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Top 5 Feature Importances:")
    for i in sorted_idx[:5]:
        print(f"    {feature_names[i]}: {importances[i]:.4f}")

    # Predict on OOS
    X_oos_scaled = scaler.transform(X_oos)
    predictions = model.predict(X_oos_scaled)
    probabilities = model.predict_proba(X_oos_scaled)[:, 1]  # HOLD 확률

    # OOS Results
    print("\n" + "="*60)
    print("  OOS RESULTS (2024-2025)")
    print("="*60)

    # Baseline: Always EXIT at TP1
    always_exit = np.full(len(oos_results), EXIT_AT_TP1)
    backtest_dynamic(oos_results, y_oos, always_exit, "Baseline: Always EXIT at TP1 (100%)")

    # Baseline: Always HOLD for TP2
    always_hold = np.full(len(oos_results), HOLD_FOR_TP2)
    backtest_dynamic(oos_results, y_oos, always_hold, "Baseline: Always HOLD for TP2 (50%+50%)")

    # ML Dynamic Exit
    backtest_dynamic(oos_results, y_oos, predictions, "ML Dynamic Exit (threshold=0.5)")

    # ML with higher threshold (more selective HOLD)
    predictions_60 = (probabilities >= 0.6).astype(int)
    backtest_dynamic(oos_results, y_oos, predictions_60, "ML Dynamic Exit (threshold=0.6)")

    predictions_70 = (probabilities >= 0.7).astype(int)
    backtest_dynamic(oos_results, y_oos, predictions_70, "ML Dynamic Exit (threshold=0.7)")

    # Oracle (optimal)
    backtest_dynamic(oos_results, y_oos, y_oos, "Oracle (Perfect Prediction)")

    # Summary
    print("\n" + "="*60)
    print("  ML Prediction Distribution (OOS)")
    print("="*60)
    print(f"  EXIT_AT_TP1: {(predictions == EXIT_AT_TP1).sum()} ({(predictions == EXIT_AT_TP1).mean()*100:.1f}%)")
    print(f"  HOLD_FOR_TP2: {(predictions == HOLD_FOR_TP2).sum()} ({(predictions == HOLD_FOR_TP2).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
