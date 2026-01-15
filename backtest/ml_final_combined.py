#!/usr/bin/env python3
"""
ML Final Combined Test
- ML Entry (threshold=0.7): 저품질 신호 필터링
- ML Dynamic Exit: TP1 도달 시 청산 vs 홀딩 결정
- Train: 2022-2023 | Test: 2024-2025

실행: python ml_final_combined.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import extract_features
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Entry labels
ENTRY_TAKE = 1
ENTRY_SKIP = 0

# Exit labels
EXIT_AT_TP1 = 0
HOLD_FOR_TP2 = 1


@dataclass
class DynamicExitFeatures:
    """TP1 도달 시점에서 수집하는 피처"""
    candles_to_tp1: int
    time_to_tp1_minutes: float
    delta_during_trade: float
    volume_during_trade: float
    delta_ratio_during: float
    volume_ratio_during: float
    momentum_at_tp1: float
    rsi_at_tp1: float
    atr_at_tp1: float
    max_favorable_excursion: float
    price_vs_tp1: float
    distance_to_tp2_pct: float
    channel_width_pct: float
    last_candle_body_pct: float
    last_candle_is_bullish: int
    is_long: int
    is_fakeout: int


def calculate_rsi(closes: np.ndarray, period: int = 14) -> float:
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


def simulate_trade_full(
    candles: pd.DataFrame,
    entry_idx: int,
    trade_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    channel_width_pct: float,
    is_fakeout: bool
) -> Tuple[Optional[DynamicExitFeatures], dict]:
    """
    전체 트레이드 시뮬레이션.
    Returns:
        - dynamic_features: TP1 도달 시 피처 (None if TP1 미도달)
        - trade_result: 트레이드 결과 정보
    """
    is_long = trade_type == 'LONG'
    max_candles = 200

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values if 'delta' in candles.columns else np.zeros(len(candles))

    hit_tp1 = False
    hit_tp2 = False
    hit_sl = False
    tp1_idx = None
    hit_be_after_tp1 = False

    cumulative_delta = 0
    cumulative_volume = 0
    max_favorable = 0

    # 평균 델타/볼륨
    lookback = 20
    if entry_idx >= lookback:
        avg_delta = np.mean(np.abs(deltas[entry_idx-lookback:entry_idx]))
        avg_volume = np.mean(volumes[entry_idx-lookback:entry_idx])
    else:
        avg_delta = 1
        avg_volume = 1

    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(candles))):
        high = highs[i]
        low = lows[i]
        close = closes[i]

        cumulative_delta += deltas[i] if i < len(deltas) else 0
        cumulative_volume += volumes[i] if i < len(volumes) else 0

        if is_long:
            favorable = (high - entry_price) / entry_price
        else:
            favorable = (entry_price - low) / entry_price
        max_favorable = max(max_favorable, favorable)

        # SL 체크 (TP1 전)
        if not hit_tp1:
            if is_long and low <= sl_price:
                hit_sl = True
                break
            elif not is_long and high >= sl_price:
                hit_sl = True
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
                break
            elif not is_long and low <= tp2_price:
                hit_tp2 = True
                break

            # BE 체크
            if is_long and low <= entry_price:
                hit_be_after_tp1 = True
                break
            elif not is_long and high >= entry_price:
                hit_be_after_tp1 = True
                break

    # 결과
    trade_result = {
        'hit_tp1': hit_tp1,
        'hit_tp2': hit_tp2,
        'hit_sl': hit_sl,
        'hit_be_after_tp1': hit_be_after_tp1,
        'entry_price': entry_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'sl_price': sl_price,
        'is_long': is_long
    }

    # TP1 미도달
    if not hit_tp1:
        return None, trade_result

    # Dynamic exit features at TP1
    candles_to_tp1 = tp1_idx - entry_idx
    time_to_tp1 = candles_to_tp1 * 15

    tp1_closes = closes[:tp1_idx+1]
    momentum_at_tp1 = (closes[tp1_idx] - closes[tp1_idx-5]) / closes[tp1_idx-5] if tp1_idx >= 5 else 0
    rsi_at_tp1 = calculate_rsi(tp1_closes)

    if tp1_idx >= 14:
        tr = np.maximum(
            highs[tp1_idx-14:tp1_idx] - lows[tp1_idx-14:tp1_idx],
            np.abs(highs[tp1_idx-14:tp1_idx] - closes[tp1_idx-15:tp1_idx-1])
        )
        atr_at_tp1 = np.mean(tr)
    else:
        atr_at_tp1 = 0

    price_vs_tp1 = (closes[tp1_idx] - tp1_price) / tp1_price

    if is_long:
        distance_to_tp2 = (tp2_price - tp1_price) / tp1_price
    else:
        distance_to_tp2 = (tp1_price - tp2_price) / tp1_price

    last_body = abs(closes[tp1_idx] - opens[tp1_idx])
    last_body_pct = last_body / closes[tp1_idx]
    last_is_bullish = 1 if closes[tp1_idx] > opens[tp1_idx] else 0

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

    return features, trade_result


def collect_all_data(df_1h, df_15m):
    """모든 트레이드 데이터 수집 (Entry 피처 + Dynamic Exit 피처 + 결과)."""
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()

    # 수집할 데이터
    entry_features_list = []
    entry_labels = []
    dynamic_features_list = []  # None if TP1 미도달
    dynamic_labels = []  # EXIT_AT_TP1 or HOLD_FOR_TP2
    trade_results = []
    timestamps = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    for i in tqdm(range(50, len(df_15m) - 250), desc='Collecting data'):
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
        fakeout_signal = htf_fakeout_map.get(htf_idx)
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
                        # Entry 피처
                        entry_feat = extract_features(df_15m, i, f_channel, 'LONG', 'FAKEOUT', fakeout_signal.extreme)
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'LONG', entry, sl, tp1, tp2)
                        entry_label = TAKE if is_win else SKIP

                        # Full simulation
                        dyn_feat, result = simulate_trade_full(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2, f_width, True
                        )

                        entry_features_list.append(entry_feat)
                        entry_labels.append(entry_label)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                        trade_results.append(result)
                        timestamps.append(df_15m.index[i])
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        entry_feat = extract_features(df_15m, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme)
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)
                        entry_label = TAKE if is_win else SKIP

                        dyn_feat, result = simulate_trade_full(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2, f_width, True
                        )

                        entry_features_list.append(entry_feat)
                        entry_labels.append(entry_label)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                        trade_results.append(result)
                        timestamps.append(df_15m.index[i])
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
                entry_feat = extract_features(df_15m, i, channel, 'LONG', 'BOUNCE', None)
                is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'LONG', entry, sl, tp1, tp2)
                entry_label = TAKE if is_win else SKIP

                dyn_feat, result = simulate_trade_full(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2, channel_width, False
                )

                entry_features_list.append(entry_feat)
                entry_labels.append(entry_label)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                trade_results.append(result)
                timestamps.append(df_15m.index[i])
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                entry_feat = extract_features(df_15m, i, channel, 'SHORT', 'BOUNCE', None)
                is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)
                entry_label = TAKE if is_win else SKIP

                dyn_feat, result = simulate_trade_full(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2, channel_width, False
                )

                entry_features_list.append(entry_feat)
                entry_labels.append(entry_label)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                trade_results.append(result)
                timestamps.append(df_15m.index[i])
                traded_entries.add(trade_key)

    return {
        'entry_features': entry_features_list,
        'entry_labels': np.array(entry_labels),
        'dynamic_features': dynamic_features_list,
        'dynamic_labels': np.array(dynamic_labels),
        'trade_results': trade_results,
        'timestamps': timestamps
    }


def entry_features_to_array(features_list):
    """Entry 피처를 배열로 변환."""
    return np.array([[
        f.channel_width_pct, f.support_touches, f.resistance_touches, f.total_touches,
        f.price_in_channel_pct, f.volume_ratio, f.delta_ratio, f.cvd_recent,
        f.volume_ma_20, f.delta_ma_20, f.atr_14, f.atr_ratio,
        f.momentum_5, f.momentum_20, f.rsi_14, f.is_bounce, f.is_long,
        f.body_size_pct, f.wick_ratio, f.is_bullish, f.hour, f.day_of_week,
        f.fakeout_depth_pct
    ] for f in features_list])


def dynamic_features_to_array(features_list):
    """Dynamic exit 피처를 배열로 변환 (None 항목 제외)."""
    valid = [(i, f) for i, f in enumerate(features_list) if f is not None]
    indices = [i for i, _ in valid]
    arr = np.array([[
        f.candles_to_tp1, f.time_to_tp1_minutes, f.delta_during_trade, f.volume_during_trade,
        f.delta_ratio_during, f.volume_ratio_during, f.momentum_at_tp1, f.rsi_at_tp1,
        f.atr_at_tp1, f.max_favorable_excursion, f.price_vs_tp1, f.distance_to_tp2_pct,
        f.channel_width_pct, f.last_candle_body_pct, f.last_candle_is_bullish,
        f.is_long, f.is_fakeout
    ] for _, f in valid])
    return indices, arr


def backtest(trade_results, entry_preds, exit_preds, exit_pred_map, label):
    """
    백테스트 실행.
    - entry_preds: TAKE(1) / SKIP(0) 배열
    - exit_preds: EXIT_AT_TP1(0) / HOLD_FOR_TP2(1) (TP1 도달 트레이드만)
    - exit_pred_map: trade_idx -> exit_pred_idx 매핑
    """
    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    wins = 0
    losses = 0
    peak = capital
    max_dd = 0
    trades_taken = 0
    trade_returns = []

    for idx, (result, take_trade) in enumerate(zip(trade_results, entry_preds)):
        if take_trade == ENTRY_SKIP:
            continue

        trades_taken += 1
        entry = result['entry_price']
        sl = result['sl_price']
        tp1 = result['tp1_price']
        tp2 = result['tp2_price']
        is_long = result['is_long']
        hit_tp1 = result['hit_tp1']
        hit_tp2 = result['hit_tp2']
        hit_sl = result['hit_sl']

        sl_dist = abs(entry - sl) / entry
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position = capital * leverage

        # PnL 계산
        if hit_sl:
            # SL 손실
            if is_long:
                pnl_pct = (sl - entry) / entry
            else:
                pnl_pct = (entry - sl) / entry
        elif not hit_tp1:
            # TP1도 못가고 타임아웃 (BE 가정)
            pnl_pct = 0
        else:
            # TP1 도달 → exit 전략 적용
            if idx in exit_pred_map:
                exit_pred = exit_preds[exit_pred_map[idx]]
            else:
                exit_pred = EXIT_AT_TP1  # fallback

            if exit_pred == EXIT_AT_TP1:
                # TP1에서 전량 청산
                if is_long:
                    pnl_pct = (tp1 - entry) / entry
                else:
                    pnl_pct = (entry - tp1) / entry
            else:  # HOLD_FOR_TP2
                if hit_tp2:
                    # 50% TP1 + 50% TP2
                    if is_long:
                        pnl_pct = 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                    else:
                        pnl_pct = 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
                else:
                    # 50% TP1 + 50% BE
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
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades: {trades_taken}")
    print(f"  Win Rate: {wr:.1f}% ({wins}W / {losses}L)")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final: ${capital:,.2f}")
    print(f"  Return: {ret:+,.1f}%")
    if len(trade_returns) > 0:
        print(f"\n  Per-trade: avg {trade_returns.mean():+.3f}%, median {np.median(trade_returns):+.3f}%")

    return capital, wr, trades_taken


def main():
    print("="*60)
    print("  ML FINAL COMBINED TEST")
    print("  ML Entry (0.7) + ML Dynamic Exit")
    print("  Train: 2022-2023 | Test: 2024-2025")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')
    print(f"  1H: {len(df_1h)}, 15m: {len(df_15m)}")
    print(f"  Range: {df_15m.index.min()} ~ {df_15m.index.max()}")

    # Collect all data
    print("\nCollecting all trade data...")
    data = collect_all_data(df_1h, df_15m)
    print(f"  Total signals: {len(data['entry_labels'])}")

    # Split IS/OOS
    years = np.array([t.year for t in data['timestamps']])
    is_mask = np.isin(years, [2022, 2023])
    oos_mask = np.isin(years, [2024, 2025])

    print(f"\n  IS (2022-2023): {is_mask.sum()} trades")
    print(f"  OOS (2024-2025): {oos_mask.sum()} trades")

    # ========== ENTRY MODEL ==========
    print("\n" + "="*60)
    print("  Training ENTRY Model (TAKE/SKIP)")
    print("="*60)

    X_entry = entry_features_to_array(data['entry_features'])
    y_entry = data['entry_labels']

    X_entry_is = X_entry[is_mask]
    y_entry_is = y_entry[is_mask]

    entry_scaler = StandardScaler()
    X_entry_is_scaled = entry_scaler.fit_transform(X_entry_is)

    entry_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    entry_model.fit(X_entry_is_scaled, y_entry_is)

    print(f"  Samples: {len(X_entry_is)}")
    print(f"  TAKE: {(y_entry_is == TAKE).sum()} ({(y_entry_is == TAKE).mean()*100:.1f}%)")
    print(f"  SKIP: {(y_entry_is == SKIP).sum()} ({(y_entry_is == SKIP).mean()*100:.1f}%)")
    print(f"  Accuracy: {entry_model.score(X_entry_is_scaled, y_entry_is):.3f}")

    # ========== DYNAMIC EXIT MODEL ==========
    print("\n" + "="*60)
    print("  Training DYNAMIC EXIT Model (EXIT/HOLD at TP1)")
    print("="*60)

    # TP1 도달한 트레이드만 사용
    dyn_indices_is, X_dyn_is = dynamic_features_to_array(
        [data['dynamic_features'][i] for i in range(len(data['dynamic_features'])) if is_mask[i]]
    )
    y_dyn_is = np.array([
        data['dynamic_labels'][i] for i in range(len(data['dynamic_labels']))
        if is_mask[i] and data['dynamic_features'][i] is not None
    ])

    # is_mask 인덱스 중 TP1 도달한 것만
    is_indices = np.where(is_mask)[0]
    dyn_is_global_indices = [is_indices[i] for i in dyn_indices_is]

    exit_scaler = StandardScaler()
    X_dyn_is_scaled = exit_scaler.fit_transform(X_dyn_is)

    exit_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    exit_model.fit(X_dyn_is_scaled, y_dyn_is)

    print(f"  Samples (TP1 hit): {len(X_dyn_is)}")
    print(f"  EXIT_AT_TP1: {(y_dyn_is == EXIT_AT_TP1).sum()} ({(y_dyn_is == EXIT_AT_TP1).mean()*100:.1f}%)")
    print(f"  HOLD_FOR_TP2: {(y_dyn_is == HOLD_FOR_TP2).sum()} ({(y_dyn_is == HOLD_FOR_TP2).mean()*100:.1f}%)")
    print(f"  Accuracy: {exit_model.score(X_dyn_is_scaled, y_dyn_is):.3f}")

    # ========== OOS PREDICTION ==========
    print("\n" + "="*60)
    print("  OOS PREDICTIONS (2024-2025)")
    print("="*60)

    # Entry predictions
    X_entry_oos = X_entry[oos_mask]
    X_entry_oos_scaled = entry_scaler.transform(X_entry_oos)
    entry_probs_oos = entry_model.predict_proba(X_entry_oos_scaled)[:, 1]

    # Exit predictions for TP1 hit trades
    oos_indices = np.where(oos_mask)[0]
    dyn_indices_oos_local, X_dyn_oos = dynamic_features_to_array(
        [data['dynamic_features'][i] for i in oos_indices]
    )

    # Mapping: global idx -> local exit pred idx
    exit_pred_map = {}
    for local_exit_idx, local_idx in enumerate(dyn_indices_oos_local):
        global_idx = oos_indices[local_idx]
        exit_pred_map[global_idx] = local_exit_idx

    if len(X_dyn_oos) > 0:
        X_dyn_oos_scaled = exit_scaler.transform(X_dyn_oos)
        exit_preds_oos = exit_model.predict(X_dyn_oos_scaled)
        exit_probs_oos = exit_model.predict_proba(X_dyn_oos_scaled)[:, 1]
    else:
        exit_preds_oos = np.array([])
        exit_probs_oos = np.array([])

    # OOS trades
    oos_results = [data['trade_results'][i] for i in oos_indices]

    # ========== BACKTEST ==========
    print("\n" + "="*60)
    print("  BACKTEST RESULTS (2024-2025)")
    print("="*60)

    # 1. Baseline (No ML, All trades, TP2 fixed)
    baseline_entry = np.ones(len(oos_results), dtype=int)
    baseline_exit = np.zeros(len(exit_preds_oos), dtype=int)  # Always EXIT at TP1
    # Actually for baseline TP2 strategy, we need HOLD
    baseline_exit_hold = np.ones(len(exit_preds_oos), dtype=int)  # Always HOLD for TP2

    # For baseline, use global indices mapping
    baseline_map = {oos_indices[local_idx]: local_exit_idx
                    for local_exit_idx, local_idx in enumerate(dyn_indices_oos_local)}

    backtest(oos_results, baseline_entry, baseline_exit_hold, baseline_map,
             "1. Baseline (No ML, All trades, HOLD for TP2)")

    # 2. ML Entry Only (threshold=0.7, HOLD for TP2)
    entry_preds_70 = (entry_probs_oos >= 0.7).astype(int)
    backtest(oos_results, entry_preds_70, baseline_exit_hold, baseline_map,
             "2. ML Entry Only (threshold=0.7, HOLD for TP2)")

    # 3. ML Dynamic Exit Only (All trades)
    backtest(oos_results, baseline_entry, exit_preds_oos, exit_pred_map,
             "3. ML Dynamic Exit Only (All trades)")

    # 4. ML Combined: Entry(0.7) + Dynamic Exit
    backtest(oos_results, entry_preds_70, exit_preds_oos, exit_pred_map,
             "4. ML COMBINED: Entry(0.7) + Dynamic Exit")

    # Additional thresholds
    print("\n" + "="*60)
    print("  ADDITIONAL THRESHOLD TESTS")
    print("="*60)

    entry_preds_60 = (entry_probs_oos >= 0.6).astype(int)
    backtest(oos_results, entry_preds_60, exit_preds_oos, exit_pred_map,
             "ML Combined: Entry(0.6) + Dynamic Exit")

    entry_preds_50 = (entry_probs_oos >= 0.5).astype(int)
    backtest(oos_results, entry_preds_50, exit_preds_oos, exit_pred_map,
             "ML Combined: Entry(0.5) + Dynamic Exit")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print("""
  1. Baseline: 모든 신호 진입, HOLD for TP2 고정
  2. ML Entry Only: 저품질 신호 필터링, HOLD for TP2 고정
  3. ML Dynamic Exit Only: 모든 신호 진입, TP1에서 동적 결정
  4. ML Combined: Entry 필터 + Dynamic Exit (최적 조합)
    """)

    # Stats
    print(f"\n  Entry Model filtered {(1 - entry_preds_70.mean())*100:.1f}% of signals (threshold=0.7)")
    if len(exit_preds_oos) > 0:
        print(f"  Exit Model chose EXIT at TP1: {(exit_preds_oos == EXIT_AT_TP1).mean()*100:.1f}%")
        print(f"  Exit Model chose HOLD for TP2: {(exit_preds_oos == HOLD_FOR_TP2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
