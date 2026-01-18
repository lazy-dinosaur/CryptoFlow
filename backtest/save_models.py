#!/usr/bin/env python3
"""
Save trained ML models for paper trading.
- Entry Model: TAKE/SKIP filter
- Dynamic Exit Model: EXIT/HOLD at TP1
- Training Data: 2024-2025 (most recent)

실행: python save_models.py
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels
from ml_exit import extract_features
from ml_entry import simulate_trade_for_entry_label, TAKE, SKIP
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple
import os

# Labels
ENTRY_TAKE = 1
ENTRY_SKIP = 0
EXIT_AT_TP1 = 0
HOLD_FOR_TP2 = 1


@dataclass
class DynamicExitFeatures:
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


def simulate_trade_full(candles, entry_idx, trade_type, entry_price, sl_price,
                        tp1_price, tp2_price, channel_width_pct, is_fakeout):
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

    cumulative_delta = 0
    cumulative_volume = 0
    max_favorable = 0

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

        cumulative_delta += deltas[i] if i < len(deltas) else 0
        cumulative_volume += volumes[i] if i < len(volumes) else 0

        if is_long:
            favorable = (high - entry_price) / entry_price
        else:
            favorable = (entry_price - low) / entry_price
        max_favorable = max(max_favorable, favorable)

        if not hit_tp1:
            if is_long and low <= sl_price:
                hit_sl = True
                break
            elif not is_long and high >= sl_price:
                hit_sl = True
                break

        if not hit_tp1:
            if is_long and high >= tp1_price:
                hit_tp1 = True
                tp1_idx = i
            elif not is_long and low <= tp1_price:
                hit_tp1 = True
                tp1_idx = i

        if hit_tp1 and not hit_tp2:
            if is_long and high >= tp2_price:
                hit_tp2 = True
                break
            elif not is_long and low <= tp2_price:
                hit_tp2 = True
                break

            if is_long and low <= entry_price:
                break
            elif not is_long and high >= entry_price:
                break

    if not hit_tp1:
        return None, hit_tp2

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

    return features, hit_tp2


def collect_training_data(df_1h, df_15m):
    """Collect training data from 2022-2023."""
    channels_dict, fakeout_signals = build_htf_channels(df_1h)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003
    tf_ratio = 4
    htf_fakeout_map = {fs.htf_idx: fs for fs in fakeout_signals}

    traded_entries = set()
    entry_features_list = []
    entry_labels = []
    dynamic_features_list = []
    dynamic_labels = []
    timestamps = []

    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    for i in tqdm(range(50, len(df_15m) - 250), desc='Collecting training data'):
        # Use 2024-2025 for training (most recent data)
        ts = df_15m.index[i]
        if ts.year not in [2024, 2025]:
            continue

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
                        entry_feat = extract_features(df_15m, i, f_channel, 'LONG', 'FAKEOUT', fakeout_signal.extreme)
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'LONG', entry, sl, tp1, tp2)

                        dyn_feat, hit_tp2 = simulate_trade_full(
                            df_15m, i, 'LONG', entry, sl, tp1, tp2, f_width, True
                        )

                        entry_features_list.append(entry_feat)
                        entry_labels.append(TAKE if is_win else SKIP)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if hit_tp2 else EXIT_AT_TP1)
                        timestamps.append(ts)
                        traded_entries.add(trade_key)
                else:
                    entry = current_close
                    sl = fakeout_signal.extreme * (1 + sl_buffer_pct)
                    tp1 = f_mid
                    tp2 = f_channel.support * 1.002

                    if sl > entry and entry > tp1:
                        entry_feat = extract_features(df_15m, i, f_channel, 'SHORT', 'FAKEOUT', fakeout_signal.extreme)
                        is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)

                        dyn_feat, hit_tp2 = simulate_trade_full(
                            df_15m, i, 'SHORT', entry, sl, tp1, tp2, f_width, True
                        )

                        entry_features_list.append(entry_feat)
                        entry_labels.append(TAKE if is_win else SKIP)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if hit_tp2 else EXIT_AT_TP1)
                        timestamps.append(ts)
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

                dyn_feat, hit_tp2 = simulate_trade_full(
                    df_15m, i, 'LONG', entry, sl, tp1, tp2, channel_width, False
                )

                entry_features_list.append(entry_feat)
                entry_labels.append(TAKE if is_win else SKIP)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if hit_tp2 else EXIT_AT_TP1)
                timestamps.append(ts)
                traded_entries.add(trade_key)

        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                entry_feat = extract_features(df_15m, i, channel, 'SHORT', 'BOUNCE', None)
                is_win, _ = simulate_trade_for_entry_label(df_15m, i, 'SHORT', entry, sl, tp1, tp2)

                dyn_feat, hit_tp2 = simulate_trade_full(
                    df_15m, i, 'SHORT', entry, sl, tp1, tp2, channel_width, False
                )

                entry_features_list.append(entry_feat)
                entry_labels.append(TAKE if is_win else SKIP)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if hit_tp2 else EXIT_AT_TP1)
                timestamps.append(ts)
                traded_entries.add(trade_key)

    return {
        'entry_features': entry_features_list,
        'entry_labels': np.array(entry_labels),
        'dynamic_features': dynamic_features_list,
        'dynamic_labels': np.array(dynamic_labels)
    }


def entry_features_to_array(features_list):
    return np.array([[
        f.channel_width_pct, f.support_touches, f.resistance_touches, f.total_touches,
        f.price_in_channel_pct, f.volume_ratio, f.delta_ratio, f.cvd_recent,
        f.volume_ma_20, f.delta_ma_20, f.atr_14, f.atr_ratio,
        f.momentum_5, f.momentum_20, f.rsi_14, f.is_bounce, f.is_long,
        f.body_size_pct, f.wick_ratio, f.is_bullish, f.hour, f.day_of_week,
        f.fakeout_depth_pct
    ] for f in features_list])


def dynamic_features_to_array(features_list):
    valid = [f for f in features_list if f is not None]
    return np.array([[
        f.candles_to_tp1, f.time_to_tp1_minutes, f.delta_during_trade, f.volume_during_trade,
        f.delta_ratio_during, f.volume_ratio_during, f.momentum_at_tp1, f.rsi_at_tp1,
        f.atr_at_tp1, f.max_favorable_excursion, f.price_vs_tp1, f.distance_to_tp2_pct,
        f.channel_width_pct, f.last_candle_body_pct, f.last_candle_is_bullish,
        f.is_long, f.is_fakeout
    ] for f in valid])


def main():
    print("="*60)
    print("  SAVE ML MODELS FOR PAPER TRADING")
    print("  Training on 2024-2025 data (most recent)")
    print("="*60)

    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df_1h = load_candles('BTCUSDT', '1h').to_pandas().set_index('time')
    df_15m = load_candles('BTCUSDT', '15m').to_pandas().set_index('time')
    print(f"  1H: {len(df_1h)}, 15m: {len(df_15m)}")

    # Collect training data (2022-2023 only)
    print("\nCollecting training data (2022-2023)...")
    data = collect_training_data(df_1h, df_15m)
    print(f"  Total training samples: {len(data['entry_labels'])}")

    # ========== TRAIN ENTRY MODEL ==========
    print("\n" + "="*60)
    print("  Training ENTRY Model")
    print("="*60)

    X_entry = entry_features_to_array(data['entry_features'])
    y_entry = data['entry_labels']

    entry_scaler = StandardScaler()
    X_entry_scaled = entry_scaler.fit_transform(X_entry)

    entry_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    entry_model.fit(X_entry_scaled, y_entry)

    print(f"  Samples: {len(X_entry)}")
    print(f"  TAKE: {(y_entry == TAKE).sum()} ({(y_entry == TAKE).mean()*100:.1f}%)")
    print(f"  SKIP: {(y_entry == SKIP).sum()} ({(y_entry == SKIP).mean()*100:.1f}%)")
    print(f"  Accuracy: {entry_model.score(X_entry_scaled, y_entry):.3f}")

    # ========== TRAIN DYNAMIC EXIT MODEL ==========
    print("\n" + "="*60)
    print("  Training DYNAMIC EXIT Model")
    print("="*60)

    # Only use samples where TP1 was hit
    valid_dyn_mask = [f is not None for f in data['dynamic_features']]
    valid_dyn_features = [f for f in data['dynamic_features'] if f is not None]
    valid_dyn_labels = data['dynamic_labels'][valid_dyn_mask]

    X_dyn = dynamic_features_to_array(valid_dyn_features)
    y_dyn = valid_dyn_labels

    exit_scaler = StandardScaler()
    X_dyn_scaled = exit_scaler.fit_transform(X_dyn)

    exit_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    exit_model.fit(X_dyn_scaled, y_dyn)

    print(f"  Samples (TP1 hit): {len(X_dyn)}")
    print(f"  EXIT_AT_TP1: {(y_dyn == EXIT_AT_TP1).sum()} ({(y_dyn == EXIT_AT_TP1).mean()*100:.1f}%)")
    print(f"  HOLD_FOR_TP2: {(y_dyn == HOLD_FOR_TP2).sum()} ({(y_dyn == HOLD_FOR_TP2).mean()*100:.1f}%)")
    print(f"  Accuracy: {exit_model.score(X_dyn_scaled, y_dyn):.3f}")

    # ========== SAVE MODELS ==========
    print("\n" + "="*60)
    print("  Saving Models")
    print("="*60)

    # Save Entry model and scaler
    entry_model_path = os.path.join(models_dir, 'entry_model.joblib')
    entry_scaler_path = os.path.join(models_dir, 'entry_scaler.joblib')
    joblib.dump(entry_model, entry_model_path)
    joblib.dump(entry_scaler, entry_scaler_path)
    print(f"  Entry model saved: {entry_model_path}")
    print(f"  Entry scaler saved: {entry_scaler_path}")

    # Save Exit model and scaler
    exit_model_path = os.path.join(models_dir, 'exit_model.joblib')
    exit_scaler_path = os.path.join(models_dir, 'exit_scaler.joblib')
    joblib.dump(exit_model, exit_model_path)
    joblib.dump(exit_scaler, exit_scaler_path)
    print(f"  Exit model saved: {exit_model_path}")
    print(f"  Exit scaler saved: {exit_scaler_path}")

    # Save feature names for reference
    entry_feature_names = [
        'channel_width_pct', 'support_touches', 'resistance_touches', 'total_touches',
        'price_in_channel_pct', 'volume_ratio', 'delta_ratio', 'cvd_recent',
        'volume_ma_20', 'delta_ma_20', 'atr_14', 'atr_ratio',
        'momentum_5', 'momentum_20', 'rsi_14', 'is_bounce', 'is_long',
        'body_size_pct', 'wick_ratio', 'is_bullish', 'hour', 'day_of_week',
        'fakeout_depth_pct'
    ]
    exit_feature_names = [
        'candles_to_tp1', 'time_to_tp1_minutes', 'delta_during_trade', 'volume_during_trade',
        'delta_ratio_during', 'volume_ratio_during', 'momentum_at_tp1', 'rsi_at_tp1',
        'atr_at_tp1', 'max_favorable_excursion', 'price_vs_tp1', 'distance_to_tp2_pct',
        'channel_width_pct', 'last_candle_body_pct', 'last_candle_is_bullish',
        'is_long', 'is_fakeout'
    ]

    import json
    metadata = {
        'entry_feature_names': entry_feature_names,
        'exit_feature_names': exit_feature_names,
        'entry_threshold': 0.7,
        'trained_on': '2022-2023',
        'entry_accuracy': float(entry_model.score(X_entry_scaled, y_entry)),
        'exit_accuracy': float(exit_model.score(X_dyn_scaled, y_dyn))
    }
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    print("\n" + "="*60)
    print("  DONE! Models ready for paper trading")
    print("="*60)


if __name__ == "__main__":
    main()
