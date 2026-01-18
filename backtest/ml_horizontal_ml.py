#!/usr/bin/env python3
"""
Horizontal Channel Strategy with ML Entry/Exit

새로운 채널 감지 로직:
1. 스윙 감지: pivothigh 스타일 (양쪽 N개 캔들 비교)
2. 저장: 최근 3개 스윙 하이/로우만
3. 채널 조건: 2개가 수평(tolerance 이내)이면 채널 형성
4. 채널 높이: 0.5% 이상

ML 전략:
- ML Entry (threshold=0.7): 저품질 신호 필터링
- ML Dynamic Exit: TP1 도달 시 청산 vs 홀딩 결정
- Train: 2022-2023 | Test: 2024-2025
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


# Labels
TAKE = 1
SKIP = 0
EXIT_AT_TP1 = 0
HOLD_FOR_TP2 = 1


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class Channel:
    support: float
    support_idx: int
    resistance: float
    resistance_idx: int
    start_idx: int


@dataclass
class EntryFeatures:
    channel_width_pct: float
    price_in_channel_pct: float
    volume_ratio: float
    delta_ratio: float
    cvd_recent: float
    volume_ma_20: float
    delta_ma_20: float
    atr_14: float
    atr_ratio: float
    momentum_5: float
    momentum_20: float
    rsi_14: float
    is_bounce: int
    is_long: int
    body_size_pct: float
    wick_ratio: float
    is_bullish: int
    hour: int
    day_of_week: int
    fakeout_depth_pct: float


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


def find_pivot_swing_points(candles: pd.DataFrame, swing_len: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find swing highs and lows using pivot method (both sides comparison).
    Similar to TradingView's ta.pivothigh/ta.pivotlow.
    """
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    for i in range(swing_len, len(candles) - swing_len):
        # Check swing high
        is_swing_high = True
        for j in range(1, swing_len + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break

        if is_swing_high:
            swing_highs.append(SwingPoint(idx=i, price=highs[i], type='high'))

        # Check swing low
        is_swing_low = True
        for j in range(1, swing_len + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break

        if is_swing_low:
            swing_lows.append(SwingPoint(idx=i, price=lows[i], type='low'))

    return swing_highs, swing_lows


def build_horizontal_channels(htf_candles: pd.DataFrame,
                               swing_len: int = 3,
                               tolerance: float = 0.005,
                               min_channel_height: float = 0.005) -> Dict[int, Channel]:
    """
    Build horizontal channels using pivot swing points.
    """
    swing_highs, swing_lows = find_pivot_swing_points(htf_candles, swing_len)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values
    htf_channel_map: Dict[int, Channel] = {}

    recent_highs: List[SwingPoint] = []
    recent_lows: List[SwingPoint] = []

    # Build map of when swings are confirmed (at idx + swing_len)
    high_confirm_map = {sh.idx + swing_len: sh for sh in swing_highs}
    low_confirm_map = {sl.idx + swing_len: sl for sl in swing_lows}

    channels_found = 0

    for i in range(len(htf_candles)):
        current_close = closes[i]

        # Check if new swing point confirmed at this index
        if i in high_confirm_map:
            sh = high_confirm_map[i]
            recent_highs.insert(0, sh)
            if len(recent_highs) > 3:
                recent_highs.pop()

        if i in low_confirm_map:
            sl = low_confirm_map[i]
            recent_lows.insert(0, sl)
            if len(recent_lows) > 3:
                recent_lows.pop()

        channel = None

        # Case 1: 2 lows are horizontal + 1 high
        if len(recent_lows) >= 2 and len(recent_highs) >= 1:
            l1 = recent_lows[0]
            l2 = recent_lows[1]
            h1 = recent_highs[0]

            low_avg = (l1.price + l2.price) / 2
            low_diff = abs(l1.price - l2.price) / low_avg

            if low_diff <= tolerance:
                support_price = low_avg
                resistance_price = h1.price
                channel_height = (resistance_price - support_price) / support_price

                if channel_height > min_channel_height:
                    start_bar = min(l2.idx, h1.idx)
                    channel = Channel(
                        support=support_price,
                        support_idx=l1.idx,
                        resistance=resistance_price,
                        resistance_idx=h1.idx,
                        start_idx=start_bar
                    )

        # Case 2: 2 highs are horizontal + 1 low
        if channel is None and len(recent_highs) >= 2 and len(recent_lows) >= 1:
            h1 = recent_highs[0]
            h2 = recent_highs[1]
            l1 = recent_lows[0]

            high_avg = (h1.price + h2.price) / 2
            high_diff = abs(h1.price - h2.price) / high_avg

            if high_diff <= tolerance:
                resistance_price = high_avg
                support_price = l1.price
                channel_height = (resistance_price - support_price) / support_price

                if channel_height > min_channel_height:
                    start_bar = min(h2.idx, l1.idx)
                    channel = Channel(
                        support=support_price,
                        support_idx=l1.idx,
                        resistance=resistance_price,
                        resistance_idx=h1.idx,
                        start_idx=start_bar
                    )

        # Check if price is within channel range
        if channel:
            if channel.support * 0.98 <= current_close <= channel.resistance * 1.02:
                htf_channel_map[i] = channel
                channels_found += 1

    print(f"  HTF Candles with Channel: {channels_found}")
    return htf_channel_map


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


def extract_entry_features(candles: pd.DataFrame, idx: int, channel: Channel,
                           trade_type: str, setup_type: str, fakeout_extreme: float = None) -> EntryFeatures:
    """Extract features for entry prediction."""
    closes = candles['close'].values
    highs = candles['high'].values
    lows = candles['low'].values
    opens = candles['open'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values if 'delta' in candles.columns else np.zeros(len(candles))

    lookback = 20
    start = max(0, idx - lookback)

    avg_volume = volumes[start:idx].mean() if idx > start else volumes[idx]
    avg_delta = np.abs(deltas[start:idx]).mean() if idx > start else abs(deltas[idx])

    # ATR
    if idx >= 14:
        tr = np.maximum(
            highs[idx-14:idx] - lows[idx-14:idx],
            np.abs(highs[idx-14:idx] - closes[idx-15:idx-1])
        )
        atr_14 = np.mean(tr)
    else:
        atr_14 = 0

    channel_width = (channel.resistance - channel.support) / channel.support
    price_in_channel = (closes[idx] - channel.support) / (channel.resistance - channel.support)

    body = abs(closes[idx] - opens[idx])
    candle_range = highs[idx] - lows[idx]
    body_size_pct = body / closes[idx] if closes[idx] > 0 else 0
    wick_ratio = (candle_range - body) / candle_range if candle_range > 0 else 0

    fakeout_depth = 0
    if fakeout_extreme and setup_type == 'FAKEOUT':
        if trade_type == 'LONG':
            fakeout_depth = (channel.support - fakeout_extreme) / channel.support * 100
        else:
            fakeout_depth = (fakeout_extreme - channel.resistance) / channel.resistance * 100

    ts = candles.index[idx]
    hour = ts.hour if hasattr(ts, 'hour') else 0
    dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0

    return EntryFeatures(
        channel_width_pct=channel_width * 100,
        price_in_channel_pct=price_in_channel,
        volume_ratio=volumes[idx] / avg_volume if avg_volume > 0 else 1,
        delta_ratio=deltas[idx] / (avg_delta + 1e-10),
        cvd_recent=deltas[start:idx].sum() if idx > start else 0,
        volume_ma_20=avg_volume,
        delta_ma_20=avg_delta,
        atr_14=atr_14,
        atr_ratio=atr_14 / closes[idx] if closes[idx] > 0 else 0,
        momentum_5=(closes[idx] - closes[idx-5]) / closes[idx-5] if idx >= 5 else 0,
        momentum_20=(closes[idx] - closes[idx-20]) / closes[idx-20] if idx >= 20 else 0,
        rsi_14=calculate_rsi(closes[:idx+1]),
        is_bounce=1 if setup_type == 'BOUNCE' else 0,
        is_long=1 if trade_type == 'LONG' else 0,
        body_size_pct=body_size_pct,
        wick_ratio=wick_ratio,
        is_bullish=1 if closes[idx] > opens[idx] else 0,
        hour=hour,
        day_of_week=dow,
        fakeout_depth_pct=fakeout_depth
    )


def simulate_trade_full(candles: pd.DataFrame, entry_idx: int, trade_type: str,
                        entry_price: float, sl_price: float, tp1_price: float, tp2_price: float,
                        channel_width_pct: float, is_fakeout: bool
                        ) -> Tuple[Optional[DynamicExitFeatures], dict]:
    """Full trade simulation with dynamic exit features."""
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

        # SL check (before TP1)
        if not hit_tp1:
            if is_long and low <= sl_price:
                hit_sl = True
                break
            elif not is_long and high >= sl_price:
                hit_sl = True
                break

        # TP1 check
        if not hit_tp1:
            if is_long and high >= tp1_price:
                hit_tp1 = True
                tp1_idx = i
            elif not is_long and low <= tp1_price:
                hit_tp1 = True
                tp1_idx = i

        # TP2 check (after TP1)
        if hit_tp1 and not hit_tp2:
            if is_long and high >= tp2_price:
                hit_tp2 = True
                break
            elif not is_long and low <= tp2_price:
                hit_tp2 = True
                break

            # BE check
            if is_long and low <= entry_price:
                hit_be_after_tp1 = True
                break
            elif not is_long and high >= entry_price:
                hit_be_after_tp1 = True
                break

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


def simulate_trade_for_entry_label(candles, idx, trade_type, entry, sl, tp1, tp2):
    """Simulate trade to determine if it's a winning entry."""
    highs = candles['high'].values
    lows = candles['low'].values

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == 'LONG':
            if lows[j] <= sl:
                return False, 0  # Loss
            if highs[j] >= tp1:
                return True, 1  # Win (at least TP1)
        else:
            if highs[j] >= sl:
                return False, 0
            if lows[j] <= tp1:
                return True, 1

    return False, 0  # Timeout


def collect_all_data(df_htf, df_ltf, htf_tf='1h', ltf_tf='15m', swing_len=3, tolerance=0.005):
    """Collect all trade data with horizontal channel detection."""
    htf_channel_map = build_horizontal_channels(df_htf, swing_len, tolerance)

    sl_buffer_pct = 0.002
    touch_threshold = 0.003

    # Calculate timeframe ratio
    tf_mins = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    traded_entries = set()

    entry_features_list = []
    entry_labels = []
    dynamic_features_list = []
    dynamic_labels = []
    trade_results = []
    timestamps = []

    ltf_highs = df_ltf['high'].values
    ltf_lows = df_ltf['low'].values
    ltf_closes = df_ltf['close'].values

    # Track pending fakeouts
    pending_breaks: List[dict] = []
    max_fakeout_wait = 5 * tf_ratio

    for i in tqdm(range(50, len(df_ltf) - 250), desc='Collecting data'):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias

        # Process pending fakeouts even without active channel
        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_fakeout_wait:
                pending_breaks.remove(pb)
                continue

            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], current_low)
                if current_close > pb['channel'].support:
                    entry = current_close
                    sl = pb['extreme'] * (1 - sl_buffer_pct)
                    mid = (pb['channel'].resistance + pb['channel'].support) / 2
                    tp1 = mid
                    tp2 = pb['channel'].resistance * 0.998
                    f_width = (pb['channel'].resistance - pb['channel'].support) / pb['channel'].support

                    trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                    if trade_key not in traded_entries and entry > sl and tp1 > entry:
                        entry_feat = extract_entry_features(df_ltf, i, pb['channel'], 'LONG', 'FAKEOUT', pb['extreme'])
                        is_win, _ = simulate_trade_for_entry_label(df_ltf, i, 'LONG', entry, sl, tp1, tp2)
                        entry_label = TAKE if is_win else SKIP

                        dyn_feat, result = simulate_trade_full(df_ltf, i, 'LONG', entry, sl, tp1, tp2, f_width, True)

                        entry_features_list.append(entry_feat)
                        entry_labels.append(entry_label)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                        trade_results.append(result)
                        timestamps.append(df_ltf.index[i])
                        traded_entries.add(trade_key)
                    pending_breaks.remove(pb)

            else:  # bull fakeout
                pb['extreme'] = max(pb['extreme'], current_high)
                if current_close < pb['channel'].resistance:
                    entry = current_close
                    sl = pb['extreme'] * (1 + sl_buffer_pct)
                    mid = (pb['channel'].resistance + pb['channel'].support) / 2
                    tp1 = mid
                    tp2 = pb['channel'].support * 1.002
                    f_width = (pb['channel'].resistance - pb['channel'].support) / pb['channel'].support

                    trade_key = (round(pb['channel'].support), round(pb['channel'].resistance), 'fakeout', i)
                    if trade_key not in traded_entries and sl > entry and entry > tp1:
                        entry_feat = extract_entry_features(df_ltf, i, pb['channel'], 'SHORT', 'FAKEOUT', pb['extreme'])
                        is_win, _ = simulate_trade_for_entry_label(df_ltf, i, 'SHORT', entry, sl, tp1, tp2)
                        entry_label = TAKE if is_win else SKIP

                        dyn_feat, result = simulate_trade_full(df_ltf, i, 'SHORT', entry, sl, tp1, tp2, f_width, True)

                        entry_features_list.append(entry_feat)
                        entry_labels.append(entry_label)
                        dynamic_features_list.append(dyn_feat)
                        dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                        trade_results.append(result)
                        timestamps.append(df_ltf.index[i])
                        traded_entries.add(trade_key)
                    pending_breaks.remove(pb)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2
        channel_width = (channel.resistance - channel.support) / channel.support

        # Check for breakouts (potential fakeouts)
        if current_close < channel.support * 0.997:
            already_tracking = any(
                pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                for pb in pending_breaks
            )
            if not already_tracking:
                pending_breaks.append({
                    'type': 'bear',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_low
                })
        elif current_close > channel.resistance * 1.003:
            already_tracking = any(
                pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                for pb in pending_breaks
            )
            if not already_tracking:
                pending_breaks.append({
                    'type': 'bull',
                    'break_idx': i,
                    'channel': channel,
                    'extreme': current_high
                })

        # Bounce entries
        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 10)
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch -> LONG
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry = current_close
            sl = channel.support * (1 - sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                entry_feat = extract_entry_features(df_ltf, i, channel, 'LONG', 'BOUNCE', None)
                is_win, _ = simulate_trade_for_entry_label(df_ltf, i, 'LONG', entry, sl, tp1, tp2)
                entry_label = TAKE if is_win else SKIP

                dyn_feat, result = simulate_trade_full(df_ltf, i, 'LONG', entry, sl, tp1, tp2, channel_width, False)

                entry_features_list.append(entry_feat)
                entry_labels.append(entry_label)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                trade_results.append(result)
                timestamps.append(df_ltf.index[i])
                traded_entries.add(trade_key)

        # BOUNCE: Resistance touch -> SHORT
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry = current_close
            sl = channel.resistance * (1 + sl_buffer_pct)
            tp1 = mid_price
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                entry_feat = extract_entry_features(df_ltf, i, channel, 'SHORT', 'BOUNCE', None)
                is_win, _ = simulate_trade_for_entry_label(df_ltf, i, 'SHORT', entry, sl, tp1, tp2)
                entry_label = TAKE if is_win else SKIP

                dyn_feat, result = simulate_trade_full(df_ltf, i, 'SHORT', entry, sl, tp1, tp2, channel_width, False)

                entry_features_list.append(entry_feat)
                entry_labels.append(entry_label)
                dynamic_features_list.append(dyn_feat)
                dynamic_labels.append(HOLD_FOR_TP2 if result['hit_tp2'] else EXIT_AT_TP1)
                trade_results.append(result)
                timestamps.append(df_ltf.index[i])
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
    return np.array([[
        f.channel_width_pct, f.price_in_channel_pct, f.volume_ratio, f.delta_ratio, f.cvd_recent,
        f.volume_ma_20, f.delta_ma_20, f.atr_14, f.atr_ratio,
        f.momentum_5, f.momentum_20, f.rsi_14, f.is_bounce, f.is_long,
        f.body_size_pct, f.wick_ratio, f.is_bullish, f.hour, f.day_of_week, f.fakeout_depth_pct
    ] for f in features_list])


def dynamic_features_to_array(features_list):
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
        if take_trade == SKIP:
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

        if hit_sl:
            if is_long:
                pnl_pct = (sl - entry) / entry
            else:
                pnl_pct = (entry - sl) / entry
        elif not hit_tp1:
            pnl_pct = 0
        else:
            if idx in exit_pred_map:
                exit_pred = exit_preds[exit_pred_map[idx]]
            else:
                exit_pred = EXIT_AT_TP1

            if exit_pred == EXIT_AT_TP1:
                if is_long:
                    pnl_pct = (tp1 - entry) / entry
                else:
                    pnl_pct = (entry - tp1) / entry
            else:
                if hit_tp2:
                    if is_long:
                        pnl_pct = 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                    else:
                        pnl_pct = 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
                else:
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
    # Parse arguments: python script.py [htf] [ltf] [swing_len] [tolerance]
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    swing_len = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    tolerance = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    print("="*60)
    print("  HORIZONTAL CHANNEL + ML TEST")
    print(f"  HTF: {htf}, LTF: {ltf}")
    print(f"  Swing Length: {swing_len}, Tolerance: {tolerance}%")
    print("  Train: 2022-2023 | Test: 2024-2025")
    print("="*60)

    print(f"\nLoading data...")
    df_htf = load_candles('BTCUSDT', htf).to_pandas().set_index('time')
    df_ltf = load_candles('BTCUSDT', ltf).to_pandas().set_index('time')
    print(f"  HTF ({htf}): {len(df_htf)}, LTF ({ltf}): {len(df_ltf)}")
    print(f"  Range: {df_ltf.index.min()} ~ {df_ltf.index.max()}")

    print("\nCollecting all trade data...")
    data = collect_all_data(df_htf, df_ltf, htf, ltf, swing_len, tolerance / 100)
    print(f"  Total signals: {len(data['entry_labels'])}")

    if len(data['entry_labels']) == 0:
        print("  No signals found!")
        return

    # Split IS/OOS
    years = np.array([t.year for t in data['timestamps']])
    is_mask = np.isin(years, [2022, 2023])
    oos_mask = np.isin(years, [2024, 2025])

    print(f"\n  IS (2022-2023): {is_mask.sum()} trades")
    print(f"  OOS (2024-2025): {oos_mask.sum()} trades")

    if is_mask.sum() == 0:
        print("  No IS data for training!")
        return

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

    dyn_indices_is, X_dyn_is = dynamic_features_to_array(
        [data['dynamic_features'][i] for i in range(len(data['dynamic_features'])) if is_mask[i]]
    )
    y_dyn_is = np.array([
        data['dynamic_labels'][i] for i in range(len(data['dynamic_labels']))
        if is_mask[i] and data['dynamic_features'][i] is not None
    ])

    if len(X_dyn_is) == 0:
        print("  No TP1 hits in IS data!")
        return

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

    if oos_mask.sum() == 0:
        print("  No OOS data!")
        return

    X_entry_oos = X_entry[oos_mask]
    X_entry_oos_scaled = entry_scaler.transform(X_entry_oos)
    entry_probs_oos = entry_model.predict_proba(X_entry_oos_scaled)[:, 1]

    oos_indices = np.where(oos_mask)[0]
    dyn_indices_oos_local, X_dyn_oos = dynamic_features_to_array(
        [data['dynamic_features'][i] for i in oos_indices]
    )

    exit_pred_map = {}
    for local_exit_idx, local_idx in enumerate(dyn_indices_oos_local):
        global_idx = oos_indices[local_idx]
        exit_pred_map[global_idx] = local_exit_idx

    if len(X_dyn_oos) > 0:
        X_dyn_oos_scaled = exit_scaler.transform(X_dyn_oos)
        exit_preds_oos = exit_model.predict(X_dyn_oos_scaled)
    else:
        exit_preds_oos = np.array([])

    oos_results = [data['trade_results'][i] for i in oos_indices]

    # ========== BACKTEST ==========
    print("\n" + "="*60)
    print("  BACKTEST RESULTS (2024-2025)")
    print("="*60)

    # 1. Baseline (No ML)
    baseline_entry = np.ones(len(oos_results), dtype=int)
    baseline_exit_hold = np.ones(len(exit_preds_oos), dtype=int)
    baseline_map = {oos_indices[local_idx]: local_exit_idx
                    for local_exit_idx, local_idx in enumerate(dyn_indices_oos_local)}

    backtest(oos_results, baseline_entry, baseline_exit_hold, baseline_map,
             "1. Baseline (No ML, HOLD for TP2)")

    # 2. ML Entry Only
    entry_preds_70 = (entry_probs_oos >= 0.7).astype(int)
    backtest(oos_results, entry_preds_70, baseline_exit_hold, baseline_map,
             "2. ML Entry Only (threshold=0.7)")

    # 3. ML Dynamic Exit Only
    backtest(oos_results, baseline_entry, exit_preds_oos, exit_pred_map,
             "3. ML Dynamic Exit Only")

    # 4. ML Combined
    backtest(oos_results, entry_preds_70, exit_preds_oos, exit_pred_map,
             "4. ML COMBINED: Entry(0.7) + Dynamic Exit")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\n  Entry Model filtered {(1 - entry_preds_70.mean())*100:.1f}% of signals (threshold=0.7)")
    if len(exit_preds_oos) > 0:
        print(f"  Exit Model chose EXIT at TP1: {(exit_preds_oos == EXIT_AT_TP1).mean()*100:.1f}%")
        print(f"  Exit Model chose HOLD for TP2: {(exit_preds_oos == HOLD_FOR_TP2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
