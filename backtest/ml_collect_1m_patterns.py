#!/usr/bin/env python3
"""
1분봉 패턴 데이터 수집기

기존 백테스트에서 모든 채널 터치 포인트를 수집하고,
각 터치에서 1분봉 피처를 추출하여 학습 데이터 생성

라벨링:
- SUCCESS: TP1 또는 TP2 도달
- FAILURE: SL 도달

출력: CSV 파일 (features + label)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ml_volume_delta_features import extract_1m_features_for_df, get_feature_names

# ============== Configuration ==============
DATA_DIR = "data/parsed/btcusdt"

# Channel detection params (1h)
MIN_TOUCHES = 2
TOUCH_TOLERANCE = 0.004  # 0.4%
MIN_CHANNEL_WIDTH = 0.008  # 0.8%
MAX_CHANNEL_WIDTH = 0.05  # 5%

# Entry params
ENTRY_TOUCH_THRESHOLD = 0.003  # 0.3%
SL_BUFFER_PCT = 0.0008


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int = 1
    resistance_touches: int = 1
    lowest_low: float = 0.0
    highest_high: float = 0.0
    confirmed: bool = False


def load_data():
    """Load all timeframe data."""
    print("Loading data...")

    df_1h = pd.read_parquet(f"{DATA_DIR}/candles_1h.parquet")
    df_15m = pd.read_parquet(f"{DATA_DIR}/candles_15m.parquet")
    df_1m = pd.read_parquet(f"{DATA_DIR}/candles_1m.parquet")

    for df in [df_1h, df_15m, df_1m]:
        df['time'] = pd.to_datetime(df['time'])

    print(f"  1h: {len(df_1h):,} candles ({df_1h['time'].min()} ~ {df_1h['time'].max()})")
    print(f"  15m: {len(df_15m):,} candles")
    print(f"  1m: {len(df_1m):,} candles")

    return df_1h, df_15m, df_1m


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[dict], List[dict]]:
    """Find swing highs and lows using confirmation method."""
    highs = candles['high'].values
    lows = candles['low'].values

    swing_highs = []
    swing_lows = []

    potential_high_idx = 0
    potential_high_price = highs[0]
    candles_since_high = 0

    potential_low_idx = 0
    potential_low_price = lows[0]
    candles_since_low = 0

    for i in range(1, len(candles)):
        if highs[i] > potential_high_price:
            potential_high_idx = i
            potential_high_price = highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            if candles_since_high == confirm_candles:
                swing_highs.append({'idx': potential_high_idx, 'price': potential_high_price})

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append({'idx': potential_low_idx, 'price': potential_low_price})

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def build_htf_channels(htf_candles: pd.DataFrame) -> Dict[int, Channel]:
    """Build evolving channels on HTF."""
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values
    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

    for i in range(len(htf_candles)):
        current_close = closes[i]

        new_high = None
        new_low = None

        for sh in swing_highs:
            if sh['idx'] + 3 == i:
                new_high = sh
                break

        for sl in swing_lows:
            if sl['idx'] + 3 == i:
                new_low = sl
                break

        valid_swing_lows = [sl for sl in swing_lows if sl['idx'] + 3 <= i]
        valid_swing_highs = [sh for sh in swing_highs if sh['idx'] + 3 <= i]

        if new_high:
            for sl in valid_swing_lows[-30:]:
                if sl['idx'] < new_high['idx'] - 100:
                    continue
                if new_high['price'] > sl['price']:
                    width_pct = (new_high['price'] - sl['price']) / sl['price']
                    if MIN_CHANNEL_WIDTH <= width_pct <= MAX_CHANNEL_WIDTH:
                        key = (new_high['idx'], sl['idx'])
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=sl['price'],
                                resistance=new_high['price'],
                                lowest_low=sl['price'],
                                highest_high=new_high['price']
                            )

        if new_low:
            for sh in valid_swing_highs[-30:]:
                if sh['idx'] < new_low['idx'] - 100:
                    continue
                if sh['price'] > new_low['price']:
                    width_pct = (sh['price'] - new_low['price']) / new_low['price']
                    if MIN_CHANNEL_WIDTH <= width_pct <= MAX_CHANNEL_WIDTH:
                        key = (sh['idx'], new_low['idx'])
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=new_low['price'],
                                resistance=sh['price'],
                                lowest_low=new_low['price'],
                                highest_high=sh['price']
                            )

        keys_to_remove = []
        for key, channel in active_channels.items():
            if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                keys_to_remove.append(key)
                continue

            if new_low and new_low['price'] < channel.resistance:
                if new_low['price'] < channel.lowest_low:
                    channel.lowest_low = new_low['price']
                    channel.support = new_low['price']
                    channel.support_touches = 1
                elif new_low['price'] > channel.lowest_low and new_low['price'] < channel.support:
                    channel.support = new_low['price']
                    channel.support_touches += 1
                elif abs(new_low['price'] - channel.support) / channel.support < TOUCH_TOLERANCE:
                    channel.support_touches += 1

            if new_high and new_high['price'] > channel.support:
                if new_high['price'] > channel.highest_high:
                    channel.highest_high = new_high['price']
                    channel.resistance = new_high['price']
                    channel.resistance_touches = 1
                elif new_high['price'] < channel.highest_high and new_high['price'] > channel.resistance:
                    channel.resistance = new_high['price']
                    channel.resistance_touches += 1
                elif abs(new_high['price'] - channel.resistance) / channel.resistance < TOUCH_TOLERANCE:
                    channel.resistance_touches += 1

            if channel.support_touches >= MIN_TOUCHES and channel.resistance_touches >= MIN_TOUCHES:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > MAX_CHANNEL_WIDTH or width_pct < MIN_CHANNEL_WIDTH:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        best_channel = None
        best_score = -1

        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            if score > best_score:
                best_score = score
                best_channel = channel

        if best_channel:
            htf_channel_map[i] = Channel(
                support=best_channel.support,
                resistance=best_channel.resistance,
                support_touches=best_channel.support_touches,
                resistance_touches=best_channel.resistance_touches,
                lowest_low=best_channel.lowest_low,
                highest_high=best_channel.highest_high,
                confirmed=best_channel.confirmed
            )

    print(f"  HTF indices with confirmed channels: {len(htf_channel_map)}")
    return htf_channel_map


def simulate_trade(candles: pd.DataFrame, idx: int, trade_type: str,
                   entry_price: float, sl_price: float, tp1_price: float, tp2_price: float) -> Optional[dict]:
    """Simulate trade with partial TP + breakeven and return detailed result."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price
    exit_reason = None
    exit_idx = None

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    exit_reason = 'SL'
                    exit_idx = j
                    break
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    exit_reason = 'BE'
                    exit_idx = j
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    exit_reason = 'TP2'
                    exit_idx = j
                    break
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    exit_reason = 'SL'
                    exit_idx = j
                    break
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    exit_reason = 'BE'
                    exit_idx = j
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    exit_reason = 'TP2'
                    exit_idx = j
                    break

    if exit_reason is None:
        return None

    return {
        'idx': idx,
        'exit_idx': exit_idx,
        'type': trade_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason,
        'hit_tp1': hit_tp1,
        'success': hit_tp1  # TP1 이상 도달 = 성공
    }


def collect_touch_data(df_1h: pd.DataFrame, df_15m: pd.DataFrame, df_1m: pd.DataFrame,
                       htf_channel_map: Dict[int, Channel],
                       ltf_timeframe: str = '15m') -> List[dict]:
    """
    모든 채널 터치 포인트를 수집하고 1분봉 피처 추출

    Args:
        df_1h: 1시간봉 데이터
        df_15m: 15분봉 (또는 5분봉) 데이터
        df_1m: 1분봉 데이터
        htf_channel_map: HTF 채널 맵
        ltf_timeframe: LTF 타임프레임 ('15m' or '5m')

    Returns:
        List[dict]: 수집된 데이터 포인트
    """
    # Index 1m data by time for quick lookup
    df_1m_indexed = df_1m.set_index('time')

    # LTF data
    ltf_highs = df_15m['high'].values
    ltf_lows = df_15m['low'].values
    ltf_closes = df_15m['close'].values

    data_points = []
    traded_entries = set()

    # HTF to LTF ratio
    if ltf_timeframe == '15m':
        tf_ratio = 4  # 1h = 4 x 15m
    else:
        tf_ratio = 12  # 1h = 12 x 5m

    print(f"\nCollecting touch points (LTF: {ltf_timeframe})...")

    for i in range(len(df_15m)):
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]
        candle_time = df_15m['time'].iloc[i]

        # Get HTF channel
        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)

        if not channel:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Trade key for duplicate prevention
        trade_key = (round(channel.support), round(channel.resistance), i // 20)
        if trade_key in traded_entries:
            continue

        direction = None
        entry_price = None
        sl_price = None
        tp1_price = None
        tp2_price = None

        # LONG: Support touch
        if current_low <= channel.support * (1 + ENTRY_TOUCH_THRESHOLD) and current_close > channel.support:
            direction = 'LONG'
            entry_price = current_close
            sl_price = channel.support * (1 - SL_BUFFER_PCT)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

        # SHORT: Resistance touch
        elif current_high >= channel.resistance * (1 - ENTRY_TOUCH_THRESHOLD) and current_close < channel.resistance:
            direction = 'SHORT'
            entry_price = current_close
            sl_price = channel.resistance * (1 + SL_BUFFER_PCT)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

        if direction is None:
            continue

        # Validate risk/reward
        if direction == 'LONG':
            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price
        else:
            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

        if risk <= 0 or reward1 <= 0:
            continue

        # Get 1m window for feature extraction
        # 15분봉 끝 시점 기준으로 이전 20개 1분봉
        end_time = candle_time + timedelta(minutes=15)
        start_time = end_time - timedelta(minutes=20)
        mask = (df_1m_indexed.index > start_time) & (df_1m_indexed.index <= end_time)
        df_1m_window = df_1m_indexed.loc[mask].reset_index()

        if len(df_1m_window) < 5:
            continue

        # Extract 1m features
        features_1m = extract_1m_features_for_df(
            df_1m_window,
            len(df_1m_window) - 1,  # Last index
            direction,
            channel.support,
            channel.resistance
        )

        # Simulate trade to get label
        trade_result = simulate_trade(df_15m, i, direction, entry_price, sl_price, tp1_price, tp2_price)

        if trade_result is None:
            continue

        # Create data point
        data_point = {
            'time': str(candle_time),
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'channel_support': channel.support,
            'channel_resistance': channel.resistance,
            'channel_width': (channel.resistance - channel.support) / channel.support,
            'support_touches': channel.support_touches,
            'resistance_touches': channel.resistance_touches,
            # Trade result
            'exit_reason': trade_result['exit_reason'],
            'hit_tp1': trade_result['hit_tp1'],
            'pnl_pct': trade_result['pnl_pct'],
            'label': 1 if trade_result['success'] else 0,  # 1 = success (TP1+), 0 = failure (SL)
            # Multi-class label for detailed analysis
            'label_detail': _get_detail_label(trade_result['exit_reason']),
        }

        # Add 1m features
        for k, v in features_1m.items():
            data_point[f'1m_{k}'] = v

        data_points.append(data_point)
        traded_entries.add(trade_key)

    print(f"  Collected {len(data_points)} touch points")
    return data_points


def _get_detail_label(exit_reason: str) -> int:
    """상세 라벨 반환"""
    label_map = {
        'SL': 0,      # 실패
        'BE': 1,      # 손익분기 (TP1 후 BE)
        'TP2': 2,     # 대성공
    }
    return label_map.get(exit_reason, 0)


def run_collection(start_date: str = '2024-01-01', end_date: str = '2025-01-01',
                   output_file: str = 'data/ml_1m_patterns.csv'):
    """데이터 수집 실행"""
    df_1h, df_15m, df_1m = load_data()

    # Filter by date
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    df_1h = df_1h[(df_1h['time'] >= start) & (df_1h['time'] < end)].reset_index(drop=True)
    df_15m = df_15m[(df_15m['time'] >= start) & (df_15m['time'] < end)].reset_index(drop=True)
    df_1m = df_1m[(df_1m['time'] >= start) & (df_1m['time'] < end)].reset_index(drop=True)

    print(f"\nCollection period: {start_date} ~ {end_date}")
    print(f"  1h candles: {len(df_1h):,}")
    print(f"  15m candles: {len(df_15m):,}")
    print(f"  1m candles: {len(df_1m):,}")

    # Build channel map
    print("\nBuilding HTF channel map...")
    htf_channel_map = build_htf_channels(df_1h)

    # Collect data
    data_points = collect_touch_data(df_1h, df_15m, df_1m, htf_channel_map, '15m')

    if not data_points:
        print("No data points collected!")
        return None

    # Convert to DataFrame
    df_data = pd.DataFrame(data_points)

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df_data)}")
    print(f"Success rate: {df_data['label'].mean() * 100:.1f}%")
    print(f"\nLabel distribution:")
    print(f"  SL (fail):     {(df_data['label'] == 0).sum():4d} ({(df_data['label'] == 0).mean()*100:.1f}%)")
    print(f"  TP1+ (success): {(df_data['label'] == 1).sum():4d} ({(df_data['label'] == 1).mean()*100:.1f}%)")

    print(f"\nDetail distribution:")
    for label, name in [(0, 'SL'), (1, 'BE'), (2, 'TP2')]:
        count = (df_data['label_detail'] == label).sum()
        pct = count / len(df_data) * 100
        print(f"  {name}: {count:4d} ({pct:.1f}%)")

    print(f"\nBy direction:")
    for direction in ['LONG', 'SHORT']:
        mask = df_data['direction'] == direction
        count = mask.sum()
        wr = df_data.loc[mask, 'label'].mean() * 100 if count > 0 else 0
        print(f"  {direction}: {count} trades, {wr:.1f}% WR")

    # Feature columns
    feature_cols = [c for c in df_data.columns if c.startswith('1m_')]
    print(f"\n1M Features ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  {col}")

    # Save to CSV
    df_data.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    return df_data


if __name__ == '__main__':
    import sys

    # Default: Training data (2022-2023)
    # Use --test for test data (2024-2025)
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test/OOS data: 2024-2025
        df = run_collection(
            start_date='2024-01-01',
            end_date='2025-06-01',
            output_file='data/ml_1m_patterns_test.csv'
        )
    else:
        # Training/IS data: 2022-2023
        df = run_collection(
            start_date='2022-01-01',
            end_date='2024-01-01',
            output_file='data/ml_1m_patterns_train.csv'
        )
