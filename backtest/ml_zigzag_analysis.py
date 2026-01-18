"""
ML Analysis for Zigzag Channel Strategy
Analyze which conditions lead to WIN vs LOSS trades

Features:
- Channel metrics (width, duration, volume, delta)
- Touch candle metrics (volume ratio, delta direction)
- Entry context (position in channel, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# === DATA CLASSES ===
@dataclass
class SwingPoint:
    idx: int
    price: float
    is_high: bool
    time: int

@dataclass
class Channel:
    high: float
    low: float
    mid: float
    width_pct: float
    swings: List[SwingPoint]
    formed_idx: int
    formed_time: int = 0

@dataclass
class TradeFeatures:
    # Signal info
    direction: str
    setup_type: str

    # Channel features
    channel_width_pct: float
    channel_duration_bars: int

    # Volume features (channel period)
    channel_avg_volume: float
    channel_total_volume: float
    channel_volume_trend: float  # slope of volume

    # Delta features (channel period)
    channel_cumulative_delta: float
    channel_delta_ratio: float  # buy_vol / total_vol

    # Touch candle features
    touch_volume: float
    touch_volume_ratio: float  # vs channel avg
    touch_delta: float
    touch_delta_pct: float  # delta / volume

    # Price context
    price_position_in_channel: float  # 0=bottom, 1=top
    distance_from_level_pct: float  # how close to S/R

    # Result
    pnl_pct: float = 0
    is_win: int = 0
    exit_reason: str = ''

# === CONFIGURATION ===
SWING_STRENGTH = 3
SWINGS_EACH_SIDE = 2
MIN_CHANNEL_WIDTH = 0.003
SL_BUFFER = 0.0005
TP1_RATIO = 0.5
TP1_QTY_PCT = 0.5

# === DATA FETCHING ===
def fetch_binance_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        current_start = data[-1][0] + 1

        if len(data) < 1000:
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    for col in ['time', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
        df[col] = pd.to_numeric(df[col])

    # Calculate delta (buy - sell)
    df['taker_sell_base'] = df['volume'] - df['taker_buy_base']
    df['delta'] = df['taker_buy_base'] - df['taker_sell_base']

    return df

# === SWING DETECTION ===
def is_swing_high(highs: np.ndarray, idx: int, strength: int) -> bool:
    if idx < strength or idx >= len(highs) - strength:
        return False
    current = highs[idx]
    for i in range(1, strength + 1):
        if highs[idx - i] >= current or highs[idx + i] >= current:
            return False
    return True

def is_swing_low(lows: np.ndarray, idx: int, strength: int) -> bool:
    if idx < strength or idx >= len(lows) - strength:
        return False
    current = lows[idx]
    for i in range(1, strength + 1):
        if lows[idx - i] <= current or lows[idx + i] <= current:
            return False
    return True

def find_next_swing_high(highs: np.ndarray, times: np.ndarray, start_idx: int, strength: int) -> Optional[SwingPoint]:
    max_idx = len(highs) - strength
    for i in range(start_idx, max_idx):
        if is_swing_high(highs, i, strength):
            return SwingPoint(idx=i, price=highs[i], is_high=True, time=times[i])
    return None

def find_next_swing_low(lows: np.ndarray, times: np.ndarray, start_idx: int, strength: int) -> Optional[SwingPoint]:
    max_idx = len(lows) - strength
    for i in range(start_idx, max_idx):
        if is_swing_low(lows, i, strength):
            return SwingPoint(idx=i, price=lows[i], is_high=False, time=times[i])
    return None

def find_zigzag_structure(htf_df: pd.DataFrame, current_close: float, strength: int, swings_each: int) -> List[SwingPoint]:
    highs = htf_df['high'].values
    lows = htf_df['low'].values
    times = htf_df['time'].values

    if len(highs) < strength * 2 + 1:
        return []

    recent_high = highs[0]
    look_for_low = current_close >= recent_high

    swings = []
    search_idx = strength
    highs_found = 0
    lows_found = 0
    max_search = len(highs) - strength
    iterations = 0

    while (highs_found < swings_each or lows_found < swings_each) and search_idx < max_search and iterations < 100:
        iterations += 1

        if look_for_low and lows_found < swings_each:
            swing = find_next_swing_low(lows, times, search_idx, strength)
            if swing:
                swings.append(swing)
                lows_found += 1
                search_idx = swing.idx + 1
                look_for_low = False
            else:
                search_idx += 1
                if search_idx >= max_search:
                    look_for_low = False
        elif not look_for_low and highs_found < swings_each:
            swing = find_next_swing_high(highs, times, search_idx, strength)
            if swing:
                swings.append(swing)
                highs_found += 1
                search_idx = swing.idx + 1
                look_for_low = True
            else:
                search_idx += 1
                if search_idx >= max_search:
                    look_for_low = True
        else:
            if lows_found >= swings_each and highs_found < swings_each:
                look_for_low = False
            elif highs_found >= swings_each and lows_found < swings_each:
                look_for_low = True
            else:
                break

    return swings

def build_channel(swings: List[SwingPoint], min_width: float, htf_df: pd.DataFrame) -> Optional[Channel]:
    if len(swings) < 2:
        return None

    prices = [s.price for s in swings]
    ch_high = max(prices)
    ch_low = min(prices)

    if ch_high == ch_low:
        return None

    width_pct = (ch_high - ch_low) / ch_low
    if width_pct < min_width:
        return None

    formed_idx = max(s.idx for s in swings)
    formed_time = htf_df.iloc[formed_idx]['time'] if formed_idx < len(htf_df) else 0

    return Channel(
        high=ch_high,
        low=ch_low,
        mid=(ch_high + ch_low) / 2,
        width_pct=width_pct,
        swings=swings,
        formed_idx=formed_idx,
        formed_time=formed_time
    )

# === FEATURE EXTRACTION ===
def extract_channel_features(ltf_df: pd.DataFrame, channel: Channel, start_time: int, end_time: int) -> Dict:
    """Extract volume/delta features for channel period"""
    mask = (ltf_df['time'] >= start_time) & (ltf_df['time'] < end_time)
    channel_data = ltf_df[mask]

    if len(channel_data) < 2:
        return {
            'channel_avg_volume': 0,
            'channel_total_volume': 0,
            'channel_volume_trend': 0,
            'channel_cumulative_delta': 0,
            'channel_delta_ratio': 0.5,
            'channel_duration_bars': 0
        }

    volumes = channel_data['volume'].values
    deltas = channel_data['delta'].values
    buy_vol = channel_data['taker_buy_base'].sum()
    total_vol = channel_data['volume'].sum()

    # Volume trend (simple linear slope)
    if len(volumes) > 1:
        x = np.arange(len(volumes))
        slope = np.polyfit(x, volumes, 1)[0]
    else:
        slope = 0

    return {
        'channel_avg_volume': volumes.mean() if len(volumes) > 0 else 0,
        'channel_total_volume': total_vol,
        'channel_volume_trend': slope,
        'channel_cumulative_delta': deltas.sum(),
        'channel_delta_ratio': buy_vol / total_vol if total_vol > 0 else 0.5,
        'channel_duration_bars': len(channel_data)
    }

def extract_touch_features(candle: pd.Series, channel: Channel, channel_features: Dict) -> Dict:
    """Extract features for the touch candle"""
    volume = candle['volume']
    delta = candle['delta']
    close = candle['close']
    low = candle['low']
    high = candle['high']

    avg_vol = channel_features['channel_avg_volume']
    vol_ratio = volume / avg_vol if avg_vol > 0 else 1

    # Price position in channel (0=bottom, 1=top)
    ch_height = channel.high - channel.low
    if ch_height > 0:
        position = (close - channel.low) / ch_height
    else:
        position = 0.5

    # Distance from level
    dist_from_support = abs(low - channel.low) / channel.low * 100
    dist_from_resistance = abs(high - channel.high) / channel.high * 100

    return {
        'touch_volume': volume,
        'touch_volume_ratio': vol_ratio,
        'touch_delta': delta,
        'touch_delta_pct': delta / volume if volume > 0 else 0,
        'price_position_in_channel': position,
        'distance_from_support_pct': dist_from_support,
        'distance_from_resistance_pct': dist_from_resistance
    }

# === TRADE SIMULATION ===
def simulate_trade_with_features(
    direction: str,
    setup_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    channel: Channel,
    ltf_df: pd.DataFrame,
    signal_idx: int,
    channel_features: Dict,
    touch_features: Dict
) -> Optional[TradeFeatures]:
    """Simulate trade and return features with result"""

    hit_tp1 = False
    partial_pnl = 0
    remaining_qty = 1.0
    entry = entry_price
    sl = sl_price

    exit_reason = ''
    final_pnl = 0

    for i in range(signal_idx + 1, len(ltf_df)):
        row = ltf_df.iloc[i]
        high = row['high']
        low = row['low']

        if direction == 'LONG':
            current_sl = entry if hit_tp1 else sl
            if low <= current_sl:
                if hit_tp1:
                    final_pnl = partial_pnl
                    exit_reason = 'BE'
                else:
                    final_pnl = ((sl - entry) / entry)
                    exit_reason = 'SL'
                break

            if not hit_tp1 and high >= tp1_price:
                partial_pnl = ((tp1_price - entry) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True

            if high >= tp2_price:
                final_pnl = ((tp2_price - entry) / entry) * remaining_qty + partial_pnl
                exit_reason = 'TP2'
                break

        else:  # SHORT
            current_sl = entry if hit_tp1 else sl
            if high >= current_sl:
                if hit_tp1:
                    final_pnl = partial_pnl
                    exit_reason = 'BE'
                else:
                    final_pnl = ((entry - sl) / entry)
                    exit_reason = 'SL'
                break

            if not hit_tp1 and low <= tp1_price:
                partial_pnl = ((entry - tp1_price) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True

            if low <= tp2_price:
                final_pnl = ((entry - tp2_price) / entry) * remaining_qty + partial_pnl
                exit_reason = 'TP2'
                break

    if not exit_reason:
        return None

    # Determine distance from level based on direction
    if direction == 'LONG':
        distance_from_level = touch_features['distance_from_support_pct']
    else:
        distance_from_level = touch_features['distance_from_resistance_pct']

    return TradeFeatures(
        direction=direction,
        setup_type=setup_type,
        channel_width_pct=channel.width_pct * 100,
        channel_duration_bars=channel_features['channel_duration_bars'],
        channel_avg_volume=channel_features['channel_avg_volume'],
        channel_total_volume=channel_features['channel_total_volume'],
        channel_volume_trend=channel_features['channel_volume_trend'],
        channel_cumulative_delta=channel_features['channel_cumulative_delta'],
        channel_delta_ratio=channel_features['channel_delta_ratio'],
        touch_volume=touch_features['touch_volume'],
        touch_volume_ratio=touch_features['touch_volume_ratio'],
        touch_delta=touch_features['touch_delta'],
        touch_delta_pct=touch_features['touch_delta_pct'],
        price_position_in_channel=touch_features['price_position_in_channel'],
        distance_from_level_pct=distance_from_level,
        pnl_pct=final_pnl * 100,
        is_win=1 if final_pnl > 0 else 0,
        exit_reason=exit_reason
    )

# === MAIN ANALYSIS ===
def run_analysis(symbol: str = "BTCUSDT", days: int = 365):
    print(f"\n{'='*60}")
    print(f"ML Analysis for Zigzag Channel Strategy")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {days} days")
    print(f"{'='*60}\n")

    # Time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    # Fetch data
    print("Fetching 1H data...")
    htf_df = fetch_binance_klines(symbol, "1h", start_time, end_time)
    print(f"  Got {len(htf_df)} 1H candles")

    print("Fetching 15m data...")
    ltf_df = fetch_binance_klines(symbol, "15m", start_time, end_time)
    print(f"  Got {len(ltf_df)} 15m candles")

    if htf_df.empty or ltf_df.empty:
        print("Error: No data fetched")
        return

    # Process
    all_features: List[TradeFeatures] = []

    htf_df = htf_df.sort_values('time').reset_index(drop=True)
    ltf_df = ltf_df.sort_values('time').reset_index(drop=True)

    print("\nExtracting features from trades...")

    prev_channel_time = 0

    for htf_idx in range(SWING_STRENGTH * 2, len(htf_df) - 1):
        htf_row = htf_df.iloc[htf_idx]
        htf_time = htf_row['time']
        htf_close = htf_row['close']

        htf_slice = htf_df.iloc[:htf_idx+1].iloc[::-1].reset_index(drop=True)
        swings = find_zigzag_structure(htf_slice, htf_close, SWING_STRENGTH, SWINGS_EACH_SIDE)

        if len(swings) < 2:
            continue

        channel = build_channel(swings, MIN_CHANNEL_WIDTH, htf_slice)
        if channel is None:
            continue

        # Get channel formation time
        channel_start_time = min(s.time for s in swings) if swings else htf_time

        # Extract channel features
        channel_features = extract_channel_features(ltf_df, channel, channel_start_time, htf_time)

        # Get LTF data for this period
        next_htf_time = htf_df.iloc[htf_idx + 1]['time'] if htf_idx + 1 < len(htf_df) else htf_time + 3600000
        ltf_slice = ltf_df[(ltf_df['time'] >= htf_time) & (ltf_df['time'] < next_htf_time)].reset_index(drop=True)

        if ltf_slice.empty:
            continue

        ch_high = channel.high
        ch_low = channel.low
        ch_height = ch_high - ch_low

        broke_support = False
        broke_resistance = False
        fakeout_extreme_low = None
        fakeout_extreme_high = None

        for i in range(len(ltf_slice)):
            row = ltf_slice.iloc[i]
            low = row['low']
            high = row['high']
            close = row['close']

            # Track breakouts
            if close < ch_low * 0.997:
                broke_support = True
                fakeout_extreme_low = low if fakeout_extreme_low is None else min(fakeout_extreme_low, low)
            if close > ch_high * 1.003:
                broke_resistance = True
                fakeout_extreme_high = high if fakeout_extreme_high is None else max(fakeout_extreme_high, high)

            # Detect signals and extract features
            signal_detected = False
            direction = ''
            setup_type = ''
            entry_price = close
            sl_price = 0
            tp1_price = 0
            tp2_price = 0

            # LONG BOUNCE
            if low <= ch_low and close > ch_low and not broke_support:
                signal_detected = True
                direction = 'LONG'
                setup_type = 'BOUNCE'
                sl_price = low * (1 - SL_BUFFER)
                tp1_price = ch_low + ch_height * TP1_RATIO
                tp2_price = ch_high

            # LONG FAKEOUT
            elif broke_support and close > ch_low:
                signal_detected = True
                direction = 'LONG'
                setup_type = 'FAKEOUT'
                sl_price = fakeout_extreme_low * (1 - SL_BUFFER) if fakeout_extreme_low else low * (1 - SL_BUFFER)
                tp1_price = ch_low + ch_height * TP1_RATIO
                tp2_price = ch_high
                broke_support = False
                fakeout_extreme_low = None

            # SHORT BOUNCE
            elif high >= ch_high and close < ch_high and not broke_resistance:
                signal_detected = True
                direction = 'SHORT'
                setup_type = 'BOUNCE'
                sl_price = high * (1 + SL_BUFFER)
                tp1_price = ch_high - ch_height * TP1_RATIO
                tp2_price = ch_low

            # SHORT FAKEOUT
            elif broke_resistance and close < ch_high:
                signal_detected = True
                direction = 'SHORT'
                setup_type = 'FAKEOUT'
                sl_price = fakeout_extreme_high * (1 + SL_BUFFER) if fakeout_extreme_high else high * (1 + SL_BUFFER)
                tp1_price = ch_high - ch_height * TP1_RATIO
                tp2_price = ch_low
                broke_resistance = False
                fakeout_extreme_high = None

            if signal_detected:
                # Extract touch features
                touch_features = extract_touch_features(row, channel, channel_features)

                # Find signal index in full LTF data
                signal_time = row['time']
                signal_idx_full = ltf_df[ltf_df['time'] == signal_time].index
                if len(signal_idx_full) == 0:
                    continue
                signal_idx_full = signal_idx_full[0]

                # Simulate trade
                trade_features = simulate_trade_with_features(
                    direction, setup_type, entry_price, sl_price, tp1_price, tp2_price,
                    channel, ltf_df, signal_idx_full, channel_features, touch_features
                )

                if trade_features:
                    all_features.append(trade_features)

    print(f"\nTotal trades with features: {len(all_features)}")

    if len(all_features) < 50:
        print("Not enough trades for ML analysis")
        return

    # Convert to DataFrame
    df = pd.DataFrame([vars(f) for f in all_features])

    # === ANALYSIS ===
    print(f"\n{'='*60}")
    print("TRADE STATISTICS")
    print(f"{'='*60}")

    wins = df[df['is_win'] == 1]
    losses = df[df['is_win'] == 0]

    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {len(wins)/len(df)*100:.1f}%")
    print(f"Total PnL: {df['pnl_pct'].sum():.2f}%")

    # === ML MODEL ===
    print(f"\n{'='*60}")
    print("ML ANALYSIS")
    print(f"{'='*60}")

    # Prepare features
    feature_cols = [
        'channel_width_pct', 'channel_duration_bars',
        'channel_avg_volume', 'channel_volume_trend',
        'channel_cumulative_delta', 'channel_delta_ratio',
        'touch_volume_ratio', 'touch_delta_pct',
        'price_position_in_channel', 'distance_from_level_pct'
    ]

    # Add categorical features
    df['is_long'] = (df['direction'] == 'LONG').astype(int)
    df['is_fakeout'] = (df['setup_type'] == 'FAKEOUT').astype(int)
    feature_cols.extend(['is_long', 'is_fakeout'])

    X = df[feature_cols].fillna(0)
    y = df['is_win']

    # Handle infinities
    X = X.replace([np.inf, -np.inf], 0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))

    # Feature Importance
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 10)")
    print(f"{'='*60}")

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"{row['feature']:30s} {row['importance']:.3f} {bar}")

    # === INSIGHTS ===
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")

    # Compare WIN vs LOSS features
    print("\nFeature Comparison (WIN vs LOSS mean):")
    for col in feature_cols[:10]:
        win_mean = wins[col].mean() if col in wins.columns else 0
        loss_mean = losses[col].mean() if col in losses.columns else 0
        diff = win_mean - loss_mean
        direction = "↑" if diff > 0 else "↓"
        print(f"  {col:30s} WIN: {win_mean:10.2f}  LOSS: {loss_mean:10.2f}  {direction}")

    # Best conditions
    print("\n--- Best Conditions for WIN ---")

    # By setup type
    for setup in ['BOUNCE', 'FAKEOUT']:
        subset = df[df['setup_type'] == setup]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            pnl = subset['pnl_pct'].sum()
            print(f"{setup}: {len(subset)} trades, {wr:.1f}% WR, {pnl:.2f}% PnL")

    # By direction
    for direction in ['LONG', 'SHORT']:
        subset = df[df['direction'] == direction]
        if len(subset) > 0:
            wr = subset['is_win'].mean() * 100
            pnl = subset['pnl_pct'].sum()
            print(f"{direction}: {len(subset)} trades, {wr:.1f}% WR, {pnl:.2f}% PnL")

    # High volume ratio trades
    high_vol = df[df['touch_volume_ratio'] > 1.5]
    low_vol = df[df['touch_volume_ratio'] <= 1.5]
    if len(high_vol) > 0:
        print(f"\nHigh Volume Ratio (>1.5x): {len(high_vol)} trades, {high_vol['is_win'].mean()*100:.1f}% WR")
    if len(low_vol) > 0:
        print(f"Low Volume Ratio (<=1.5x): {len(low_vol)} trades, {low_vol['is_win'].mean()*100:.1f}% WR")

    # Delta direction alignment
    long_trades = df[df['direction'] == 'LONG']
    short_trades = df[df['direction'] == 'SHORT']

    if len(long_trades) > 0:
        long_positive_delta = long_trades[long_trades['touch_delta'] > 0]
        long_negative_delta = long_trades[long_trades['touch_delta'] <= 0]
        print(f"\nLONG + Positive Delta: {len(long_positive_delta)} trades, {long_positive_delta['is_win'].mean()*100:.1f}% WR")
        print(f"LONG + Negative Delta: {len(long_negative_delta)} trades, {long_negative_delta['is_win'].mean()*100:.1f}% WR")

    if len(short_trades) > 0:
        short_positive_delta = short_trades[short_trades['touch_delta'] > 0]
        short_negative_delta = short_trades[short_trades['touch_delta'] <= 0]
        print(f"\nSHORT + Positive Delta: {len(short_positive_delta)} trades, {short_positive_delta['is_win'].mean()*100:.1f}% WR")
        print(f"SHORT + Negative Delta: {len(short_negative_delta)} trades, {short_negative_delta['is_win'].mean()*100:.1f}% WR")

    print(f"\n{'='*60}")

    return df, rf, importance

if __name__ == "__main__":
    # 1년 데이터로 분석
    run_analysis("BTCUSDT", days=365)
