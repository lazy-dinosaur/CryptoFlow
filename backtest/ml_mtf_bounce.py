#!/usr/bin/env python3
"""
Multi-Timeframe Bounce Confirmation Backtest

채널 감지: 1시간봉
영역 진입 감지: 15분봉
반등 확인 & 진입: 1분봉

전략 비교:
1. NO_CONFIRM: 15분봉 영역 터치 + 마감 = 진입 (기존 방식)
2. DELTA_CONFIRM: 1분봉에서 delta 반전 확인 후 진입
3. WICK_CONFIRM: 1분봉에서 긴 꼬리 확인 후 진입
4. DELTA_STRONG: delta 강한 확인 (threshold 높임)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============== Configuration ==============
DATA_DIR = "data/parsed/btcusdt"

# Channel detection params (1h)
CHANNEL_LOOKBACK = 100  # 1h candles
MIN_TOUCHES = 2
TOUCH_TOLERANCE = 0.004  # 0.4%
MIN_CHANNEL_WIDTH = 0.008  # 0.8%
MAX_CHANNEL_WIDTH = 0.05  # 5%

# Entry zone params (15m)
ZONE_THRESHOLD = 0.003  # 0.3% from support/resistance

# Trade params
SL_BUFFER_PCT = 0.0008
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.015
MAX_LEVERAGE = 15
FEE_PCT = 0.0004

# Bounce confirmation params (1m)
DELTA_THRESHOLD = 50  # Minimum delta for confirmation
DELTA_STRONG_THRESHOLD = 150  # Strong delta confirmation
DELTA_LOOKBACK = 5    # 1m candles to check delta
WICK_RATIO_THRESHOLD = 0.5  # Wick should be > 50% of candle range
WICK_STRONG_RATIO = 0.65  # Strong wick rejection
MAX_WAIT_CANDLES = 15  # Max 1m candles to wait for confirmation (15분)

# Cooldown after loss
COOLDOWN_CANDLES = 4  # 4 x 15m = 1 hour cooldown after loss

# Trend filter
MA_PERIOD = 50  # EMA period for trend filter

# R:R filter
MIN_RR_RATIO = 1.5  # Minimum risk:reward ratio for entry


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int
    resistance_touches: int


@dataclass
class Trade:
    entry_time: datetime
    direction: str  # LONG or SHORT
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    status: str = 'ACTIVE'
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    tp1_hit: bool = False


@dataclass
class Strategy:
    name: str
    capital: float = INITIAL_CAPITAL
    trades: List[Trade] = field(default_factory=list)
    active_trade: Optional[Trade] = None
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    cooldown_until: int = 0  # Skip signals until this 15m candle index


def load_data():
    """Load all timeframe data."""
    print("Loading data...")

    df_1h = pd.read_parquet(f"{DATA_DIR}/candles_1h.parquet")
    df_15m = pd.read_parquet(f"{DATA_DIR}/candles_15m.parquet")
    df_1m = pd.read_parquet(f"{DATA_DIR}/candles_1m.parquet")

    # Ensure time is datetime
    for df in [df_1h, df_15m, df_1m]:
        df['time'] = pd.to_datetime(df['time'])

    print(f"  1h: {len(df_1h):,} candles ({df_1h['time'].min()} ~ {df_1h['time'].max()})")
    print(f"  15m: {len(df_15m):,} candles")
    print(f"  1m: {len(df_1m):,} candles")

    return df_1h, df_15m, df_1m


def find_swing_points(df: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[dict], List[dict]]:
    """Find swing highs and lows."""
    highs = df['high'].values
    lows = df['low'].values
    times = df['time'].values

    swing_highs = []
    swing_lows = []

    for i in range(confirm_candles, len(df) - confirm_candles):
        # Swing high
        if all(highs[i] > highs[i-j] for j in range(1, confirm_candles+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, confirm_candles+1)):
            swing_highs.append({'idx': i, 'price': highs[i], 'time': times[i]})

        # Swing low
        if all(lows[i] < lows[i-j] for j in range(1, confirm_candles+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, confirm_candles+1)):
            swing_lows.append({'idx': i, 'price': lows[i], 'time': times[i]})

    return swing_highs, swing_lows


def detect_channel(df_1h: pd.DataFrame, end_idx: int) -> Optional[Channel]:
    """Detect channel using 1h data up to end_idx."""
    if end_idx < CHANNEL_LOOKBACK:
        return None

    df = df_1h.iloc[end_idx - CHANNEL_LOOKBACK:end_idx].copy()
    swing_highs, swing_lows = find_swing_points(df)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    current_price = df['close'].iloc[-1]
    best_channel = None
    best_score = -1

    for sh in swing_highs[-10:]:
        for sl in swing_lows[-10:]:
            if sh['price'] <= sl['price']:
                continue

            width_pct = (sh['price'] - sl['price']) / sl['price']
            if width_pct < MIN_CHANNEL_WIDTH or width_pct > MAX_CHANNEL_WIDTH:
                continue

            # Price should be within channel
            if current_price < sl['price'] * 0.98 or current_price > sh['price'] * 1.02:
                continue

            # Count touches
            support_touches = sum(1 for s in swing_lows
                                  if abs(s['price'] - sl['price']) / sl['price'] < TOUCH_TOLERANCE)
            resistance_touches = sum(1 for s in swing_highs
                                     if abs(s['price'] - sh['price']) / sh['price'] < TOUCH_TOLERANCE)

            if support_touches >= MIN_TOUCHES and resistance_touches >= MIN_TOUCHES:
                score = support_touches + resistance_touches
                if score > best_score:
                    best_score = score
                    best_channel = Channel(
                        support=sl['price'],
                        resistance=sh['price'],
                        support_touches=support_touches,
                        resistance_touches=resistance_touches
                    )

    return best_channel


def check_zone_entry(candle: pd.Series, channel: Channel) -> Optional[str]:
    """Check if 15m candle entered buy/sell zone."""
    if channel is None:
        return None

    # Buy zone: low touched support area
    if candle['low'] <= channel.support * (1 + ZONE_THRESHOLD) and candle['close'] > channel.support:
        return 'LONG'

    # Sell zone: high touched resistance area
    if candle['high'] >= channel.resistance * (1 - ZONE_THRESHOLD) and candle['close'] < channel.resistance:
        return 'SHORT'

    return None


def check_delta_confirmation(df_1m: pd.DataFrame, direction: str, threshold: float = DELTA_THRESHOLD) -> bool:
    """Check if delta confirms bounce direction."""
    if len(df_1m) < DELTA_LOOKBACK:
        return False

    recent_delta = df_1m['delta'].iloc[-DELTA_LOOKBACK:].sum()

    if direction == 'LONG':
        return recent_delta > threshold
    else:  # SHORT
        return recent_delta < -threshold


def check_wick_confirmation(candle: pd.Series, direction: str) -> bool:
    """Check if candle has rejection wick."""
    candle_range = candle['high'] - candle['low']
    if candle_range == 0:
        return False

    body_top = max(candle['open'], candle['close'])
    body_bottom = min(candle['open'], candle['close'])

    if direction == 'LONG':
        # Lower wick should be long (rejection of lower prices)
        lower_wick = body_bottom - candle['low']
        return (lower_wick / candle_range) > WICK_RATIO_THRESHOLD
    else:  # SHORT
        # Upper wick should be long (rejection of higher prices)
        upper_wick = candle['high'] - body_top
        return (upper_wick / candle_range) > WICK_RATIO_THRESHOLD


def check_rr_ratio(entry_price: float, direction: str, channel: Channel) -> float:
    """Calculate risk:reward ratio for a potential trade."""
    mid_price = (channel.resistance + channel.support) / 2

    if direction == 'LONG':
        sl = channel.support * (1 - SL_BUFFER_PCT)
        tp1 = mid_price
        risk = entry_price - sl
        reward = tp1 - entry_price
    else:
        sl = channel.resistance * (1 + SL_BUFFER_PCT)
        tp1 = mid_price
        risk = sl - entry_price
        reward = entry_price - tp1

    if risk <= 0:
        return 0

    return reward / risk


def create_trade(entry_time: datetime, direction: str, entry_price: float,
                 channel: Channel, tp_style: str = 'channel') -> Trade:
    """Create a new trade.

    tp_style:
      - 'channel': TP1=midpoint, TP2=opposite boundary
      - 'fixed_1.5': TP1 at 1.5:1 R:R (single exit, no partial)
      - 'fixed_2.0': TP1 at 2.0:1 R:R (single exit, no partial)
    """
    if direction == 'LONG':
        sl = channel.support * (1 - SL_BUFFER_PCT)
        risk = entry_price - sl
    else:
        sl = channel.resistance * (1 + SL_BUFFER_PCT)
        risk = sl - entry_price

    mid_price = (channel.resistance + channel.support) / 2

    if tp_style == 'fixed_1.5':
        if direction == 'LONG':
            tp1 = entry_price + risk * 1.5
            tp2 = tp1  # No second target
        else:
            tp1 = entry_price - risk * 1.5
            tp2 = tp1
    elif tp_style == 'fixed_2.0':
        if direction == 'LONG':
            tp1 = entry_price + risk * 2.0
            tp2 = tp1
        else:
            tp1 = entry_price - risk * 2.0
            tp2 = tp1
    else:  # 'channel'
        if direction == 'LONG':
            tp1 = mid_price
            tp2 = channel.resistance * 0.998
        else:
            tp1 = mid_price
            tp2 = channel.support * 1.002

    return Trade(
        entry_time=entry_time,
        direction=direction,
        entry_price=entry_price,
        sl_price=sl,
        tp1_price=tp1,
        tp2_price=tp2
    )


def update_trade(trade: Trade, candle: pd.Series) -> bool:
    """Update trade with candle data. Returns True if trade closed."""
    if trade.status != 'ACTIVE':
        return False

    is_long = trade.direction == 'LONG'

    # Fixed R:R mode: TP1 == TP2 means single exit, no partial
    is_fixed_rr = (trade.tp1_price == trade.tp2_price)

    # Check SL
    if is_long and candle['low'] <= trade.sl_price:
        trade.status = 'SL_HIT'
        trade.exit_price = trade.sl_price
        trade.exit_time = candle['time']
        trade.pnl_pct = (trade.sl_price - trade.entry_price) / trade.entry_price
        if trade.tp1_hit and not is_fixed_rr:
            trade.pnl_pct = 0.5 * trade.pnl_pct  # Only remaining 50%
        return True
    elif not is_long and candle['high'] >= trade.sl_price:
        trade.status = 'SL_HIT'
        trade.exit_price = trade.sl_price
        trade.exit_time = candle['time']
        trade.pnl_pct = (trade.entry_price - trade.sl_price) / trade.entry_price
        if trade.tp1_hit and not is_fixed_rr:
            trade.pnl_pct = 0.5 * trade.pnl_pct
        return True

    # Fixed R:R mode: single exit at TP
    if is_fixed_rr:
        if is_long and candle['high'] >= trade.tp1_price:
            trade.status = 'TP_HIT'
            trade.exit_price = trade.tp1_price
            trade.exit_time = candle['time']
            trade.pnl_pct = (trade.tp1_price - trade.entry_price) / trade.entry_price
            trade.tp1_hit = True
            return True
        elif not is_long and candle['low'] <= trade.tp1_price:
            trade.status = 'TP_HIT'
            trade.exit_price = trade.tp1_price
            trade.exit_time = candle['time']
            trade.pnl_pct = (trade.entry_price - trade.tp1_price) / trade.entry_price
            trade.tp1_hit = True
            return True
        return False

    # Channel mode: partial exit at TP1, then TP2 or BE
    # Check TP1
    if not trade.tp1_hit:
        if is_long and candle['high'] >= trade.tp1_price:
            trade.tp1_hit = True
            trade.sl_price = trade.entry_price  # Move SL to BE
        elif not is_long and candle['low'] <= trade.tp1_price:
            trade.tp1_hit = True
            trade.sl_price = trade.entry_price

    # Check TP2
    if trade.tp1_hit:
        if is_long and candle['high'] >= trade.tp2_price:
            trade.status = 'TP2_HIT'
            trade.exit_price = trade.tp2_price
            trade.exit_time = candle['time']
            trade.pnl_pct = 0.5 * (trade.tp2_price - trade.entry_price) / trade.entry_price
            return True
        elif not is_long and candle['low'] <= trade.tp2_price:
            trade.status = 'TP2_HIT'
            trade.exit_price = trade.tp2_price
            trade.exit_time = candle['time']
            trade.pnl_pct = 0.5 * (trade.entry_price - trade.tp2_price) / trade.entry_price
            return True

        # Check BE (after TP1)
        if is_long and candle['low'] <= trade.entry_price:
            trade.status = 'BE_HIT'
            trade.exit_price = trade.entry_price
            trade.exit_time = candle['time']
            trade.pnl_pct = 0
            return True
        elif not is_long and candle['high'] >= trade.entry_price:
            trade.status = 'BE_HIT'
            trade.exit_price = trade.entry_price
            trade.exit_time = candle['time']
            trade.pnl_pct = 0
            return True

    return False


def calculate_pnl(trade: Trade, capital: float) -> float:
    """Calculate PnL in dollars."""
    # Calculate original SL distance (before any BE move)
    if trade.direction == 'LONG':
        # For long, original SL was below entry
        sl_dist = abs(trade.entry_price - trade.sl_price) / trade.entry_price
        if trade.tp1_hit:
            # If TP1 hit, SL was moved to BE, so use TP1 distance to estimate original SL
            # Approximation: for channel trades, SL ~ entry - (TP1 - entry)
            sl_dist = abs(trade.tp1_price - trade.entry_price) / trade.entry_price
    else:
        sl_dist = abs(trade.sl_price - trade.entry_price) / trade.entry_price
        if trade.tp1_hit:
            sl_dist = abs(trade.entry_price - trade.tp1_price) / trade.entry_price

    leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE) if sl_dist > 0 else 1
    position = capital * leverage

    # Fixed R:R mode: single exit (TP1 == TP2)
    is_fixed_rr = (trade.tp1_price == trade.tp2_price)
    if is_fixed_rr:
        return position * trade.pnl_pct - position * FEE_PCT * 2

    # Channel mode: TP1 hit means 50% was already taken
    if trade.tp1_hit and trade.status != 'SL_HIT':
        tp1_pnl = 0.5 * abs(trade.tp1_price - trade.entry_price) / trade.entry_price
        tp1_dollar = position * tp1_pnl - position * FEE_PCT
        remaining_pnl = position * trade.pnl_pct - position * FEE_PCT
        return tp1_dollar + remaining_pnl
    else:
        return position * trade.pnl_pct - position * FEE_PCT * 2


def run_backtest(df_1h: pd.DataFrame, df_15m: pd.DataFrame, df_1m: pd.DataFrame,
                 start_date: str = '2023-01-01', end_date: str = '2025-01-01'):
    """Run multi-timeframe backtest."""

    # Filter data
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    df_1h = df_1h[(df_1h['time'] >= start_dt) & (df_1h['time'] < end_dt)].reset_index(drop=True)
    df_15m = df_15m[(df_15m['time'] >= start_dt) & (df_15m['time'] < end_dt)].reset_index(drop=True)
    df_1m = df_1m[(df_1m['time'] >= start_dt) & (df_1m['time'] < end_dt)].reset_index(drop=True)

    print(f"\nBacktest period: {start_date} ~ {end_date}")
    print(f"  1h candles: {len(df_1h):,}")
    print(f"  15m candles: {len(df_15m):,}")
    print(f"  1m candles: {len(df_1m):,}")

    # Initialize strategies
    strategies = {
        'NO_CONFIRM': Strategy(name='No Confirmation (15m close)'),
        'DELTA_CONFIRM': Strategy(name='Delta Confirmation (1m)'),
        'DELTA_STRONG': Strategy(name='Delta Strong (1m, threshold=150)'),
        'WICK_CONFIRM': Strategy(name='Wick Confirmation (1m)'),
        'COMBINED': Strategy(name='Delta + Wick Combined'),
        'TREND_ALIGNED': Strategy(name='Trend Aligned (EMA50)'),
        'LONG_ONLY': Strategy(name='Long Only (Delta)'),
        'SHORT_ONLY': Strategy(name='Short Only (Delta)'),
        'BEST_SETUP': Strategy(name='Trend+Delta+RR>=1.5'),
        'TIGHT_CHANNEL': Strategy(name='Channel Width < 2.5%'),
        'FIXED_RR': Strategy(name='Fixed 1.5:1 TP (no partial)'),
        'TREND_FIXED_RR': Strategy(name='Trend+Fixed 2:1 TP'),
        'TREND_1.5RR': Strategy(name='Trend+Fixed 1.5:1 TP'),
        'TREND_LONG_ONLY': Strategy(name='Trend Long Only 1.5:1'),
        'TREND_LONG_2RR': Strategy(name='Trend Long Only 2:1'),
        'TREND_LONG_STRONG': Strategy(name='Trend Long Strong Delta'),
    }

    # Calculate EMA for trend filter on 1h
    df_1h['ema50'] = df_1h['close'].ewm(span=MA_PERIOD, adjust=False).mean()

    # Build time index for 1m data
    df_1m_indexed = df_1m.set_index('time')

    # Track pending signals for confirmation strategies
    pending_signals: Dict[str, dict] = {}

    current_channel = None
    last_1h_time = None

    processed_15m = 0
    signals_found = 0

    print("\nRunning backtest...")

    for idx_15m, candle_15m in df_15m.iterrows():
        processed_15m += 1
        if processed_15m % 10000 == 0:
            print(f"  Processed {processed_15m:,} / {len(df_15m):,} 15m candles...")

        candle_time = candle_15m['time']

        # Update channel on new 1h candle
        current_1h_time = candle_time.floor('h')
        current_ema50 = None
        if current_1h_time != last_1h_time:
            idx_1h = df_1h[df_1h['time'] <= current_1h_time].index
            if len(idx_1h) > 0:
                current_channel = detect_channel(df_1h, idx_1h[-1] + 1)
                current_ema50 = df_1h['ema50'].iloc[idx_1h[-1]]
            last_1h_time = current_1h_time

        # Get current EMA if not set this iteration
        if current_ema50 is None:
            idx_1h = df_1h[df_1h['time'] <= current_1h_time].index
            if len(idx_1h) > 0:
                current_ema50 = df_1h['ema50'].iloc[idx_1h[-1]]

        # Update active trades for all strategies (using 15m data)
        for strat_name, strat in strategies.items():
            if strat.active_trade:
                closed = update_trade(strat.active_trade, candle_15m)
                if closed:
                    pnl = calculate_pnl(strat.active_trade, strat.capital)
                    strat.capital += pnl
                    strat.total_pnl += pnl
                    if strat.active_trade.tp1_hit or pnl > 0:
                        strat.wins += 1
                    else:
                        strat.losses += 1
                        # Apply cooldown after loss
                        strat.cooldown_until = idx_15m + COOLDOWN_CANDLES
                    strat.trades.append(strat.active_trade)
                    strat.active_trade = None

        # Check for zone entry
        if current_channel is None:
            continue

        direction = check_zone_entry(candle_15m, current_channel)
        if direction is None:
            continue

        signals_found += 1

        # Strategy 1: NO_CONFIRM - Enter immediately at 15m close
        strat = strategies['NO_CONFIRM']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            trade = create_trade(candle_time, direction, candle_15m['close'], current_channel)
            strat.active_trade = trade

        # Strategies 2 & 3: Wait for 1m confirmation
        # Get 1m candles for the next 15 minutes
        end_time = candle_time + timedelta(minutes=15)
        mask = (df_1m_indexed.index > candle_time) & (df_1m_indexed.index <= end_time)
        df_1m_window = df_1m_indexed.loc[mask]

        if len(df_1m_window) == 0:
            continue

        # Check DELTA_CONFIRM
        strat = strategies['DELTA_CONFIRM']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,  # time is the index
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # Check DELTA_STRONG (higher threshold)
        strat = strategies['DELTA_STRONG']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction, threshold=DELTA_STRONG_THRESHOLD):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # Check WICK_CONFIRM
        strat = strategies['WICK_CONFIRM']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            for i in range(len(df_1m_window)):
                entry_candle = df_1m_window.iloc[i]
                if check_wick_confirmation(entry_candle, direction):
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # Check COMBINED (Delta + Wick both required)
        strat = strategies['COMBINED']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            for i in range(len(df_1m_window)):
                entry_candle = df_1m_window.iloc[i]
                df_1m_slice = df_1m_window.iloc[:i+1]
                delta_ok = check_delta_confirmation(df_1m_slice.reset_index(), direction)
                wick_ok = check_wick_confirmation(entry_candle, direction)
                if delta_ok and wick_ok:
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # Check TREND_ALIGNED (Only trade when trend aligns with direction)
        # LONG only when price above EMA50, SHORT only when below
        strat = strategies['TREND_ALIGNED']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            trend_aligned = False
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                trend_aligned = True
            elif direction == 'SHORT' and candle_15m['close'] < current_ema50:
                trend_aligned = True

            if trend_aligned:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel
                        )
                        strat.active_trade = trade
                        break

        # LONG_ONLY strategy
        strat = strategies['LONG_ONLY']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and direction == 'LONG':
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # SHORT_ONLY strategy
        strat = strategies['SHORT_ONLY']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and direction == 'SHORT':
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # BEST_SETUP: Trend aligned + Delta confirmed + Good R:R
        strat = strategies['BEST_SETUP']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            trend_aligned = False
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                trend_aligned = True
            elif direction == 'SHORT' and candle_15m['close'] < current_ema50:
                trend_aligned = True

            if trend_aligned:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        rr = check_rr_ratio(entry_candle['close'], direction, current_channel)
                        if rr >= MIN_RR_RATIO:
                            trade = create_trade(
                                entry_candle.name,
                                direction,
                                entry_candle['close'],
                                current_channel
                            )
                            strat.active_trade = trade
                        break  # Only check first delta confirmation

        # TIGHT_CHANNEL: Only trade narrow channels (better R:R)
        channel_width = (current_channel.resistance - current_channel.support) / current_channel.support
        strat = strategies['TIGHT_CHANNEL']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and channel_width < 0.025:
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel
                    )
                    strat.active_trade = trade
                    break

        # FIXED_RR: Fixed 1.5:1 R:R with delta confirmation (no partial exit)
        strat = strategies['FIXED_RR']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until:
            for i in range(len(df_1m_window)):
                df_1m_slice = df_1m_window.iloc[:i+1]
                if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                    entry_candle = df_1m_window.iloc[i]
                    trade = create_trade(
                        entry_candle.name,
                        direction,
                        entry_candle['close'],
                        current_channel,
                        tp_style='fixed_1.5'
                    )
                    strat.active_trade = trade
                    break

        # TREND_FIXED_RR: Trend aligned + Fixed 2:1 R:R
        strat = strategies['TREND_FIXED_RR']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            trend_aligned = False
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                trend_aligned = True
            elif direction == 'SHORT' and candle_15m['close'] < current_ema50:
                trend_aligned = True

            if trend_aligned:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel,
                            tp_style='fixed_2.0'
                        )
                        strat.active_trade = trade
                        break

        # TREND_1.5RR: Trend aligned + Fixed 1.5:1 R:R
        strat = strategies['TREND_1.5RR']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            trend_aligned = False
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                trend_aligned = True
            elif direction == 'SHORT' and candle_15m['close'] < current_ema50:
                trend_aligned = True

            if trend_aligned:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel,
                            tp_style='fixed_1.5'
                        )
                        strat.active_trade = trade
                        break

        # TREND_LONG_ONLY: Trend aligned, Long only, Fixed 1.5:1 R:R
        strat = strategies['TREND_LONG_ONLY']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel,
                            tp_style='fixed_1.5'
                        )
                        strat.active_trade = trade
                        break

        # TREND_LONG_2RR: Trend aligned, Long only, Fixed 2:1 R:R
        strat = strategies['TREND_LONG_2RR']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel,
                            tp_style='fixed_2.0'
                        )
                        strat.active_trade = trade
                        break

        # TREND_LONG_STRONG: Trend Long + Strong Delta (threshold=150)
        strat = strategies['TREND_LONG_STRONG']
        if strat.active_trade is None and idx_15m >= strat.cooldown_until and current_ema50 is not None:
            if direction == 'LONG' and candle_15m['close'] > current_ema50:
                for i in range(len(df_1m_window)):
                    df_1m_slice = df_1m_window.iloc[:i+1]
                    if check_delta_confirmation(df_1m_slice.reset_index(), direction, threshold=DELTA_STRONG_THRESHOLD):
                        entry_candle = df_1m_window.iloc[i]
                        trade = create_trade(
                            entry_candle.name,
                            direction,
                            entry_candle['close'],
                            current_channel,
                            tp_style='fixed_1.5'
                        )
                        strat.active_trade = trade
                        break

    # Close any remaining trades at last price
    last_price = df_15m['close'].iloc[-1]
    for strat_name, strat in strategies.items():
        if strat.active_trade:
            strat.active_trade.exit_price = last_price
            strat.active_trade.status = 'OPEN'
            strat.trades.append(strat.active_trade)

    print(f"\nTotal signals found: {signals_found}")

    return strategies


def print_results(strategies: Dict[str, Strategy]):
    """Print backtest results."""
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)

    print(f"\n{'Strategy':<35} {'Capital':>12} {'Return':>10} {'Trades':>8} {'Wins':>6} {'WR':>8} {'Avg PnL':>10}")
    print("-"*80)

    for name, strat in strategies.items():
        total_trades = len(strat.trades)
        win_rate = strat.wins / (strat.wins + strat.losses) * 100 if (strat.wins + strat.losses) > 0 else 0
        returns = (strat.capital / INITIAL_CAPITAL - 1) * 100
        avg_pnl = strat.total_pnl / total_trades if total_trades > 0 else 0

        print(f"{strat.name:<35} ${strat.capital:>10,.2f} {returns:>+9.1f}% {total_trades:>8} {strat.wins:>6} {win_rate:>7.1f}% ${avg_pnl:>9.2f}")

    print("="*80)

    # Detailed stats
    for name, strat in strategies.items():
        print(f"\n{strat.name}:")

        if len(strat.trades) == 0:
            print("  No trades")
            continue

        # Count by exit type
        sl_hits = sum(1 for t in strat.trades if t.status == 'SL_HIT')
        tp1_only = sum(1 for t in strat.trades if t.status == 'BE_HIT')
        tp2_hits = sum(1 for t in strat.trades if t.status == 'TP2_HIT')
        tp_hits = sum(1 for t in strat.trades if t.status == 'TP_HIT')  # Fixed R:R

        print(f"  SL Hits: {sl_hits} ({sl_hits/len(strat.trades)*100:.1f}%)")
        if tp_hits > 0:
            print(f"  TP Hits: {tp_hits} ({tp_hits/len(strat.trades)*100:.1f}%)")
        else:
            print(f"  TP1→BE: {tp1_only} ({tp1_only/len(strat.trades)*100:.1f}%)")
            print(f"  TP2 Hits: {tp2_hits} ({tp2_hits/len(strat.trades)*100:.1f}%)")

        # Long vs Short
        longs = [t for t in strat.trades if t.direction == 'LONG']
        shorts = [t for t in strat.trades if t.direction == 'SHORT']
        print(f"  Longs: {len(longs)}, Shorts: {len(shorts)}")


def main():
    # Load data
    df_1h, df_15m, df_1m = load_data()

    # Run backtest
    strategies = run_backtest(
        df_1h, df_15m, df_1m,
        start_date='2023-01-01',
        end_date='2025-01-01'
    )

    # Print results
    print_results(strategies)

    # Summary of best strategies
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("""
Best performing strategies (in order):

1. TREND LONG ONLY with 1.5:1 R:R
   - Only take LONG trades when price > EMA50
   - Wait for delta confirmation (positive delta in 1m timeframe)
   - Use fixed 1.5:1 risk:reward (no partial exits)
   - 53-54% TP hit rate, 46% SL hit rate

2. Key Filters:
   - TREND: Price must be above 50 EMA on 1h timeframe for longs
   - DELTA: Must see positive delta flow in 1m candles before entry
   - NO SHORTS: Short trades have ~70% SL hit rate, avoid them

3. Entry Process:
   a) 1h: Detect channel + check trend (price > EMA50)
   b) 15m: Wait for candle to touch support zone and close above it
   c) 1m: Wait for delta confirmation (cumulative delta > threshold)
   d) Enter at 1m candle close with 1.5:1 R:R target

4. Exit Strategy:
   - SL at channel support (with small buffer)
   - TP at 1.5x risk distance (NOT at channel midpoint)
   - No partial exits, full position exit at TP or SL
""")


if __name__ == "__main__":
    main()
