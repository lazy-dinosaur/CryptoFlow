"""
False Breakout / Liquidity Sweep Strategy
- Identifies swing highs/lows to form a range
- Detects false breakouts with weak volume/delta
- Enters on return to range with multi-target profit taking
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .base import Strategy, Signal, SignalType


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    price: float
    time: pd.Timestamp
    type: str  # 'high' or 'low'
    volume: float
    delta: float


@dataclass
class Range:
    """Represents a trading range."""
    high: SwingPoint
    low: SwingPoint
    mid_price: float


class FalseBreakoutStrategy(Strategy):
    """
    False Breakout Strategy with Volume/Delta Confirmation.

    SHORT Setup:
    1. Range forms (swing high + swing low)
    2. Price breaks above swing high
    3. Weakness detected: low volume OR negative delta (long absorption)
    4. Price returns below swing high
    5. Entry: SHORT
    6. SL: Above new high
    7. TP1: Mid range, TP2: Swing low

    LONG Setup (opposite):
    1. Range forms
    2. Price breaks below swing low
    3. Weakness: low volume OR positive delta (short absorption)
    4. Price returns above swing low
    5. Entry: LONG
    6. SL: Below new low
    7. TP1: Mid range, TP2: Swing high
    """

    @property
    def name(self) -> str:
        return "FalseBreakout"

    def __init__(self, params=None):
        default_params = {
            'swing_lookback': 3,        # Bars to look back/forward for swing detection (좌우 3개)
            'min_range_pct': 0.005,     # Minimum range size (0.5%)
            'volume_threshold': 0.8,    # Breakout volume < 80% of swing volume = weak
            'absorption_threshold': 0,  # Delta threshold for absorption
            'sl_buffer_pct': 0.001,     # SL buffer above/below extreme (0.1%)
            'tp1_ratio': 0.5,           # TP1 at 50% of range (mid)
            'tp2_ratio': 1.0,           # TP2 at 100% of range (opposite extreme)
            'use_tp1_only': False,      # If True, only use TP1
            'fixed_rr': False,          # If True, use fixed SL/TP percentages
            'sl_pct': 0.01,             # Fixed SL percentage (1%)
            'tp_pct': 0.03,             # Fixed TP percentage (3%) - 1:3 R:R
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

        # State tracking
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.current_range: Optional[Range] = None
        self.breakout_candle: Optional[dict] = None
        self.breakout_direction: Optional[str] = None  # 'up' or 'down'
        self.waiting_for_return: bool = False

        # Pre-calculated swings (calculated once)
        self._swings_calculated: bool = False
        self._all_swing_highs: List[SwingPoint] = []
        self._all_swing_lows: List[SwingPoint] = []

    def _precalculate_swings(self, full_history: pd.DataFrame):
        """
        Pre-calculate all swing points (time-series correct).
        Swing is confirmed N bars AFTER it occurs.

        Swing High at index i is confirmed at index i+N when:
        - high[i] > all highs in [i-N : i-1] (higher than N previous)
        - high[i] > all highs in [i+1 : i+N] (higher than N following)

        We record the confirmation_index so we know when it became visible.
        """
        n = self.params['swing_lookback']

        if len(full_history) < n * 2 + 1:
            return

        # Use numpy for fast vectorized computation
        highs = full_history['high'].values
        lows = full_history['low'].values

        # Rolling max/min for efficiency
        # For each position, check if it's a local max/min
        for i in range(n, len(full_history) - n):
            # Swing High: higher than N before AND N after
            before_max = highs[i-n:i].max()
            after_max = highs[i+1:i+n+1].max()

            if highs[i] > before_max and highs[i] > after_max:
                row = full_history.iloc[i]
                self._all_swing_highs.append(SwingPoint(
                    index=i,
                    price=row['high'],
                    time=full_history.index[i],
                    type='high',
                    volume=row['volume'],
                    delta=row['delta']
                ))

            # Swing Low: lower than N before AND N after
            before_min = lows[i-n:i].min()
            after_min = lows[i+1:i+n+1].min()

            if lows[i] < before_min and lows[i] < after_min:
                row = full_history.iloc[i]
                self._all_swing_lows.append(SwingPoint(
                    index=i,
                    price=row['low'],
                    time=full_history.index[i],
                    type='low',
                    volume=row['volume'],
                    delta=row['delta']
                ))

        self._swings_calculated = True
        print(f"  Found {len(self._all_swing_highs)} swing highs, {len(self._all_swing_lows)} swing lows")

    def _get_recent_swings(self, current_idx: int) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Get swing points that are confirmed by current index.
        A swing at index i is confirmed at i + N (lookback).
        """
        n = self.params['swing_lookback']
        # Swing is confirmed N bars after it occurs
        # So at current_idx, we can see swings up to index (current_idx - n)
        max_swing_idx = current_idx - n

        recent_highs = [s for s in self._all_swing_highs if s.index <= max_swing_idx]
        recent_lows = [s for s in self._all_swing_lows if s.index <= max_swing_idx]
        return recent_highs, recent_lows

    def _get_current_range(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> Optional[Range]:
        """Get the most recent valid range."""
        if not swing_highs or not swing_lows:
            return None

        # Get most recent swing high and low
        recent_high = swing_highs[-1]
        recent_low = swing_lows[-1]

        # Check minimum range size
        range_pct = (recent_high.price - recent_low.price) / recent_low.price
        if range_pct < self.params['min_range_pct']:
            return None

        mid_price = (recent_high.price + recent_low.price) / 2

        return Range(
            high=recent_high,
            low=recent_low,
            mid_price=mid_price
        )

    def _check_breakout_weakness(self, candle: pd.Series, direction: str,
                                  range_obj: Range, avg_volume: float) -> Tuple[bool, str]:
        """
        Check if breakout shows weakness.
        Absorption = market orders trapped at wicks
        Returns (is_weak, reason)
        """
        reasons = []

        # Calculate wick sizes
        body_top = max(candle['open'], candle['close'])
        body_bottom = min(candle['open'], candle['close'])
        upper_wick = candle['high'] - body_top
        lower_wick = body_bottom - candle['low']
        body_size = body_top - body_bottom
        candle_range = candle['high'] - candle['low']

        # Wick ratio (wick / total candle range)
        upper_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
        lower_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

        if direction == 'up':
            # Breaking above high - check for weakness
            # 1. Long absorption: Upper wick + negative delta
            #    = Market longs entered at high, got absorbed by sellers
            if upper_wick_ratio > 0.3 and candle['delta'] < 0:
                reasons.append(f"LongAbsorption(wick={upper_wick_ratio:.0%}, Δ={candle['delta']:.0f})")

            # 2. Negative delta alone (strong absorption)
            elif candle['delta'] < -abs(avg_volume * 0.1):  # Significant negative delta
                reasons.append(f"StrongAbsorption(Δ={candle['delta']:.0f})")

            # 3. Volume lower than swing high volume
            if candle['volume'] < range_obj.high.volume * self.params['volume_threshold']:
                reasons.append(f"LowVol({candle['volume']:.0f}<{range_obj.high.volume:.0f})")

            # 4. Long upper wick without follow-through
            if upper_wick_ratio > 0.5:
                reasons.append(f"Rejection(wick={upper_wick_ratio:.0%})")

        else:  # direction == 'down'
            # Breaking below low - check for weakness
            # 1. Short absorption: Lower wick + positive delta
            #    = Market shorts entered at low, got absorbed by buyers
            if lower_wick_ratio > 0.3 and candle['delta'] > 0:
                reasons.append(f"ShortAbsorption(wick={lower_wick_ratio:.0%}, Δ={candle['delta']:.0f})")

            # 2. Positive delta alone (strong absorption)
            elif candle['delta'] > abs(avg_volume * 0.1):  # Significant positive delta
                reasons.append(f"StrongAbsorption(Δ={candle['delta']:.0f})")

            # 3. Volume lower than swing low volume
            if candle['volume'] < range_obj.low.volume * self.params['volume_threshold']:
                reasons.append(f"LowVol({candle['volume']:.0f}<{range_obj.low.volume:.0f})")

            # 4. Long lower wick without follow-through
            if lower_wick_ratio > 0.5:
                reasons.append(f"Rejection(wick={lower_wick_ratio:.0%})")

        is_weak = len(reasons) >= 1  # At least 1 weakness signal
        return is_weak, ", ".join(reasons) if reasons else ""

    def precalculate_all_swings(self, full_candles: pd.DataFrame):
        """Call this before backtesting with full candle data."""
        self._precalculate_swings(full_candles)

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        # Need enough history
        min_history = self.params['swing_lookback'] * 4
        if len(history) < min_history:
            return Signal(SignalType.NONE)

        # Don't enter if already in position
        if self.has_position():
            return Signal(SignalType.NONE)

        # Get swing points up to current position
        current_idx = len(history)
        swing_highs, swing_lows = self._get_recent_swings(current_idx)

        # Get current range
        current_range = self._get_current_range(swing_highs, swing_lows)
        if not current_range:
            self.waiting_for_return = False
            self.breakout_candle = None
            return Signal(SignalType.NONE)

        price = candle['close']
        high = candle['high']
        low = candle['low']

        # Calculate average volume for reference
        avg_volume = history['volume'].tail(20).mean()

        # === State Machine ===

        # State 1: Looking for breakout
        if not self.waiting_for_return:
            # Check for breakout above range high
            if high > current_range.high.price:
                is_weak, reason = self._check_breakout_weakness(
                    candle, 'up', current_range, avg_volume
                )
                if is_weak:
                    self.waiting_for_return = True
                    self.breakout_direction = 'up'
                    self.breakout_candle = {
                        'high': high,
                        'low': low,
                        'close': price,
                        'reason': reason
                    }
                    self.current_range = current_range

            # Check for breakout below range low
            elif low < current_range.low.price:
                is_weak, reason = self._check_breakout_weakness(
                    candle, 'down', current_range, avg_volume
                )
                if is_weak:
                    self.waiting_for_return = True
                    self.breakout_direction = 'down'
                    self.breakout_candle = {
                        'high': high,
                        'low': low,
                        'close': price,
                        'reason': reason
                    }
                    self.current_range = current_range

            return Signal(SignalType.NONE)

        # State 2: Waiting for return to range
        if self.waiting_for_return and self.current_range:
            range_obj = self.current_range
            sl_buffer = self.params['sl_buffer_pct']

            # SHORT Setup: Price broke up, now returning below high
            if self.breakout_direction == 'up':
                # Price returned below the range high
                if price < range_obj.high.price:
                    # Reset state
                    self.waiting_for_return = False
                    breakout_high = self.breakout_candle['high']
                    reason = self.breakout_candle['reason']
                    self.breakout_candle = None

                    # Calculate levels
                    if self.params['fixed_rr']:
                        # Fixed R:R (e.g., 1:3)
                        sl = price * (1 + self.params['sl_pct'])
                        tp = price * (1 - self.params['tp_pct'])
                    else:
                        # Structure-based levels
                        sl = breakout_high * (1 + sl_buffer)
                        tp1 = range_obj.mid_price
                        tp2 = range_obj.low.price
                        tp = tp1 if self.params['use_tp1_only'] else tp2

                    return Signal(
                        type=SignalType.SHORT,
                        entry_price=price,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=0.7,
                        reason=f"FalseBreakout UP: {reason}"
                    )

                # Breakout continued - cancel setup
                if high > self.breakout_candle['high'] * 1.005:  # 0.5% higher
                    self.waiting_for_return = False
                    self.breakout_candle = None

            # LONG Setup: Price broke down, now returning above low
            elif self.breakout_direction == 'down':
                # Price returned above the range low
                if price > range_obj.low.price:
                    # Reset state
                    self.waiting_for_return = False
                    breakout_low = self.breakout_candle['low']
                    reason = self.breakout_candle['reason']
                    self.breakout_candle = None

                    # Calculate levels
                    if self.params['fixed_rr']:
                        # Fixed R:R (e.g., 1:3)
                        sl = price * (1 - self.params['sl_pct'])
                        tp = price * (1 + self.params['tp_pct'])
                    else:
                        # Structure-based levels
                        sl = breakout_low * (1 - sl_buffer)
                        tp1 = range_obj.mid_price
                        tp2 = range_obj.high.price
                        tp = tp1 if self.params['use_tp1_only'] else tp2

                    return Signal(
                        type=SignalType.LONG,
                        entry_price=price,
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=0.7,
                        reason=f"FalseBreakout DOWN: {reason}"
                    )

                # Breakout continued - cancel setup
                if low < self.breakout_candle['low'] * 0.995:  # 0.5% lower
                    self.waiting_for_return = False
                    self.breakout_candle = None

        return Signal(SignalType.NONE)
