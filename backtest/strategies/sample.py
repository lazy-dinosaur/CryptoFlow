"""
Sample Delta-Based Strategy
Demonstrates how to use the backtesting framework
"""

import pandas as pd
import numpy as np
from .base import Strategy, Signal, SignalType


class SampleDeltaStrategy(Strategy):
    """
    Simple delta + moving average crossover strategy.

    Entry conditions:
    - LONG: Delta positive (buyers dominant) + Price above SMA
    - SHORT: Delta negative (sellers dominant) + Price below SMA

    Exit conditions:
    - Stop Loss: 1% from entry
    - Take Profit: 2% from entry (2:1 R:R)
    """

    @property
    def name(self) -> str:
        return "SampleDelta"

    def __init__(self, params=None):
        default_params = {
            'sma_period': 20,
            'delta_threshold': 0,  # Minimum delta for entry
            'stop_loss_pct': 0.01,  # 1%
            'take_profit_pct': 0.02,  # 2%
            'min_volume': 0,  # Minimum volume filter
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        # Need enough history for SMA
        sma_period = self.params['sma_period']
        if len(history) < sma_period:
            return Signal(SignalType.NONE)

        # Calculate SMA
        sma = history['close'].tail(sma_period).mean()

        # Current values
        price = candle['close']
        delta = candle['delta']
        volume = candle['volume']

        # Volume filter
        if volume < self.params['min_volume']:
            return Signal(SignalType.NONE)

        # Don't enter if already in position
        if self.has_position():
            return Signal(SignalType.NONE)

        # Entry logic
        stop_loss_pct = self.params['stop_loss_pct']
        take_profit_pct = self.params['take_profit_pct']

        # LONG condition: Positive delta + price above SMA
        if delta > self.params['delta_threshold'] and price > sma:
            return Signal(
                type=SignalType.LONG,
                entry_price=price,
                stop_loss=price * (1 - stop_loss_pct),
                take_profit=price * (1 + take_profit_pct),
                confidence=min(abs(delta) / 100, 1.0),  # Normalize
                reason=f"Delta={delta:.2f}, Price>{sma:.2f}"
            )

        # SHORT condition: Negative delta + price below SMA
        if delta < -self.params['delta_threshold'] and price < sma:
            return Signal(
                type=SignalType.SHORT,
                entry_price=price,
                stop_loss=price * (1 + stop_loss_pct),
                take_profit=price * (1 - take_profit_pct),
                confidence=min(abs(delta) / 100, 1.0),
                reason=f"Delta={delta:.2f}, Price<{sma:.2f}"
            )

        return Signal(SignalType.NONE)
