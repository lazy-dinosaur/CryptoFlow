#!/usr/bin/env python3
"""
Backtesting Engine
Runs strategies against historical candle data
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Type
from datetime import datetime
import json
import os

from strategies.base import Strategy, Signal, SignalType, Position


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    type: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    reason: str


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'total_pnl_pct': round(self.total_pnl_pct * 100, 2),
            'avg_pnl_pct': round(self.avg_pnl_pct * 100, 4),
            'max_drawdown': round(self.max_drawdown * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'profit_factor': round(self.profit_factor, 2)
        }

    def summary(self) -> str:
        return f"""
═══════════════════════════════════════════════════
  Backtest Results: {self.strategy_name}
═══════════════════════════════════════════════════
  Symbol: {self.symbol} | Timeframe: {self.timeframe}
  Period: {self.start_date} to {self.end_date}
───────────────────────────────────────────────────
  Total Trades:    {self.total_trades}
  Winning Trades:  {self.winning_trades}
  Losing Trades:   {self.losing_trades}
  Win Rate:        {self.win_rate * 100:.1f}%
───────────────────────────────────────────────────
  Total PnL:       {self.total_pnl_pct * 100:.2f}%
  Avg Trade PnL:   {self.avg_pnl_pct * 100:.4f}%
  Max Drawdown:    {self.max_drawdown * 100:.2f}%
  Sharpe Ratio:    {self.sharpe_ratio:.2f}
  Profit Factor:   {self.profit_factor:.2f}
═══════════════════════════════════════════════════
"""


class BacktestEngine:
    """Engine for running backtests."""

    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0004):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital for backtesting
            commission: Commission rate per trade (0.0004 = 0.04% for Binance futures)
        """
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, strategy: Strategy, candles: pd.DataFrame,
            symbol: str = "UNKNOWN", timeframe: str = "unknown") -> BacktestResult:
        """
        Run backtest with given strategy and data.

        Args:
            strategy: Strategy instance to test
            candles: DataFrame with OHLCV + delta data
            symbol: Symbol being tested
            timeframe: Timeframe of the data

        Returns:
            BacktestResult with performance metrics
        """
        trades: List[Trade] = []
        equity = self.initial_capital
        equity_curve = [equity]
        peak_equity = equity

        # Iterate through candles
        for i in range(len(candles)):
            current = candles.iloc[i]
            history = candles.iloc[:i] if i > 0 else pd.DataFrame()

            # Check for exit if in position
            if strategy.has_position():
                exit_signal = strategy.check_exit(current)
                if exit_signal and exit_signal.type == SignalType.CLOSE:
                    # Close position
                    trade = self._close_position(strategy, current, exit_signal.reason)
                    trades.append(trade)

                    # Update equity
                    equity *= (1 + trade.pnl_pct - self.commission * 2)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    continue

            # Get signal from strategy
            signal = strategy.on_candle(current, history)

            # Execute signal
            if signal.type in [SignalType.LONG, SignalType.SHORT]:
                strategy.enter_position(signal, current)

        # Close any remaining position at end
        if strategy.has_position():
            trade = self._close_position(strategy, candles.iloc[-1], "End of Data")
            trades.append(trade)
            equity *= (1 + trade.pnl_pct - self.commission * 2)
            equity_curve.append(equity)

        # Calculate metrics
        return self._calculate_results(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            trades=trades,
            equity_curve=equity_curve
        )

    def _close_position(self, strategy: Strategy, candle: pd.Series, reason: str) -> Trade:
        """Close current position and return trade record."""
        pos = strategy.position
        exit_price = candle['close']

        # Calculate PnL
        if pos.type == SignalType.LONG:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:  # SHORT
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl = self.initial_capital * pnl_pct

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=candle.name,
            type=pos.type.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason
        )

        strategy.exit_position()
        return trade

    def _calculate_results(self, strategy_name: str, symbol: str, timeframe: str,
                          candles: pd.DataFrame, trades: List[Trade],
                          equity_curve: List[float]) -> BacktestResult:
        """Calculate performance metrics."""

        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=str(candles.index[0].date()),
                end_date=str(candles.index[-1].date()),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                avg_pnl_pct=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
                trades=[],
                equity_curve=equity_curve
            )

        # Basic stats
        pnls = [t.pnl_pct for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_pnl_pct = sum(pnls)
        win_rate = len(winning) / len(trades) if trades else 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified, assuming 0 risk-free rate)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=str(candles.index[0].date()),
            end_date=str(candles.index[-1].date()),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=sum(t.pnl for t in trades),
            total_pnl_pct=total_pnl_pct,
            avg_pnl_pct=np.mean(pnls),
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            trades=trades,
            equity_curve=equity_curve
        )


def save_results(result: BacktestResult, output_dir: str = None):
    """Save backtest results to file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.strategy_name}_{result.symbol}_{result.timeframe}_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    # Convert trades to serializable format
    trades_data = [
        {
            'entry_time': str(t.entry_time),
            'exit_time': str(t.exit_time),
            'type': t.type,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'reason': t.reason
        }
        for t in result.trades
    ]

    output = {
        **result.to_dict(),
        'trades': trades_data,
        'equity_curve': result.equity_curve
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath
