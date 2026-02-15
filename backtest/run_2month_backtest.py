#!/usr/bin/env python3
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared.channel_builder import build_channels as _build_htf_map, Channel

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "server",
    "data",
    "cryptoflow.db",
)


@dataclass
class BacktestResult:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    final_capital: float


def load_candles_from_db(timeframe: int, days: int = 60):
    conn = sqlite3.connect(DB_PATH)
    candles_per_day = 24 * 60 // timeframe
    limit = days * candles_per_day

    query = f"""
        SELECT time, open, high, low, close, volume, delta
        FROM (
            SELECT time, open, high, low, close, volume, delta
            FROM candles_{timeframe}
            WHERE symbol = 'BINANCE:BTCUSDT'
            ORDER BY time DESC
            LIMIT ?
        ) sub ORDER BY time
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    return df


def build_channels(df_1h) -> Dict[int, Channel]:
    return _build_htf_map(
        df_1h["high"].values,
        df_1h["low"].values,
        df_1h["close"].values,
    )


def simulate_trade(
    df_15m, entry_idx, direction, entry_price, sl_price, tp1_price, tp2_price
):
    tp1_hit = False
    exit_price = None
    exit_reason = None

    max_hold = 150
    for j in range(entry_idx + 1, min(entry_idx + max_hold, len(df_15m))):
        high = df_15m["high"].iloc[j]
        low = df_15m["low"].iloc[j]

        if direction == "LONG":
            if low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                break
            if not tp1_hit and high >= tp1_price:
                tp1_hit = True
            if high >= tp2_price:
                exit_price = tp2_price
                exit_reason = "TP2"
                break
        else:
            if high >= sl_price:
                exit_price = sl_price
                exit_reason = "SL"
                break
            if not tp1_hit and low <= tp1_price:
                tp1_hit = True
            if low <= tp2_price:
                exit_price = tp2_price
                exit_reason = "TP2"
                break

    if exit_price is None:
        exit_price = df_15m["close"].iloc[
            min(entry_idx + max_hold - 1, len(df_15m) - 1)
        ]
        exit_reason = "TIMEOUT"

    # Calculate component pnl percentages
    if direction == "LONG":
        sl_pct = (sl_price - entry_price) / entry_price
        tp1_pct = (tp1_price - entry_price) / entry_price
        tp2_pct = (tp2_price - entry_price) / entry_price
        exit_pct = (exit_price - entry_price) / entry_price
    else:
        sl_pct = (entry_price - sl_price) / entry_price
        tp1_pct = (entry_price - tp1_price) / entry_price
        tp2_pct = (entry_price - tp2_price) / entry_price
        exit_pct = (entry_price - exit_price) / entry_price

    # 50/50 partial profit calculation (no BE stop)
    if exit_reason == "SL":
        if tp1_hit:
            pnl_pct = 0.5 * tp1_pct + 0.5 * sl_pct
        else:
            pnl_pct = sl_pct
    elif exit_reason == "TP2":
        pnl_pct = 0.5 * tp1_pct + 0.5 * tp2_pct
    elif exit_reason == "TIMEOUT":
        if tp1_hit:
            pnl_pct = 0.5 * tp1_pct + 0.5 * exit_pct
        else:
            pnl_pct = exit_pct

    sl_dist = abs(sl_price - entry_price) / entry_price

    return {
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "tp1_hit": tp1_hit,
        "sl_dist": sl_dist,
    }


def run_backtest(df_1h, df_15m) -> BacktestResult:
    htf_map = build_channels(df_1h)

    one_hour_ms = 3600000
    htf_time_to_idx = {}
    for idx in range(len(df_1h)):
        t = int(df_1h["time"].iloc[idx])
        htf_time_to_idx[t] = idx

    touch_th = 0.003
    sl_buffer = 0.0008
    signal_cooldown_ms = 20 * 15 * 60 * 1000
    traded = set()
    trades = []

    for i in tqdm(range(len(df_15m)), desc="Backtesting"):
        candle_time_ms = int(df_15m["time"].iloc[i])
        htf_time = (candle_time_ms // one_hour_ms) * one_hour_ms
        htf_idx = htf_time_to_idx.get(htf_time)
        if htf_idx is None:
            continue
        ch = htf_map.get(htf_idx - 1)
        if not ch:
            continue
        if (ch.resistance - ch.support) / ch.support < 0.015:
            continue

        close = df_15m["close"].iloc[i]
        high = df_15m["high"].iloc[i]
        low = df_15m["low"].iloc[i]
        candle_time = int(df_15m["time"].iloc[i])
        key = (
            round(ch.support),
            round(ch.resistance),
            candle_time // signal_cooldown_ms,
        )
        if key in traded:
            continue

        mid = (ch.resistance + ch.support) / 2

        if low <= ch.support * (1 + touch_th) and close > ch.support:
            entry = close
            sl = ch.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = ch.resistance * 0.998
            if entry > sl and tp1 > entry:
                result = simulate_trade(df_15m, i, "LONG", entry, sl, tp1, tp2)
                trades.append(result)
                traded.add(key)
        elif high >= ch.resistance * (1 - touch_th) and close < ch.resistance:
            entry = close
            sl = ch.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = ch.support * 1.002
            if sl > entry and entry > tp1:
                result = simulate_trade(df_15m, i, "SHORT", entry, sl, tp1, tp2)
                trades.append(result)
                traded.add(key)

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 10000)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 20
    fee_pct = 0.0004
    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for t in trades:
        leverage = min(risk_pct / t["sl_dist"] if t["sl_dist"] > 0 else 1, max_leverage)
        position_size = capital * leverage
        gross_pnl = position_size * t["pnl_pct"]
        fees = position_size * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    total_pnl = (capital - 10000) / 10000 * 100
    avg_pnl = total_pnl / len(trades)
    win_rate = wins / len(trades) * 100 if trades else 0

    return BacktestResult(
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        max_drawdown=max_dd,
        final_capital=capital,
    )


if __name__ == "__main__":
    days = 60

    print("=" * 60)
    print("BACKTEST: Last " + str(days) + " days")
    print("=" * 60)

    print("\nLoading data from DB...")
    df_1h = load_candles_from_db(60, days)
    df_15m = load_candles_from_db(15, days)

    print("1H candles: " + str(len(df_1h)))
    print("15M candles: " + str(len(df_15m)))
    print(
        "Period: "
        + str(df_1h["datetime"].iloc[0])
        + " ~ "
        + str(df_1h["datetime"].iloc[-1])
    )

    print("\nRunning backtest...")
    result = run_backtest(df_1h, df_15m)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Total Trades: " + str(result.total_trades))
    print("Wins: " + str(result.wins) + " | Losses: " + str(result.losses))
    print("Win Rate: " + str(round(result.win_rate, 1)) + "%")
    print("Total PnL: " + str(round(result.total_pnl, 2)) + "%")
    print("Avg PnL per Trade: " + str(round(result.avg_pnl, 2)) + "%")
    print("Max Drawdown: " + str(round(result.max_drawdown, 2)) + "%")
    print("Final Capital: $" + "{:,.2f}".format(result.final_capital))
    print("=" * 60)
