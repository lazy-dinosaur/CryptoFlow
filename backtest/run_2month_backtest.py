#!/usr/bin/env python3
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm

DB_PATH = "/home/ubuntu/projects/CryptoFlow/server/data/cryptoflow.db"


@dataclass
class Channel:
    support: float
    resistance: float
    support_idx: int = 0
    resistance_idx: int = 0
    lowest_low: float = 0
    highest_high: float = 0
    support_touches: int = 1
    resistance_touches: int = 1
    confirmed: bool = False


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
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)

    query = f"""
        SELECT time, open, high, low, close, volume, delta
        FROM candles_{timeframe}
        WHERE symbol = 'BINANCE:BTCUSDT' AND time >= ? AND time <= ?
        ORDER BY time
    """
    df = pd.read_sql_query(query, conn, params=[start_ts, end_ts])
    conn.close()
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    return df


def find_swing_points(candles, confirm_candles=3):
    highs = candles["high"].values
    lows = candles["low"].values
    swing_highs, swing_lows = [], []

    pot_high_idx, pot_high_price, since_high = 0, highs[0], 0
    pot_low_idx, pot_low_price, since_low = 0, lows[0], 0

    for i in range(1, len(candles)):
        if highs[i] > pot_high_price:
            pot_high_idx, pot_high_price, since_high = i, highs[i], 0
        else:
            since_high += 1
            if since_high == confirm_candles:
                swing_highs.append({"idx": pot_high_idx, "price": pot_high_price})

        if lows[i] < pot_low_price:
            pot_low_idx, pot_low_price, since_low = i, lows[i], 0
        else:
            since_low += 1
            if since_low == confirm_candles:
                swing_lows.append({"idx": pot_low_idx, "price": pot_low_price})

        if since_high >= confirm_candles:
            pot_high_price, pot_high_idx, since_high = highs[i], i, 0
        if since_low >= confirm_candles:
            pot_low_price, pot_low_idx, since_low = lows[i], i, 0

    return swing_highs, swing_lows


def build_channels(df_1h) -> Dict[int, Channel]:
    max_w, min_w, touch_th = 0.05, 0.015, 0.004
    swing_highs, swing_lows = find_swing_points(df_1h)
    closes = df_1h["close"].values

    active: Dict[tuple, Channel] = {}
    htf_map: Dict[int, Channel] = {}

    for idx in range(len(df_1h)):
        close = closes[idx]
        new_high = next((sh for sh in swing_highs if sh["idx"] + 3 == idx), None)
        new_low = next((sl for sl in swing_lows if sl["idx"] + 3 == idx), None)
        valid_lows = [sl for sl in swing_lows if sl["idx"] + 3 <= idx]
        valid_highs = [sh for sh in swing_highs if sh["idx"] + 3 <= idx]

        if new_high:
            for sl in valid_lows[-30:]:
                if sl["idx"] < new_high["idx"] - 100:
                    continue
                if new_high["price"] > sl["price"]:
                    w = (new_high["price"] - sl["price"]) / sl["price"]
                    if min_w <= w <= max_w:
                        key = (new_high["idx"], sl["idx"])
                        if key not in active:
                            active[key] = Channel(
                                support=sl["price"],
                                resistance=new_high["price"],
                                support_idx=sl["idx"],
                                resistance_idx=new_high["idx"],
                                lowest_low=sl["price"],
                                highest_high=new_high["price"],
                            )

        if new_low:
            for sh in valid_highs[-30:]:
                if sh["idx"] < new_low["idx"] - 100:
                    continue
                if sh["price"] > new_low["price"]:
                    w = (sh["price"] - new_low["price"]) / new_low["price"]
                    if min_w <= w <= max_w:
                        key = (sh["idx"], new_low["idx"])
                        if key not in active:
                            active[key] = Channel(
                                support=new_low["price"],
                                resistance=sh["price"],
                                support_idx=new_low["idx"],
                                resistance_idx=sh["idx"],
                                lowest_low=new_low["price"],
                                highest_high=sh["price"],
                            )

        to_remove = []
        for key, ch in active.items():
            if close < ch.lowest_low * 0.96 or close > ch.highest_high * 1.04:
                to_remove.append(key)
                continue

            if new_low and new_low["price"] < ch.resistance:
                if new_low["price"] < ch.lowest_low:
                    ch.lowest_low = ch.support = new_low["price"]
                    ch.support_touches = 1
                elif ch.lowest_low < new_low["price"] < ch.support:
                    ch.support = new_low["price"]
                    ch.support_touches += 1
                elif abs(new_low["price"] - ch.support) / ch.support < touch_th:
                    ch.support_touches += 1

            if new_high and new_high["price"] > ch.support:
                if new_high["price"] > ch.highest_high:
                    ch.highest_high = ch.resistance = new_high["price"]
                    ch.resistance_touches = 1
                elif ch.resistance < new_high["price"] < ch.highest_high:
                    ch.resistance = new_high["price"]
                    ch.resistance_touches += 1
                elif abs(new_high["price"] - ch.resistance) / ch.resistance < touch_th:
                    ch.resistance_touches += 1

            if ch.support_touches >= 2 and ch.resistance_touches >= 2:
                ch.confirmed = True
            w = (ch.resistance - ch.support) / ch.support
            if not (min_w <= w <= max_w):
                to_remove.append(key)

        for key in to_remove:
            del active[key]

        candidates = [
            (
                ch.support_touches + ch.resistance_touches,
                (ch.resistance - ch.support) / ch.support,
                ch,
            )
            for ch in active.values()
            if ch.confirmed and ch.support * 0.98 <= close <= ch.resistance * 1.02
        ]

        if candidates:
            max_score = max(c[0] for c in candidates)
            top = [c for c in candidates if c[0] == max_score]
            htf_map[idx] = min(top, key=lambda c: c[1])[2]

    return htf_map


def simulate_trade(
    df_15m, entry_idx, direction, entry_price, sl_price, tp1_price, tp2_price
):
    tp1_hit = False
    exit_price = None
    exit_reason = None

    for j in range(entry_idx + 1, min(entry_idx + 100, len(df_15m))):
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
        exit_price = df_15m["close"].iloc[min(entry_idx + 99, len(df_15m) - 1)]
        exit_reason = "TIMEOUT"

    if direction == "LONG":
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    if tp1_hit and exit_reason == "SL":
        pnl_pct = (
            pnl_pct * 0.5
            + (
                (tp1_price - entry_price) / entry_price
                if direction == "LONG"
                else (entry_price - tp1_price) / entry_price
            )
            * 0.5
        )

    return {"pnl_pct": pnl_pct, "exit_reason": exit_reason, "tp1_hit": tp1_hit}


def run_backtest(df_1h, df_15m) -> BacktestResult:
    htf_map = build_channels(df_1h)

    touch_th = 0.003
    sl_buffer = 0.0008
    traded = set()
    trades = []

    for i in tqdm(range(len(df_15m)), desc="Backtesting"):
        htf_idx = i // 4
        ch = htf_map.get(htf_idx - 1)
        if not ch:
            continue
        if (ch.resistance - ch.support) / ch.support < 0.015:
            continue

        close = df_15m["close"].iloc[i]
        high = df_15m["high"].iloc[i]
        low = df_15m["low"].iloc[i]
        key = (round(ch.support), round(ch.resistance), i // 20)
        if key in traded:
            continue

        mid = (ch.resistance + ch.support) / 2

        if low <= ch.support * (1 + touch_th) and close > ch.support:
            entry = close
            sl = ch.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = ch.resistance * 0.998
            result = simulate_trade(df_15m, i, "LONG", entry, sl, tp1, tp2)
            trades.append(result)
            traded.add(key)
        elif high >= ch.resistance * (1 - touch_th) and close < ch.resistance:
            entry = close
            sl = ch.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = ch.support * 1.002
            result = simulate_trade(df_15m, i, "SHORT", entry, sl, tp1, tp2)
            trades.append(result)
            traded.add(key)

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 10000)

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004
    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for t in trades:
        risk_amount = capital * risk_pct
        leverage = min(
            risk_pct / abs(t["pnl_pct"]) if t["pnl_pct"] != 0 else 1, max_leverage
        )
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
