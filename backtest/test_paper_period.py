#!/usr/bin/env python3
"""
Test backtest for paper trading period (2026-01-21 ~ 2026-02-04)
Uses OCI SQLite DB data instead of parquet files
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ===== Configuration =====
DB_PATH = "/home/ubuntu/projects/CryptoFlow/server/data/cryptoflow.db"
SYMBOL = "BINANCE:BTCUSDT"

# Paper trading period
START_DATE = "2026-01-21"
END_DATE = "2026-02-04"

# Strategy parameters (must match paper trading)
TOUCH_THRESHOLD = 0.003
SL_BUFFER_PCT = 0.0008
MIN_CHANNEL_WIDTH = 0.015
MAX_CHANNEL_WIDTH = 0.05
RISK_PCT = 0.015
MAX_LEVERAGE = 15
FEE_PCT = 0.0004


@dataclass
class Channel:
    support: float
    resistance: float
    support_touches: int = 1
    resistance_touches: int = 1
    highest_high: float = 0
    lowest_low: float = 0
    confirmed: bool = False


def load_candles(
    db_path: str, symbol: str, table: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Load candles from SQLite DB"""
    conn = sqlite3.connect(db_path)

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(
        datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp()
        * 1000
    )

    query = f"""
        SELECT time, open, high, low, close, volume 
        FROM {table} 
        WHERE symbol = ? AND time >= ? AND time <= ?
        ORDER BY time
    """

    df = pd.read_sql_query(query, conn, params=(symbol, start_ts, end_ts))
    conn.close()

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")

    return df


def find_swing_points(
    df: pd.DataFrame, confirm_candles: int = 3
) -> Tuple[List[dict], List[dict]]:
    """Find swing highs and lows"""
    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    potential_high = None
    potential_low = None
    high_confirm = 0
    low_confirm = 0

    for i in range(len(df)):
        if potential_high is None or highs[i] > potential_high["price"]:
            potential_high = {"idx": i, "price": highs[i]}
            high_confirm = 0
        else:
            high_confirm += 1
            if high_confirm >= confirm_candles:
                swing_highs.append(potential_high)
                potential_high = {"idx": i, "price": highs[i]}
                high_confirm = 0

        if potential_low is None or lows[i] < potential_low["price"]:
            potential_low = {"idx": i, "price": lows[i]}
            low_confirm = 0
        else:
            low_confirm += 1
            if low_confirm >= confirm_candles:
                swing_lows.append(potential_low)
                potential_low = {"idx": i, "price": lows[i]}
                low_confirm = 0

    return swing_highs, swing_lows


def build_htf_channels(
    htf_df: pd.DataFrame, tiebreaker: str = "narrow"
) -> Dict[int, Channel]:
    """Build HTF channel map (matches paper trading logic)"""
    swing_highs, swing_lows = find_swing_points(htf_df)

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

    highs = htf_df["high"].values
    lows = htf_df["low"].values
    closes = htf_df["close"].values

    for i in range(len(htf_df)):
        new_high = None
        new_low = None

        for sh in swing_highs:
            if sh["idx"] + 3 == i:
                new_high = sh
                break

        for sl in swing_lows:
            if sl["idx"] + 3 == i:
                new_low = sl
                break

        # Create new channels
        if new_high:
            for sl in swing_lows:
                if sl["idx"] < new_high["idx"] and sl["idx"] >= new_high["idx"] - 30:
                    width_pct = (new_high["price"] - sl["price"]) / sl["price"]
                    if MIN_CHANNEL_WIDTH <= width_pct <= MAX_CHANNEL_WIDTH:
                        key = (new_high["idx"], sl["idx"])
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=sl["price"],
                                resistance=new_high["price"],
                                highest_high=new_high["price"],
                                lowest_low=sl["price"],
                            )

        if new_low:
            for sh in swing_highs:
                if sh["idx"] < new_low["idx"] and sh["idx"] >= new_low["idx"] - 30:
                    width_pct = (sh["price"] - new_low["price"]) / new_low["price"]
                    if MIN_CHANNEL_WIDTH <= width_pct <= MAX_CHANNEL_WIDTH:
                        key = (sh["idx"], new_low["idx"])
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=new_low["price"],
                                resistance=sh["price"],
                                highest_high=sh["price"],
                                lowest_low=new_low["price"],
                            )

        # Update existing channels
        current_close = closes[i]
        keys_to_remove = []

        for key, channel in active_channels.items():
            # Invalidate if price breaks out
            if (
                current_close < channel.lowest_low * 0.96
                or current_close > channel.highest_high * 1.04
            ):
                keys_to_remove.append(key)
                continue

            # Update with new swing points
            if new_low and new_low["price"] > channel.support:
                if new_low["price"] < channel.lowest_low:
                    channel.lowest_low = new_low["price"]
                    channel.support = new_low["price"]
                    channel.support_touches = 1
                elif new_low["price"] < channel.support:
                    channel.support = new_low["price"]
                    channel.support_touches += 1
                elif abs(new_low["price"] - channel.support) / channel.support < 0.004:
                    channel.support_touches += 1

            if new_high and new_high["price"] < channel.resistance:
                if new_high["price"] > channel.highest_high:
                    channel.highest_high = new_high["price"]
                    channel.resistance = new_high["price"]
                    channel.resistance_touches = 1
                elif new_high["price"] > channel.resistance:
                    channel.resistance = new_high["price"]
                    channel.resistance_touches += 1
                elif (
                    abs(new_high["price"] - channel.resistance) / channel.resistance
                    < 0.004
                ):
                    channel.resistance_touches += 1

            # Check confirmation
            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            # Check width still valid
            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > MAX_CHANNEL_WIDTH or width_pct < MIN_CHANNEL_WIDTH:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Select best channel (NARROW tiebreaker - matches paper trading)
        candidates = []
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if (
                current_close < channel.support * 0.98
                or current_close > channel.resistance * 1.02
            ):
                continue
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        best_channel = None
        if candidates:
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]

            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            else:
                # NARROW tiebreaker
                best_channel = min(top_candidates, key=lambda c: c[1])[2]

        htf_channel_map[i] = best_channel

    return htf_channel_map


def simulate_trade(
    ltf_df: pd.DataFrame,
    idx: int,
    trade_type: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
) -> dict:
    """Simulate a single trade"""
    highs = ltf_df["high"].values
    lows = ltf_df["low"].values

    hit_tp1 = False
    pnl_pct = 0.0
    outcome = 0
    current_sl = sl_price
    exit_idx = idx
    status = "TIMEOUT"

    for j in range(idx + 1, min(idx + 150, len(ltf_df))):
        if trade_type == "LONG":
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -(abs(entry_price - sl_price) / entry_price)
                    outcome = 0
                    exit_idx = j
                    status = "SL_HIT"
                    break
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * ((tp1_price - entry_price) / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    outcome = 0.5
                    exit_idx = j
                    status = "BE_HIT"
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * ((tp2_price - entry_price) / entry_price)
                    outcome = 1
                    exit_idx = j
                    status = "TP2_HIT"
                    break
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -(abs(entry_price - sl_price) / entry_price)
                    outcome = 0
                    exit_idx = j
                    status = "SL_HIT"
                    break
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * ((entry_price - tp1_price) / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    outcome = 0.5
                    exit_idx = j
                    status = "BE_HIT"
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * ((entry_price - tp2_price) / entry_price)
                    outcome = 1
                    exit_idx = j
                    status = "TP2_HIT"
                    break

    return {
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "status": status,
        "exit_idx": exit_idx,
        "hit_tp1": hit_tp1,
    }


def run_backtest(htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> dict:
    """Run backtest for the period"""
    print(f"\n{'=' * 60}")
    print(f"  Backtest: {START_DATE} ~ {END_DATE}")
    print(f"  HTF candles: {len(htf_df)}, LTF candles: {len(ltf_df)}")
    print(f"{'=' * 60}")

    # Build HTF channels
    print("\nBuilding HTF channels...")
    htf_channel_map = build_htf_channels(htf_df, tiebreaker="narrow")
    confirmed_count = sum(1 for c in htf_channel_map.values() if c is not None)
    print(f"  Confirmed channel mappings: {confirmed_count}")

    # Collect setups
    tf_ratio = 4  # 1h / 15m = 4
    ltf_closes = ltf_df["close"].values
    ltf_highs = ltf_df["high"].values
    ltf_lows = ltf_df["low"].values

    trades = []
    traded_entries = set()

    print("\nScanning for setups...")
    for i in range(tf_ratio, len(ltf_df)):
        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Lookahead bias prevention

        if channel is None:
            continue

        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        # Check channel width
        width_pct = (channel.resistance - channel.support) / channel.support
        if width_pct < MIN_CHANNEL_WIDTH:
            continue

        mid_price = (channel.resistance + channel.support) / 2

        # Cooldown key
        trade_key = (
            round(channel.support),
            round(channel.resistance),
            "bounce",
            i // 20,
        )
        if trade_key in traded_entries:
            continue

        # Support bounce → LONG
        if (
            current_low <= channel.support * (1 + TOUCH_THRESHOLD)
            and current_close > channel.support
        ):
            entry_price = current_close
            sl_price = channel.support * (1 - SL_BUFFER_PCT)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            if entry_price > sl_price and tp1_price > entry_price:
                result = simulate_trade(
                    ltf_df, i, "LONG", entry_price, sl_price, tp1_price, tp2_price
                )
                result["direction"] = "LONG"
                result["entry_price"] = entry_price
                result["entry_idx"] = i
                result["entry_time"] = ltf_df.index[i]
                result["channel_support"] = channel.support
                result["channel_resistance"] = channel.resistance
                result["channel_width"] = width_pct * 100
                trades.append(result)
                traded_entries.add(trade_key)

        # Resistance bounce → SHORT
        elif (
            current_high >= channel.resistance * (1 - TOUCH_THRESHOLD)
            and current_close < channel.resistance
        ):
            entry_price = current_close
            sl_price = channel.resistance * (1 + SL_BUFFER_PCT)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            if sl_price > entry_price and entry_price > tp1_price:
                result = simulate_trade(
                    ltf_df, i, "SHORT", entry_price, sl_price, tp1_price, tp2_price
                )
                result["direction"] = "SHORT"
                result["entry_price"] = entry_price
                result["entry_idx"] = i
                result["entry_time"] = ltf_df.index[i]
                result["channel_support"] = channel.support
                result["channel_resistance"] = channel.resistance
                result["channel_width"] = width_pct * 100
                trades.append(result)
                traded_entries.add(trade_key)

    print(f"  Found {len(trades)} trades")

    if not trades:
        print("\n  No trades found!")
        return {}

    # Calculate results with capital management
    df = pd.DataFrame(trades)

    capital = 10000
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
        sl_dist = (
            abs(
                trade["entry_price"]
                - (
                    trade["channel_support"] * (1 - SL_BUFFER_PCT)
                    if trade["direction"] == "LONG"
                    else trade["channel_resistance"] * (1 + SL_BUFFER_PCT)
                )
            )
            / trade["entry_price"]
        )
        if sl_dist <= 0:
            continue

        leverage = min(RISK_PCT / sl_dist, MAX_LEVERAGE)
        position_value = capital * leverage

        gross_pnl = position_value * trade["pnl_pct"]
        fees = position_value * FEE_PCT * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Period: {START_DATE} ~ {END_DATE}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Final Capital: ${capital:,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"{'=' * 60}")

    # Status breakdown
    print(f"\n  Status Breakdown:")
    status_counts = df["status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"    {status}: {count} ({pct:.1f}%)")

    # Direction breakdown
    print(f"\n  Direction Breakdown:")
    for direction in ["LONG", "SHORT"]:
        dir_df = df[df["direction"] == direction]
        if len(dir_df) > 0:
            dir_wins = len(dir_df[dir_df["outcome"] > 0])
            dir_wr = dir_wins / len(dir_df) * 100
            print(f"    {direction}: {len(dir_df)} trades, WR {dir_wr:.1f}%")

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "final_capital": capital,
        "total_return": total_return,
    }


def main():
    print(f"Loading data from {DB_PATH}...")

    # Load HTF (1h) and LTF (15m) candles
    htf_df = load_candles(DB_PATH, SYMBOL, "candles_60", START_DATE, END_DATE)
    ltf_df = load_candles(DB_PATH, SYMBOL, "candles_15", START_DATE, END_DATE)

    print(f"  1h candles: {len(htf_df)} ({htf_df.index.min()} ~ {htf_df.index.max()})")
    print(f"  15m candles: {len(ltf_df)} ({ltf_df.index.min()} ~ {ltf_df.index.max()})")

    if len(htf_df) < 50 or len(ltf_df) < 200:
        print("Not enough data for backtest!")
        return

    results = run_backtest(htf_df, ltf_df)

    if results:
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON WITH PAPER TRADING")
        print(f"{'=' * 60}")
        print(f"  Paper Trading (old): 42 trades, 26.5% WR, -29.08% return")
        print(
            f"  Backtest:           {results['trades']} trades, {results['win_rate']:.1f}% WR, {results['total_return']:+.2f}% return"
        )
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
