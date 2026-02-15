#!/usr/bin/env python3
"""Compare paper trading trades with backtest trades"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "/home/ubuntu/projects/CryptoFlow/server/data/cryptoflow.db"
PAPER_DB = "/home/ubuntu/projects/CryptoFlow/server/data/ml_paper_trading.db.bak.20260204_114840"


def load_paper_trades():
    conn = sqlite3.connect(PAPER_DB)
    df = pd.read_sql_query(
        """
        SELECT timestamp, direction, entry_price, sl_price, tp1_price, tp2_price,
               status, pnl_pct, channel_support, channel_resistance
        FROM trades ORDER BY timestamp
    """,
        conn,
    )
    conn.close()
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def load_candles(table, symbol, start_ts, end_ts):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"""
        SELECT time, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ? AND time >= ? AND time <= ?
        ORDER BY time
    """,
        conn,
        params=(symbol, start_ts, end_ts),
    )
    conn.close()
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.set_index("time")


def main():
    paper_df = load_paper_trades()

    print("=" * 80)
    print("  PAPER TRADING vs BACKTEST COMPARISON")
    print("=" * 80)

    # Remove exact duplicates
    unique_paper = paper_df.drop_duplicates(
        subset=["timestamp", "direction", "entry_price"]
    )
    print(f"\nPaper Trading: {len(paper_df)} total, {len(unique_paper)} unique trades")
    print(f"Duplicates removed: {len(paper_df) - len(unique_paper)}")

    # Group duplicates
    dup_counts = paper_df.groupby(["timestamp", "direction", "entry_price"]).size()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        print(f"\nDuplicate entries found:")
        for (ts, direction, price), count in dups.items():
            time = pd.to_datetime(ts, unit="ms")
            print(f"  {time} | {direction} @ {price:.1f} | {count}x duplicated")

    print("\n" + "=" * 80)
    print("  UNIQUE TRADES DETAIL")
    print("=" * 80)

    # Status summary
    status_counts = unique_paper["status"].value_counts()
    print(f"\nStatus breakdown:")
    for status, count in status_counts.items():
        pct = count / len(unique_paper) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Direction summary
    print(f"\nDirection breakdown:")
    for direction in ["LONG", "SHORT"]:
        dir_df = unique_paper[unique_paper["direction"] == direction]
        if len(dir_df) > 0:
            wins = len(dir_df[dir_df["pnl_pct"] > 0])
            wr = wins / len(dir_df) * 100
            print(f"  {direction}: {len(dir_df)} trades, {wins} wins, WR {wr:.1f}%")

    # Calculate win rate
    wins = len(unique_paper[unique_paper["pnl_pct"] > 0])
    losses = len(unique_paper[unique_paper["pnl_pct"] <= 0])
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"  CORRECTED RESULTS (after removing duplicates)")
    print(f"{'=' * 80}")
    print(f"  Unique Trades: {len(unique_paper)}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win Rate: {wr:.1f}%")

    # Channel width analysis
    unique_paper["width_pct"] = (
        (unique_paper["channel_resistance"] - unique_paper["channel_support"])
        / unique_paper["channel_support"]
        * 100
    )
    narrow = unique_paper[unique_paper["width_pct"] < 1.5]
    print(
        f"\n  Trades with channel width < 1.5%: {len(narrow)} ({len(narrow) / len(unique_paper) * 100:.1f}%)"
    )
    if len(narrow) > 0:
        narrow_wins = len(narrow[narrow["pnl_pct"] > 0])
        narrow_wr = narrow_wins / len(narrow) * 100
        print(f"    Narrow channel WR: {narrow_wr:.1f}%")

    print("\n" + "=" * 80)
    print("  TRADE LIST (first 25)")
    print("=" * 80)
    print(
        f"{'Time':<20} {'Dir':<6} {'Entry':>10} {'SL':>10} {'Status':<10} {'PnL%':>8} {'Width%':>7}"
    )
    print("-" * 80)

    for _, row in unique_paper.head(25).iterrows():
        print(
            f"{row['time'].strftime('%Y-%m-%d %H:%M'):<20} {row['direction']:<6} {row['entry_price']:>10.1f} {row['sl_price']:>10.1f} {row['status']:<10} {row['pnl_pct'] * 100:>+7.2f}% {row['width_pct']:>6.2f}%"
        )


if __name__ == "__main__":
    main()
