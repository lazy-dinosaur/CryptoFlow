#!/usr/bin/env python3
"""
Binance Trades Data Parser (Polars - High Performance)
Converts downloaded trades CSV files to candles with delta calculation
"""

import os
import polars as pl
from glob import glob
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Optional
import shutil

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "binance")
PARSED_DIR = os.path.join(os.path.dirname(__file__), "data", "parsed")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "data", "temp")

# Timeframes in minutes
TIMEFRAMES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def process_csv_file(args: tuple) -> dict:
    """Process a single CSV file using Polars."""
    csv_path, timeframes = args
    filename = os.path.basename(csv_path)

    try:
        # Read CSV with Polars (much faster than pandas)
        df = pl.read_csv(
            csv_path,
            columns=["price", "qty", "time", "is_buyer_maker"],
            dtypes={
                "price": pl.Float64,
                "qty": pl.Float64,
                "time": pl.Int64,
                "is_buyer_maker": pl.Boolean,
            },
        )

        # Convert time to datetime
        df = df.with_columns(pl.from_epoch("time", time_unit="ms").alias("time"))

        # Calculate buy/sell volumes
        df = df.with_columns(
            [
                pl.when(~pl.col("is_buyer_maker"))
                .then(pl.col("qty"))
                .otherwise(0.0)
                .alias("buy_volume"),
                pl.when(pl.col("is_buyer_maker"))
                .then(pl.col("qty"))
                .otherwise(0.0)
                .alias("sell_volume"),
            ]
        )

        trades_count = len(df)
        result = {"filename": filename, "trades": trades_count, "candles": {}}

        # Aggregate to each timeframe
        for tf_name in timeframes:
            tf_minutes = TIMEFRAMES[tf_name]
            interval = f"{tf_minutes}m"

            candles = (
                df.group_by_dynamic("time", every=interval)
                .agg(
                    [
                        pl.col("price").first().alias("open"),
                        pl.col("price").max().alias("high"),
                        pl.col("price").min().alias("low"),
                        pl.col("price").last().alias("close"),
                        pl.col("qty").sum().alias("volume"),
                        pl.col("buy_volume").sum().alias("buy_volume"),
                        pl.col("sell_volume").sum().alias("sell_volume"),
                        pl.col("price").count().alias("trade_count"),
                    ]
                )
                .with_columns(
                    (pl.col("buy_volume") - pl.col("sell_volume")).alias("delta")
                )
                .drop_nulls()
            )

            result["candles"][tf_name] = candles

        return result

    except Exception as e:
        return {"filename": filename, "error": str(e)}


def merge_candles(results_list: list, timeframe: str) -> pl.DataFrame:
    """Merge candles from multiple files using Polars."""
    all_candles = []
    for r in results_list:
        if "candles" in r and timeframe in r["candles"]:
            all_candles.append(r["candles"][timeframe])

    if not all_candles:
        return pl.DataFrame()

    combined = pl.concat(all_candles)

    # Group by time and aggregate (handle overlaps)
    merged = (
        combined.group_by("time")
        .agg(
            [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("buy_volume").sum(),
                pl.col("sell_volume").sum(),
                pl.col("trade_count").sum(),
            ]
        )
        .with_columns((pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"))
        .sort("time")
    )

    return merged


def parse_symbol_data(
    symbol: str, timeframes: Optional[list] = None, workers: Optional[int] = None
):
    """Parse all CSV data for a symbol into candles using Polars."""

    if timeframes is None:
        timeframes = list(TIMEFRAMES.keys())

    if workers is None:
        workers = min(cpu_count(), 6)

    # Find CSV files in temp directory
    csv_pattern = os.path.join(TEMP_DIR, f"{symbol}-trades-*.csv")
    csv_files = sorted(glob(csv_pattern))

    if not csv_files:
        print(f"No CSV files found in {TEMP_DIR}")
        print("Run unzip first or use download_data.py")
        return

    print(f"\n{'='*60}")
    print(f"  BTCUSDT Data Parser (Polars)")
    print(f"{'='*60}")
    print(f"  Files: {len(csv_files)}")
    print(f"  Workers: {workers}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"{'='*60}\n")

    # Process CSVs in parallel
    print("Processing trades data...")
    args_list = [(csv, timeframes) for csv in csv_files]

    results = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_csv_file, args_list),
            total=len(args_list),
            desc="Processing",
            unit="file",
        ):
            if "error" in result:
                tqdm.write(f"  ✗ {result['filename']}: {result['error']}")
            else:
                tqdm.write(f"  ✓ {result['filename']}: {result['trades']:,} trades")
            results.append(result)

    # Filter successful results
    valid_results = [r for r in results if "candles" in r]
    print(f"\nProcessed {len(valid_results)}/{len(csv_files)} files successfully")

    if not valid_results:
        print("No data processed successfully")
        return

    # Merge and save
    print("\nMerging and saving candles...")
    symbol_dir = os.path.join(PARSED_DIR, symbol.lower())
    os.makedirs(symbol_dir, exist_ok=True)

    for tf_name in tqdm(timeframes, desc="Saving", unit="tf"):
        merged = merge_candles(valid_results, tf_name)
        if len(merged) > 0:
            output_path = os.path.join(symbol_dir, f"candles_{tf_name}.parquet")
            merged.write_parquet(output_path)

            first_date = merged["time"].min()
            last_date = merged["time"].max()
            tqdm.write(
                f"  ✓ {tf_name}: {len(merged):,} candles ({first_date} ~ {last_date})"
            )

    print(f"\n{'='*60}")
    print(f"  Complete! Data saved to: {symbol_dir}")
    print(f"{'='*60}\n")
    return symbol_dir


def load_candles(symbol: str, timeframe: str) -> pl.DataFrame:
    """Load parsed candles for a symbol and timeframe."""
    path = os.path.join(PARSED_DIR, symbol.lower(), f"candles_{timeframe}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No parsed data found at {path}. Run parse_data.py first."
        )

    return pl.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(
        description="Parse Binance trades data to candles (Polars)"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help=f"Timeframes to generate (default: all). Options: {list(TIMEFRAMES.keys())}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 6)",
    )

    args = parser.parse_args()

    parse_symbol_data(
        symbol=args.symbol, timeframes=args.timeframes, workers=args.workers
    )


if __name__ == "__main__":
    main()
