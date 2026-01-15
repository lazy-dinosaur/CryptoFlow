#!/usr/bin/env python3
"""
Binance Data Vision Downloader
Downloads historical trades data for backtesting
"""

import os
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Configuration
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/trades"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "binance")

def download_file(symbol: str, year: int, month: int) -> dict:
    """Download a single month's trades data."""
    month_str = f"{month:02d}"
    filename = f"{symbol}-trades-{year}-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    filepath = os.path.join(DATA_DIR, filename)

    # Skip if already exists
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return {"file": filename, "status": "exists", "size_mb": round(size_mb, 1)}

    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 404:
            return {"file": filename, "status": "not_found"}

        response.raise_for_status()

        # Save file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Downloaded {filename} ({size_mb:.1f} MB)")
        return {"file": filename, "status": "downloaded", "size_mb": round(size_mb, 1)}

    except Exception as e:
        return {"file": filename, "status": "error", "error": str(e)}

def download_range(symbol: str, start_year: int, start_month: int,
                   end_year: int, end_month: int, max_workers: int = 4):
    """Download data for a range of months."""

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate list of (year, month) tuples
    months_to_download = []
    current_year, current_month = start_year, start_month

    while (current_year, current_month) <= (end_year, end_month):
        months_to_download.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    print(f"Downloading {len(months_to_download)} months of {symbol} data...")
    print(f"Range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
    print("=" * 50)

    results = []

    # Download in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, symbol, year, month): (year, month)
            for year, month in months_to_download
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary:")

    downloaded = [r for r in results if r["status"] == "downloaded"]
    existed = [r for r in results if r["status"] == "exists"]
    not_found = [r for r in results if r["status"] == "not_found"]
    errors = [r for r in results if r["status"] == "error"]

    total_size = sum(r.get("size_mb", 0) for r in downloaded + existed)

    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Already existed: {len(existed)}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total size: {total_size:.1f} MB")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Download Binance historical trades data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--start", default="2024-01", help="Start month (YYYY-MM)")
    parser.add_argument("--end", default="2025-12", help="End month (YYYY-MM)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")

    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    end_year, end_month = map(int, args.end.split("-"))

    download_range(
        symbol=args.symbol,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()
