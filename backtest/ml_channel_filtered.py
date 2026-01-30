#!/usr/bin/env python3
"""
Filtered Channel Strategy Backtest

Tests the improvements identified in strategy research:
1. Channel width filter: min 1.5% (narrow channels had 50% WR)
2. Volume filter: volume_ratio > 1.0 (high volume = higher WR)
3. CVD filter: 20-candle CVD aligned with trade direction (77% vs 67% WR)

Baseline: ml_channel_tiebreaker_proper.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str


@dataclass
class Channel:
    support: float
    support_idx: int
    resistance: float
    resistance_idx: int
    lowest_low: float
    highest_high: float
    support_touches: int = 1
    resistance_touches: int = 1
    confirmed: bool = False


def find_swing_points(
    candles: pd.DataFrame, confirm_candles: int = 3
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find swing highs and lows on HTF."""
    highs = candles["high"].values
    lows = candles["low"].values

    swing_highs = []
    swing_lows = []

    potential_high_idx = 0
    potential_high_price = highs[0]
    candles_since_high = 0

    potential_low_idx = 0
    potential_low_price = lows[0]
    candles_since_low = 0

    for i in range(1, len(candles)):
        if highs[i] > potential_high_price:
            potential_high_idx = i
            potential_high_price = highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            if candles_since_high == confirm_candles:
                swing_highs.append(
                    SwingPoint(
                        idx=potential_high_idx, price=potential_high_price, type="high"
                    )
                )

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append(
                    SwingPoint(
                        idx=potential_low_idx, price=potential_low_price, type="low"
                    )
                )

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def build_htf_channels(
    htf_candles: pd.DataFrame,
    max_channel_width: float = 0.05,
    min_channel_width: float = 0.008,
    touch_threshold: float = 0.004,
    tiebreaker: str = "first",
) -> Dict[int, Channel]:
    """Build evolving channels on HTF."""
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    highs = htf_candles["high"].values
    lows = htf_candles["low"].values
    closes = htf_candles["close"].values

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

    for i in range(len(htf_candles)):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        new_high = None
        new_low = None

        for sh in swing_highs:
            if sh.idx + 3 == i:
                new_high = sh
                break

        for sl in swing_lows:
            if sl.idx + 3 == i:
                new_low = sl
                break

        valid_swing_lows = [sl for sl in swing_lows if sl.idx + 3 <= i]
        valid_swing_highs = [sh for sh in swing_highs if sh.idx + 3 <= i]

        if new_high:
            for sl in valid_swing_lows[-30:]:
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (new_high.idx, sl.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=sl.price,
                                support_idx=sl.idx,
                                resistance=new_high.price,
                                resistance_idx=new_high.idx,
                                lowest_low=sl.price,
                                highest_high=new_high.price,
                            )

        if new_low:
            for sh in valid_swing_highs[-30:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (sh.idx, new_low.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=new_low.price,
                                support_idx=new_low.idx,
                                resistance=sh.price,
                                resistance_idx=sh.idx,
                                lowest_low=new_low.price,
                                highest_high=sh.price,
                            )

        keys_to_remove = []
        for key, channel in active_channels.items():
            if (
                current_close < channel.lowest_low * 0.96
                or current_close > channel.highest_high * 1.04
            ):
                keys_to_remove.append(key)
                continue

            if new_low and new_low.price < channel.resistance:
                if new_low.price < channel.lowest_low:
                    channel.lowest_low = new_low.price
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches = 1
                elif (
                    new_low.price > channel.lowest_low
                    and new_low.price < channel.support
                ):
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches += 1
                elif (
                    abs(new_low.price - channel.support) / channel.support
                    < touch_threshold
                ):
                    channel.support_touches += 1

            if new_high and new_high.price > channel.support:
                if new_high.price > channel.highest_high:
                    channel.highest_high = new_high.price
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches = 1
                elif (
                    new_high.price < channel.highest_high
                    and new_high.price > channel.resistance
                ):
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches += 1
                elif (
                    abs(new_high.price - channel.resistance) / channel.resistance
                    < touch_threshold
                ):
                    channel.resistance_touches += 1

            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

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

        if candidates:
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]

            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            elif tiebreaker == "narrow":
                best_channel = min(top_candidates, key=lambda c: c[1])[2]
            else:
                best_channel = top_candidates[0][2]

            htf_channel_map[i] = best_channel

    return htf_channel_map


def simulate_trade(
    candles,
    idx,
    trade_type,
    entry_price,
    sl_price,
    tp1_price,
    tp2_price,
    channel,
    volumes,
    deltas,
    avg_volume,
    avg_delta,
    cvd_recent,
    opens,
    closes,
):
    """Simulate trade with partial TP + breakeven."""
    highs = candles["high"].values
    lows = candles["low"].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)
    rr_ratio = reward2 / risk if risk > 0 else 0

    outcome = 0
    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price

    for j in range(idx + 1, min(idx + 150, len(candles))):
        if trade_type == "LONG":
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    break
                if highs[j] >= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if lows[j] <= current_sl:
                    outcome = 0.5
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break
        else:
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    break
                if lows[j] <= tp1_price:
                    pnl_pct += 0.5 * (reward1 / entry_price)
                    hit_tp1 = True
                    current_sl = entry_price
            else:
                if highs[j] >= current_sl:
                    outcome = 0.5
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break

    width_pct = (channel.resistance - channel.support) / channel.support

    return {
        "idx": idx,
        "type": trade_type,
        "entry": entry_price,
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "rr_ratio": rr_ratio,
        "pnl_pct": pnl_pct,
        "channel_width": width_pct,
        "total_touches": channel.support_touches + channel.resistance_touches,
        "volume_at_entry": volumes[idx],
        "volume_ratio": volumes[idx] / avg_volume if avg_volume > 0 else 1,
        "delta_at_entry": deltas[idx],
        "cvd_recent": cvd_recent,
        "body_bullish": 1 if closes[idx] > opens[idx] else 0,
        "outcome": outcome,
    }


def collect_filtered_setups(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008,
    # Filter parameters
    min_channel_width_filter: float = 0.015,  # 1.5% min
    min_volume_ratio: float = 1.0,  # Only high volume
    require_cvd_alignment: bool = True,  # CVD must align
    quiet: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    Collect setups with optional filters.
    Returns (filtered_setups, all_setups) for comparison.
    """
    htf_channel_map = build_htf_channels(htf_candles, tiebreaker="first")

    ltf_highs = ltf_candles["high"].values
    ltf_lows = ltf_candles["low"].values
    ltf_closes = ltf_candles["close"].values
    ltf_opens = ltf_candles["open"].values
    ltf_volumes = ltf_candles["volume"].values
    ltf_deltas = ltf_candles["delta"].values

    all_setups = []
    filtered_setups = []
    traded_entries = set()

    tf_mins = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"Filtering: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)

        if not channel:
            continue

        # Historical features
        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist["volume"].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist["delta"].mean() if len(hist) > 0 else 0
        cvd_recent = hist["delta"].sum() if len(hist) > 0 else 0

        mid_price = (channel.resistance + channel.support) / 2
        channel_width = (channel.resistance - channel.support) / channel.support
        volume_ratio = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1

        trade_key = (
            round(channel.support),
            round(channel.resistance),
            "bounce",
            i // 20,
        )
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch (LONG)
        if (
            current_low <= channel.support * (1 + touch_threshold)
            and current_close > channel.support
        ):
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price

            if risk > 0 and reward1 > 0:
                # CVD alignment for LONG: positive CVD (buyers dominant)
                cvd_aligned = cvd_recent > 0

                setup = simulate_trade(
                    ltf_candles,
                    i,
                    "LONG",
                    entry_price,
                    sl_price,
                    tp1_price,
                    tp2_price,
                    channel,
                    ltf_volumes,
                    ltf_deltas,
                    avg_volume,
                    avg_delta,
                    cvd_recent,
                    ltf_opens,
                    ltf_closes,
                )

                if setup:
                    setup["cvd_aligned"] = cvd_aligned
                    all_setups.append(setup)
                    traded_entries.add(trade_key)

                    # Apply filters
                    passes_width = channel_width >= min_channel_width_filter
                    passes_volume = volume_ratio >= min_volume_ratio
                    passes_cvd = cvd_aligned if require_cvd_alignment else True

                    if passes_width and passes_volume and passes_cvd:
                        filtered_setups.append(setup)

        # BOUNCE: Resistance touch (SHORT)
        elif (
            current_high >= channel.resistance * (1 - touch_threshold)
            and current_close < channel.resistance
        ):
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

            if risk > 0 and reward1 > 0:
                # CVD alignment for SHORT: negative CVD (sellers dominant)
                cvd_aligned = cvd_recent < 0

                setup = simulate_trade(
                    ltf_candles,
                    i,
                    "SHORT",
                    entry_price,
                    sl_price,
                    tp1_price,
                    tp2_price,
                    channel,
                    ltf_volumes,
                    ltf_deltas,
                    avg_volume,
                    avg_delta,
                    cvd_recent,
                    ltf_opens,
                    ltf_closes,
                )

                if setup:
                    setup["cvd_aligned"] = cvd_aligned
                    all_setups.append(setup)
                    traded_entries.add(trade_key)

                    # Apply filters
                    passes_width = channel_width >= min_channel_width_filter
                    passes_volume = volume_ratio >= min_volume_ratio
                    passes_cvd = cvd_aligned if require_cvd_alignment else True

                    if passes_width and passes_volume and passes_cvd:
                        filtered_setups.append(setup)

    return filtered_setups, all_setups


def run_backtest(df: pd.DataFrame, label: str = "") -> dict:
    """Run backtest and return results."""
    if len(df) == 0:
        print(f"  {label}: No trades")
        return {"trades": 0, "return": 0, "max_dd": 0, "win_rate": 0, "avg_pnl": 0}

    capital = 10000
    risk_pct = 0.015
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
        sl_dist = abs(trade["entry"] - trade["sl"]) / trade["entry"]
        if sl_dist <= 0:
            continue

        leverage = min(risk_pct / sl_dist, max_leverage)
        position_value = capital * leverage

        gross_pnl = position_value * trade["pnl_pct"]
        fees = position_value * fee_pct * 2
        net_pnl = gross_pnl - fees

        capital += net_pnl
        capital = max(capital, 0)

        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if capital <= 0:
            break

    total_return = (capital - 10000) / 10000 * 100
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_pnl = df["pnl_pct"].mean() * 100

    print(f"\n  {label}:")
    print(f"    Trades: {len(df)}, Avg PnL: {avg_pnl:+.4f}%")
    print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd * 100:.1f}%")
    print(f"    Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"    Final: ${capital:,.2f}")

    return {
        "trades": len(df),
        "return": total_return,
        "max_dd": max_dd * 100,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "final_capital": capital,
    }


def main():
    htf = "1h"
    ltf = "15m"

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   Filtered Channel Strategy Backtest                                   ║
║   Testing: Channel Width, Volume Ratio, CVD Alignment                  ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index("time")
    htf_candles = htf_candles[htf_candles.index.year == 2024]
    print(f"  Loaded {len(htf_candles):,} HTF candles (2024)")

    print(f"\nLoading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index("time")
    ltf_candles = ltf_candles[ltf_candles.index.year == 2024]
    print(f"  Loaded {len(ltf_candles):,} LTF candles (2024)")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}")

    results = {}

    # ========================================
    # Test 1: Baseline (no filters)
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 1: BASELINE (No Filters)")
    print("=" * 70)

    _, all_setups = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0,
        min_volume_ratio=0,
        require_cvd_alignment=False,
        quiet=True,
    )
    df_baseline = pd.DataFrame(all_setups)
    results["baseline"] = run_backtest(df_baseline, "Baseline")

    # ========================================
    # Test 2: Channel Width Filter Only (>= 1.5%)
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 2: Channel Width Filter (>= 1.5%)")
    print("=" * 70)

    filtered_width, _ = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0.015,
        min_volume_ratio=0,
        require_cvd_alignment=False,
        quiet=True,
    )
    df_width = pd.DataFrame(filtered_width)
    results["width_only"] = run_backtest(df_width, "Width >= 1.5%")

    # ========================================
    # Test 3: Volume Filter Only (ratio >= 1.0)
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 3: Volume Filter (ratio >= 1.0)")
    print("=" * 70)

    filtered_volume, _ = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0,
        min_volume_ratio=1.0,
        require_cvd_alignment=False,
        quiet=True,
    )
    df_volume = pd.DataFrame(filtered_volume)
    results["volume_only"] = run_backtest(df_volume, "Volume >= 1.0x")

    # ========================================
    # Test 4: CVD Filter Only
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 4: CVD Alignment Filter")
    print("=" * 70)

    filtered_cvd, _ = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0,
        min_volume_ratio=0,
        require_cvd_alignment=True,
        quiet=True,
    )
    df_cvd = pd.DataFrame(filtered_cvd)
    results["cvd_only"] = run_backtest(df_cvd, "CVD Aligned")

    # ========================================
    # Test 5: Combined Filters (All three)
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 5: COMBINED FILTERS (Width + Volume + CVD)")
    print("=" * 70)

    filtered_all, _ = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0.015,
        min_volume_ratio=1.0,
        require_cvd_alignment=True,
        quiet=True,
    )
    df_all = pd.DataFrame(filtered_all)
    results["combined"] = run_backtest(df_all, "All Filters Combined")

    # ========================================
    # Test 6: Width + CVD (no volume - might reduce trades too much)
    # ========================================
    print("\n" + "=" * 70)
    print("  TEST 6: Width + CVD (no volume filter)")
    print("=" * 70)

    filtered_width_cvd, _ = collect_filtered_setups(
        htf_candles,
        ltf_candles,
        htf,
        ltf,
        min_channel_width_filter=0.015,
        min_volume_ratio=0,
        require_cvd_alignment=True,
        quiet=True,
    )
    df_width_cvd = pd.DataFrame(filtered_width_cvd)
    results["width_cvd"] = run_backtest(df_width_cvd, "Width + CVD")

    # ========================================
    # Summary Comparison
    # ========================================
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Filter':<25} {'Trades':>8} {'Win Rate':>10} {'Avg PnL':>10} {'Return':>12} {'Max DD':>10}"
    )
    print("-" * 75)

    for name, res in results.items():
        print(
            f"{name:<25} {res['trades']:>8} {res['win_rate']:>9.1f}% {res['avg_pnl']:>+9.4f}% {res['return']:>+11.1f}% {res['max_dd']:>9.1f}%"
        )

    # Find best
    best_return = max(results.items(), key=lambda x: x[1]["return"])
    best_wr = max(results.items(), key=lambda x: x[1]["win_rate"])

    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    print(f"\n  Best Return: {best_return[0]} ({best_return[1]['return']:+.1f}%)")
    print(f"  Best Win Rate: {best_wr[0]} ({best_wr[1]['win_rate']:.1f}%)")

    # Trade-off analysis
    baseline_ret = results["baseline"]["return"]
    for name, res in results.items():
        if name == "baseline":
            continue
        trade_reduction = (1 - res["trades"] / results["baseline"]["trades"]) * 100
        return_change = res["return"] - baseline_ret
        print(f"\n  {name}:")
        print(f"    Trade reduction: {trade_reduction:.1f}%")
        print(f"    Return change: {return_change:+.1f}%")
        print(f"    Worth it: {'YES' if return_change > 0 else 'NO'}")


if __name__ == "__main__":
    main()
