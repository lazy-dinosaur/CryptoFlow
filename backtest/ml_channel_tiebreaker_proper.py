#!/usr/bin/env python3
"""
Multi-Timeframe Proper Channel Strategy

HTF (1H): Channel detection with evolving S/R
LTF (15m): Entry execution

Channel Logic:
1. Support = lowest swing low that contains subsequent lows
2. If new low < current support → support updates
3. If new low > lowest but < previous → channel tightens
4. Same for resistance

Trade Types:
- BOUNCE: Price touches HTF S/R on LTF and bounces
- FAKEOUT: Price breaks HTF S/R, comes back → tight SL

Exit Strategy:
- 50% at mid-channel
- Move SL to breakeven
- 50% at opposite channel edge
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


@dataclass
class FakeoutSignal:
    htf_idx: int
    type: str  # 'bull' or 'bear'
    channel: Channel
    extreme: float  # Fakeout extreme price


def build_htf_channels(
    htf_candles: pd.DataFrame,
    max_channel_width: float = 0.05,  # 5%
    min_channel_width: float = 0.015,
    touch_threshold: float = 0.004,
    tiebreaker: str = "first",
) -> Tuple[Dict[int, Channel], List[FakeoutSignal]]:
    """
    Build evolving channels on HTF.
    Returns dict mapping HTF candle index to active confirmed channel.
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    highs = htf_candles["high"].values
    lows = htf_candles["low"].values
    closes = htf_candles["close"].values

    # Track all channels
    active_channels: Dict[tuple, Channel] = {}

    # Result: HTF index -> best confirmed channel
    htf_channel_map: Dict[int, Channel] = {}

    # Track fakeouts in first pass
    fakeout_signals: List[FakeoutSignal] = []
    pending_breaks: List[dict] = []  # Breakouts waiting for return
    max_fakeout_wait_htf = 5

    for i in range(len(htf_candles)):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]

        # Find new swing points confirmed at this index
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

        # Create new channel from H-L pair
        # Get swing points confirmed by current index (3 candles after peak)
        # A swing point at idx is confirmed at idx+3, so we can only use it at i if idx+3 <= i
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

        # Process pending breakouts for fakeout detection BEFORE channel invalidation
        for pb in pending_breaks[:]:
            candles_since = i - pb["break_idx"]
            if candles_since > max_fakeout_wait_htf:
                pending_breaks.remove(pb)
                continue

            # Update extreme
            if pb["type"] == "bear":
                pb["extreme"] = min(pb["extreme"], current_low)
                # Check if price returned inside channel
                if current_close > pb["channel"].support:
                    fakeout_signals.append(
                        FakeoutSignal(
                            htf_idx=i,
                            type="bear",
                            channel=pb["channel"],
                            extreme=pb["extreme"],
                        )
                    )
                    pending_breaks.remove(pb)
            else:  # bull
                pb["extreme"] = max(pb["extreme"], current_high)
                if current_close < pb["channel"].resistance:
                    fakeout_signals.append(
                        FakeoutSignal(
                            htf_idx=i,
                            type="bull",
                            channel=pb["channel"],
                            extreme=pb["extreme"],
                        )
                    )
                    pending_breaks.remove(pb)

        # Check for new breakouts on confirmed channels
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            # Bear breakout
            if current_close < channel.support * 0.997:
                # Check not already tracking this breakout
                already_tracking = any(
                    pb["channel"].support == channel.support
                    and pb["channel"].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append(
                        {
                            "type": "bear",
                            "break_idx": i,
                            "channel": Channel(
                                support=channel.support,
                                support_idx=channel.support_idx,
                                resistance=channel.resistance,
                                resistance_idx=channel.resistance_idx,
                                lowest_low=channel.lowest_low,
                                highest_high=channel.highest_high,
                                support_touches=channel.support_touches,
                                resistance_touches=channel.resistance_touches,
                                confirmed=True,
                            ),
                            "extreme": current_low,
                        }
                    )
            # Bull breakout
            elif current_close > channel.resistance * 1.003:
                already_tracking = any(
                    pb["channel"].support == channel.support
                    and pb["channel"].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append(
                        {
                            "type": "bull",
                            "break_idx": i,
                            "channel": Channel(
                                support=channel.support,
                                support_idx=channel.support_idx,
                                resistance=channel.resistance,
                                resistance_idx=channel.resistance_idx,
                                lowest_low=channel.lowest_low,
                                highest_high=channel.highest_high,
                                support_touches=channel.support_touches,
                                resistance_touches=channel.resistance_touches,
                                confirmed=True,
                            ),
                            "extreme": current_high,
                        }
                    )

        # Update existing channels
        keys_to_remove = []
        for key, channel in active_channels.items():
            # Remove if price broke through significantly
            if (
                current_close < channel.lowest_low * 0.96
                or current_close > channel.highest_high * 1.04
            ):
                keys_to_remove.append(key)
                continue

            # Update with new swing points (evolving logic)
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

            # Check confirmation
            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            # Check width still valid
            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Find best confirmed channel for this HTF index
        # Collect all valid candidates with scores
        candidates = []

        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue

            # Price should be inside channel
            if (
                current_close < channel.support * 0.98
                or current_close > channel.resistance * 1.02
            ):
                continue

            # Score by total touches
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        if candidates:
            # Find max score
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]

            # Apply tiebreaker
            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            elif tiebreaker == "narrow":
                # Select narrowest channel
                best_channel = min(top_candidates, key=lambda c: c[1])[2]
            else:  # 'first' - original behavior
                best_channel = top_candidates[0][2]

            htf_channel_map[i] = best_channel

    confirmed_count = len(set(id(c) for c in htf_channel_map.values()))
    print(f"  HTF Confirmed Channels: {confirmed_count}")
    print(f"  HTF Fakeout Signals: {len(fakeout_signals)}")

    return htf_channel_map, fakeout_signals


def htf_to_ltf_idx(htf_idx: int, htf_tf: str, ltf_tf: str) -> int:
    """Convert HTF index to LTF index."""
    tf_mins = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
    ratio = tf_mins[htf_tf] / tf_mins[ltf_tf]
    return int(htf_idx * ratio)


def collect_mtf_setups(
    htf_candles: pd.DataFrame,
    ltf_candles: pd.DataFrame,
    htf_tf: str = "1h",
    ltf_tf: str = "15m",
    touch_threshold: float = 0.003,
    sl_buffer_pct: float = 0.0008,
    quiet: bool = False,
    tiebreaker: str = "first",
) -> List[dict]:
    """Collect setups using MTF analysis."""

    # Build HTF channels and fakeout signals
    htf_channel_map, htf_fakeout_signals = build_htf_channels(
        htf_candles, tiebreaker=tiebreaker
    )

    # LTF data
    ltf_highs = ltf_candles["high"].values
    ltf_lows = ltf_candles["low"].values
    ltf_closes = ltf_candles["close"].values
    ltf_opens = ltf_candles["open"].values
    ltf_volumes = ltf_candles["volume"].values
    ltf_deltas = ltf_candles["delta"].values

    setups = []
    traded_entries = set()

    # Build LTF index to HTF channel mapping
    tf_mins = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    # Build HTF fakeout signal map: htf_idx -> FakeoutSignal
    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"MTF: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        # Get HTF channel for current LTF candle
        # IMPORTANT: Use htf_idx - 1 to avoid lookahead bias
        # We can only use channel info from CLOSED HTF candles
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

        # FAKEOUT DISABLED - avg PnL was -0.10% after fixing lookahead bias
        # Only using BOUNCE entries which have +1.60% avg PnL

        # Check for bounce entries
        trade_key = (
            round(channel.support),
            round(channel.resistance),
            "bounce",
            i // 20,
        )
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch
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
                setup = simulate_trade(
                    ltf_candles,
                    i,
                    "LONG",
                    entry_price,
                    sl_price,
                    tp1_price,
                    tp2_price,
                    channel,
                    None,
                    0,
                    "BOUNCE",
                    ltf_volumes,
                    ltf_deltas,
                    avg_volume,
                    avg_delta,
                    cvd_recent,
                    ltf_opens,
                    ltf_closes,
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

        # BOUNCE: Resistance touch
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
                setup = simulate_trade(
                    ltf_candles,
                    i,
                    "SHORT",
                    entry_price,
                    sl_price,
                    tp1_price,
                    tp2_price,
                    channel,
                    None,
                    0,
                    "BOUNCE",
                    ltf_volumes,
                    ltf_deltas,
                    avg_volume,
                    avg_delta,
                    cvd_recent,
                    ltf_opens,
                    ltf_closes,
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

    return setups


def simulate_trade(
    candles,
    idx,
    trade_type,
    entry_price,
    sl_price,
    tp1_price,
    tp2_price,
    channel,
    fakeout_extreme,
    candles_to_reclaim,
    setup_type,
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
    fakeout_depth = 0
    if fakeout_extreme:
        if trade_type == "LONG":
            fakeout_depth = (channel.support - fakeout_extreme) / channel.support * 100
        else:
            fakeout_depth = (
                (fakeout_extreme - channel.resistance) / channel.resistance * 100
            )

    return {
        "idx": idx,
        "type": trade_type,
        "setup_type": setup_type,
        "entry": entry_price,
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "rr_ratio": rr_ratio,
        "pnl_pct": pnl_pct,
        "channel_width": width_pct,
        "total_touches": channel.support_touches + channel.resistance_touches,
        "fakeout_depth_pct": fakeout_depth,
        "candles_to_reclaim": candles_to_reclaim,
        "volume_at_entry": volumes[idx],
        "volume_ratio": volumes[idx] / avg_volume if avg_volume > 0 else 1,
        "delta_at_entry": deltas[idx],
        "delta_ratio": deltas[idx] / (abs(avg_delta) + 1),
        "cvd_recent": cvd_recent,
        "body_bullish": 1 if closes[idx] > opens[idx] else 0,
        "outcome": outcome,
    }


def run_backtest(df: pd.DataFrame, label: str = ""):
    """Run backtest on given dataframe."""
    if len(df) == 0:
        print(f"  {label}: No trades")
        return

    capital = 10000
    risk_pct = 0.015  # 1.5% risk per trade
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
    actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_pnl = df["pnl_pct"].mean() * 100

    print(f"\n  {label}:")
    print(f"    Trades: {len(df)}, Avg PnL: {avg_pnl:+.4f}%")
    print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd * 100:.1f}%")
    print(f"    Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
    print(f"    Final: ${capital:,.2f}")


def analyze_results(setups: List[dict], ltf_candles: pd.DataFrame):
    """Analyze results with IS/OOS split."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    # Get timestamps for each setup
    df["time"] = ltf_candles.index[df["idx"].values]
    df["year"] = pd.to_datetime(df["time"]).dt.year

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    bounce_df = df[df["setup_type"] == "BOUNCE"]
    fakeout_df = df[df["setup_type"] == "FAKEOUT"]
    print(f"\n  BOUNCE:  {len(bounce_df)}")
    print(f"  FAKEOUT: {len(fakeout_df)}")

    full_wins = len(df[df["outcome"] == 1])
    partial_wins = len(df[df["outcome"] == 0.5])
    losses = len(df[df["outcome"] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins / len(df) * 100:.1f}%)")
    print(
        f"  Partial wins (TP1):  {partial_wins} ({partial_wins / len(df) * 100:.1f}%)"
    )
    print(f"  Losses:              {losses} ({losses / len(df) * 100:.1f}%)")

    avg_pnl = df["pnl_pct"].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}%")
    print(f"  Avg R:R: {df['rr_ratio'].mean():.2f}")

    for stype in ["BOUNCE", "FAKEOUT"]:
        subset = df[df["setup_type"] == stype]
        if len(subset) > 0:
            avg = subset["pnl_pct"].mean() * 100
            wr = (subset["outcome"] >= 0.5).mean() * 100
            print(
                f"\n  {stype}: {len(subset)} trades, Avg PnL: {avg:+.4f}%, WR: {wr:.1f}%"
            )

    # Split by year for IS/OOS
    years = sorted(df["year"].unique())
    print(f"\n  Years in data: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # IS = 2024, OOS = 2025
    is_df = df[df["year"] == 2024]
    oos_df = df[df["year"] == 2025]

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS (1.5% risk, with fees)")
    print("=" * 60)

    # Full backtest
    run_backtest(df, "FULL (All Data)")

    # IS/OOS split
    run_backtest(is_df, "IN-SAMPLE (2024)")
    run_backtest(oos_df, "OUT-OF-SAMPLE (2025) ⭐")


def run_tiebreaker_comparison(
    htf_candles, ltf_candles, htf_tf, ltf_tf, tiebreaker, label
):
    """Run backtest with specific tiebreaker and return results."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    setups = collect_mtf_setups(
        htf_candles, ltf_candles, htf_tf, ltf_tf, quiet=True, tiebreaker=tiebreaker
    )

    if not setups:
        print("  No setups found!")
        return None

    df = pd.DataFrame(setups)
    df["time"] = ltf_candles.index[df["idx"].values]
    df["year"] = pd.to_datetime(df["time"]).dt.year

    # Calculate stats
    bounce_df = df[df["setup_type"] == "BOUNCE"]
    fakeout_df = df[df["setup_type"] == "FAKEOUT"]

    bounce_wr = (bounce_df["outcome"] >= 0.5).mean() * 100 if len(bounce_df) > 0 else 0
    fakeout_wr = (
        (fakeout_df["outcome"] >= 0.5).mean() * 100 if len(fakeout_df) > 0 else 0
    )
    total_wr = (df["outcome"] >= 0.5).mean() * 100

    avg_pnl = df["pnl_pct"].mean() * 100
    bounce_avg = bounce_df["pnl_pct"].mean() * 100 if len(bounce_df) > 0 else 0
    fakeout_avg = fakeout_df["pnl_pct"].mean() * 100 if len(fakeout_df) > 0 else 0

    print(f"\n  Total: {len(df)} trades, Avg PnL: {avg_pnl:+.4f}%")
    print(
        f"  BOUNCE:  {len(bounce_df)} trades, WR: {bounce_wr:.1f}%, Avg: {bounce_avg:+.4f}%"
    )
    print(
        f"  FAKEOUT: {len(fakeout_df)} trades, WR: {fakeout_wr:.1f}%, Avg: {fakeout_avg:+.4f}%"
    )
    print(f"  Overall WR: {total_wr:.1f}%")

    # Run backtest
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

    print(f"\n  Backtest (1.5% risk, fees):")
    print(f"    Return: {total_return:+.1f}%")
    print(f"    Max DD: {max_dd * 100:.1f}%")
    print(f"    Final: ${capital:,.2f}")

    return {
        "total_trades": len(df),
        "bounce_trades": len(bounce_df),
        "fakeout_trades": len(fakeout_df),
        "bounce_wr": bounce_wr,
        "fakeout_wr": fakeout_wr,
        "total_wr": total_wr,
        "return": total_return,
        "max_dd": max_dd * 100,
        "final_capital": capital,
    }


def main():
    htf = "1h"
    ltf = "15m"

    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   Tiebreaker Comparison: FIRST vs NARROW                              ║
║   원본 ml_channel_proper_mtf.py 로직 유지                              ║
║   같은 score일 때 tiebreaker만 다르게 적용                              ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index("time")

    # Filter to 2024 only for fair comparison with original
    htf_candles = htf_candles[htf_candles.index.year == 2024]
    print(f"  Loaded {len(htf_candles):,} candles (2024 only)")

    print(f"\nLoading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index("time")

    # Filter to 2024 only
    ltf_candles = ltf_candles[ltf_candles.index.year == 2024]
    print(f"  Loaded {len(ltf_candles):,} candles (2024 only)")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}")

    # Run both tiebreakers
    results_first = run_tiebreaker_comparison(
        htf_candles, ltf_candles, htf, ltf, "first", "FIRST (기존 방식)"
    )
    results_narrow = run_tiebreaker_comparison(
        htf_candles, ltf_candles, htf, ltf, "narrow", "NARROW (좁은 채널 우선)"
    )

    # Comparison summary
    print("\n" + "=" * 70)
    print("  비교 요약")
    print("=" * 70)

    if results_first and results_narrow:
        print(f"\n{'항목':<20} {'FIRST':>15} {'NARROW':>15} {'차이':>15}")
        print("-" * 65)
        print(
            f"{'총 매매':<20} {results_first['total_trades']:>15} {results_narrow['total_trades']:>15} {results_narrow['total_trades'] - results_first['total_trades']:>+15}"
        )
        print(
            f"{'BOUNCE':<20} {results_first['bounce_trades']:>15} {results_narrow['bounce_trades']:>15}"
        )
        print(
            f"{'FAKEOUT':<20} {results_first['fakeout_trades']:>15} {results_narrow['fakeout_trades']:>15}"
        )
        print(
            f"{'BOUNCE WR':<20} {results_first['bounce_wr']:>14.1f}% {results_narrow['bounce_wr']:>14.1f}% {results_narrow['bounce_wr'] - results_first['bounce_wr']:>+14.1f}%"
        )
        print(
            f"{'FAKEOUT WR':<20} {results_first['fakeout_wr']:>14.1f}% {results_narrow['fakeout_wr']:>14.1f}% {results_narrow['fakeout_wr'] - results_first['fakeout_wr']:>+14.1f}%"
        )
        print(
            f"{'전체 WR':<20} {results_first['total_wr']:>14.1f}% {results_narrow['total_wr']:>14.1f}% {results_narrow['total_wr'] - results_first['total_wr']:>+14.1f}%"
        )
        print(
            f"{'수익률':<20} {results_first['return']:>+14.1f}% {results_narrow['return']:>+14.1f}% {results_narrow['return'] - results_first['return']:>+14.1f}%"
        )
        print(
            f"{'최대 DD':<20} {results_first['max_dd']:>14.1f}% {results_narrow['max_dd']:>14.1f}%"
        )
        print(
            f"{'최종 자본':<20} ${results_first['final_capital']:>13,.0f} ${results_narrow['final_capital']:>13,.0f}"
        )

        print("\n" + "=" * 70)
        print("  결론")
        print("=" * 70)

        if results_narrow["return"] > results_first["return"]:
            print(f"\n  ✅ NARROW 방식이 더 좋음!")
            print(
                f"     수익률: {results_narrow['return']:+.1f}% vs {results_first['return']:+.1f}%"
            )
            print(
                f"     승률: {results_narrow['total_wr']:.1f}% vs {results_first['total_wr']:.1f}%"
            )
        elif results_narrow["return"] < results_first["return"]:
            print(f"\n  ✅ FIRST 방식이 더 좋음!")
            print(
                f"     수익률: {results_first['return']:+.1f}% vs {results_narrow['return']:+.1f}%"
            )
            print(
                f"     승률: {results_first['total_wr']:.1f}% vs {results_narrow['total_wr']:.1f}%"
            )
        else:
            print(f"\n  ⚖️ 두 방식이 동일한 결과!")


if __name__ == "__main__":
    main()
