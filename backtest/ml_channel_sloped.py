#!/usr/bin/env python3
"""
Sloped Channel Strategy (Ascending/Descending Channels)

Channel Types:
1. Ascending Channel: Higher Highs + Higher Lows (Uptrend)
   - LONG on support touch (with trend)
   - SHORT on resistance touch (counter-trend, optional)

2. Descending Channel: Lower Highs + Lower Lows (Downtrend)
   - SHORT on resistance touch (with trend)
   - LONG on support touch (counter-trend, optional)

Entry/Exit:
- Same as horizontal channel: Partial TP + Breakeven SL
- Prefer trading WITH the trend for higher win rate
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


@dataclass
class SwingPoint:
    idx: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class TrendLine:
    """A trendline connecting two swing points."""
    start_idx: int
    start_price: float
    end_idx: int
    end_price: float
    slope: float  # Price change per candle
    type: str  # 'support' or 'resistance'

    def price_at(self, idx: int) -> float:
        """Get the trendline price at a given index."""
        return self.start_price + self.slope * (idx - self.start_idx)


@dataclass
class SlopedChannel:
    """An ascending or descending channel."""
    support_line: TrendLine
    resistance_line: TrendLine
    direction: str  # 'ascending' or 'descending'
    start_idx: int
    touches: int = 2
    confirmed: bool = False

    def get_support_at(self, idx: int) -> float:
        return self.support_line.price_at(idx)

    def get_resistance_at(self, idx: int) -> float:
        return self.resistance_line.price_at(idx)

    def get_mid_at(self, idx: int) -> float:
        return (self.get_support_at(idx) + self.get_resistance_at(idx)) / 2

    def get_width_at(self, idx: int) -> float:
        return self.get_resistance_at(idx) - self.get_support_at(idx)


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find swing highs and lows."""
    highs = candles['high'].values
    lows = candles['low'].values

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
                swing_highs.append(SwingPoint(idx=potential_high_idx, price=potential_high_price, type='high'))

        if lows[i] < potential_low_price:
            potential_low_idx = i
            potential_low_price = lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append(SwingPoint(idx=potential_low_idx, price=potential_low_price, type='low'))

        if candles_since_high >= confirm_candles:
            potential_high_price = highs[i]
            potential_high_idx = i
            candles_since_high = 0

        if candles_since_low >= confirm_candles:
            potential_low_price = lows[i]
            potential_low_idx = i
            candles_since_low = 0

    return swing_highs, swing_lows


def detect_trend_structure(swing_highs: List[SwingPoint], swing_lows: List[SwingPoint],
                            lookback: int = 5) -> List[dict]:
    """
    Detect trend structure: HH/HL (uptrend) or LH/LL (downtrend).

    Returns list of trend segments with start/end indices and direction.
    """
    trends = []

    # Need at least 2 highs and 2 lows
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return trends

    # Combine and sort swing points by index
    all_swings = [(sh.idx, sh.price, 'high') for sh in swing_highs] + \
                 [(sl.idx, sl.price, 'low') for sl in swing_lows]
    all_swings.sort(key=lambda x: x[0])

    # Sliding window to detect trends
    for i in range(len(all_swings) - 3):
        window = all_swings[i:i+4]

        # Get highs and lows in window
        window_highs = [(idx, price) for idx, price, t in window if t == 'high']
        window_lows = [(idx, price) for idx, price, t in window if t == 'low']

        if len(window_highs) >= 2 and len(window_lows) >= 2:
            # Check for uptrend (HH + HL)
            hh = window_highs[-1][1] > window_highs[-2][1]  # Higher High
            hl = window_lows[-1][1] > window_lows[-2][1]    # Higher Low

            # Check for downtrend (LH + LL)
            lh = window_highs[-1][1] < window_highs[-2][1]  # Lower High
            ll = window_lows[-1][1] < window_lows[-2][1]    # Lower Low

            if hh and hl:
                trends.append({
                    'start_idx': window[0][0],
                    'end_idx': window[-1][0],
                    'direction': 'ascending',
                    'highs': window_highs,
                    'lows': window_lows
                })
            elif lh and ll:
                trends.append({
                    'start_idx': window[0][0],
                    'end_idx': window[-1][0],
                    'direction': 'descending',
                    'highs': window_highs,
                    'lows': window_lows
                })

    return trends


def build_sloped_channels(htf_candles: pd.DataFrame,
                          min_channel_width_pct: float = 0.01,
                          max_channel_width_pct: float = 0.08,
                          min_slope: float = 0.00001) -> Dict[int, SlopedChannel]:
    """
    Build sloped channels from trend structures.

    Returns dict mapping HTF candle index to active sloped channel.
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values

    # Detect trends
    trends = detect_trend_structure(swing_highs, swing_lows)
    print(f"  Trend Segments: {len(trends)}")

    # Build channels from trends
    channels: List[SlopedChannel] = []

    for trend in trends:
        highs = trend['highs']
        lows = trend['lows']
        direction = trend['direction']

        if len(highs) < 2 or len(lows) < 2:
            continue

        # Build resistance line from highs
        h1_idx, h1_price = highs[-2]
        h2_idx, h2_price = highs[-1]

        if h2_idx == h1_idx:
            continue

        resistance_slope = (h2_price - h1_price) / (h2_idx - h1_idx)

        # Build support line from lows
        l1_idx, l1_price = lows[-2]
        l2_idx, l2_price = lows[-1]

        if l2_idx == l1_idx:
            continue

        support_slope = (l2_price - l1_price) / (l2_idx - l1_idx)

        # Validate channel
        # Both lines should have similar slope direction
        if direction == 'ascending':
            if resistance_slope <= min_slope or support_slope <= min_slope:
                continue
        else:  # descending
            if resistance_slope >= -min_slope or support_slope >= -min_slope:
                continue

        # Create trendlines
        resistance_line = TrendLine(
            start_idx=h1_idx,
            start_price=h1_price,
            end_idx=h2_idx,
            end_price=h2_price,
            slope=resistance_slope,
            type='resistance'
        )

        support_line = TrendLine(
            start_idx=l1_idx,
            start_price=l1_price,
            end_idx=l2_idx,
            end_price=l2_price,
            slope=support_slope,
            type='support'
        )

        # Check channel width
        start_idx = min(h1_idx, l1_idx)
        mid_idx = (h2_idx + l2_idx) // 2
        width_at_mid = resistance_line.price_at(mid_idx) - support_line.price_at(mid_idx)
        width_pct = width_at_mid / support_line.price_at(mid_idx) if support_line.price_at(mid_idx) > 0 else 0

        if width_pct < min_channel_width_pct or width_pct > max_channel_width_pct:
            continue

        channel = SlopedChannel(
            support_line=support_line,
            resistance_line=resistance_line,
            direction=direction,
            start_idx=start_idx,
            touches=4,  # 2 highs + 2 lows
            confirmed=True
        )

        channels.append(channel)

    print(f"  Sloped Channels: {len(channels)} ({sum(1 for c in channels if c.direction == 'ascending')} ascending, {sum(1 for c in channels if c.direction == 'descending')} descending)")

    # Map channels to HTF indices
    htf_channel_map: Dict[int, SlopedChannel] = {}

    for i in range(len(htf_candles)):
        current_close = closes[i]

        # Find active channel at this index
        best_channel = None
        best_score = -1

        for channel in channels:
            # Channel must have started
            if i < channel.start_idx:
                continue

            # Channel shouldn't be too old (200 candles max)
            if i - channel.start_idx > 200:
                continue

            # Price should be inside channel
            support_price = channel.get_support_at(i)
            resistance_price = channel.get_resistance_at(i)

            # Validate prices are positive and channel is still valid
            if support_price <= 0 or resistance_price <= support_price:
                continue

            if current_close < support_price * 0.97 or current_close > resistance_price * 1.03:
                continue

            # Score by touches
            score = channel.touches
            if score > best_score:
                best_score = score
                best_channel = channel

        if best_channel:
            htf_channel_map[i] = best_channel

    return htf_channel_map


def collect_sloped_setups(htf_candles: pd.DataFrame,
                          ltf_candles: pd.DataFrame,
                          htf_tf: str = "1h",
                          ltf_tf: str = "15m",
                          touch_threshold: float = 0.004,
                          sl_buffer_pct: float = 0.001,
                          trade_with_trend_only: bool = False) -> List[dict]:
    """
    Collect setups from sloped channels.

    Args:
        trade_with_trend_only: If True, only trade in trend direction
            (LONG on ascending support, SHORT on descending resistance)
    """
    htf_channel_map = build_sloped_channels(htf_candles)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    setups = []
    traded_entries = set()

    iterator = tqdm(range(len(ltf_candles)), desc=f"Sloped: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx)

        if not channel:
            continue

        # Get dynamic S/R levels at current index
        support_price = channel.get_support_at(htf_idx)
        resistance_price = channel.get_resistance_at(htf_idx)
        mid_price = channel.get_mid_at(htf_idx)

        # Validate
        if support_price <= 0 or resistance_price <= support_price:
            continue

        # Historical features
        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 else 0

        trade_key = (round(support_price), round(resistance_price), channel.direction, i // 20)
        if trade_key in traded_entries:
            continue

        # =====================================================================
        # ASCENDING CHANNEL
        # =====================================================================
        if channel.direction == 'ascending':
            # Support touch → LONG (with trend)
            if current_low <= support_price * (1 + touch_threshold) and current_close > support_price:
                entry_price = current_close
                sl_price = support_price * (1 - sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = resistance_price * 0.998

                risk = entry_price - sl_price
                reward1 = tp1_price - entry_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price)
                    if setup:
                        setup['setup_type'] = 'TREND_BOUNCE'
                        setup['channel_direction'] = 'ascending'
                        setup['trade_direction'] = 'with_trend'
                        setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                        setups.append(setup)
                        traded_entries.add(trade_key)

            # Resistance touch → SHORT (counter-trend, optional)
            elif not trade_with_trend_only:
                if current_high >= resistance_price * (1 - touch_threshold) and current_close < resistance_price:
                    entry_price = current_close
                    sl_price = resistance_price * (1 + sl_buffer_pct)
                    tp1_price = mid_price
                    tp2_price = support_price * 1.002

                    risk = sl_price - entry_price
                    reward1 = entry_price - tp1_price

                    if risk > 0 and reward1 > 0:
                        setup = simulate_trade(ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price)
                        if setup:
                            setup['setup_type'] = 'COUNTER_BOUNCE'
                            setup['channel_direction'] = 'ascending'
                            setup['trade_direction'] = 'counter_trend'
                            setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                            setups.append(setup)
                            traded_entries.add(trade_key)

        # =====================================================================
        # DESCENDING CHANNEL
        # =====================================================================
        elif channel.direction == 'descending':
            # Resistance touch → SHORT (with trend)
            if current_high >= resistance_price * (1 - touch_threshold) and current_close < resistance_price:
                entry_price = current_close
                sl_price = resistance_price * (1 + sl_buffer_pct)
                tp1_price = mid_price
                tp2_price = support_price * 1.002

                risk = sl_price - entry_price
                reward1 = entry_price - tp1_price

                if risk > 0 and reward1 > 0:
                    setup = simulate_trade(ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price)
                    if setup:
                        setup['setup_type'] = 'TREND_BOUNCE'
                        setup['channel_direction'] = 'descending'
                        setup['trade_direction'] = 'with_trend'
                        setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                        setups.append(setup)
                        traded_entries.add(trade_key)

            # Support touch → LONG (counter-trend, optional)
            elif not trade_with_trend_only:
                if current_low <= support_price * (1 + touch_threshold) and current_close > support_price:
                    entry_price = current_close
                    sl_price = support_price * (1 - sl_buffer_pct)
                    tp1_price = mid_price
                    tp2_price = resistance_price * 0.998

                    risk = entry_price - sl_price
                    reward1 = tp1_price - entry_price

                    if risk > 0 and reward1 > 0:
                        setup = simulate_trade(ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price)
                        if setup:
                            setup['setup_type'] = 'COUNTER_BOUNCE'
                            setup['channel_direction'] = 'descending'
                            setup['trade_direction'] = 'counter_trend'
                            setup['volume_ratio'] = ltf_volumes[i] / avg_volume if avg_volume > 0 else 1
                            setups.append(setup)
                            traded_entries.add(trade_key)

    return setups


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price):
    """Simulate a trade with partial TP."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)
    rr_ratio = reward2 / risk if risk > 0 else 0

    outcome = 0
    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price

    for j in range(idx + 1, min(idx + 100, len(candles))):
        if trade_type == 'LONG':
            if not hit_tp1:
                if lows[j] <= current_sl:
                    pnl_pct = -risk / entry_price
                    outcome = 0
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
        else:  # SHORT
            if not hit_tp1:
                if highs[j] >= current_sl:
                    pnl_pct = -risk / entry_price
                    outcome = 0
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

    return {
        'idx': idx,
        'type': trade_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'rr_ratio': rr_ratio,
        'pnl_pct': pnl_pct,
        'outcome': outcome
    }


def analyze_results(setups: List[dict], ltf_candles: pd.DataFrame):
    """Analyze sloped channel results."""
    if not setups:
        print("  No setups found!")
        return

    df = pd.DataFrame(setups)

    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    print(f"\n  Total Setups: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # By channel direction
    asc = df[df['channel_direction'] == 'ascending']
    desc = df[df['channel_direction'] == 'descending']
    print(f"\n  Ascending Channel:  {len(asc)} trades")
    print(f"  Descending Channel: {len(desc)} trades")

    # By trade direction
    with_trend = df[df['trade_direction'] == 'with_trend']
    counter_trend = df[df['trade_direction'] == 'counter_trend']

    print(f"\n  With Trend:    {len(with_trend)} trades")
    print(f"  Counter Trend: {len(counter_trend)} trades")

    # Stats by trade direction
    for direction in ['with_trend', 'counter_trend']:
        subset = df[df['trade_direction'] == direction]
        if len(subset) > 0:
            avg = subset['pnl_pct'].mean() * 100
            wr = (subset['outcome'] >= 0.5).mean() * 100
            print(f"\n  {direction.upper().replace('_', ' ')}:")
            print(f"    Trades: {len(subset)}, Avg PnL: {avg:+.4f}%, WR: {wr:.1f}%")

    # Overall
    full_wins = len(df[df['outcome'] == 1])
    partial_wins = len(df[df['outcome'] == 0.5])
    losses = len(df[df['outcome'] == 0])

    print(f"\n  Full wins (TP2):     {full_wins} ({full_wins/len(df)*100:.1f}%)")
    print(f"  Partial wins (TP1):  {partial_wins} ({partial_wins/len(df)*100:.1f}%)")
    print(f"  Losses:              {losses} ({losses/len(df)*100:.1f}%)")

    avg_pnl = df['pnl_pct'].mean() * 100
    print(f"\n  Avg PnL per trade: {avg_pnl:+.4f}%")

    years = sorted(df['year'].unique())
    print(f"\n  Years: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # Backtest
    def run_backtest(subset: pd.DataFrame, label: str):
        if len(subset) == 0:
            print(f"\n  {label}: No trades")
            return

        subset = subset.sort_values('idx')

        capital = 10000
        risk_pct = 0.01
        max_leverage = 15
        fee_pct = 0.0004

        peak = capital
        max_dd = 0
        wins = 0
        losses = 0

        for _, trade in subset.iterrows():
            sl_dist = abs(trade['entry'] - trade['sl']) / trade['entry']
            if sl_dist <= 0:
                continue

            leverage = min(risk_pct / sl_dist, max_leverage)
            position_value = capital * leverage

            gross_pnl = position_value * trade['pnl_pct']
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
                print(f"    *** BANKRUPT ***")
                break

        total_return = (capital - 10000) / 10000 * 100
        actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        avg_pnl = subset['pnl_pct'].mean() * 100

        print(f"\n  {label}:")
        print(f"    Trades: {len(subset)}, Avg PnL: {avg_pnl:+.4f}%")
        print(f"    Return: {total_return:+.1f}%, Max DD: {max_dd*100:.1f}%")
        print(f"    Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
        print(f"    Final: ${capital:,.2f}")

    print("\n" + "="*60)
    print("  BACKTEST (1% risk, with fees)")
    print("="*60)

    run_backtest(df, "FULL (All Data)")

    is_df = df[df['year'] == 2024]
    oos_df = df[df['year'] == 2025]

    run_backtest(is_df, "IN-SAMPLE (2024)")
    run_backtest(oos_df, "OUT-OF-SAMPLE (2025)")

    # By trade direction
    print("\n" + "="*60)
    print("  BY TRADE DIRECTION")
    print("="*60)

    run_backtest(with_trend, "WITH TREND")
    run_backtest(counter_trend, "COUNTER TREND")


def main():
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    trend_only = "--trend-only" in sys.argv

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   SLOPED Channel Strategy (Ascending/Descending)          ║
║   HTF ({htf}) channels + LTF ({ltf}) entries                   ║
║                                                           ║
║   Ascending:  LONG on support (with trend)                ║
║   Descending: SHORT on resistance (with trend)            ║
║                                                           ║
║   Trade with trend only: {str(trend_only):<5}                          ║
╚═══════════════════════════════════════════════════════════╝
""")

    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(htf_candles):,} candles\n")

    print(f"Loading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(ltf_candles):,} candles")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}\n")

    setups = collect_sloped_setups(
        htf_candles, ltf_candles, htf, ltf,
        trade_with_trend_only=trend_only
    )

    print("\n" + "="*60)
    print("  SLOPED CHANNEL RESULTS")
    print("="*60)

    analyze_results(setups, ltf_candles)


if __name__ == "__main__":
    main()
