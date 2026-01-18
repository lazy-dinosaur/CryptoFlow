#!/usr/bin/env python3
"""
Hybrid Strategy - Market Regime Detection + Strategy Switching

Market Regime Detection:
- ADX < 25: Ranging → Channel Strategy
- ADX >= 25: Trending → Order Block Strategy

Combines the best of both worlds:
- Channel strategy for sideways markets
- Order Block strategy for trending markets
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


# =============================================================================
# Market Regime Detection
# =============================================================================

def calculate_adx(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX (Average Directional Index) for trend strength."""
    high = candles['high']
    low = candles['low']
    close = candles['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx


def detect_market_regime(candles: pd.DataFrame, adx_threshold: float = 25) -> pd.Series:
    """
    Detect market regime: 'trending' or 'ranging'

    Returns Series with regime for each candle.
    """
    adx = calculate_adx(candles)
    regime = pd.Series(index=candles.index, dtype=str)
    regime[adx >= adx_threshold] = 'trending'
    regime[adx < adx_threshold] = 'ranging'
    regime = regime.fillna('ranging')  # Default to ranging
    return regime, adx


# =============================================================================
# Channel Strategy (for Ranging Markets)
# =============================================================================

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


# =============================================================================
# Order Block Strategy (for Trending Markets)
# =============================================================================

@dataclass
class OrderBlock:
    idx: int
    type: str  # 'demand' or 'supply'
    top: float
    bottom: float
    volume: float
    delta: float
    move_size: float
    tested: bool = False


def find_order_blocks(candles: pd.DataFrame,
                      min_move_pct: float = 0.02,
                      lookback: int = 5) -> List[OrderBlock]:
    """Find order blocks in the data."""
    order_blocks = []

    closes = candles['close'].values
    opens = candles['open'].values
    highs = candles['high'].values
    lows = candles['low'].values
    volumes = candles['volume'].values
    deltas = candles['delta'].values

    avg_volumes = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for i in range(lookback, len(candles) - lookback):
        # Bullish move → Demand Zone
        future_high = highs[i+1:i+lookback+1].max()
        move_up = (future_high - closes[i]) / closes[i]

        if move_up >= min_move_pct:
            for j in range(i, max(i-lookback, 0), -1):
                is_bearish = closes[j] < opens[j]
                has_volume = volumes[j] >= avg_volumes[j]

                if is_bearish and has_volume:
                    order_blocks.append(OrderBlock(
                        idx=j,
                        type='demand',
                        top=max(opens[j], closes[j]),
                        bottom=min(opens[j], closes[j]),
                        volume=volumes[j],
                        delta=deltas[j],
                        move_size=move_up
                    ))
                    break

        # Bearish move → Supply Zone
        future_low = lows[i+1:i+lookback+1].min()
        move_down = (closes[i] - future_low) / closes[i]

        if move_down >= min_move_pct:
            for j in range(i, max(i-lookback, 0), -1):
                is_bullish = closes[j] > opens[j]
                has_volume = volumes[j] >= avg_volumes[j]

                if is_bullish and has_volume:
                    order_blocks.append(OrderBlock(
                        idx=j,
                        type='supply',
                        top=max(opens[j], closes[j]),
                        bottom=min(opens[j], closes[j]),
                        volume=volumes[j],
                        delta=deltas[j],
                        move_size=move_down
                    ))
                    break

    return order_blocks


# =============================================================================
# Hybrid Strategy
# =============================================================================

def run_hybrid_strategy(htf_candles: pd.DataFrame,
                        ltf_candles: pd.DataFrame,
                        htf_tf: str = "1h",
                        ltf_tf: str = "15m",
                        adx_threshold: float = 25) -> Tuple[List[dict], List[dict]]:
    """
    Run hybrid strategy with market regime switching.

    Returns:
        - channel_trades: Trades from channel strategy (ranging)
        - ob_trades: Trades from order block strategy (trending)
    """
    # Detect market regime on HTF
    regime, adx = detect_market_regime(htf_candles, adx_threshold)

    ranging_pct = (regime == 'ranging').mean() * 100
    trending_pct = (regime == 'trending').mean() * 100
    print(f"  Market Regime: {ranging_pct:.1f}% Ranging, {trending_pct:.1f}% Trending")
    print(f"  ADX Stats: min={adx.min():.1f}, max={adx.max():.1f}, mean={adx.mean():.1f}, median={adx.median():.1f}")

    # Setup LTF data
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    # Build channels for ranging periods
    swing_highs, swing_lows = find_swing_points(htf_candles)
    print(f"  HTF Swing Points: {len(swing_highs)} highs, {len(swing_lows)} lows")

    # Build order blocks for trending periods
    order_blocks = find_order_blocks(htf_candles)
    print(f"  HTF Order Blocks: {len(order_blocks)}")

    # Track active channels
    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

    # Build channel map
    closes = htf_candles['close'].values
    highs = htf_candles['high'].values
    lows = htf_candles['low'].values

    for i in range(len(htf_candles)):
        current_close = closes[i]

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

        if new_high:
            for sl in swing_lows[-30:]:
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if 0.008 <= width_pct <= 0.05:
                        key = (new_high.idx, sl.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=sl.price,
                                support_idx=sl.idx,
                                resistance=new_high.price,
                                resistance_idx=new_high.idx,
                                lowest_low=sl.price,
                                highest_high=new_high.price
                            )

        if new_low:
            for sh in swing_highs[-30:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if 0.008 <= width_pct <= 0.05:
                        key = (sh.idx, new_low.idx)
                        if key not in active_channels:
                            active_channels[key] = Channel(
                                support=new_low.price,
                                support_idx=new_low.idx,
                                resistance=sh.price,
                                resistance_idx=sh.idx,
                                lowest_low=new_low.price,
                                highest_high=sh.price
                            )

        # Update channels
        keys_to_remove = []
        for key, channel in active_channels.items():
            if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                keys_to_remove.append(key)
                continue

            touch_threshold = 0.004
            if new_low and new_low.price < channel.resistance:
                if abs(new_low.price - channel.support) / channel.support < touch_threshold:
                    channel.support_touches += 1

            if new_high and new_high.price > channel.support:
                if abs(new_high.price - channel.resistance) / channel.resistance < touch_threshold:
                    channel.resistance_touches += 1

            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > 0.05 or width_pct < 0.008:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Best channel
        best_channel = None
        best_score = -1
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            if score > best_score:
                best_score = score
                best_channel = channel

        if best_channel:
            htf_channel_map[i] = best_channel

    print(f"  HTF Confirmed Channels: {len(set(id(c) for c in htf_channel_map.values()))}")

    # Trading simulation
    channel_trades = []
    ob_trades = []
    traded_entries = set()
    last_entry_idx = -100
    min_entry_gap = 4

    # Active OBs
    active_obs = order_blocks.copy()

    iterator = tqdm(range(len(ltf_candles)), desc=f"Hybrid: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        if htf_idx >= len(regime):
            continue

        current_regime = regime.iloc[htf_idx]

        if i - last_entry_idx < min_entry_gap:
            continue

        # Historical features
        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]

        # =====================================================================
        # RANGING: Channel Strategy
        # =====================================================================
        if current_regime == 'ranging':
            channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias
            if channel:
                mid_price = (channel.resistance + channel.support) / 2
                touch_threshold = 0.003
                sl_buffer = 0.001

                trade_key = (round(channel.support), round(channel.resistance), 'channel', i // 20)
                if trade_key not in traded_entries:
                    # Support touch → LONG
                    if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
                        entry_price = current_close
                        sl_price = channel.support * (1 - sl_buffer)
                        tp1_price = mid_price
                        tp2_price = channel.resistance * 0.998

                        trade = simulate_trade(
                            ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price
                        )
                        if trade:
                            trade['strategy'] = 'CHANNEL'
                            trade['regime'] = 'ranging'
                            channel_trades.append(trade)
                            traded_entries.add(trade_key)
                            last_entry_idx = i

                    # Resistance touch → SHORT
                    elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
                        entry_price = current_close
                        sl_price = channel.resistance * (1 + sl_buffer)
                        tp1_price = mid_price
                        tp2_price = channel.support * 1.002

                        trade = simulate_trade(
                            ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price
                        )
                        if trade:
                            trade['strategy'] = 'CHANNEL'
                            trade['regime'] = 'ranging'
                            channel_trades.append(trade)
                            traded_entries.add(trade_key)
                            last_entry_idx = i

        # =====================================================================
        # TRENDING: Order Block Strategy
        # =====================================================================
        else:  # trending
            for ob in active_obs[:]:
                if ob.tested:
                    continue

                # Check if price reached OB
                ob_htf_idx = ob.idx
                ob_ltf_start = ob_htf_idx * tf_ratio

                # Skip if OB is in the future
                if ob_ltf_start > i:
                    continue

                # Skip if OB is too old
                if i - ob_ltf_start > 500:  # ~5 days for 15m
                    active_obs.remove(ob)
                    continue

                trade_key = (round(ob.bottom), round(ob.top), 'ob', ob.idx)
                if trade_key in traded_entries:
                    continue

                # Demand zone (LONG)
                if ob.type == 'demand':
                    if current_low <= ob.top and current_close > ob.bottom:
                        entry_price = current_close
                        sl_price = ob.bottom * 0.999
                        tp_price = entry_price + 2 * (entry_price - sl_price)  # 2:1 R:R

                        trade = simulate_trade(
                            ltf_candles, i, 'LONG', entry_price, sl_price,
                            (entry_price + tp_price) / 2, tp_price  # TP1 at 1:1, TP2 at 2:1
                        )
                        if trade:
                            trade['strategy'] = 'ORDERBLOCK'
                            trade['regime'] = 'trending'
                            ob_trades.append(trade)
                            traded_entries.add(trade_key)
                            last_entry_idx = i
                            ob.tested = True

                # Supply zone (SHORT)
                elif ob.type == 'supply':
                    if current_high >= ob.bottom and current_close < ob.top:
                        entry_price = current_close
                        sl_price = ob.top * 1.001
                        tp_price = entry_price - 2 * (sl_price - entry_price)

                        trade = simulate_trade(
                            ltf_candles, i, 'SHORT', entry_price, sl_price,
                            (entry_price + tp_price) / 2, tp_price
                        )
                        if trade:
                            trade['strategy'] = 'ORDERBLOCK'
                            trade['regime'] = 'trending'
                            ob_trades.append(trade)
                            traded_entries.add(trade_key)
                            last_entry_idx = i
                            ob.tested = True

    return channel_trades, ob_trades


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price):
    """Simulate a trade with partial TP."""
    highs = candles['high'].values
    lows = candles['low'].values

    risk = abs(entry_price - sl_price)
    if risk <= 0:
        return None

    reward1 = abs(tp1_price - entry_price)
    reward2 = abs(tp2_price - entry_price)

    pnl_pct = 0
    hit_tp1 = False
    current_sl = sl_price

    for j in range(idx + 1, min(idx + 100, len(candles))):
        if trade_type == 'LONG':
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
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    break
        else:  # SHORT
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
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    break

    return {
        'idx': idx,
        'type': trade_type,
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'pnl_pct': pnl_pct
    }


def analyze_results(channel_trades: List[dict], ob_trades: List[dict], ltf_candles: pd.DataFrame):
    """Analyze hybrid strategy results."""
    all_trades = channel_trades + ob_trades

    if not all_trades:
        print("  No trades!")
        return

    df = pd.DataFrame(all_trades)
    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    print(f"\n  Total Trades: {len(df)}")
    print(f"  CHANNEL trades: {len(channel_trades)} (Ranging)")
    print(f"  ORDERBLOCK trades: {len(ob_trades)} (Trending)")

    # By strategy
    for strategy in ['CHANNEL', 'ORDERBLOCK']:
        subset = df[df['strategy'] == strategy]
        if len(subset) > 0:
            wins = len(subset[subset['pnl_pct'] > 0])
            wr = wins / len(subset) * 100
            avg_pnl = subset['pnl_pct'].mean() * 100
            print(f"\n  {strategy}:")
            print(f"    Trades: {len(subset)}, WR: {wr:.1f}%, Avg PnL: {avg_pnl:+.4f}%")

    # Overall stats
    wins = len(df[df['pnl_pct'] > 0])
    losses = len(df[df['pnl_pct'] <= 0])
    win_rate = wins / len(df) * 100
    avg_pnl = df['pnl_pct'].mean() * 100

    print(f"\n  Overall Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Overall Avg PnL: {avg_pnl:+.4f}%")

    years = sorted(df['year'].unique())
    print(f"\n  Years: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # Backtest
    def run_backtest(subset: pd.DataFrame, label: str):
        if len(subset) == 0:
            print(f"\n  {label}: No trades")
            return

        # Sort by time
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

    # By strategy backtest
    print("\n" + "="*60)
    print("  BY STRATEGY")
    print("="*60)

    channel_df = df[df['strategy'] == 'CHANNEL']
    ob_df = df[df['strategy'] == 'ORDERBLOCK']

    run_backtest(channel_df, "CHANNEL (Ranging)")
    run_backtest(ob_df, "ORDERBLOCK (Trending)")


def main():
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"
    adx_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 25

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   HYBRID Strategy - Market Regime Switching               ║
║   HTF ({htf}) + LTF ({ltf}) | ADX Threshold: {adx_threshold}            ║
║                                                           ║
║   Ranging (ADX < {adx_threshold}): Channel Strategy              ║
║   Trending (ADX >= {adx_threshold}): Order Block Strategy         ║
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

    channel_trades, ob_trades = run_hybrid_strategy(
        htf_candles, ltf_candles, htf, ltf, adx_threshold
    )

    print("\n" + "="*60)
    print("  HYBRID RESULTS")
    print("="*60)

    analyze_results(channel_trades, ob_trades, ltf_candles)


if __name__ == "__main__":
    main()
