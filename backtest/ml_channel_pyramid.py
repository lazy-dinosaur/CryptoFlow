#!/usr/bin/env python3
"""
Channel Pyramid Strategy - Position Accumulation within Channel

Strategy:
1. Enter on each channel touch (accumulate positions)
2. TP1 (33%): Mid-channel → SL to breakeven
3. TP2 (33%): Channel edge → Hold remaining
4. TP3 (34%): Flag target (channel height extension after breakout)

Position Pyramiding:
- Add position on each valid channel touch
- Each position manages its own TP1/TP2
- All remaining positions exit at flag target on channel breakout
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
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
    # For flag target calculation
    entry_move_start: float = 0  # Price before channel formed
    entry_move_end: float = 0    # Price at channel entry


@dataclass
class Position:
    """Individual position within pyramid."""
    entry_idx: int
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    sl_price: float
    tp1_price: float  # Mid-channel
    tp2_price: float  # Channel edge
    tp3_price: float  # Flag target
    size_pct: float = 1.0  # Position size as fraction
    tp1_hit: bool = False
    tp2_hit: bool = False
    closed: bool = False
    pnl_pct: float = 0.0


@dataclass
class FakeoutSignal:
    htf_idx: int
    type: str
    channel: Channel
    extreme: float


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


def build_htf_channels(htf_candles: pd.DataFrame,
                       max_channel_width: float = 0.05,
                       min_channel_width: float = 0.008,
                       touch_threshold: float = 0.004) -> Tuple[Dict[int, Channel], List[FakeoutSignal]]:
    """Build evolving channels on HTF with entry move tracking for flag targets."""
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    highs = htf_candles['high'].values
    lows = htf_candles['low'].values
    closes = htf_candles['close'].values

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}
    fakeout_signals: List[FakeoutSignal] = []
    pending_breaks: List[dict] = []
    max_fakeout_wait_htf = 5

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

        # Create new channels
        if new_high:
            for sl in swing_lows[-30:]:
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (new_high.idx, sl.idx)
                        if key not in active_channels:
                            # Calculate entry move (flagpole)
                            lookback = max(0, sl.idx - 50)
                            entry_move_start = closes[lookback] if lookback < len(closes) else sl.price

                            active_channels[key] = Channel(
                                support=sl.price,
                                support_idx=sl.idx,
                                resistance=new_high.price,
                                resistance_idx=new_high.idx,
                                lowest_low=sl.price,
                                highest_high=new_high.price,
                                entry_move_start=entry_move_start,
                                entry_move_end=sl.price
                            )

        if new_low:
            for sh in swing_highs[-30:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if min_channel_width <= width_pct <= max_channel_width:
                        key = (sh.idx, new_low.idx)
                        if key not in active_channels:
                            lookback = max(0, new_low.idx - 50)
                            entry_move_start = closes[lookback] if lookback < len(closes) else sh.price

                            active_channels[key] = Channel(
                                support=new_low.price,
                                support_idx=new_low.idx,
                                resistance=sh.price,
                                resistance_idx=sh.idx,
                                lowest_low=new_low.price,
                                highest_high=sh.price,
                                entry_move_start=entry_move_start,
                                entry_move_end=sh.price
                            )

        # Process pending fakeouts
        for pb in pending_breaks[:]:
            candles_since = i - pb['break_idx']
            if candles_since > max_fakeout_wait_htf:
                pending_breaks.remove(pb)
                continue

            if pb['type'] == 'bear':
                pb['extreme'] = min(pb['extreme'], current_low)
                if current_close > pb['channel'].support:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bear',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)
            else:
                pb['extreme'] = max(pb['extreme'], current_high)
                if current_close < pb['channel'].resistance:
                    fakeout_signals.append(FakeoutSignal(
                        htf_idx=i,
                        type='bull',
                        channel=pb['channel'],
                        extreme=pb['extreme']
                    ))
                    pending_breaks.remove(pb)

        # Check for breakouts
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.997:
                already_tracking = any(
                    pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append({
                        'type': 'bear',
                        'break_idx': i,
                        'channel': Channel(
                            support=channel.support,
                            support_idx=channel.support_idx,
                            resistance=channel.resistance,
                            resistance_idx=channel.resistance_idx,
                            lowest_low=channel.lowest_low,
                            highest_high=channel.highest_high,
                            support_touches=channel.support_touches,
                            resistance_touches=channel.resistance_touches,
                            confirmed=True,
                            entry_move_start=channel.entry_move_start,
                            entry_move_end=channel.entry_move_end
                        ),
                        'extreme': current_low
                    })
            elif current_close > channel.resistance * 1.003:
                already_tracking = any(
                    pb['channel'].support == channel.support and pb['channel'].resistance == channel.resistance
                    for pb in pending_breaks
                )
                if not already_tracking:
                    pending_breaks.append({
                        'type': 'bull',
                        'break_idx': i,
                        'channel': Channel(
                            support=channel.support,
                            support_idx=channel.support_idx,
                            resistance=channel.resistance,
                            resistance_idx=channel.resistance_idx,
                            lowest_low=channel.lowest_low,
                            highest_high=channel.highest_high,
                            support_touches=channel.support_touches,
                            resistance_touches=channel.resistance_touches,
                            confirmed=True,
                            entry_move_start=channel.entry_move_start,
                            entry_move_end=channel.entry_move_end
                        ),
                        'extreme': current_high
                    })

        # Update channels
        keys_to_remove = []
        for key, channel in active_channels.items():
            if current_close < channel.lowest_low * 0.96 or current_close > channel.highest_high * 1.04:
                keys_to_remove.append(key)
                continue

            if new_low and new_low.price < channel.resistance:
                if new_low.price < channel.lowest_low:
                    channel.lowest_low = new_low.price
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches = 1
                elif new_low.price > channel.lowest_low and new_low.price < channel.support:
                    channel.support = new_low.price
                    channel.support_idx = new_low.idx
                    channel.support_touches += 1
                elif abs(new_low.price - channel.support) / channel.support < touch_threshold:
                    channel.support_touches += 1

            if new_high and new_high.price > channel.support:
                if new_high.price > channel.highest_high:
                    channel.highest_high = new_high.price
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches = 1
                elif new_high.price < channel.highest_high and new_high.price > channel.resistance:
                    channel.resistance = new_high.price
                    channel.resistance_idx = new_high.idx
                    channel.resistance_touches += 1
                elif abs(new_high.price - channel.resistance) / channel.resistance < touch_threshold:
                    channel.resistance_touches += 1

            if channel.support_touches >= 2 and channel.resistance_touches >= 2:
                channel.confirmed = True

            width_pct = (channel.resistance - channel.support) / channel.support
            if width_pct > max_channel_width or width_pct < min_channel_width:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del active_channels[key]

        # Find best channel
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

    confirmed_count = len(set(id(c) for c in htf_channel_map.values()))
    print(f"  HTF Confirmed Channels: {confirmed_count}")
    print(f"  HTF Fakeout Signals: {len(fakeout_signals)}")

    return htf_channel_map, fakeout_signals


def calculate_flag_target(channel: Channel, direction: str) -> float:
    """Calculate flag target based on entry move (flagpole)."""
    channel_height = channel.resistance - channel.support

    # Use channel height as minimum extension
    if direction == 'LONG':
        # Upside breakout target
        return channel.resistance + channel_height
    else:
        # Downside breakout target
        return channel.support - channel_height


def simulate_pyramid_strategy(htf_candles: pd.DataFrame,
                               ltf_candles: pd.DataFrame,
                               htf_channel_map: Dict[int, Channel],
                               htf_fakeout_signals: List[FakeoutSignal],
                               htf_tf: str = "1h",
                               ltf_tf: str = "15m",
                               touch_threshold: float = 0.003,
                               sl_buffer_pct: float = 0.001) -> List[dict]:
    """
    Simulate pyramid strategy with position accumulation.
    """
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    htf_fakeout_map = {fs.htf_idx: fs for fs in htf_fakeout_signals}

    # Track all trades (completed)
    completed_trades = []

    # Active positions (pyramided)
    active_positions: List[Position] = []

    # Track current channel for pyramiding
    current_channel: Optional[Channel] = None
    current_channel_key: Optional[tuple] = None
    last_entry_idx = -100  # Prevent too frequent entries

    # Minimum candles between entries
    min_entry_gap = 4  # 4 LTF candles = 1 hour for 15m

    iterator = tqdm(range(len(ltf_candles)), desc=f"Pyramid: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)  # Fix lookahead bias

        # Update active positions
        for pos in active_positions[:]:
            if pos.closed:
                continue

            if pos.direction == 'LONG':
                # Check SL
                if current_low <= pos.sl_price:
                    if pos.tp1_hit:
                        # SL at breakeven after TP1
                        pos.pnl_pct += 0  # No additional loss
                    else:
                        # Full loss
                        risk = pos.entry_price - pos.sl_price
                        pos.pnl_pct = -risk / pos.entry_price
                    pos.closed = True
                    completed_trades.append({
                        'idx': pos.entry_idx,
                        'exit_idx': i,
                        'type': 'LONG',
                        'entry': pos.entry_price,
                        'exit': pos.sl_price if not pos.tp1_hit else pos.entry_price,
                        'pnl_pct': pos.pnl_pct,
                        'exit_reason': 'SL_BE' if pos.tp1_hit else 'SL'
                    })
                    continue

                # Check TP1
                if not pos.tp1_hit and current_high >= pos.tp1_price:
                    reward1 = pos.tp1_price - pos.entry_price
                    pos.pnl_pct += 0.33 * (reward1 / pos.entry_price)
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price  # Move SL to breakeven

                # Check TP2
                if pos.tp1_hit and not pos.tp2_hit and current_high >= pos.tp2_price:
                    reward2 = pos.tp2_price - pos.entry_price
                    pos.pnl_pct += 0.33 * (reward2 / pos.entry_price)
                    pos.tp2_hit = True

                # Check TP3 (flag target)
                if pos.tp2_hit and current_high >= pos.tp3_price:
                    reward3 = pos.tp3_price - pos.entry_price
                    pos.pnl_pct += 0.34 * (reward3 / pos.entry_price)
                    pos.closed = True
                    completed_trades.append({
                        'idx': pos.entry_idx,
                        'exit_idx': i,
                        'type': 'LONG',
                        'entry': pos.entry_price,
                        'exit': pos.tp3_price,
                        'pnl_pct': pos.pnl_pct,
                        'exit_reason': 'TP3_FLAG'
                    })

            else:  # SHORT
                if current_high >= pos.sl_price:
                    if pos.tp1_hit:
                        pos.pnl_pct += 0
                    else:
                        risk = pos.sl_price - pos.entry_price
                        pos.pnl_pct = -risk / pos.entry_price
                    pos.closed = True
                    completed_trades.append({
                        'idx': pos.entry_idx,
                        'exit_idx': i,
                        'type': 'SHORT',
                        'entry': pos.entry_price,
                        'exit': pos.sl_price if not pos.tp1_hit else pos.entry_price,
                        'pnl_pct': pos.pnl_pct,
                        'exit_reason': 'SL_BE' if pos.tp1_hit else 'SL'
                    })
                    continue

                if not pos.tp1_hit and current_low <= pos.tp1_price:
                    reward1 = pos.entry_price - pos.tp1_price
                    pos.pnl_pct += 0.33 * (reward1 / pos.entry_price)
                    pos.tp1_hit = True
                    pos.sl_price = pos.entry_price

                if pos.tp1_hit and not pos.tp2_hit and current_low <= pos.tp2_price:
                    reward2 = pos.entry_price - pos.tp2_price
                    pos.pnl_pct += 0.33 * (reward2 / pos.entry_price)
                    pos.tp2_hit = True

                if pos.tp2_hit and current_low <= pos.tp3_price:
                    reward3 = pos.entry_price - pos.tp3_price
                    pos.pnl_pct += 0.34 * (reward3 / pos.entry_price)
                    pos.closed = True
                    completed_trades.append({
                        'idx': pos.entry_idx,
                        'exit_idx': i,
                        'type': 'SHORT',
                        'entry': pos.entry_price,
                        'exit': pos.tp3_price,
                        'pnl_pct': pos.pnl_pct,
                        'exit_reason': 'TP3_FLAG'
                    })

        # Remove closed positions
        active_positions = [p for p in active_positions if not p.closed]

        if not channel:
            continue

        # Check for new entry (pyramid)
        if i - last_entry_idx < min_entry_gap:
            continue

        channel_key = (round(channel.support), round(channel.resistance))
        mid_price = (channel.resistance + channel.support) / 2

        # BOUNCE entry at support
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998
            tp3_price = calculate_flag_target(channel, 'LONG')

            risk = entry_price - sl_price
            if risk > 0 and tp1_price > entry_price:
                pos = Position(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction='LONG',
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    tp3_price=tp3_price
                )
                active_positions.append(pos)
                last_entry_idx = i
                current_channel = channel
                current_channel_key = channel_key

        # BOUNCE entry at resistance
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002
            tp3_price = calculate_flag_target(channel, 'SHORT')

            risk = sl_price - entry_price
            if risk > 0 and tp1_price < entry_price:
                pos = Position(
                    entry_idx=i,
                    entry_price=entry_price,
                    direction='SHORT',
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    tp3_price=tp3_price
                )
                active_positions.append(pos)
                last_entry_idx = i
                current_channel = channel
                current_channel_key = channel_key

        # Fakeout entry
        fakeout_signal = htf_fakeout_map.get(htf_idx - 1)  # Fix lookahead bias
        if fakeout_signal and i % tf_ratio == 0:
            f_channel = fakeout_signal.channel
            f_mid = (f_channel.resistance + f_channel.support) / 2

            if fakeout_signal.type == 'bear':
                entry_price = current_close
                sl_price = fakeout_signal.extreme * (1 - sl_buffer_pct)
                tp1_price = f_mid
                tp2_price = f_channel.resistance * 0.998
                tp3_price = calculate_flag_target(f_channel, 'LONG')

                risk = entry_price - sl_price
                if risk > 0 and tp1_price > entry_price:
                    pos = Position(
                        entry_idx=i,
                        entry_price=entry_price,
                        direction='LONG',
                        sl_price=sl_price,
                        tp1_price=tp1_price,
                        tp2_price=tp2_price,
                        tp3_price=tp3_price
                    )
                    active_positions.append(pos)
                    last_entry_idx = i

            else:  # bull fakeout
                entry_price = current_close
                sl_price = fakeout_signal.extreme * (1 + sl_buffer_pct)
                tp1_price = f_mid
                tp2_price = f_channel.support * 1.002
                tp3_price = calculate_flag_target(f_channel, 'SHORT')

                risk = sl_price - entry_price
                if risk > 0 and tp1_price < entry_price:
                    pos = Position(
                        entry_idx=i,
                        entry_price=entry_price,
                        direction='SHORT',
                        sl_price=sl_price,
                        tp1_price=tp1_price,
                        tp2_price=tp2_price,
                        tp3_price=tp3_price
                    )
                    active_positions.append(pos)
                    last_entry_idx = i

    # Close remaining positions at last price
    for pos in active_positions:
        if not pos.closed:
            final_price = ltf_closes[-1]
            if pos.direction == 'LONG':
                remaining_pct = 0.34 if pos.tp2_hit else (0.67 if pos.tp1_hit else 1.0)
                pos.pnl_pct += remaining_pct * (final_price - pos.entry_price) / pos.entry_price
            else:
                remaining_pct = 0.34 if pos.tp2_hit else (0.67 if pos.tp1_hit else 1.0)
                pos.pnl_pct += remaining_pct * (pos.entry_price - final_price) / pos.entry_price

            completed_trades.append({
                'idx': pos.entry_idx,
                'exit_idx': len(ltf_candles) - 1,
                'type': pos.direction,
                'entry': pos.entry_price,
                'exit': final_price,
                'pnl_pct': pos.pnl_pct,
                'exit_reason': 'EOD'
            })

    return completed_trades


def analyze_results(trades: List[dict], ltf_candles: pd.DataFrame):
    """Analyze pyramid strategy results."""
    if not trades:
        print("  No trades!")
        return

    df = pd.DataFrame(trades)

    print(f"\n  Total Trades: {len(df)}")
    print(f"  LONG:  {len(df[df['type'] == 'LONG'])}")
    print(f"  SHORT: {len(df[df['type'] == 'SHORT'])}")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason in df['exit_reason'].unique():
        count = len(df[df['exit_reason'] == reason])
        print(f"    {reason}: {count} ({count/len(df)*100:.1f}%)")

    # PnL stats
    wins = len(df[df['pnl_pct'] > 0])
    losses = len(df[df['pnl_pct'] <= 0])
    win_rate = wins / len(df) * 100

    avg_pnl = df['pnl_pct'].mean() * 100
    avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() * 100 if wins > 0 else 0
    avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() * 100 if losses > 0 else 0

    print(f"\n  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg PnL: {avg_pnl:+.4f}%")
    print(f"  Avg Win: {avg_win:+.4f}%")
    print(f"  Avg Loss: {avg_loss:+.4f}%")

    # Get timestamps
    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    years = sorted(df['year'].unique())
    print(f"\n  Years: {years}")
    print(f"  Trades by year: {df.groupby('year').size().to_dict()}")

    # Backtest
    print("\n" + "="*60)
    print("  BACKTEST (1% risk per trade, with fees)")
    print("="*60)

    def run_backtest(subset: pd.DataFrame, label: str):
        if len(subset) == 0:
            print(f"\n  {label}: No trades")
            return

        capital = 10000
        risk_pct = 0.01
        max_leverage = 15
        fee_pct = 0.0004

        peak = capital
        max_dd = 0
        wins = 0
        losses = 0

        for _, trade in subset.iterrows():
            # Estimate SL distance from pnl (rough)
            sl_dist = 0.01  # Default 1%
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

    run_backtest(df, "FULL (All Data)")

    is_df = df[df['year'] == 2024]
    oos_df = df[df['year'] == 2025]

    run_backtest(is_df, "IN-SAMPLE (2024)")
    run_backtest(oos_df, "OUT-OF-SAMPLE (2025)")


def main():
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Channel PYRAMID Strategy                                ║
║   HTF ({htf}) channels + LTF ({ltf}) entries                   ║
║   TP1 (33%) + TP2 (33%) + TP3 FLAG (34%)                  ║
║   Position Accumulation within Channel                    ║
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

    htf_channel_map, htf_fakeout_signals = build_htf_channels(htf_candles)

    trades = simulate_pyramid_strategy(
        htf_candles, ltf_candles,
        htf_channel_map, htf_fakeout_signals,
        htf, ltf
    )

    print("\n" + "="*60)
    print("  PYRAMID RESULTS")
    print("="*60)

    analyze_results(trades, ltf_candles)


if __name__ == "__main__":
    main()
