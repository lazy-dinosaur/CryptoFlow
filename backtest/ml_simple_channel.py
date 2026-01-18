#!/usr/bin/env python3
"""
Simple Channel Detection Backtest (3-Point + FAKEOUT Only)

채널 감지 로직:
1. 최근 피봇 3개로 채널 형성
   - 상승 파동: 저-고-저 패턴
   - 하락 파동: 고-저-고 패턴
2. 최소 채널 폭: 0.3%

Entry Types:
- FAKEOUT ONLY: 이탈 후 복귀

Exit Strategy:
- TP1: 채널 50% (50% 물량 청산)
- TP2: 반대 채널 경계 (나머지 50%)
- TP1 도달 시 SL을 Entry로 이동 (본절)
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles


# === CONFIGURATION ===
SWING_STRENGTH = 3           # 피봇 감지 강도 (양쪽 N개 캔들 비교)
MIN_CHANNEL_WIDTH = 0.003    # 3포인트 채널 최소 폭 (0.3%)
SL_BUFFER = 0.0005           # SL 버퍼 (0.05%)
TP1_RATIO = 0.5              # TP1 = 채널 50%
TP1_QTY_PCT = 0.5            # TP1에서 50% 청산


@dataclass
class SwingPoint:
    idx: int
    price: float
    is_high: bool  # True = Pivot High, False = Pivot Low


@dataclass
class SimpleChannel:
    support: float
    resistance: float
    width_pct: float
    num_points: int  # 2 or 3
    wave_type: str  # 'ascending' (저→고) or 'descending' (고→저)
    swings: List[SwingPoint]
    formed_idx: int


@dataclass
class Trade:
    idx: int
    direction: str  # 'LONG' or 'SHORT'
    setup_type: str  # 'BOUNCE' or 'FAKEOUT'
    entry: float
    sl: float
    tp1: float
    tp2: float
    channel: SimpleChannel
    pnl_pct: float = 0.0
    exit_reason: str = ''
    hit_tp1: bool = False


# === PIVOT DETECTION ===
def find_pivots(candles: pd.DataFrame, strength: int = 3) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find pivot highs and lows using both-side comparison.
    Similar to TradingView's ta.pivothigh/ta.pivotlow.
    """
    highs = candles['high'].values
    lows = candles['low'].values

    pivot_highs = []
    pivot_lows = []

    for i in range(strength, len(candles) - strength):
        # Check pivot high
        is_pivot_high = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_pivot_high = False
                break

        if is_pivot_high:
            pivot_highs.append(SwingPoint(idx=i, price=highs[i], is_high=True))

        # Check pivot low
        is_pivot_low = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_pivot_low = False
                break

        if is_pivot_low:
            pivot_lows.append(SwingPoint(idx=i, price=lows[i], is_high=False))

    return pivot_highs, pivot_lows


def get_recent_pivots_at_idx(pivot_highs: List[SwingPoint],
                              pivot_lows: List[SwingPoint],
                              current_idx: int,
                              strength: int) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Get confirmed pivots up to current index."""
    # Pivot is confirmed strength candles after the pivot point
    confirm_offset = strength

    confirmed_highs = [p for p in pivot_highs if p.idx + confirm_offset <= current_idx]
    confirmed_lows = [p for p in pivot_lows if p.idx + confirm_offset <= current_idx]

    return confirmed_highs, confirmed_lows


def build_simple_channel(pivot_highs: List[SwingPoint],
                          pivot_lows: List[SwingPoint],
                          current_idx: int,
                          strength: int,
                          min_width: float = 0.003) -> Optional[SimpleChannel]:
    """
    Build simple channel from most recent pivots.

    Step 1: 최근 Pivot High 1개 + Pivot Low 1개
    Step 2: 파동 순서 확인 (상승/하락)
    Step 3: 이격 체크 및 채널 생성
    """
    confirmed_highs, confirmed_lows = get_recent_pivots_at_idx(
        pivot_highs, pivot_lows, current_idx, strength
    )

    if not confirmed_highs or not confirmed_lows:
        return None

    # Get most recent high and low
    recent_high = confirmed_highs[-1]
    recent_low = confirmed_lows[-1]

    # Determine wave order
    if recent_low.idx < recent_high.idx:
        # 저→고 = 상승 파동 (ascending)
        wave_type = 'ascending'
    else:
        # 고→저 = 하락 파동 (descending)
        wave_type = 'descending'

    # Calculate gap
    gap_pct = (recent_high.price - recent_low.price) / recent_low.price

    # 3-point channel only
    if gap_pct < min_width:
        return None  # 너무 좁음

    if wave_type == 'ascending':
        # 상승 파동 (저→고): 이전 Low 찾기 → 저-고-저 패턴
        prev_lows = [p for p in confirmed_lows if p.idx < recent_low.idx]
        if not prev_lows:
            return None

        prev_low = prev_lows[-1]
        support = min(recent_low.price, prev_low.price)
        resistance = recent_high.price
        swings = [prev_low, recent_high, recent_low]
        num_points = 3
    else:
        # 하락 파동 (고→저): 이전 High 찾기 → 고-저-고 패턴
        prev_highs = [p for p in confirmed_highs if p.idx < recent_high.idx]
        if not prev_highs:
            return None

        prev_high = prev_highs[-1]
        support = recent_low.price
        resistance = max(recent_high.price, prev_high.price)
        swings = [prev_high, recent_low, recent_high]
        num_points = 3

    width_pct = (resistance - support) / support

    if width_pct < min_width:
        return None

    formed_idx = max(s.idx for s in swings)

    return SimpleChannel(
        support=support,
        resistance=resistance,
        width_pct=width_pct,
        num_points=num_points,
        wave_type=wave_type,
        swings=swings,
        formed_idx=formed_idx
    )


# === SIGNAL DETECTION ===
def detect_signals(ltf_candles: pd.DataFrame,
                   channel: SimpleChannel,
                   start_idx: int,
                   end_idx: int) -> List[Trade]:
    """
    Detect FAKEOUT signals only within LTF candles.
    """
    trades = []

    if channel is None:
        return trades

    support = channel.support
    resistance = channel.resistance
    ch_height = resistance - support

    # Track fakeout state
    broke_support = False
    broke_resistance = False
    fakeout_extreme_low = None
    fakeout_extreme_high = None

    highs = ltf_candles['high'].values
    lows = ltf_candles['low'].values
    closes = ltf_candles['close'].values

    for i in range(start_idx, min(end_idx, len(ltf_candles))):
        high = highs[i]
        low = lows[i]
        close = closes[i]

        # === FAKEOUT TRACKING ===
        # Broke below support?
        if close < support * 0.997:  # 0.3% margin
            broke_support = True
            fakeout_extreme_low = low if fakeout_extreme_low is None else min(fakeout_extreme_low, low)

        # Broke above resistance?
        if close > resistance * 1.003:
            broke_resistance = True
            fakeout_extreme_high = high if fakeout_extreme_high is None else max(fakeout_extreme_high, high)

        # === LONG FAKEOUT ===
        # Was below support, now closed back above
        if broke_support and close > support:
            sl = fakeout_extreme_low * (1 - SL_BUFFER) if fakeout_extreme_low else low * (1 - SL_BUFFER)
            tp1 = support + ch_height * TP1_RATIO
            tp2 = resistance

            trades.append(Trade(
                idx=i,
                direction='LONG',
                setup_type='FAKEOUT',
                entry=close,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                channel=channel
            ))
            broke_support = False
            fakeout_extreme_low = None

        # === SHORT FAKEOUT ===
        # Was above resistance, now closed back below
        if broke_resistance and close < resistance:
            sl = fakeout_extreme_high * (1 + SL_BUFFER) if fakeout_extreme_high else high * (1 + SL_BUFFER)
            tp1 = resistance - ch_height * TP1_RATIO
            tp2 = support

            trades.append(Trade(
                idx=i,
                direction='SHORT',
                setup_type='FAKEOUT',
                entry=close,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                channel=channel
            ))
            broke_resistance = False
            fakeout_extreme_high = None

    return trades


# === TRADE SIMULATION ===
def simulate_trade(trade: Trade, ltf_candles: pd.DataFrame) -> Trade:
    """
    Simulate trade with TP1/TP2/SL.
    TP1 도달 시 SL을 Entry로 이동 (본절)
    """
    highs = ltf_candles['high'].values
    lows = ltf_candles['low'].values

    direction = trade.direction
    entry = trade.entry
    sl = trade.sl
    tp1 = trade.tp1
    tp2 = trade.tp2

    hit_tp1 = False
    partial_pnl = 0.0
    remaining_qty = 1.0

    for i in range(trade.idx + 1, min(trade.idx + 200, len(ltf_candles))):
        high = highs[i]
        low = lows[i]

        if direction == 'LONG':
            # Check SL (after TP1: breakeven, before TP1: original SL)
            current_sl = entry if hit_tp1 else sl
            if low <= current_sl:
                if hit_tp1:
                    # Breakeven exit - only keep TP1 profit
                    trade.pnl_pct = partial_pnl
                    trade.exit_reason = 'BE'
                    trade.hit_tp1 = True
                else:
                    trade.pnl_pct = ((sl - entry) / entry)
                    trade.exit_reason = 'SL'
                return trade

            # Check TP1
            if not hit_tp1 and high >= tp1:
                partial_pnl = ((tp1 - entry) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True

            # Check TP2
            if high >= tp2:
                trade.pnl_pct = ((tp2 - entry) / entry) * remaining_qty + partial_pnl
                trade.exit_reason = 'TP2' if hit_tp1 else 'TP2_FULL'
                trade.hit_tp1 = hit_tp1
                return trade

        else:  # SHORT
            # Check SL (after TP1: breakeven, before TP1: original SL)
            current_sl = entry if hit_tp1 else sl
            if high >= current_sl:
                if hit_tp1:
                    # Breakeven exit - only keep TP1 profit
                    trade.pnl_pct = partial_pnl
                    trade.exit_reason = 'BE'
                    trade.hit_tp1 = True
                else:
                    trade.pnl_pct = ((entry - sl) / entry)
                    trade.exit_reason = 'SL'
                return trade

            # Check TP1
            if not hit_tp1 and low <= tp1:
                partial_pnl = ((entry - tp1) / entry) * TP1_QTY_PCT
                remaining_qty = 1 - TP1_QTY_PCT
                hit_tp1 = True

            # Check TP2
            if low <= tp2:
                trade.pnl_pct = ((entry - tp2) / entry) * remaining_qty + partial_pnl
                trade.exit_reason = 'TP2' if hit_tp1 else 'TP2_FULL'
                trade.hit_tp1 = hit_tp1
                return trade

    # Timeout - close at current price
    if hit_tp1:
        trade.pnl_pct = partial_pnl
        trade.exit_reason = 'TIMEOUT_TP1'
        trade.hit_tp1 = True
    else:
        last_close = ltf_candles['close'].iloc[min(trade.idx + 199, len(ltf_candles) - 1)]
        if direction == 'LONG':
            trade.pnl_pct = ((last_close - entry) / entry)
        else:
            trade.pnl_pct = ((entry - last_close) / entry)
        trade.exit_reason = 'TIMEOUT'

    return trade


# === MAIN BACKTEST ===
def run_backtest(htf: str = "1h", ltf: str = "15m"):
    """Run backtest with 3-point channel + FAKEOUT only."""
    print(f"""
{'='*60}
  Simple Channel Detection Backtest (3-Point + FAKEOUT Only)
{'='*60}
  HTF: {htf} (channel detection)
  LTF: {ltf} (entry)
  Swing Strength: {SWING_STRENGTH}
  Min Channel Width: {MIN_CHANNEL_WIDTH*100:.1f}%
  SL Buffer: {SL_BUFFER*100:.2f}%
  TP1: {TP1_RATIO*100:.0f}% of channel, close {TP1_QTY_PCT*100:.0f}%
  Strategy: 3-Point Channels + FAKEOUT entries only
{'='*60}
""")

    # Load data
    print(f"Loading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(htf_candles):,} candles")

    print(f"Loading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(ltf_candles):,} candles")
    print(f"  Date range: {ltf_candles.index[0]} ~ {ltf_candles.index[-1]}\n")

    # Find all pivots in HTF data
    print("Finding HTF pivots...")
    pivot_highs, pivot_lows = find_pivots(htf_candles, SWING_STRENGTH)
    print(f"  Pivot Highs: {len(pivot_highs)}")
    print(f"  Pivot Lows: {len(pivot_lows)}")

    # Calculate timeframe ratio
    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf] // tf_mins[ltf]

    # Process candles
    all_trades: List[Trade] = []
    channels_found = 0

    traded_channels = set()  # Track unique channels to avoid duplicate trades

    print("\nProcessing...")

    for htf_idx in tqdm(range(SWING_STRENGTH * 2, len(htf_candles) - 1), desc="HTF candles"):
        # Build channel at current HTF candle
        channel = build_simple_channel(
            pivot_highs, pivot_lows, htf_idx, SWING_STRENGTH,
            MIN_CHANNEL_WIDTH
        )

        if channel is None:
            continue

        # Check if this is a new channel
        channel_key = (round(channel.support, 2), round(channel.resistance, 2))
        if channel_key in traded_channels:
            continue

        channels_found += 1

        # Get LTF range for this HTF candle
        ltf_start = htf_idx * tf_ratio
        ltf_end = (htf_idx + 1) * tf_ratio

        # Detect signals in LTF
        signals = detect_signals(ltf_candles, channel, ltf_start, ltf_end)

        for signal in signals:
            # Simulate trade
            trade = simulate_trade(signal, ltf_candles)

            if trade.exit_reason:  # Trade completed
                all_trades.append(trade)
                traded_channels.add(channel_key)

    # === RESULTS ===
    print(f"\n{'='*60}")
    print("RESULTS (3-Point Channels + FAKEOUT Only)")
    print(f"{'='*60}")
    print(f"Channels Found: {channels_found}")
    print(f"Trades Executed: {len(all_trades)}")

    if not all_trades:
        print("No trades to analyze")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame([{
        'idx': t.idx,
        'direction': t.direction,
        'setup_type': t.setup_type,
        'entry': t.entry,
        'sl': t.sl,
        'tp1': t.tp1,
        'tp2': t.tp2,
        'pnl_pct': t.pnl_pct,
        'exit_reason': t.exit_reason,
        'hit_tp1': t.hit_tp1,
        'channel_width': t.channel.width_pct,
        'wave_type': t.channel.wave_type
    } for t in all_trades])

    # Basic stats
    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]

    total_pnl = df['pnl_pct'].sum() * 100
    avg_pnl = df['pnl_pct'].mean() * 100
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    avg_win = wins['pnl_pct'].mean() * 100 if len(wins) > 0 else 0
    avg_loss = losses['pnl_pct'].mean() * 100 if len(losses) > 0 else 0

    print(f"\nTotal PnL: {total_pnl:.2f}%")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)}/{len(df)})")
    print(f"Avg Win: {avg_win:.2f}%")
    print(f"Avg Loss: {avg_loss:.2f}%")
    print(f"Avg Trade: {avg_pnl:.4f}%")

    # By direction (LONG vs SHORT)
    print(f"\n--- By Direction ---")
    for direction in ['LONG', 'SHORT']:
        subset = df[df['direction'] == direction]
        if len(subset) > 0:
            pnl = subset['pnl_pct'].sum() * 100
            w = len(subset[subset['pnl_pct'] > 0])
            wr = w / len(subset) * 100
            print(f"{direction}: {len(subset)} trades, {pnl:.2f}% PnL, {wr:.1f}% WR")

    # By exit reason
    print(f"\n--- By Exit Reason ---")
    for reason in ['TP2', 'TP2_FULL', 'BE', 'SL', 'TIMEOUT', 'TIMEOUT_TP1']:
        subset = df[df['exit_reason'] == reason]
        if len(subset) > 0:
            pnl = subset['pnl_pct'].sum() * 100
            print(f"{reason}: {len(subset)} trades, {pnl:.2f}% PnL")

    # By wave type
    print(f"\n--- By Wave Type ---")
    for wave in ['ascending', 'descending']:
        subset = df[df['wave_type'] == wave]
        if len(subset) > 0:
            pnl = subset['pnl_pct'].sum() * 100
            w = len(subset[subset['pnl_pct'] > 0])
            wr = w / len(subset) * 100
            print(f"{wave}: {len(subset)} trades, {pnl:.2f}% PnL, {wr:.1f}% WR")

    # Get timestamps for year split
    df['time'] = ltf_candles.index[df['idx'].values]
    df['year'] = pd.to_datetime(df['time']).dt.year

    print(f"\n--- By Year ---")
    for year in sorted(df['year'].unique()):
        subset = df[df['year'] == year]
        if len(subset) > 0:
            pnl = subset['pnl_pct'].sum() * 100
            w = len(subset[subset['pnl_pct'] > 0])
            wr = w / len(subset) * 100
            print(f"{year}: {len(subset)} trades, {pnl:.2f}% PnL, {wr:.1f}% WR")

    # Backtest with risk management
    print(f"\n{'='*60}")
    print("BACKTEST WITH RISK MANAGEMENT")
    print(f"{'='*60}")

    capital = 10000
    risk_pct = 0.015  # 1.5% risk per trade
    max_leverage = 15
    fee_pct = 0.0004

    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for _, trade in df.iterrows():
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

    print(f"  Start: $10,000")
    print(f"  Final: ${capital:,.2f}")
    print(f"  Return: {total_return:+.1f}%")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Win Rate: {actual_wr:.1f}% ({wins}W / {losses}L)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Parse arguments: python script.py [htf] [ltf]
    htf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    ltf = sys.argv[2] if len(sys.argv) > 2 else "15m"

    run_backtest(htf, ltf)
