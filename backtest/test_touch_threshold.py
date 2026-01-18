#!/usr/bin/env python3
"""
Touch Threshold 테스트 (NO ML)

다양한 touch_threshold 값으로 백테스트:
- 0.1%, 0.15%, 0.2%, 0.25%, 0.3%
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
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
    resistance: float
    support_touches: int
    resistance_touches: int
    confirmed: bool


def find_swing_points(candles: pd.DataFrame, confirm_candles: int = 3):
    highs = candles['high'].values
    lows = candles['low'].values
    
    swing_highs, swing_lows = [], []
    
    potential_high_idx, potential_high_price = 0, highs[0]
    candles_since_high = 0
    potential_low_idx, potential_low_price = 0, lows[0]
    candles_since_low = 0
    
    for i in range(1, len(candles)):
        if highs[i] > potential_high_price:
            potential_high_idx, potential_high_price = i, highs[i]
            candles_since_high = 0
        else:
            candles_since_high += 1
            if candles_since_high == confirm_candles:
                swing_highs.append(SwingPoint(idx=potential_high_idx, price=potential_high_price, type='high'))
        
        if lows[i] < potential_low_price:
            potential_low_idx, potential_low_price = i, lows[i]
            candles_since_low = 0
        else:
            candles_since_low += 1
            if candles_since_low == confirm_candles:
                swing_lows.append(SwingPoint(idx=potential_low_idx, price=potential_low_price, type='low'))
        
        if candles_since_high >= confirm_candles:
            potential_high_price, potential_high_idx = highs[i], i
            candles_since_high = 0
        if candles_since_low >= confirm_candles:
            potential_low_price, potential_low_idx = lows[i], i
            candles_since_low = 0
    
    return swing_highs, swing_lows


def build_channels(htf_candles: pd.DataFrame) -> Dict[int, Channel]:
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)
    
    closes = htf_candles['close'].values
    active_channels = {}
    htf_channel_map = {}
    
    for i in range(len(htf_candles)):
        current_close = closes[i]
        
        new_high = next((sh for sh in swing_highs if sh.idx + 3 == i), None)
        new_low = next((sl for sl in swing_lows if sl.idx + 3 == i), None)
        
        valid_lows = [sl for sl in swing_lows if sl.idx + 3 <= i]
        valid_highs = [sh for sh in swing_highs if sh.idx + 3 <= i]
        
        if new_high:
            for sl in valid_lows[-30:]:
                if sl.idx < new_high.idx - 100 or new_high.price <= sl.price:
                    continue
                width_pct = (new_high.price - sl.price) / sl.price
                if 0.008 <= width_pct <= 0.05:
                    key = (new_high.idx, sl.idx)
                    if key not in active_channels:
                        active_channels[key] = {
                            'support': sl.price, 'resistance': new_high.price,
                            'lowest': sl.price, 'highest': new_high.price,
                            's_touches': 1, 'r_touches': 1
                        }
        
        if new_low:
            for sh in valid_highs[-30:]:
                if sh.idx < new_low.idx - 100 or sh.price <= new_low.price:
                    continue
                width_pct = (sh.price - new_low.price) / new_low.price
                if 0.008 <= width_pct <= 0.05:
                    key = (sh.idx, new_low.idx)
                    if key not in active_channels:
                        active_channels[key] = {
                            'support': new_low.price, 'resistance': sh.price,
                            'lowest': new_low.price, 'highest': sh.price,
                            's_touches': 1, 'r_touches': 1
                        }
        
        # Update channels
        keys_to_remove = []
        for key, ch in active_channels.items():
            if current_close < ch['lowest'] * 0.96 or current_close > ch['highest'] * 1.04:
                keys_to_remove.append(key)
                continue
            
            if new_low and new_low.price < ch['resistance']:
                if abs(new_low.price - ch['support']) / ch['support'] < 0.004:
                    ch['s_touches'] += 1
            
            if new_high and new_high.price > ch['support']:
                if abs(new_high.price - ch['resistance']) / ch['resistance'] < 0.004:
                    ch['r_touches'] += 1
            
            width_pct = (ch['resistance'] - ch['support']) / ch['support']
            if width_pct > 0.05 or width_pct < 0.008:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del active_channels[key]
        
        # Select best channel
        candidates = []
        for key, ch in active_channels.items():
            confirmed = ch['s_touches'] >= 2 and ch['r_touches'] >= 2
            if not confirmed:
                continue
            if current_close < ch['support'] * 0.98 or current_close > ch['resistance'] * 1.02:
                continue
            score = ch['s_touches'] + ch['r_touches']
            candidates.append((score, Channel(
                support=ch['support'], resistance=ch['resistance'],
                support_touches=ch['s_touches'], resistance_touches=ch['r_touches'],
                confirmed=True
            )))
        
        if candidates:
            htf_channel_map[i] = max(candidates, key=lambda c: c[0])[1]
    
    return htf_channel_map


def simulate_trade(highs, lows, idx, direction, entry, sl, tp1, tp2):
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'loss', (sl - entry) / entry
            if highs[j] >= tp1:
                # Check TP2 after TP1
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:  # BE hit
                        return 'partial', 0.5 * (tp1 - entry) / entry
                    if highs[k] >= tp2:
                        return 'full', 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry
                return 'partial', 0.5 * (tp1 - entry) / entry
        else:
            if highs[j] >= sl:
                return 'loss', (entry - sl) / entry
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return 'partial', 0.5 * (entry - tp1) / entry
                    if lows[k] <= tp2:
                        return 'full', 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry
                return 'partial', 0.5 * (entry - tp1) / entry
    return None, 0


def collect_trades(htf_candles, ltf_candles, channel_map, touch_threshold, sl_buffer=0.0008, tf_ratio=4):
    trades = []
    traded_keys = set()
    
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    
    for i in range(50, len(ltf_candles) - 150):
        htf_idx = i // tf_ratio
        channel = channel_map.get(htf_idx - 1)  # Avoid lookahead
        
        if not channel:
            continue
        
        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        mid = (channel.resistance + channel.support) / 2
        
        bounce_key = (round(channel.support), round(channel.resistance), i // 20)
        if bounce_key in traded_keys:
            continue
        
        # Support touch → LONG
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998
            
            # 실제 터치 거리 계산
            touch_distance = (low - channel.support) / channel.support
            
            if entry > sl and tp1 > entry:
                result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'LONG', entry, sl, tp1, tp2)
                if result:
                    trades.append({
                        'direction': 'LONG', 'entry': entry, 'sl': sl,
                        'result': result, 'pnl': pnl, 'touch_distance': touch_distance
                    })
                    traded_keys.add(bounce_key)
        
        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002
            
            touch_distance = (channel.resistance - high) / channel.resistance
            
            if sl > entry and entry > tp1:
                result, pnl = simulate_trade(ltf_highs, ltf_lows, i, 'SHORT', entry, sl, tp1, tp2)
                if result:
                    trades.append({
                        'direction': 'SHORT', 'entry': entry, 'sl': sl,
                        'result': result, 'pnl': pnl, 'touch_distance': touch_distance
                    })
                    traded_keys.add(bounce_key)
    
    return trades


def backtest(trades, label):
    if not trades:
        return {'trades': 0, 'wr': 0, 'avg_pnl': 0, 'ret': 0}
    
    capital = 10000
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004
    
    wins, losses = 0, 0
    
    for t in trades:
        sl_dist = abs(t['entry'] - t['sl']) / t['entry']
        if sl_dist <= 0:
            continue
        
        lev = min(risk_pct / sl_dist, max_lev)
        position = capital * lev
        
        pnl = position * t['pnl'] - position * fee_pct * 2
        capital += pnl
        capital = max(capital, 0)
        
        if t['result'] != 'loss':
            wins += 1
        else:
            losses += 1
    
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ret = (capital / 10000 - 1) * 100
    avg_pnl = np.mean([t['pnl'] for t in trades]) * 100
    avg_touch = np.mean([t['touch_distance'] for t in trades]) * 100
    
    return {
        'trades': total, 'wr': wr, 'avg_pnl': avg_pnl, 
        'ret': ret, 'avg_touch_dist': avg_touch
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   Touch Threshold 테스트 (NO ML)                                  ║
║   다양한 touch_threshold로 BOUNCE 전략 백테스트                    ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')
    
    # Split by year
    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]
    htf_2025 = htf_all[htf_all.index.year == 2025]
    ltf_2025 = ltf_all[ltf_all.index.year == 2025]
    
    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")
    print(f"  2025: HTF={len(htf_2025)}, LTF={len(ltf_2025)}")
    
    # Build channels once
    print("\nBuilding channels...")
    channel_map_2024 = build_channels(htf_2024)
    channel_map_2025 = build_channels(htf_2025)
    print(f"  2024 channels: {len(channel_map_2024)}")
    print(f"  2025 channels: {len(channel_map_2025)}")
    
    # Test different thresholds
    thresholds = [0.001, 0.0015, 0.002, 0.0025, 0.003]
    
    print("\n" + "="*90)
    print(f"  {'Threshold':<12} | {'2024':^38} | {'2025 (OOS)':^38}")
    print(f"  {'':12} | {'Trades':>8} {'WR':>8} {'AvgPnL':>10} {'Return':>10} | {'Trades':>8} {'WR':>8} {'AvgPnL':>10} {'Return':>10}")
    print("="*90)
    
    results = []
    for thresh in thresholds:
        # 2024
        trades_2024 = collect_trades(htf_2024, ltf_2024, channel_map_2024, thresh)
        r_2024 = backtest(trades_2024, f"2024 thresh={thresh}")
        
        # 2025
        trades_2025 = collect_trades(htf_2025, ltf_2025, channel_map_2025, thresh)
        r_2025 = backtest(trades_2025, f"2025 thresh={thresh}")
        
        print(f"  {thresh*100:.2f}%        | {r_2024['trades']:>8} {r_2024['wr']:>7.1f}% {r_2024['avg_pnl']:>+9.3f}% {r_2024['ret']:>+9.1f}% | {r_2025['trades']:>8} {r_2025['wr']:>7.1f}% {r_2025['avg_pnl']:>+9.3f}% {r_2025['ret']:>+9.1f}%")
        
        results.append({
            'threshold': thresh,
            '2024_trades': r_2024['trades'], '2024_wr': r_2024['wr'], '2024_avg_pnl': r_2024['avg_pnl'], '2024_ret': r_2024['ret'],
            '2025_trades': r_2025['trades'], '2025_wr': r_2025['wr'], '2025_avg_pnl': r_2025['avg_pnl'], '2025_ret': r_2025['ret']
        })
    
    print("="*90)
    
    # Find best threshold by 2024 avg_pnl
    best = max(results, key=lambda x: x['2024_avg_pnl'])
    print(f"\n  Best by 2024 Avg PnL: {best['threshold']*100:.2f}%")
    print(f"    2024: {best['2024_trades']} trades, {best['2024_wr']:.1f}% WR, {best['2024_avg_pnl']:+.3f}% avg")
    print(f"    2025: {best['2025_trades']} trades, {best['2025_wr']:.1f}% WR, {best['2025_avg_pnl']:+.3f}% avg")
    
    # Trade count vs quality tradeoff
    print("\n" + "="*90)
    print("  Trade Count vs Quality 분석")
    print("="*90)
    
    for r in results:
        trades_per_day_2024 = r['2024_trades'] / 365
        trades_per_day_2025 = r['2025_trades'] / (18)  # ~18 days in Jan 2025
        print(f"  {r['threshold']*100:.2f}%: 2024 {trades_per_day_2024:.1f}건/일, 2025 {trades_per_day_2025:.1f}건/일")


if __name__ == "__main__":
    main()
