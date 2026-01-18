#!/usr/bin/env python3
"""
터치 거리 분석 - 채널 폭 대비 비율

질문: 채널 폭이 1%일 때 적절한 터치 거리는?
분석: 실제 바운스 매매의 터치 거리 분포 확인
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
from ml_channel_proper_mtf import build_htf_channels


def collect_bounce_data(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """모든 바운스 시도의 터치 거리와 결과 수집."""
    data = []
    
    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    
    sl_buffer = 0.0008
    
    for i in range(50, len(ltf_candles) - 150):
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx - 1)
        
        if not channel:
            continue
        
        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        
        channel_width = (channel.resistance - channel.support) / channel.support
        mid = (channel.resistance + channel.support) / 2
        
        # Support 근처 (0.5% 이내)
        if low <= channel.support * 1.005 and close > channel.support:
            touch_distance = (low - channel.support) / channel.support
            touch_ratio = touch_distance / channel_width  # 채널폭 대비 비율
            
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998
            
            if entry > sl and tp1 > entry:
                # Simulate outcome
                success = False
                for j in range(i+1, min(i+150, len(ltf_highs))):
                    if ltf_lows[j] <= sl:
                        break
                    if ltf_highs[j] >= tp1:
                        success = True
                        break
                
                data.append({
                    'direction': 'LONG',
                    'touch_distance_pct': touch_distance * 100,
                    'touch_ratio': touch_ratio,
                    'channel_width_pct': channel_width * 100,
                    'success': success
                })
        
        # Resistance 근처 (0.5% 이내)
        elif high >= channel.resistance * 0.995 and close < channel.resistance:
            touch_distance = (channel.resistance - high) / channel.resistance
            touch_ratio = touch_distance / channel_width
            
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002
            
            if sl > entry and entry > tp1:
                success = False
                for j in range(i+1, min(i+150, len(ltf_highs))):
                    if ltf_highs[j] >= sl:
                        break
                    if ltf_lows[j] <= tp1:
                        success = True
                        break
                
                data.append({
                    'direction': 'SHORT',
                    'touch_distance_pct': touch_distance * 100,
                    'touch_ratio': touch_ratio,
                    'channel_width_pct': channel_width * 100,
                    'success': success
                })
    
    return pd.DataFrame(data)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   터치 거리 분석 - 채널 폭 대비 비율                               ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')
    
    # Use 2024 for analysis
    htf_2024 = htf_all[htf_all.index.year == 2024]
    ltf_2024 = ltf_all[ltf_all.index.year == 2024]
    
    print(f"  2024: HTF={len(htf_2024)}, LTF={len(ltf_2024)}")
    
    # Build channels
    print("\nBuilding channels...")
    channels_dict, _ = build_htf_channels(htf_2024)
    print(f"  Channels: {len(channels_dict)}")
    
    # Collect bounce data
    print("\nCollecting bounce data...")
    df = collect_bounce_data(htf_2024, ltf_2024, channels_dict)
    print(f"  Total samples: {len(df)}")
    
    if len(df) == 0:
        print("No data collected!")
        return
    
    # 1. 채널 폭 분포
    print("\n" + "="*60)
    print("  1. 채널 폭 분포")
    print("="*60)
    print(f"  Mean:   {df['channel_width_pct'].mean():.2f}%")
    print(f"  Median: {df['channel_width_pct'].median():.2f}%")
    print(f"  Min:    {df['channel_width_pct'].min():.2f}%")
    print(f"  Max:    {df['channel_width_pct'].max():.2f}%")
    
    # 2. 터치 거리 분포 (절대값)
    print("\n" + "="*60)
    print("  2. 터치 거리 분포 (절대값, S/R에서의 거리)")
    print("="*60)
    print(f"  Mean:   {df['touch_distance_pct'].mean():.3f}%")
    print(f"  Median: {df['touch_distance_pct'].median():.3f}%")
    print(f"  25%:    {df['touch_distance_pct'].quantile(0.25):.3f}%")
    print(f"  75%:    {df['touch_distance_pct'].quantile(0.75):.3f}%")
    
    # 3. 터치 거리 vs 승률 (절대값)
    print("\n" + "="*60)
    print("  3. 터치 거리 vs 승률 (절대값)")
    print("="*60)
    
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    df['distance_bin'] = pd.cut(df['touch_distance_pct'], bins)
    
    for bin_range in df.groupby('distance_bin', observed=True):
        bin_name, group = bin_range
        if len(group) > 0:
            wr = group['success'].mean() * 100
            print(f"  {bin_name}: {len(group):>5}건, WR: {wr:>5.1f}%")
    
    # 4. 터치 비율 vs 승률 (채널폭 대비)
    print("\n" + "="*60)
    print("  4. 터치 비율 vs 승률 (채널폭 대비)")
    print("="*60)
    
    ratio_bins = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
    df['ratio_bin'] = pd.cut(df['touch_ratio'], ratio_bins)
    
    for bin_range in df.groupby('ratio_bin', observed=True):
        bin_name, group = bin_range
        if len(group) > 0:
            wr = group['success'].mean() * 100
            print(f"  {bin_name}: {len(group):>5}건, WR: {wr:>5.1f}%")
    
    # 5. 최적 터치 조건 찾기
    print("\n" + "="*60)
    print("  5. 최적 터치 조건 탐색")
    print("="*60)
    
    print("\n  [절대값 기준]")
    for max_dist in [0.05, 0.1, 0.15, 0.2, 0.3]:
        subset = df[df['touch_distance_pct'] <= max_dist]
        if len(subset) > 10:
            wr = subset['success'].mean() * 100
            print(f"  touch <= {max_dist:.2f}%: {len(subset):>5}건, WR: {wr:>5.1f}%")
    
    print("\n  [채널폭 대비 비율 기준]")
    for max_ratio in [0.02, 0.05, 0.1, 0.15, 0.2]:
        subset = df[df['touch_ratio'] <= max_ratio]
        if len(subset) > 10:
            wr = subset['success'].mean() * 100
            avg_dist = subset['touch_distance_pct'].mean()
            print(f"  ratio <= {max_ratio*100:.0f}%: {len(subset):>5}건, WR: {wr:>5.1f}%, avg touch: {avg_dist:.3f}%")
    
    # 6. 결론
    print("\n" + "="*60)
    print("  6. 결론")
    print("="*60)
    
    # Best by WR with reasonable trade count
    best_absolute = None
    best_ratio = None
    
    for max_dist in [0.05, 0.1, 0.15, 0.2]:
        subset = df[df['touch_distance_pct'] <= max_dist]
        if len(subset) >= 50:
            wr = subset['success'].mean() * 100
            if best_absolute is None or wr > best_absolute[1]:
                best_absolute = (max_dist, wr, len(subset))
    
    for max_ratio in [0.02, 0.05, 0.1, 0.15]:
        subset = df[df['touch_ratio'] <= max_ratio]
        if len(subset) >= 50:
            wr = subset['success'].mean() * 100
            if best_ratio is None or wr > best_ratio[1]:
                best_ratio = (max_ratio, wr, len(subset))
    
    if best_absolute:
        print(f"  절대값 기준 최적: touch <= {best_absolute[0]:.2f}% (WR: {best_absolute[1]:.1f}%, {best_absolute[2]}건)")
    if best_ratio:
        print(f"  비율 기준 최적: ratio <= {best_ratio[0]*100:.0f}% (WR: {best_ratio[1]:.1f}%, {best_ratio[2]}건)")


if __name__ == "__main__":
    main()
