#!/usr/bin/env python3
"""
같은 방향 제한 테스트
- 기존: 무제한 동시 포지션
- 새로운: LONG/SHORT 각 1개씩만 허용 (헷지만 가능)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles
from ml_channel_proper_mtf import build_htf_channels


def simulate_trade(highs, lows, idx, direction, entry, sl, tp1, tp2):
    """매매 시뮬레이션 - (결과, pnl, 청산인덱스) 반환."""
    for j in range(idx + 1, min(idx + 150, len(highs))):
        if direction == 'LONG':
            if lows[j] <= sl:
                return 'SL', (sl - entry) / entry, j
            if highs[j] >= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if lows[k] <= entry:
                        return 'TP1_BE', 0.5 * (tp1 - entry) / entry, k
                    if highs[k] >= tp2:
                        return 'TP1_TP2', 0.5 * (tp1 - entry) / entry + 0.5 * (tp2 - entry) / entry, k
                return 'TP1_BE', 0.5 * (tp1 - entry) / entry, j + 50
        else:
            if highs[j] >= sl:
                return 'SL', (entry - sl) / entry, j
            if lows[j] <= tp1:
                for k in range(j, min(j + 100, len(highs))):
                    if highs[k] >= entry:
                        return 'TP1_BE', 0.5 * (entry - tp1) / entry, k
                    if lows[k] <= tp2:
                        return 'TP1_TP2', 0.5 * (entry - tp1) / entry + 0.5 * (entry - tp2) / entry, k
                return 'TP1_BE', 0.5 * (entry - tp1) / entry, j + 50
    return None, 0, idx + 150


def collect_all_signals(htf_candles, ltf_candles, channels_dict, tf_ratio=4):
    """모든 시그널 수집 (필터링 없이)."""
    signals = []

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values

    sl_buffer = 0.0008
    touch_threshold = 0.003

    for i in range(100, len(ltf_candles) - 200):
        htf_idx = i // tf_ratio
        channel = channels_dict.get(htf_idx - 1)

        if not channel:
            continue

        close = ltf_closes[i]
        high = ltf_highs[i]
        low = ltf_lows[i]
        mid = (channel.resistance + channel.support) / 2

        # Support touch → LONG
        if low <= channel.support * (1 + touch_threshold) and close > channel.support:
            entry = close
            sl = channel.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = channel.resistance * 0.998

            if entry > sl and tp1 > entry:
                signals.append({
                    'idx': i,
                    'direction': 'LONG',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'channel_key': (round(channel.support), round(channel.resistance)),
                })

        # Resistance touch → SHORT
        elif high >= channel.resistance * (1 - touch_threshold) and close < channel.resistance:
            entry = close
            sl = channel.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = channel.support * 1.002

            if sl > entry and entry > tp1:
                signals.append({
                    'idx': i,
                    'direction': 'SHORT',
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'channel_key': (round(channel.support), round(channel.resistance)),
                })

    return signals


def backtest_with_limit(signals, ltf_candles, mode='unlimited'):
    """
    백테스트 실행
    mode: 'unlimited' | 'direction_limit' | 'single'
    """
    risk_pct = 0.015
    max_lev = 15
    fee_pct = 0.0004

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values

    # 활성 포지션 추적
    active_long = None   # (exit_idx, ...)
    active_short = None

    traded_keys = set()  # 같은 채널 중복 방지

    wins, losses = 0, 0
    trade_pnls = []
    skipped = 0

    for s in signals:
        # 채널 중복 체크
        bounce_key = (s['channel_key'][0], s['channel_key'][1], s['idx'] // 20)
        if bounce_key in traded_keys:
            continue

        # 활성 포지션 업데이트
        if active_long and s['idx'] >= active_long:
            active_long = None
        if active_short and s['idx'] >= active_short:
            active_short = None

        # 포지션 제한 체크
        if mode == 'single':
            if active_long or active_short:
                skipped += 1
                continue
        elif mode == 'direction_limit':
            if s['direction'] == 'LONG' and active_long:
                skipped += 1
                continue
            if s['direction'] == 'SHORT' and active_short:
                skipped += 1
                continue
        # unlimited: 제한 없음

        # 매매 시뮬레이션
        result, pnl, exit_idx = simulate_trade(
            ltf_highs, ltf_lows, s['idx'],
            s['direction'], s['entry'], s['sl'], s['tp1'], s['tp2']
        )

        if result is None:
            continue

        # 포지션 등록
        if s['direction'] == 'LONG':
            active_long = exit_idx
        else:
            active_short = exit_idx

        traded_keys.add(bounce_key)

        # 수익 계산
        sl_dist = abs(s['entry'] - s['sl']) / s['entry']
        lev = min(risk_pct / sl_dist, max_lev) if sl_dist > 0 else 1
        gross_return = pnl * lev
        net_return = gross_return - (fee_pct * 2 * lev)
        trade_pnls.append(net_return * 100)

        if net_return > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    avg_pnl = np.mean(trade_pnls) if trade_pnls else 0
    total_return = sum(trade_pnls)

    return {
        'trades': total,
        'skipped': skipped,
        'wins': wins,
        'losses': losses,
        'wr': wr,
        'avg_pnl': avg_pnl,
        'total_return': total_return,
    }


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   포지션 제한 테스트                                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Load data
    print("Loading data...")
    htf_all = load_candles("BTCUSDT", "1h").to_pandas().set_index('time')
    ltf_all = load_candles("BTCUSDT", "15m").to_pandas().set_index('time')

    print("Building channels...")
    channels_all, _ = build_htf_channels(htf_all)

    print("Collecting signals...")
    signals = collect_all_signals(htf_all, ltf_all, channels_all)
    print(f"  Total signals: {len(signals)}")

    # 테스트
    print("\n" + "="*70)
    print("  📊 포지션 제한별 결과 비교")
    print("="*70)

    modes = [
        ('unlimited', '무제한'),
        ('direction_limit', 'LONG/SHORT 각 1개'),
        ('single', '단일 포지션만'),
    ]

    print(f"\n  {'모드':<25} | {'매매':>6} | {'스킵':>6} | {'WR':>7} | {'Avg PnL':>9} | {'총수익':>10}")
    print("-"*70)

    for mode, label in modes:
        result = backtest_with_limit(signals, ltf_all, mode)
        print(f"  {label:<25} | {result['trades']:>6} | {result['skipped']:>6} | {result['wr']:>6.1f}% | {result['avg_pnl']:>+8.2f}% | {result['total_return']:>+9.1f}%")

    print("\n" + "="*70)
    print("  💡 분석")
    print("="*70)

    unlimited = backtest_with_limit(signals, ltf_all, 'unlimited')
    limited = backtest_with_limit(signals, ltf_all, 'direction_limit')
    single = backtest_with_limit(signals, ltf_all, 'single')

    print(f"""
  무제한 vs LONG/SHORT 각 1개:
  - 매매 수: {unlimited['trades']} → {limited['trades']} ({limited['trades'] - unlimited['trades']:+d})
  - 총수익: {unlimited['total_return']:+.1f}% → {limited['total_return']:+.1f}% ({limited['total_return'] - unlimited['total_return']:+.1f}%)
  - WR: {unlimited['wr']:.1f}% → {limited['wr']:.1f}%

  LONG/SHORT 각 1개 vs 단일 포지션:
  - 매매 수: {limited['trades']} → {single['trades']} ({single['trades'] - limited['trades']:+d})
  - 총수익: {limited['total_return']:+.1f}% → {single['total_return']:+.1f}% ({single['total_return'] - limited['total_return']:+.1f}%)
""")


if __name__ == "__main__":
    main()
