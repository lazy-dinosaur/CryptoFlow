#!/usr/bin/env python3
"""Compare backtest results WITH vs WITHOUT cooldown"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from run_2month_backtest import load_candles_from_db, build_channels, simulate_trade
from shared.channel_builder import build_channels as _build_htf_map
from tqdm import tqdm

def run_with_cooldown(df_1h, df_15m, use_cooldown=True):
    htf_map = build_channels(df_1h)
    touch_th = 0.003
    sl_buffer = 0.0008
    signal_cooldown_ms = 20 * 15 * 60 * 1000
    traded = set()
    trades = []

    for i in range(len(df_15m)):
        htf_idx = i // 4
        ch = htf_map.get(htf_idx - 1)
        if not ch:
            continue
        if (ch.resistance - ch.support) / ch.support < 0.015:
            continue

        close = df_15m["close"].iloc[i]
        high = df_15m["high"].iloc[i]
        low = df_15m["low"].iloc[i]
        candle_time = int(df_15m["time"].iloc[i])

        if use_cooldown:
            key = (round(ch.support), round(ch.resistance), candle_time // signal_cooldown_ms)
            if key in traded:
                continue

        mid = (ch.resistance + ch.support) / 2

        if low <= ch.support * (1 + touch_th) and close > ch.support:
            entry = close
            sl = ch.support * (1 - sl_buffer)
            tp1 = mid
            tp2 = ch.resistance * 0.998
            if entry > sl and tp1 > entry:
                result = simulate_trade(df_15m, i, "LONG", entry, sl, tp1, tp2)
                trades.append(result)
                if use_cooldown:
                    traded.add(key)
        elif high >= ch.resistance * (1 - touch_th) and close < ch.resistance:
            entry = close
            sl = ch.resistance * (1 + sl_buffer)
            tp1 = mid
            tp2 = ch.support * 1.002
            if sl > entry and entry > tp1:
                result = simulate_trade(df_15m, i, "SHORT", entry, sl, tp1, tp2)
                trades.append(result)
                if use_cooldown:
                    traded.add(key)

    if not trades:
        return {"trades": 0, "wr": 0, "pnl": 0, "max_dd": 0, "final": 10000}

    capital = 10000
    risk_pct = 0.015
    max_leverage = 20
    fee_pct = 0.0004
    peak = capital
    max_dd = 0
    wins = 0
    losses = 0

    for t in trades:
        leverage = min(risk_pct / t["sl_dist"] if t["sl_dist"] > 0 else 1, max_leverage)
        position_size = capital * leverage
        gross_pnl = position_size * t["pnl_pct"]
        fees = position_size * fee_pct * 2
        net_pnl = gross_pnl - fees
        capital += net_pnl
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd
        if net_pnl > 0:
            wins += 1
        else:
            losses += 1

    total_pnl = (capital - 10000) / 10000 * 100
    win_rate = wins / len(trades) * 100

    exit_reasons = {}
    for t in trades:
        r = t["exit_reason"]
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        "trades": len(trades), "wins": wins, "losses": losses,
        "wr": round(win_rate, 1), "pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2), "final": round(capital, 2),
        "exits": exit_reasons
    }

if __name__ == "__main__":
    days = 60
    print("Loading data...")
    df_1h = load_candles_from_db(60, days)
    df_15m = load_candles_from_db(15, days)
    print(f"1H: {len(df_1h)}, 15M: {len(df_15m)}")
    print(f"Period: {df_1h['datetime'].iloc[0]} ~ {df_1h['datetime'].iloc[-1]}")

    print("\n" + "=" * 60)
    print("WITH COOLDOWN (5hr)")
    print("=" * 60)
    r1 = run_with_cooldown(df_1h, df_15m, use_cooldown=True)
    print(f"Trades: {r1['trades']}  W/L: {r1['wins']}/{r1['losses']}")
    print(f"Win Rate: {r1['wr']}%")
    print(f"PnL: {r1['pnl']}%  Final: ${r1['final']:,.2f}")
    print(f"Max DD: {r1['max_dd']}%")
    print(f"Exits: {r1['exits']}")

    print("\n" + "=" * 60)
    print("WITHOUT COOLDOWN")
    print("=" * 60)
    r2 = run_with_cooldown(df_1h, df_15m, use_cooldown=False)
    print(f"Trades: {r2['trades']}  W/L: {r2['wins']}/{r2['losses']}")
    print(f"Win Rate: {r2['wr']}%")
    print(f"PnL: {r2['pnl']}%  Final: ${r2['final']:,.2f}")
    print(f"Max DD: {r2['max_dd']}%")
    print(f"Exits: {r2['exits']}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Trades: {r1['trades']} -> {r2['trades']} ({r2['trades'] - r1['trades']:+d})")
    print(f"Win Rate: {r1['wr']}% -> {r2['wr']}% ({r2['wr'] - r1['wr']:+.1f}%p)")
    print(f"PnL: {r1['pnl']}% -> {r2['pnl']}% ({r2['pnl'] - r1['pnl']:+.2f}%)")
    print(f"Max DD: {r1['max_dd']}% -> {r2['max_dd']}%")
