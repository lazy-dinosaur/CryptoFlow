#!/usr/bin/env python3
import sys

sys.path.insert(0, ".")

from ml_channel_tiebreaker_proper import run_backtest
from parse_data import load_candles
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print("=" * 60)
print(
    "BACKTEST: "
    + start_date.strftime("%Y-%m-%d")
    + " ~ "
    + end_date.strftime("%Y-%m-%d")
)
print("=" * 60)

df_1h = load_candles(
    "BINANCE:BTCUSDT",
    "1h",
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
)
df_15m = load_candles(
    "BINANCE:BTCUSDT",
    "15m",
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
)

print("1H candles: " + str(len(df_1h)))
print("15M candles: " + str(len(df_15m)))

result = run_backtest(
    df_1h, df_15m, htf_tf="1h", ltf_tf="15m", tiebreaker="narrow", quiet=False
)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print("Total Trades: " + str(result.total_trades))
print("Wins: " + str(result.wins) + " | Losses: " + str(result.losses))
print("Win Rate: " + str(round(result.win_rate, 1)) + "%")
print("Total PnL: " + str(round(result.total_pnl, 2)) + "%")
print("Avg PnL per Trade: " + str(round(result.avg_pnl, 2)) + "%")
print("Max Drawdown: " + str(round(result.max_drawdown, 2)) + "%")
print("Final Capital: $" + "{:,.2f}".format(result.final_capital))
print("=" * 60)
