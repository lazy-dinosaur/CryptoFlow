"""
Shared channel detection logic used by both backtest and paper trading.

Builds an htf_map: Dict[int, Channel] where each key is a 1H candle index
and the value is the best confirmed channel at that point in time.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Channel:
    support: float
    resistance: float
    support_idx: int = 0
    resistance_idx: int = 0
    lowest_low: float = 0
    highest_high: float = 0
    support_touches: int = 1
    resistance_touches: int = 1
    confirmed: bool = False


MAX_CHANNEL_WIDTH = 0.05
MIN_CHANNEL_WIDTH = 0.015
TOUCH_THRESHOLD = 0.004


def find_swing_points(
    highs, lows, confirm_candles: int = 3
) -> Tuple[List[dict], List[dict]]:
    swing_highs, swing_lows = [], []

    pot_high_idx, pot_high_price, since_high = 0, highs[0], 0
    pot_low_idx, pot_low_price, since_low = 0, lows[0], 0

    for i in range(1, len(highs)):
        if highs[i] > pot_high_price:
            pot_high_idx, pot_high_price, since_high = i, highs[i], 0
        else:
            since_high += 1
            if since_high == confirm_candles:
                swing_highs.append({"idx": pot_high_idx, "price": pot_high_price})

        if lows[i] < pot_low_price:
            pot_low_idx, pot_low_price, since_low = i, lows[i], 0
        else:
            since_low += 1
            if since_low == confirm_candles:
                swing_lows.append({"idx": pot_low_idx, "price": pot_low_price})

        if since_high >= confirm_candles:
            pot_high_price, pot_high_idx, since_high = highs[i], i, 0

        if since_low >= confirm_candles:
            pot_low_price, pot_low_idx, since_low = lows[i], i, 0

    return swing_highs, swing_lows


def build_channels(highs, lows, closes) -> Dict[int, Channel]:
    """
    Build htf_map by iterating through all candles sequentially.
    Returns dict mapping candle index -> best confirmed channel at that index.

    Args:
        highs: array of high prices
        lows: array of low prices
        closes: array of close prices
    """
    swing_highs, swing_lows = find_swing_points(highs, lows)

    active: Dict[tuple, Channel] = {}
    htf_map: Dict[int, Channel] = {}

    for idx in range(len(closes)):
        close = closes[idx]
        new_high = next((sh for sh in swing_highs if sh["idx"] + 3 == idx), None)
        new_low = next((sl for sl in swing_lows if sl["idx"] + 3 == idx), None)
        valid_lows = [sl for sl in swing_lows if sl["idx"] + 3 <= idx]
        valid_highs = [sh for sh in swing_highs if sh["idx"] + 3 <= idx]

        if new_high:
            for sl in valid_lows[-30:]:
                if sl["idx"] < new_high["idx"] - 100:
                    continue
                if new_high["price"] > sl["price"]:
                    w = (new_high["price"] - sl["price"]) / sl["price"]
                    if MIN_CHANNEL_WIDTH <= w <= MAX_CHANNEL_WIDTH:
                        key = (new_high["idx"], sl["idx"])
                        if key not in active:
                            active[key] = Channel(
                                support=sl["price"],
                                resistance=new_high["price"],
                                support_idx=sl["idx"],
                                resistance_idx=new_high["idx"],
                                lowest_low=sl["price"],
                                highest_high=new_high["price"],
                            )

        if new_low:
            for sh in valid_highs[-30:]:
                if sh["idx"] < new_low["idx"] - 100:
                    continue
                if sh["price"] > new_low["price"]:
                    w = (sh["price"] - new_low["price"]) / new_low["price"]
                    if MIN_CHANNEL_WIDTH <= w <= MAX_CHANNEL_WIDTH:
                        key = (sh["idx"], new_low["idx"])
                        if key not in active:
                            active[key] = Channel(
                                support=new_low["price"],
                                resistance=sh["price"],
                                support_idx=new_low["idx"],
                                resistance_idx=sh["idx"],
                                lowest_low=new_low["price"],
                                highest_high=sh["price"],
                            )

        to_remove = []
        for key, ch in active.items():
            if close < ch.lowest_low * 0.96 or close > ch.highest_high * 1.04:
                to_remove.append(key)
                continue

            if new_low and new_low["price"] < ch.resistance:
                if new_low["price"] < ch.lowest_low:
                    ch.lowest_low = ch.support = new_low["price"]
                    ch.support_touches = 1
                elif ch.lowest_low < new_low["price"] < ch.support:
                    ch.support = new_low["price"]
                    ch.support_touches += 1
                elif abs(new_low["price"] - ch.support) / ch.support < TOUCH_THRESHOLD:
                    ch.support_touches += 1

            if new_high and new_high["price"] > ch.support:
                if new_high["price"] > ch.highest_high:
                    ch.highest_high = ch.resistance = new_high["price"]
                    ch.resistance_touches = 1
                elif ch.resistance < new_high["price"] < ch.highest_high:
                    ch.resistance = new_high["price"]
                    ch.resistance_touches += 1
                elif (
                    abs(new_high["price"] - ch.resistance) / ch.resistance
                    < TOUCH_THRESHOLD
                ):
                    ch.resistance_touches += 1

            if ch.support_touches >= 2 and ch.resistance_touches >= 2:
                ch.confirmed = True
            w = (ch.resistance - ch.support) / ch.support
            if not (MIN_CHANNEL_WIDTH <= w <= MAX_CHANNEL_WIDTH):
                to_remove.append(key)

        for key in to_remove:
            del active[key]

        candidates = [
            (
                ch.support_touches + ch.resistance_touches,
                (ch.resistance - ch.support) / ch.support,
                ch,
            )
            for ch in active.values()
            if ch.confirmed and ch.support * 0.98 <= close <= ch.resistance * 1.02
        ]

        if candidates:
            max_score = max(c[0] for c in candidates)
            top = [c for c in candidates if c[0] == max_score]
            htf_map[idx] = min(top, key=lambda c: c[1])[2]

    return htf_map
