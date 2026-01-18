#!/usr/bin/env python3
"""
ML Multi-Timeframe Bounce Strategy with 1m Volume/Delta Pattern Filtering

기반: ml_channel_tiebreaker_proper.py (NARROW tiebreaker)
추가: 1분봉 볼륨/델타 패턴 ML 필터링

HTF (1H): Channel detection with evolving S/R
LTF (15m): Entry execution
1m: ML-based volume/delta pattern analysis for entry filtering

Usage:
    python ml_mtf_bounce.py          # Baseline only
    python ml_mtf_bounce.py --ml     # Baseline + ML comparison
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

# Try to import ML features module
try:
    from ml_volume_delta_features import extract_1m_features, calculate_dynamic_sl, get_feature_names
    HAS_ML_FEATURES = True
except ImportError:
    HAS_ML_FEATURES = False
    print("[WARN] ml_volume_delta_features not found, ML features disabled")

# ============== Configuration ==============
MODELS_DIR = "models"
ML_ENTRY_THRESHOLD = 0.5


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
    """Find swing highs and lows on HTF."""
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
                       touch_threshold: float = 0.004,
                       tiebreaker: str = 'narrow') -> Dict[int, Channel]:
    """
    Build evolving channels on HTF.
    Returns dict mapping HTF candle index to active confirmed channel.
    """
    swing_highs, swing_lows = find_swing_points(htf_candles, confirm_candles=3)

    print(f"  HTF Swing Highs: {len(swing_highs)}")
    print(f"  HTF Swing Lows: {len(swing_lows)}")

    closes = htf_candles['close'].values

    active_channels: Dict[tuple, Channel] = {}
    htf_channel_map: Dict[int, Channel] = {}

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

        valid_swing_lows = [sl for sl in swing_lows if sl.idx + 3 <= i]
        valid_swing_highs = [sh for sh in swing_highs if sh.idx + 3 <= i]

        if new_high:
            for sl in valid_swing_lows[-30:]:
                if sl.idx < new_high.idx - 100:
                    continue
                if new_high.price > sl.price:
                    width_pct = (new_high.price - sl.price) / sl.price
                    if min_channel_width <= width_pct <= max_channel_width:
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
            for sh in valid_swing_highs[-30:]:
                if sh.idx < new_low.idx - 100:
                    continue
                if sh.price > new_low.price:
                    width_pct = (sh.price - new_low.price) / new_low.price
                    if min_channel_width <= width_pct <= max_channel_width:
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

        candidates = []
        for key, channel in active_channels.items():
            if not channel.confirmed:
                continue
            if current_close < channel.support * 0.98 or current_close > channel.resistance * 1.02:
                continue
            score = channel.support_touches + channel.resistance_touches
            width_pct = (channel.resistance - channel.support) / channel.support
            candidates.append((score, width_pct, channel))

        if candidates:
            max_score = max(c[0] for c in candidates)
            top_candidates = [c for c in candidates if c[0] == max_score]

            if len(top_candidates) == 1:
                best_channel = top_candidates[0][2]
            elif tiebreaker == 'narrow':
                best_channel = min(top_candidates, key=lambda c: c[1])[2]
            else:
                best_channel = top_candidates[0][2]

            htf_channel_map[i] = best_channel

    confirmed_count = len(set(id(c) for c in htf_channel_map.values()))
    print(f"  HTF Confirmed Channels: {confirmed_count}")

    return htf_channel_map


def simulate_trade(candles, idx, trade_type, entry_price, sl_price, tp1_price, tp2_price,
                   channel, volumes, deltas, avg_volume, avg_delta, cvd_recent, opens, closes):
    """Simulate trade with partial TP + breakeven."""
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

    for j in range(idx + 1, min(idx + 150, len(candles))):
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
                    outcome = 0.5
                    break
                if highs[j] >= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break
        else:
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
                    outcome = 0.5
                    break
                if lows[j] <= tp2_price:
                    pnl_pct += 0.5 * (reward2 / entry_price)
                    outcome = 1
                    break

    width_pct = (channel.resistance - channel.support) / channel.support

    return {
        'idx': idx,
        'type': trade_type,
        'setup_type': 'BOUNCE',
        'entry': entry_price,
        'sl': sl_price,
        'tp1': tp1_price,
        'tp2': tp2_price,
        'rr_ratio': rr_ratio,
        'pnl_pct': pnl_pct,
        'channel_width': width_pct,
        'total_touches': channel.support_touches + channel.resistance_touches,
        'volume_at_entry': volumes[idx],
        'volume_ratio': volumes[idx] / avg_volume if avg_volume > 0 else 1,
        'delta_at_entry': deltas[idx],
        'delta_ratio': deltas[idx] / (abs(avg_delta) + 1),
        'cvd_recent': cvd_recent,
        'body_bullish': 1 if closes[idx] > opens[idx] else 0,
        'outcome': outcome
    }


def collect_setups_baseline(htf_candles: pd.DataFrame,
                            ltf_candles: pd.DataFrame,
                            htf_tf: str = "1h",
                            ltf_tf: str = "15m",
                            touch_threshold: float = 0.003,
                            sl_buffer_pct: float = 0.0008,
                            quiet: bool = False,
                            tiebreaker: str = 'narrow') -> List[dict]:
    """
    Collect setups using MTF analysis (same as ml_channel_tiebreaker_proper.py).
    This is the BASELINE strategy.
    """
    htf_channel_map = build_htf_channels(htf_candles, tiebreaker=tiebreaker)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

    setups = []
    traded_entries = set()

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"Baseline: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)

        if not channel:
            continue

        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 and 'delta' in hist.columns else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 and 'delta' in hist.columns else 0

        mid_price = (channel.resistance + channel.support) / 2

        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        # BOUNCE: Support touch
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price

            if risk > 0 and reward1 > 0:
                setup = simulate_trade(
                    ltf_candles, i, 'LONG', entry_price, sl_price, tp1_price, tp2_price,
                    channel, ltf_volumes, ltf_deltas, avg_volume, avg_delta, cvd_recent, ltf_opens, ltf_closes
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

        # BOUNCE: Resistance touch
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

            if risk > 0 and reward1 > 0:
                setup = simulate_trade(
                    ltf_candles, i, 'SHORT', entry_price, sl_price, tp1_price, tp2_price,
                    channel, ltf_volumes, ltf_deltas, avg_volume, avg_delta, cvd_recent, ltf_opens, ltf_closes
                )
                if setup:
                    setups.append(setup)
                    traded_entries.add(trade_key)

    return setups


def load_ml_model():
    """Load trained ML reversal model."""
    model_path = os.path.join(MODELS_DIR, 'reversal_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'reversal_scaler.joblib')
    meta_path = os.path.join(MODELS_DIR, 'reversal_meta.joblib')

    if not os.path.exists(model_path):
        print(f"[ML] Model not found at {model_path}")
        return None, None, None, ML_ENTRY_THRESHOLD

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        threshold = meta.get('best_threshold', ML_ENTRY_THRESHOLD)
        feature_names = meta.get('feature_names', [])
        print(f"[ML] Model loaded: threshold={threshold:.2f}, features={len(feature_names)}")
        return model, scaler, feature_names, threshold
    except Exception as e:
        print(f"[ML] Failed to load model: {e}")
        return None, None, None, ML_ENTRY_THRESHOLD


def collect_setups_ml(htf_candles: pd.DataFrame,
                      ltf_candles: pd.DataFrame,
                      candles_1m: pd.DataFrame,
                      ml_model, ml_scaler, feature_names: List[str],
                      ml_threshold: float = ML_ENTRY_THRESHOLD,
                      htf_tf: str = "1h",
                      ltf_tf: str = "15m",
                      touch_threshold: float = 0.003,
                      sl_buffer_pct: float = 0.0008,
                      use_dynamic_sl: bool = False,
                      quiet: bool = False,
                      tiebreaker: str = 'narrow') -> List[dict]:
    """
    Collect setups with ML-based 1m volume/delta pattern filtering.
    """
    if not HAS_ML_FEATURES or ml_model is None:
        print("[ML] ML features not available, using baseline")
        return collect_setups_baseline(htf_candles, ltf_candles, htf_tf, ltf_tf,
                                       touch_threshold, sl_buffer_pct, quiet, tiebreaker)

    htf_channel_map = build_htf_channels(htf_candles, tiebreaker=tiebreaker)

    ltf_highs = ltf_candles['high'].values
    ltf_lows = ltf_candles['low'].values
    ltf_closes = ltf_candles['close'].values
    ltf_opens = ltf_candles['open'].values
    ltf_volumes = ltf_candles['volume'].values
    ltf_deltas = ltf_candles['delta'].values if 'delta' in ltf_candles.columns else np.zeros(len(ltf_candles))

    # Index 1m candles by time
    candles_1m_indexed = candles_1m.set_index('time') if 'time' in candles_1m.columns else candles_1m

    setups = []
    traded_entries = set()
    ml_filtered = 0
    ml_passed = 0

    tf_mins = {'5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}
    tf_ratio = tf_mins[htf_tf] // tf_mins[ltf_tf]
    ltf_minutes = tf_mins[ltf_tf]

    iterator = range(len(ltf_candles))
    if not quiet:
        iterator = tqdm(iterator, desc=f"ML Filter: {htf_tf}→{ltf_tf}")

    for i in iterator:
        current_close = ltf_closes[i]
        current_high = ltf_highs[i]
        current_low = ltf_lows[i]

        htf_idx = i // tf_ratio
        channel = htf_channel_map.get(htf_idx - 1)

        if not channel:
            continue

        hist_start = max(0, i - 20)
        hist = ltf_candles.iloc[hist_start:i]
        avg_volume = hist['volume'].mean() if len(hist) > 0 else ltf_volumes[i]
        avg_delta = hist['delta'].mean() if len(hist) > 0 and 'delta' in hist.columns else 0
        cvd_recent = hist['delta'].sum() if len(hist) > 0 and 'delta' in hist.columns else 0

        mid_price = (channel.resistance + channel.support) / 2

        trade_key = (round(channel.support), round(channel.resistance), 'bounce', i // 20)
        if trade_key in traded_entries:
            continue

        direction = None
        entry_price = None
        sl_price = None
        tp1_price = None
        tp2_price = None

        # BOUNCE: Support touch
        if current_low <= channel.support * (1 + touch_threshold) and current_close > channel.support:
            direction = 'LONG'
            entry_price = current_close
            sl_price = channel.support * (1 - sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.resistance * 0.998

        # BOUNCE: Resistance touch
        elif current_high >= channel.resistance * (1 - touch_threshold) and current_close < channel.resistance:
            direction = 'SHORT'
            entry_price = current_close
            sl_price = channel.resistance * (1 + sl_buffer_pct)
            tp1_price = mid_price
            tp2_price = channel.support * 1.002

        if direction is None:
            continue

        # Validate risk/reward
        if direction == 'LONG':
            risk = entry_price - sl_price
            reward1 = tp1_price - entry_price
        else:
            risk = sl_price - entry_price
            reward1 = entry_price - tp1_price

        if risk <= 0 or reward1 <= 0:
            continue

        # Get 1m window for ML features
        candle_time = ltf_candles.index[i]
        try:
            end_time = candle_time + pd.Timedelta(minutes=ltf_minutes)
            start_time = end_time - pd.Timedelta(minutes=20)
            mask = (candles_1m_indexed.index > start_time) & (candles_1m_indexed.index <= end_time)
            df_1m_window = candles_1m_indexed.loc[mask].reset_index()
        except Exception:
            df_1m_window = pd.DataFrame()

        if len(df_1m_window) < 5:
            # Not enough 1m data, skip this trade
            ml_filtered += 1
            continue

        # Extract 1m features
        features_1m = extract_1m_features(
            df_1m_window,
            direction,
            channel.support,
            channel.resistance
        )

        # Build feature vector
        feature_values = []
        for fname in feature_names:
            if fname.startswith('1m_'):
                key = fname[3:]
                feature_values.append(features_1m.get(key, 0.0))
            elif fname == 'channel_width':
                feature_values.append((channel.resistance - channel.support) / channel.support)
            elif fname == 'support_touches':
                feature_values.append(channel.support_touches)
            elif fname == 'resistance_touches':
                feature_values.append(channel.resistance_touches)
            elif fname == 'is_long':
                feature_values.append(1 if direction == 'LONG' else 0)
            else:
                feature_values.append(0.0)

        # Predict
        X = np.array([feature_values])
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        X_scaled = ml_scaler.transform(X)
        prob = ml_model.predict_proba(X_scaled)[0, 1]

        # Filter by threshold
        if prob < ml_threshold:
            ml_filtered += 1
            continue

        ml_passed += 1

        # Dynamic SL (optional)
        if use_dynamic_sl and len(df_1m_window) > 0:
            sl_price = calculate_dynamic_sl(
                df_1m_window,
                direction,
                channel.support,
                channel.resistance,
                0.0003
            )

        # Simulate trade
        setup = simulate_trade(
            ltf_candles, i, direction, entry_price, sl_price, tp1_price, tp2_price,
            channel, ltf_volumes, ltf_deltas, avg_volume, avg_delta, cvd_recent, ltf_opens, ltf_closes
        )
        if setup:
            setup['ml_prob'] = prob
            setups.append(setup)
            traded_entries.add(trade_key)

    print(f"  [ML] Passed: {ml_passed}, Filtered: {ml_filtered} (threshold={ml_threshold:.2f})")

    return setups


def run_backtest(setups: List[dict], label: str = "") -> dict:
    """Run backtest on setups and return results."""
    if not setups:
        print(f"  {label}: No trades")
        return {'trades': 0, 'wr': 0, 'return_pct': 0, 'max_dd': 0, 'final_capital': 10000}

    df = pd.DataFrame(setups)

    capital = 10000
    risk_pct = 0.015
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
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    return {
        'trades': len(df),
        'wr': wr,
        'return_pct': total_return,
        'max_dd': max_dd * 100,
        'final_capital': capital
    }


def run_baseline_backtest(year: int = 2024):
    """Run baseline backtest (same as ml_channel_tiebreaker_proper.py NARROW)."""
    print(f"\n{'='*70}")
    print(f"  BASELINE BACKTEST (ml_channel_tiebreaker_proper.py NARROW)")
    print(f"{'='*70}")

    htf = "1h"
    ltf = "15m"

    print(f"\nLoading {htf} data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    htf_candles = htf_candles[htf_candles.index.year == year]
    print(f"  Loaded {len(htf_candles):,} candles ({year} only)")

    print(f"\nLoading {ltf} data...")
    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    ltf_candles = ltf_candles[ltf_candles.index.year == year]
    print(f"  Loaded {len(ltf_candles):,} candles")

    print(f"\nCollecting setups...")
    setups = collect_setups_baseline(htf_candles, ltf_candles, htf, ltf, tiebreaker='narrow')

    result = run_backtest(setups, "BASELINE")

    print(f"\n  Results:")
    print(f"    Trades: {result['trades']}")
    print(f"    Win Rate: {result['wr']:.1f}%")
    print(f"    Return: {result['return_pct']:+.1f}%")
    print(f"    Max DD: {result['max_dd']:.1f}%")
    print(f"    Final Capital: ${result['final_capital']:,.2f}")

    return result


def run_ml_backtest(year: int = 2024, thresholds: List[float] = None):
    """Run ML backtest comparing baseline vs ML filtered."""
    print(f"\n{'='*70}")
    print(f"  ML BACKTEST COMPARISON")
    print(f"{'='*70}")

    htf = "1h"
    ltf = "15m"

    if thresholds is None:
        thresholds = [0.4, 0.5, 0.6, 0.7]

    print(f"\nLoading data...")
    htf_candles_pl = load_candles("BTCUSDT", htf)
    htf_candles = htf_candles_pl.to_pandas().set_index('time')
    htf_candles = htf_candles[htf_candles.index.year == year]

    ltf_candles_pl = load_candles("BTCUSDT", ltf)
    ltf_candles = ltf_candles_pl.to_pandas().set_index('time')
    ltf_candles = ltf_candles[ltf_candles.index.year == year]

    candles_1m_pl = load_candles("BTCUSDT", "1m")
    candles_1m = candles_1m_pl.to_pandas()
    candles_1m['time'] = pd.to_datetime(candles_1m['time'])
    candles_1m = candles_1m[candles_1m['time'].dt.year == year]

    print(f"  1h: {len(htf_candles):,} candles")
    print(f"  15m: {len(ltf_candles):,} candles")
    print(f"  1m: {len(candles_1m):,} candles")

    # Load ML model
    ml_model, ml_scaler, feature_names, default_threshold = load_ml_model()

    results = {}

    # Baseline
    print(f"\n[BASELINE]")
    setups_baseline = collect_setups_baseline(htf_candles, ltf_candles, htf, ltf, quiet=True, tiebreaker='narrow')
    results['BASELINE'] = run_backtest(setups_baseline, "BASELINE")
    print(f"  Trades: {results['BASELINE']['trades']}, WR: {results['BASELINE']['wr']:.1f}%, Return: {results['BASELINE']['return_pct']:+.1f}%")

    # ML filtered
    if ml_model is not None:
        for thresh in thresholds:
            name = f'ML_T{int(thresh*100)}'
            print(f"\n[{name}]")
            setups_ml = collect_setups_ml(
                htf_candles, ltf_candles, candles_1m,
                ml_model, ml_scaler, feature_names,
                ml_threshold=thresh,
                htf_tf=htf, ltf_tf=ltf,
                quiet=True, tiebreaker='narrow'
            )
            results[name] = run_backtest(setups_ml, name)
            print(f"  Trades: {results[name]['trades']}, WR: {results[name]['wr']:.1f}%, Return: {results[name]['return_pct']:+.1f}%")

        # With dynamic SL
        print(f"\n[ML_T50_DYN_SL]")
        setups_ml_dyn = collect_setups_ml(
            htf_candles, ltf_candles, candles_1m,
            ml_model, ml_scaler, feature_names,
            ml_threshold=0.5,
            htf_tf=htf, ltf_tf=ltf,
            use_dynamic_sl=True,
            quiet=True, tiebreaker='narrow'
        )
        results['ML_T50_DYN_SL'] = run_backtest(setups_ml_dyn, "ML_T50_DYN_SL")
        print(f"  Trades: {results['ML_T50_DYN_SL']['trades']}, WR: {results['ML_T50_DYN_SL']['wr']:.1f}%, Return: {results['ML_T50_DYN_SL']['return_pct']:+.1f}%")

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<20} {'Trades':>8} {'WR':>8} {'Return':>12} {'Max DD':>8} {'Final':>15}")
    print("-" * 80)

    for name, res in sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True):
        print(f"{name:<20} {res['trades']:>8} {res['wr']:>7.1f}% {res['return_pct']:>+11.1f}% {res['max_dd']:>7.1f}% ${res['final_capital']:>14,.2f}")

    # Comparison
    baseline = results.get('BASELINE', {})
    best_ml = max([(k, v) for k, v in results.items() if k.startswith('ML_')],
                  key=lambda x: x[1].get('return_pct', -999), default=(None, None))

    if best_ml[0] and baseline:
        print(f"\n{'='*80}")
        print(f"  COMPARISON: BASELINE vs {best_ml[0]}")
        print(f"{'='*80}")
        print(f"  Baseline:  {baseline['trades']} trades, {baseline['wr']:.1f}% WR, {baseline['return_pct']:+.1f}% return")
        print(f"  {best_ml[0]}:  {best_ml[1]['trades']} trades, {best_ml[1]['wr']:.1f}% WR, {best_ml[1]['return_pct']:+.1f}% return")
        print(f"  WR improvement: {best_ml[1]['wr'] - baseline['wr']:+.1f}%")
        print(f"  Return improvement: {best_ml[1]['return_pct'] - baseline['return_pct']:+.1f}%")

    return results


if __name__ == '__main__':
    import sys

    if '--ml' in sys.argv:
        if '--oos' in sys.argv:
            # OOS test: 2024-2025
            print("\n" + "="*70)
            print("  OUT-OF-SAMPLE TEST (2024)")
            print("="*70)
            run_ml_backtest(year=2024)
        else:
            # Default: 2024
            run_ml_backtest(year=2024)
    else:
        run_baseline_backtest(year=2024)
