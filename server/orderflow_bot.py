import pandas as pd
import numpy as np
import json
from ml_trainer import find_support_zones, find_resistance_zones, TOLERANCE

class OrderflowBot:
    """
    Multi-Factor Scoring Bot with Intelligent SL/TP
    
    SL: ATR-based (volatility adjusted)
    TP: Next zone target (or ATR-based if no zone)
    RR: Score-dependent (higher score = bigger target)
    """
    
    def __init__(self):
        self.min_score = 5  # Raised from 4 for higher quality signals
        self.atr_period = 14
        self.sl_atr_multiplier = 1.5  # SL = 1.5x ATR below entry
        self.min_rr = 1.2             # Don't trade if RR < 1.2
    
    def analyze(self, candles):
        if not candles or len(candles) < 52:
            return {'signal': False, 'message': 'Not enough data'}
        df = self._prepare_df(candles)
        return self._analyze_index(df, len(df) - 1)
    
    def backtest(self, candles):
        if not candles or len(candles) < 52:
            return []
        df = self._prepare_df(candles)
        signals = []
        for i in range(51, len(df)):
            result = self._analyze_index(df, i)
            if result.get('signal'):
                result['candleTime'] = int(df.iloc[i]['time'])
                signals.append(result)
        return signals
    
    def _prepare_df(self, candles):
        df = pd.DataFrame(candles)
        column_map = {
            'buyVolume': 'buy_volume',
            'sellVolume': 'sell_volume',
            'tradeCount': 'trade_count'
        }
        df = df.rename(columns=column_map)
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(self.atr_period).mean()
        
        # Calculate EMAs for trend filter
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df
    
    def _parse_clusters(self, clusters):
        if not clusters:
            return []
        if isinstance(clusters, str):
            try:
                return json.loads(clusters)
            except:
                return []
        return clusters if isinstance(clusters, list) else []
    
    def _get_poc(self, clusters):
        if not clusters:
            return None, 0
        max_vol = 0
        poc_price = None
        for c in clusters:
            total = c.get('buy', 0) + c.get('sell', 0)
            if total > max_vol:
                max_vol = total
                poc_price = c.get('price')
        return poc_price, max_vol
    
    def _get_imbalances(self, clusters, threshold=3.0):
        buy_imbalances = []
        sell_imbalances = []
        for c in clusters:
            buy = c.get('buy', 0)
            sell = c.get('sell', 0)
            price = c.get('price')
            if sell > 0 and buy / sell >= threshold:
                buy_imbalances.append(price)
            if buy > 0 and sell / buy >= threshold:
                sell_imbalances.append(price)
        return buy_imbalances, sell_imbalances
    
    def _count_stacked_imbalances(self, imbalance_prices, candle_low, candle_high, tick_size=10):
        if len(imbalance_prices) < 2:
            return 0
        sorted_prices = sorted(imbalance_prices)
        max_stack = 1
        current_stack = 1
        for i in range(1, len(sorted_prices)):
            if abs(sorted_prices[i] - sorted_prices[i-1]) <= tick_size * 2:
                current_stack += 1
                max_stack = max(max_stack, current_stack)
            else:
                current_stack = 1
        return max_stack
    
    def _find_next_resistance(self, resistance_zones, entry_price):
        """Find nearest resistance above entry price."""
        candidates = [z for z in resistance_zones if z['price'] > entry_price]
        if not candidates:
            return None
        return min(candidates, key=lambda z: z['price'] - entry_price)
    
    def _find_next_support(self, support_zones, entry_price):
        """Find nearest support below entry price."""
        candidates = [z for z in support_zones if z['price'] < entry_price]
        if not candidates:
            return None
        return max(candidates, key=lambda z: z['price'])
    
    def _calculate_sl_tp(self, direction, entry, atr, score, support_zones, resistance_zones, candle_low, candle_high):
        """
        Calculate intelligent SL and TP.
        
        SL: ATR-based with structure (swing low/high)
        TP: Next zone or ATR-based, adjusted by score
        """
        
        # Score-based RR target
        if score >= 6:
            target_rr = 2.5  # High confidence
        elif score >= 5:
            target_rr = 2.0
        else:
            target_rr = 1.5  # Minimum confidence
        
        if direction == 'LONG':
            # SL: Below swing low, at least 1x ATR below entry
            structure_sl = candle_low * 0.999
            atr_sl = entry - (atr * self.sl_atr_multiplier)
            sl = min(structure_sl, atr_sl)  # Use the tighter one
            
            risk = entry - sl
            if risk <= 0:
                return None, None, None
            
            # TP: Next resistance zone or ATR-based
            next_resist = self._find_next_resistance(resistance_zones, entry)
            if next_resist:
                zone_tp = next_resist['price'] * 0.999  # Just before zone
                zone_rr = (zone_tp - entry) / risk
                # Use zone if it gives reasonable RR
                if zone_rr >= self.min_rr:
                    return float(sl), float(zone_tp), float(zone_rr)
            
            # Fallback: ATR-based TP with score-adjusted RR
            tp = entry + (risk * target_rr)
            return float(sl), float(tp), float(target_rr)
        
        else:  # SHORT
            # SL: Above swing high, at least 1x ATR above entry
            structure_sl = candle_high * 1.001
            atr_sl = entry + (atr * self.sl_atr_multiplier)
            sl = max(structure_sl, atr_sl)
            
            risk = sl - entry
            if risk <= 0:
                return None, None, None
            
            # TP: Next support zone or ATR-based
            next_supp = self._find_next_support(support_zones, entry)
            if next_supp:
                zone_tp = next_supp['price'] * 1.001  # Just before zone
                zone_rr = (entry - zone_tp) / risk
                if zone_rr >= self.min_rr:
                    return float(sl), float(zone_tp), float(zone_rr)
            
            # Fallback
            tp = entry - (risk * target_rr)
            return float(sl), float(tp), float(target_rr)
    
    def _analyze_index(self, df, idx):
        if idx < 51:
            return {'signal': False}
        
        current = df.iloc[idx]
        
        # Basic data
        open_p = current['open']
        high = current['high']
        low = current['low']
        close = current['close']
        volume = current['volume']
        delta = current.get('delta', 0)
        buy_vol = current.get('buy_volume', 0)
        sell_vol = current.get('sell_volume', 0)
        trade_count = current.get('trade_count', 0)
        atr = current.get('atr', 0)
        
        if atr == 0 or pd.isna(atr):
            atr = (high - low)  # Fallback
        
        # Clusters
        clusters = self._parse_clusters(current.get('clusters'))
        poc_price, poc_vol = self._get_poc(clusters)
        buy_imbalances, sell_imbalances = self._get_imbalances(clusters)
        
        # Averages
        lookback = df.iloc[max(0, idx-20):idx]
        avg_vol = lookback['volume'].mean() if len(lookback) > 0 else volume
        avg_body = (lookback['close'] - lookback['open']).abs().mean()
        avg_trades = lookback['trade_count'].mean() if 'trade_count' in lookback.columns else trade_count
        
        # Candle props
        body = abs(close - open_p)
        candle_range = high - low
        
        # Zones
        support_zones = find_support_zones(df, idx)
        resistance_zones = find_resistance_zones(df, idx)
        
        touching_support = None
        touching_resistance = None
        
        for zone in support_zones:
            if zone['strength'] >= 2 and abs(low - zone['price']) < close * TOLERANCE:
                touching_support = zone
                break
        
        for zone in resistance_zones:
            if zone['strength'] >= 2 and abs(high - zone['price']) < close * TOLERANCE:
                touching_resistance = zone
                break
        
        # SCORING SYSTEM FOR LONG (Weighted factors)
        # ============================================
        long_score = 0
        long_reasons = []
        
        # Trend filter: EMA20 > EMA50 for bullish trend
        ema20 = current.get('ema20', 0)
        ema50 = current.get('ema50', 0)
        bullish_trend = ema20 > ema50 if ema20 > 0 and ema50 > 0 else True
        bearish_trend = ema20 < ema50 if ema20 > 0 and ema50 > 0 else True
        
        # +2: Delta Divergence (strong reversal signal)
        if delta < -avg_vol * 0.1 and close >= open_p:
            long_score += 2
            long_reasons.append("DeltaDiv(+2)")
        
        if volume > avg_vol * 1.5:
            long_score += 1
            long_reasons.append("VolSpike")
        
        if avg_body > 0 and body < avg_body * 0.6:
            long_score += 1
            long_reasons.append("SmallBody")
        
        if poc_price and candle_range > 0:
            poc_position = (poc_price - low) / candle_range
            if poc_position < 0.3:
                long_score += 1
                long_reasons.append("POCLow")
        
        # +2: Zone Touch (zones are proven reliable)
        if touching_support:
            long_score += 2
            long_reasons.append(f"Supp{touching_support['strength']}(+2)")
        
        buy_stack = self._count_stacked_imbalances(buy_imbalances, low, high)
        if buy_stack >= 2:
            long_score += 1
            long_reasons.append(f"BuyStack{buy_stack}")
        
        if avg_trades > 0 and trade_count > avg_trades * 1.5:
            long_score += 1
            long_reasons.append("HighTrades")
        
        # ============================================
        # SCORING SYSTEM FOR SHORT (Weighted factors)
        # ============================================
        short_score = 0
        short_reasons = []
        
        # +2: Delta Divergence (strong reversal signal)
        if delta > avg_vol * 0.1 and close <= open_p:
            short_score += 2
            short_reasons.append("DeltaDiv(+2)")
        
        if volume > avg_vol * 1.5:
            short_score += 1
            short_reasons.append("VolSpike")
        
        if avg_body > 0 and body < avg_body * 0.6:
            short_score += 1
            short_reasons.append("SmallBody")
        
        if poc_price and candle_range > 0:
            poc_position = (poc_price - low) / candle_range
            if poc_position > 0.7:
                short_score += 1
                short_reasons.append("POCHigh")
        
        # +2: Zone Touch (zones are proven reliable)
        if touching_resistance:
            short_score += 2
            short_reasons.append(f"Resist{touching_resistance['strength']}(+2)")
        
        sell_stack = self._count_stacked_imbalances(sell_imbalances, low, high)
        if sell_stack >= 2:
            short_score += 1
            short_reasons.append(f"SellStack{sell_stack}")
        
        if avg_trades > 0 and trade_count > avg_trades * 1.5:
            short_score += 1
            short_reasons.append("HighTrades")
        
        # ============================================
        # SIGNAL DECISION WITH TREND FILTER + SL/TP
        # ============================================
        
        # LONG: Requires bullish trend (EMA20 > EMA50)
        if long_score >= self.min_score and long_score > short_score and bullish_trend:
            entry = float(close)
            sl, tp, rr = self._calculate_sl_tp(
                'LONG', entry, atr, long_score,
                support_zones, resistance_zones, low, high
            )
            
            if sl is None or rr < self.min_rr:
                return {'signal': False}
            
            return {
                'signal': True,
                'type': 'LONG',
                'setupType': f"Score {long_score}/7: {', '.join(long_reasons)}",
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'rr': rr,
                'confidence': min(0.95, 0.5 + (long_score * 0.07)),
                'zonePrice': float(touching_support['price']) if touching_support else float(low),
                'source': 'OrderflowBot',
                'score': long_score
            }
        
        # SHORT: Requires bearish trend (EMA20 < EMA50)
        if short_score >= self.min_score and short_score > long_score and bearish_trend:
            entry = float(close)
            sl, tp, rr = self._calculate_sl_tp(
                'SHORT', entry, atr, short_score,
                support_zones, resistance_zones, low, high
            )
            
            if sl is None or rr < self.min_rr:
                return {'signal': False}
            
            return {
                'signal': True,
                'type': 'SHORT',
                'setupType': f"Score {short_score}/7: {', '.join(short_reasons)}",
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'rr': rr,
                'confidence': min(0.95, 0.5 + (short_score * 0.07)),
                'zonePrice': float(touching_resistance['price']) if touching_resistance else float(high),
                'source': 'OrderflowBot',
                'score': short_score
            }
        
        return {'signal': False}
