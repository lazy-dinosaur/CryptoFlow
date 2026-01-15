"""
ML-Based Strategy with Confidence Filter
Only trades when model confidence exceeds threshold
"""

import pandas as pd
import numpy as np
import joblib
import os
from .base import Strategy, Signal, SignalType


class MLStrategy(Strategy):
    """
    ML-based strategy that uses XGBoost predictions.
    Only enters trades when model confidence > threshold.
    """

    @property
    def name(self) -> str:
        return "MLStrategy"

    def __init__(self, params=None):
        default_params = {
            'model_path': None,
            'scaler_path': None,
            'features_path': None,
            'confidence_threshold': 0.6,  # Only trade when > 60% confident
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.02,
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

        # Load model
        self.model = joblib.load(self.params['model_path'])
        self.scaler = joblib.load(self.params['scaler_path'])
        self.feature_names = joblib.load(self.params['features_path'])

        # Feature cache
        self.features_cache = None

    def _create_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """Create features from history (same as training)."""
        df = history.copy()
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_pct'] = (df['high'] - df['low']) / df['close']
        features['close_open_pct'] = (df['close'] - df['open']) / df['open']

        # Moving averages
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}_dist'] = (df['close'] - sma) / sma

        # EMA
        for period in [5, 10, 20]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}_dist'] = (df['close'] - ema) / ema

        # Volatility
        features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        features['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()

        # Volume features
        volume_sma = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / volume_sma

        # Delta features
        features['delta'] = df['delta']
        features['delta_sma_5'] = df['delta'].rolling(5).mean()
        features['delta_sma_10'] = df['delta'].rolling(10).mean()
        features['delta_std_10'] = df['delta'].rolling(10).std()
        features['cumulative_delta_10'] = df['delta'].rolling(10).sum()
        features['cumulative_delta_20'] = df['delta'].rolling(20).sum()

        # Delta momentum
        features['delta_momentum'] = df['delta'] - df['delta'].shift(1)
        features['delta_acceleration'] = features['delta_momentum'] - features['delta_momentum'].shift(1)

        # Buy/Sell ratio
        features['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-10)
        features['buy_sell_ratio_sma_5'] = features['buy_sell_ratio'].rolling(5).mean()

        # RSI
        delta_price = df['close'].diff()
        gain = (delta_price.where(delta_price > 0, 0)).rolling(14).mean()
        loss = (-delta_price.where(delta_price < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        features['bb_width'] = (bb_upper - bb_lower) / sma_20
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'delta_lag_{lag}'] = features['delta'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)

        return features

    def on_candle(self, candle: pd.Series, history: pd.DataFrame) -> Signal:
        # Need enough history
        if len(history) < 60:
            return Signal(SignalType.NONE)

        # Don't enter if already in position
        if self.has_position():
            return Signal(SignalType.NONE)

        # Create features for all history + current candle
        full_history = pd.concat([history, candle.to_frame().T])
        features = self._create_features(full_history)

        # Get latest features
        latest = features.iloc[-1:][self.feature_names]

        if latest.isna().any().any():
            return Signal(SignalType.NONE)

        # Scale and predict
        scaled = self.scaler.transform(latest)
        prob = self.model.predict_proba(scaled)[0]

        prob_down, prob_up = prob[0], prob[1]
        threshold = self.params['confidence_threshold']

        price = candle['close']
        stop_loss_pct = self.params['stop_loss_pct']
        take_profit_pct = self.params['take_profit_pct']

        # LONG: High confidence of UP
        if prob_up > threshold:
            return Signal(
                type=SignalType.LONG,
                entry_price=price,
                stop_loss=price * (1 - stop_loss_pct),
                take_profit=price * (1 + take_profit_pct),
                confidence=prob_up,
                reason=f"ML Prob UP: {prob_up:.1%}"
            )

        # SHORT: High confidence of DOWN
        if prob_down > threshold:
            return Signal(
                type=SignalType.SHORT,
                entry_price=price,
                stop_loss=price * (1 + stop_loss_pct),
                take_profit=price * (1 - take_profit_pct),
                confidence=prob_down,
                reason=f"ML Prob DOWN: {prob_down:.1%}"
            )

        return Signal(SignalType.NONE)
