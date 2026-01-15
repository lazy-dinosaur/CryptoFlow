#!/usr/bin/env python3
"""
ML Training Script for Price Direction Prediction
Uses historical candle data to train a model
"""

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from parse_data import load_candles

# Model save directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def create_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Create features from OHLCV + delta data."""

    features = pd.DataFrame(index=df.index)

    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_pct'] = (df['high'] - df['low']) / df['close']
    features['close_open_pct'] = (df['close'] - df['open']) / df['open']

    # Moving averages
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'sma_{period}_dist'] = (df['close'] - features[f'sma_{period}']) / features[f'sma_{period}']

    # EMA
    for period in [5, 10, 20]:
        features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        features[f'ema_{period}_dist'] = (df['close'] - features[f'ema_{period}']) / features[f'ema_{period}']

    # Volatility
    features['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    features['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()

    # Volume features
    features['volume_sma_20'] = df['volume'].rolling(20).mean()
    features['volume_ratio'] = df['volume'] / features['volume_sma_20']

    # Delta features (unique to our data)
    features['delta'] = df['delta']
    features['delta_sma_5'] = df['delta'].rolling(5).mean()
    features['delta_sma_10'] = df['delta'].rolling(10).mean()
    features['delta_std_10'] = df['delta'].rolling(10).std()
    features['cumulative_delta_10'] = df['delta'].rolling(10).sum()
    features['cumulative_delta_20'] = df['delta'].rolling(20).sum()

    # Delta momentum
    features['delta_momentum'] = df['delta'] - df['delta'].shift(1)
    features['delta_acceleration'] = features['delta_momentum'] - features['delta_momentum'].shift(1)

    # Buy/Sell volume ratio
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
    features['bb_upper'] = sma_20 + 2 * std_20
    features['bb_lower'] = sma_20 - 2 * std_20
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
    features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)

    # Lagged features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'delta_lag_{lag}'] = features['delta'].shift(lag)
        features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)

    # Drop intermediate columns (keep only ratios and normalized values)
    cols_to_drop = [c for c in features.columns if c.startswith('sma_') and not c.endswith('_dist')]
    cols_to_drop += [c for c in features.columns if c.startswith('ema_') and not c.endswith('_dist')]
    cols_to_drop += ['volume_sma_20', 'bb_upper', 'bb_lower']
    features = features.drop(columns=cols_to_drop, errors='ignore')

    return features


def create_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Create labels for prediction.

    Args:
        df: DataFrame with 'close' column
        horizon: How many candles ahead to predict
        threshold: Minimum % change to count as up/down (0 = any movement)

    Returns:
        Series with labels: 1 = up, 0 = down
    """
    future_returns = df['close'].shift(-horizon) / df['close'] - 1

    if threshold > 0:
        # 3-class: up (1), neutral (0.5), down (0)
        labels = pd.Series(0.5, index=df.index)
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = 0
    else:
        # Binary: up (1), down (0)
        labels = (future_returns > 0).astype(int)

    return labels


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model and print results."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("  Model Evaluation")
    print("="*60)

    print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, y_pred)*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test, y_pred)*100:.2f}%")
    print(f"  F1 Score:  {f1_score(y_test, y_pred)*100:.2f}%")

    print("\n  Classification Report:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    # Feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]

    print("\n  Top 15 Features:")
    print("-"*60)
    for i, idx in enumerate(indices):
        print(f"  {i+1:2}. {feature_names[idx]:25} {importance[idx]:.4f}")

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML model for price prediction")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (candles)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           CryptoFlow ML Training                          ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Load data
    print(f"Loading {args.symbol} {args.timeframe} data...")
    candles_pl = load_candles(args.symbol, args.timeframe)
    df = candles_pl.to_pandas().set_index('time')
    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {df.index[0]} to {df.index[-1]}")

    # Create features and labels
    print("\nCreating features...")
    features = create_features(df)
    labels = create_labels(df, horizon=args.horizon)

    # Combine and drop NaN
    data = features.copy()
    data['label'] = labels
    data = data.dropna()

    print(f"  Features: {len(features.columns)}")
    print(f"  Samples after cleaning: {len(data):,}")

    # Split data (time-based, not random)
    split_idx = int(len(data) * (1 - args.test_size))

    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False
    )

    print(f"\n  Train: {len(X_train):,} samples ({train_data.index[0].date()} ~ {train_data.index[split_idx-1].date()})")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples ({test_data.index[0].date()} ~ {test_data.index[-1].date()})")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\nTraining XGBoost model...")
    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test, X_train.columns.tolist())

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"xgb_{args.symbol}_{args.timeframe}_{timestamp}.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{args.symbol}_{args.timeframe}_{timestamp}.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, f"features_{args.symbol}_{args.timeframe}_{timestamp}.joblib"))

    print(f"\n{'='*60}")
    print(f"  Model saved to: {model_path}")
    print(f"{'='*60}\n")

    # Label distribution
    print("Label Distribution:")
    print(f"  Train - UP: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%), DOWN: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)")
    print(f"  Test  - UP: {(y_test==1).sum()} ({(y_test==1).mean()*100:.1f}%), DOWN: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
