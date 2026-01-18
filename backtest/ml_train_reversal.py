#!/usr/bin/env python3
"""
1분봉 반전 패턴 ML 모델 학습

수집된 데이터를 사용하여 채널 터치 시 반전 확률을 예측하는 모델 학습

Models:
- RandomForest
- XGBoost (optional)

Output:
- models/reversal_model.joblib
- models/reversal_scaler.joblib
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed, using RandomForest only")

# Paths
DATA_FILE_TRAIN = 'data/ml_1m_patterns_train.csv'  # 2022-2023 (IS)
DATA_FILE_TEST = 'data/ml_1m_patterns_test.csv'    # 2024-2025 (OOS)
MODELS_DIR = 'models'


def load_data(file_path: str = DATA_FILE_TRAIN) -> pd.DataFrame:
    """Load collected pattern data."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} samples")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and labels for training.

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
    """
    # 1M features only (new approach)
    feature_cols_1m = [c for c in df.columns if c.startswith('1m_')]

    # Additional context features
    context_cols = [
        'channel_width',
        'support_touches',
        'resistance_touches',
    ]

    # Add direction encoding
    df['is_long'] = (df['direction'] == 'LONG').astype(int)
    context_cols.append('is_long')

    # Combine all features
    feature_cols = feature_cols_1m + context_cols

    # Check which columns exist
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]

    if missing_cols:
        print(f"[WARN] Missing columns: {missing_cols}")

    print(f"\nUsing {len(available_cols)} features:")
    for col in available_cols:
        print(f"  - {col}")

    X = df[available_cols].values
    y = df['label'].values
    feature_names = available_cols

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    return X, y, feature_names


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train RandomForest classifier."""
    print("\n" + "=" * 60)
    print("TRAINING: RandomForest")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['SL (0)', 'TP1+ (1)']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {auc:.4f}")

    # Feature importance
    print("\nFeature Importance (Top 15):")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return rf, importance


def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost classifier."""
    if not HAS_XGB:
        print("[SKIP] XGBoost not available")
        return None, None

    print("\n" + "=" * 60)
    print("TRAINING: XGBoost")
    print("=" * 60)

    # Calculate scale_pos_weight for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['SL (0)', 'TP1+ (1)']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC: {auc:.4f}")

    # Feature importance
    print("\nFeature Importance (Top 15):")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return xgb, importance


def find_optimal_threshold(model, X_test, y_test, feature_names=None):
    """
    Find optimal probability threshold for trading.

    Goal: Maximize win rate while maintaining enough trades
    """
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.3, 0.9, 0.05)
    results = []

    for thresh in thresholds:
        mask = y_prob >= thresh
        if mask.sum() == 0:
            continue

        y_pred = (y_prob >= thresh).astype(int)
        filtered_y_test = y_test[mask]
        filtered_y_pred = y_pred[mask]

        if len(filtered_y_test) == 0:
            continue

        accuracy = (filtered_y_pred == filtered_y_test).mean()
        win_rate = filtered_y_test.mean()  # Actual win rate when taking filtered trades
        trade_count = mask.sum()
        filter_pct = trade_count / len(y_test) * 100

        results.append({
            'threshold': thresh,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'trades': trade_count,
            'filter_pct': filter_pct
        })

    df_results = pd.DataFrame(results)
    print("\nThreshold Analysis:")
    print(f"{'Thresh':>8} {'Accuracy':>10} {'WinRate':>10} {'Trades':>8} {'Filter%':>10}")
    print("-" * 50)

    for _, row in df_results.iterrows():
        print(f"{row['threshold']:>8.2f} {row['accuracy']*100:>9.1f}% {row['win_rate']*100:>9.1f}% {int(row['trades']):>8} {row['filter_pct']:>9.1f}%")

    # Find best threshold (balance between win rate and trade count)
    # Weight: win_rate * sqrt(trade_count_normalized)
    if len(df_results) > 0:
        df_results['trade_norm'] = df_results['trades'] / df_results['trades'].max()
        df_results['score'] = df_results['win_rate'] * np.sqrt(df_results['trade_norm'])
        best_idx = df_results['score'].idxmax()
        best_thresh = df_results.loc[best_idx, 'threshold']
        print(f"\n  Best threshold: {best_thresh:.2f}")
        print(f"    Win Rate: {df_results.loc[best_idx, 'win_rate']*100:.1f}%")
        print(f"    Trades: {int(df_results.loc[best_idx, 'trades'])}")
        return best_thresh

    return 0.5


def cross_validate_model(model, X, y, n_splits=5):
    """Perform time series cross-validation."""
    print("\n" + "=" * 60)
    print("TIME SERIES CROSS-VALIDATION")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')

    print(f"\nROC AUC scores: {scores}")
    print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return scores.mean()


def save_model(model, scaler, feature_names, best_threshold, model_name='reversal'):
    """Save trained model and scaler."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f'{model_name}_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, f'{model_name}_scaler.joblib')
    meta_path = os.path.join(MODELS_DIR, f'{model_name}_meta.joblib')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        'feature_names': feature_names,
        'best_threshold': best_threshold,
    }, meta_path)

    print(f"\n[SAVED] Model: {model_path}")
    print(f"[SAVED] Scaler: {scaler_path}")
    print(f"[SAVED] Metadata: {meta_path}")


def run_training(data_file: str = DATA_FILE_TRAIN):
    """Run full training pipeline."""
    # Load data
    df = load_data(data_file)

    # Data summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Success rate (baseline): {df['label'].mean()*100:.1f}%")
    print(f"LONG: {(df['direction']=='LONG').sum()}, SHORT: {(df['direction']=='SHORT').sum()}")

    # Prepare features
    X, y, feature_names = prepare_features(df)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-based split (last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Train success rate: {y_train.mean()*100:.1f}%")
    print(f"Test success rate: {y_test.mean()*100:.1f}%")

    # Train models
    rf_model, rf_importance = train_random_forest(X_train, y_train, X_test, y_test, feature_names)

    xgb_model = None
    if HAS_XGB:
        xgb_model, xgb_importance = train_xgboost(X_train, y_train, X_test, y_test, feature_names)

    # Find optimal threshold
    best_model = xgb_model if xgb_model is not None else rf_model
    best_threshold = find_optimal_threshold(best_model, X_test, y_test)

    # Cross-validation
    cv_score = cross_validate_model(best_model, X_scaled, y)

    # Save best model
    model_name = 'reversal_xgb' if xgb_model is not None else 'reversal_rf'
    save_model(best_model, scaler, feature_names, best_threshold, model_name)

    # Also save as generic 'reversal' for easy loading
    save_model(best_model, scaler, feature_names, best_threshold, 'reversal')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {model_name}")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"CV ROC AUC: {cv_score:.4f}")

    return best_model, scaler, feature_names, best_threshold


if __name__ == '__main__':
    run_training()
