"""
CryptoFlow ML Signal Predictor
Trains an XGBoost model to predict signal success based on orderflow features.
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

# Check if sklearn/xgboost available, if not provide installation instructions
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import xgboost as xgb
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install scikit-learn xgboost pandas numpy")
    exit(1)

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cryptoflow.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'signal_model.pkl')
SYMBOL = None  # None = all symbols
TIMEFRAME = 1  # 1-minute candles
MIN_RR = 1.0  # Lowered from 1.5 to get more samples
TOLERANCE = 0.0002  # Increased tolerance for zone proximity

# DB connection helper with timeout to prevent corruption
DB_TIMEOUT = 30  # seconds

def get_db_connection(db_path):
    """Create DB connection with proper timeout and WAL mode"""
    conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=30000')
    return conn

def load_candles(db_path, symbol, timeframe):
    """Load candles from SQLite database."""
    conn = get_db_connection(db_path)
    
    if symbol:
        query = f"""
            SELECT * FROM candles_{timeframe} 
            WHERE symbol = ? 
            ORDER BY time ASC
        """
        df = pd.read_sql_query(query, conn, params=[symbol])
    else:
        query = f"""
            SELECT * FROM candles_{timeframe} 
            ORDER BY symbol, time ASC
        """
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    # Parse clusters JSON
    df['clusters'] = df['clusters'].apply(lambda x: json.loads(x) if x else {})
    
    print(f"Loaded {len(df)} candles for {symbol or 'ALL SYMBOLS'} ({timeframe}m)")
    return df

def extract_features(df, idx, depth_data=None, whale_trades=None):
    """Extract ML features for a single candle."""
    if idx < 5:  # Need history
        return None
    
    candle = df.iloc[idx]
    prev_candles = df.iloc[max(0, idx-5):idx]
    
    features = {}
    
    # Basic OHLCV
    features['is_green'] = 1 if candle['close'] > candle['open'] else 0
    features['body_size'] = abs(candle['close'] - candle['open'])
    features['range'] = candle['high'] - candle['low']
    features['upper_wick'] = candle['high'] - max(candle['open'], candle['close'])
    features['lower_wick'] = min(candle['open'], candle['close']) - candle['low']
    
    # Wick ratios
    if features['range'] > 0:
        features['lower_wick_ratio'] = features['lower_wick'] / features['range']
        features['upper_wick_ratio'] = features['upper_wick'] / features['range']
        features['body_ratio'] = features['body_size'] / features['range']
    else:
        features['lower_wick_ratio'] = 0
        features['upper_wick_ratio'] = 0
        features['body_ratio'] = 0
    
    # Delta (Orderflow)
    features['delta'] = candle['delta']
    features['delta_pct'] = candle['delta'] / candle['volume'] if candle['volume'] > 0 else 0
    features['buy_volume_pct'] = candle['buy_volume'] / candle['volume'] if candle['volume'] > 0 else 0
    
    # Volume comparison
    avg_volume = prev_candles['volume'].mean() if len(prev_candles) > 0 else candle['volume']
    features['volume_ratio'] = candle['volume'] / avg_volume if avg_volume > 0 else 1
    
    # Delta trend (last 5 candles)
    features['delta_sum_5'] = prev_candles['delta'].sum()
    features['delta_avg_5'] = prev_candles['delta'].mean() if len(prev_candles) > 0 else 0
    
    # CVD (Cumulative Volume Delta) - NEW ORDER FLOW FEATURE
    if len(prev_candles) >= 5:
        cvd_values = prev_candles['delta'].cumsum()
        features['cvd_5'] = cvd_values.iloc[-1] if len(cvd_values) > 0 else 0
        features['cvd_slope'] = (cvd_values.iloc[-1] - cvd_values.iloc[0]) / 5 if len(cvd_values) >= 2 else 0
        features['cvd_momentum'] = features['cvd_5'] / (abs(features['delta_sum_5']) + 1)
    else:
        features['cvd_5'] = 0
        features['cvd_slope'] = 0
        features['cvd_momentum'] = 0
    
    # Price momentum
    if len(prev_candles) >= 5:
        features['price_change_5'] = candle['close'] - prev_candles.iloc[0]['close']
        features['high_5'] = prev_candles['high'].max()
        features['low_5'] = prev_candles['low'].min()
        features['range_position'] = (candle['close'] - features['low_5']) / (features['high_5'] - features['low_5']) if (features['high_5'] - features['low_5']) > 0 else 0.5
    else:
        features['price_change_5'] = 0
        features['high_5'] = candle['high']
        features['low_5'] = candle['low']
        features['range_position'] = 0.5
    
    # Cluster analysis (Imbalances)
    clusters = candle['clusters']
    total_imbalances = 0
    buy_imbalances = 0
    sell_imbalances = 0
    
    for price_str, cluster in clusters.items():
        ask = cluster.get('ask', 0)
        bid = cluster.get('bid', 0)
        if ask > 0 and bid > 0:
            if ask >= bid * 2:
                buy_imbalances += 1
            elif bid >= ask * 2:
                sell_imbalances += 1
        total_imbalances = buy_imbalances + sell_imbalances
    
    features['buy_imbalances'] = buy_imbalances
    features['sell_imbalances'] = sell_imbalances
    features['imbalance_ratio'] = buy_imbalances / (total_imbalances + 1)
    
    # Trade intensity
    features['trade_count'] = candle['trade_count']
    avg_trades = prev_candles['trade_count'].mean() if len(prev_candles) > 0 else candle['trade_count']
    features['trade_intensity'] = candle['trade_count'] / avg_trades if avg_trades > 0 else 1
    
    # ========== NEW FEATURES: DEPTH HEATMAP ==========
    features.update(extract_depth_features(candle, depth_data))
    
    # ========== NEW FEATURES: WHALE TRADES ==========
    features.update(extract_whale_features(candle, whale_trades, prev_candles))
    
    return features


def extract_depth_features(candle, depth_data):
    """Extract features from depth heatmap data (liquidity walls)."""
    features = {
        'bid_wall_volume': 0,
        'ask_wall_volume': 0,
        'bid_wall_distance': 0,
        'ask_wall_distance': 0,
        'liquidity_imbalance': 0,
        'nearest_support_strength': 0,
        'nearest_resistance_strength': 0
    }
    
    if not depth_data:
        return features
    
    price = candle['close']
    candle_time = candle['time']
    
    # Find closest depth snapshot to candle time
    closest_snapshot = None
    min_time_diff = float('inf')
    
    for snapshot in depth_data:
        time_diff = abs(snapshot.get('time', 0) - candle_time)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_snapshot = snapshot
    
    if not closest_snapshot or min_time_diff > 60000:  # Max 1 min diff
        return features
    
    bids = closest_snapshot.get('bids', [])
    asks = closest_snapshot.get('asks', [])
    
    # Analyze bids (support)
    total_bid_vol = 0
    nearest_big_bid = None
    for level in bids:
        p, q = level.get('p', 0), level.get('q', 0)
        if p < price:
            total_bid_vol += q
            if q > 5 and (nearest_big_bid is None or p > nearest_big_bid['p']):
                nearest_big_bid = {'p': p, 'q': q}
    
    if nearest_big_bid:
        features['bid_wall_volume'] = nearest_big_bid['q']
        features['bid_wall_distance'] = (price - nearest_big_bid['p']) / price * 100
        features['nearest_support_strength'] = nearest_big_bid['q']
    
    # Analyze asks (resistance)
    total_ask_vol = 0
    nearest_big_ask = None
    for level in asks:
        p, q = level.get('p', 0), level.get('q', 0)
        if p > price:
            total_ask_vol += q
            if q > 5 and (nearest_big_ask is None or p < nearest_big_ask['p']):
                nearest_big_ask = {'p': p, 'q': q}
    
    if nearest_big_ask:
        features['ask_wall_volume'] = nearest_big_ask['q']
        features['ask_wall_distance'] = (nearest_big_ask['p'] - price) / price * 100
        features['nearest_resistance_strength'] = nearest_big_ask['q']
    
    # Liquidity imbalance
    if total_bid_vol + total_ask_vol > 0:
        features['liquidity_imbalance'] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
    
    return features


def extract_whale_features(candle, whale_trades, prev_candles):
    """Extract features from whale (large) trades."""
    features = {
        'whale_buy_volume': 0,
        'whale_sell_volume': 0,
        'whale_count_5': 0,
        'whale_net_delta': 0,
        'whale_intensity': 0
    }
    
    if not whale_trades:
        return features
    
    candle_time = candle['time']
    candle_end = candle_time + 60000  # 1 minute candle
    
    # 5-candle lookback for whale activity
    lookback_start = candle_time - 5 * 60000
    
    whale_buys = 0
    whale_sells = 0
    count = 0
    
    for trade in whale_trades:
        trade_time = trade.get('time', 0)
        if lookback_start <= trade_time <= candle_end:
            qty = trade.get('quantity', 0)
            is_buy = trade.get('isBuyerMaker', False) == False  # Taker buy
            
            if is_buy:
                whale_buys += qty
            else:
                whale_sells += qty
            count += 1
    
    features['whale_buy_volume'] = whale_buys
    features['whale_sell_volume'] = whale_sells
    features['whale_count_5'] = count
    features['whale_net_delta'] = whale_buys - whale_sells
    
    # Calculate intensity relative to normal volume
    avg_volume = prev_candles['volume'].mean() if len(prev_candles) > 0 else candle['volume']
    total_whale = whale_buys + whale_sells
    features['whale_intensity'] = total_whale / avg_volume if avg_volume > 0 else 0
    
    return features

def find_support_zones(df, idx):
    """Find active support zones at candle index."""
    zones = []
    
    for i in range(max(0, idx-50), idx):
        candle = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else candle
        
        # Absorption pattern: Strong rejection with high volume
        if (candle['low'] < prev['low'] and 
            candle['close'] > candle['open'] and
            candle['delta'] > 0):
            
            zone_price = candle['low']
            
            # Check if zone is still valid (not broken)
            is_broken = False
            for j in range(i+1, idx):
                if df.iloc[j]['close'] < zone_price * 0.999:
                    is_broken = True
                    break
            
            if not is_broken:
                zones.append({
                    'price': zone_price,
                    'idx': i,
                    'strength': candle['volume']
                })
    
    return zones

def find_resistance_zones(df, idx):
    """Find active resistance zones at candle index."""
    zones = []
    
    for i in range(max(0, idx-50), idx):
        candle = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else candle
        
        # Absorption pattern: Strong rejection with high volume
        if (candle['high'] > prev['high'] and 
            candle['close'] < candle['open'] and
            candle['delta'] < 0):
            
            zone_price = candle['high']
            
            # Check if zone is still valid (not broken)
            is_broken = False
            for j in range(i+1, idx):
                if df.iloc[j]['close'] > zone_price * 1.001:
                    is_broken = True
                    break
            
            if not is_broken:
                zones.append({
                    'price': zone_price,
                    'idx': i,
                    'strength': candle['volume']
                })
    
    return zones

def backtest_signal(df, idx, entry, sl, tp, direction='LONG'):
    """Backtest a signal and return result."""
    for j in range(idx + 1, len(df)):
        future = df.iloc[j]
        if direction == 'LONG':
            if future['low'] <= sl:
                return 'loss'
            if future['high'] >= tp:
                return 'win'
        else:  # SHORT
            if future['high'] >= sl:
                return 'loss'
            if future['low'] <= tp:
                return 'win'
    return 'active'

def load_depth_data(db_path, symbol='BTCUSDT'):
    """Load depth heatmap snapshots from database."""
    try:
        conn = get_db_connection(db_path)
        query = """
            SELECT time, bids, asks 
            FROM heatmap_snapshots 
            WHERE symbol = ? 
            ORDER BY time ASC
        """
        rows = conn.execute(query, [symbol]).fetchall()
        conn.close()
        
        depth_data = []
        for row in rows:
            try:
                depth_data.append({
                    'time': row[0],
                    'bids': json.loads(row[1]) if row[1] else [],
                    'asks': json.loads(row[2]) if row[2] else []
                })
            except:
                pass
        
        print(f"Loaded {len(depth_data)} depth snapshots")
        return depth_data
    except Exception as e:
        print(f"Error loading depth data: {e}")
        return []


def load_whale_trades(db_path, symbol='btcusdt', threshold=5.0):
    """Load whale (large) trades from database."""
    try:
        conn = get_db_connection(db_path)
        query = """
            SELECT time, price, quantity, is_buyer_maker 
            FROM big_trades 
            WHERE symbol = ? AND quantity >= ?
            ORDER BY time ASC
        """
        rows = conn.execute(query, [symbol, threshold]).fetchall()
        conn.close()
        
        whale_trades = []
        for row in rows:
            whale_trades.append({
                'time': row[0],
                'price': row[1],
                'quantity': row[2],
                'isBuyerMaker': row[3] == 1
            })
        
        print(f"Loaded {len(whale_trades)} whale trades (>= {threshold} BTC)")
        return whale_trades
    except Exception as e:
        print(f"Error loading whale trades: {e}")
        return []


def generate_training_data(df, depth_data=None, whale_trades=None):
    """Generate training dataset with features and labels (LONG + SHORT)."""
    X = []
    y = []

    print("Generating training data with enhanced features (LONG + SHORT)...")

    long_count = 0
    short_count = 0

    for idx in range(10, len(df) - 10):  # Need buffer on both sides
        candle = df.iloc[idx]

        support_zones = find_support_zones(df, idx)
        resistance_zones = find_resistance_zones(df, idx)

        # ========== LONG SETUP ==========
        is_green = candle['close'] > candle['open']
        has_positive_delta = candle['delta'] > 0

        if is_green and has_positive_delta:
            touching_support = None
            for zone in support_zones:
                if abs(candle['low'] - zone['price']) < candle['close'] * TOLERANCE:
                    touching_support = zone
                    break

            if touching_support:
                entry = candle['close']
                sl = min(candle['low'], touching_support['price']) * 0.9997
                risk = entry - sl

                if risk > 0:
                    target_resistance = None
                    min_dist = float('inf')
                    for zone in resistance_zones:
                        if zone['price'] > entry:
                            dist = zone['price'] - entry
                            if dist < min_dist:
                                min_dist = dist
                                target_resistance = zone

                    if target_resistance and min_dist >= risk * MIN_RR:
                        tp = target_resistance['price']
                        result = backtest_signal(df, idx, entry, sl, tp, 'LONG')

                        if result != 'active':
                            features = extract_features(df, idx, depth_data, whale_trades)
                            if features:
                                features['zone_distance'] = abs(candle['low'] - touching_support['price'])
                                features['zone_age'] = idx - touching_support['idx']
                                features['zone_strength'] = touching_support['strength']
                                features['risk_reward'] = min_dist / risk
                                features['is_short'] = 0

                                X.append(features)
                                y.append(1 if result == 'win' else 0)
                                long_count += 1

        # ========== SHORT SETUP ==========
        is_red = candle['close'] < candle['open']
        has_negative_delta = candle['delta'] < 0

        if is_red and has_negative_delta:
            touching_resistance = None
            for zone in resistance_zones:
                if abs(candle['high'] - zone['price']) < candle['close'] * TOLERANCE:
                    touching_resistance = zone
                    break

            if touching_resistance:
                entry = candle['close']
                sl = max(candle['high'], touching_resistance['price']) * 1.0003
                risk = sl - entry

                if risk > 0:
                    target_support = None
                    min_dist = float('inf')
                    for zone in support_zones:
                        if zone['price'] < entry:
                            dist = entry - zone['price']
                            if dist < min_dist:
                                min_dist = dist
                                target_support = zone

                    if target_support and min_dist >= risk * MIN_RR:
                        tp = target_support['price']
                        result = backtest_signal(df, idx, entry, sl, tp, 'SHORT')

                        if result != 'active':
                            features = extract_features(df, idx, depth_data, whale_trades)
                            if features:
                                features['zone_distance'] = abs(candle['high'] - touching_resistance['price'])
                                features['zone_age'] = idx - touching_resistance['idx']
                                features['zone_strength'] = touching_resistance['strength']
                                features['risk_reward'] = min_dist / risk
                                features['is_short'] = 1

                                X.append(features)
                                y.append(1 if result == 'win' else 0)
                                short_count += 1

    # Calculate separate win rates
    long_wins = 0
    long_total = 0
    short_wins = 0
    short_total = 0

    for i, features in enumerate(X):
        if features.get('is_short', 0) == 1:
            short_total += 1
            if y[i] == 1:
                short_wins += 1
        else:
            long_total += 1
            if y[i] == 1:
                long_wins += 1

    long_winrate = (long_wins / long_total * 100) if long_total > 0 else 0
    short_winrate = (short_wins / short_total * 100) if short_total > 0 else 0
    total_winrate = (sum(y) / len(y) * 100) if y else 0

    print(f"Generated {len(X)} training samples (LONG: {long_count}, SHORT: {short_count})")
    print(f"Win rates - Total: {total_winrate:.1f}% | LONG: {long_winrate:.1f}% | SHORT: {short_winrate:.1f}%")

    stats = {
        'total_samples': len(X),
        'long_samples': long_total,
        'short_samples': short_total,
        'long_wins': long_wins,
        'short_wins': short_wins,
        'long_winrate': long_winrate,
        'short_winrate': short_winrate,
        'total_winrate': total_winrate
    }

    return pd.DataFrame(X), np.array(y), stats

def train_model(X, y):
    """Train XGBoost classifier."""
    if len(X) < 20:
        print("Not enough training samples!")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate LONG/SHORT accuracy separately
    long_correct = 0
    long_total = 0
    short_correct = 0
    short_total = 0

    for i in range(len(X_test)):
        is_short = X_test.iloc[i].get('is_short', 0) == 1
        correct = y_pred[i] == y_test.iloc[i] if hasattr(y_test, 'iloc') else y_pred[i] == y_test[i]

        if is_short:
            short_total += 1
            if correct:
                short_correct += 1
        else:
            long_total += 1
            if correct:
                long_correct += 1

    long_accuracy = (long_correct / long_total * 100) if long_total > 0 else 0
    short_accuracy = (short_correct / short_total * 100) if short_total > 0 else 0

    print(f"\nModel Accuracy: {accuracy*100:.1f}%")
    print(f"  LONG Accuracy: {long_accuracy:.1f}% ({long_correct}/{long_total})")
    print(f"  SHORT Accuracy: {short_accuracy:.1f}% ({short_correct}/{short_total})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    model_stats = {
        'accuracy': accuracy * 100,
        'long_accuracy': long_accuracy,
        'short_accuracy': short_accuracy,
        'long_test_count': long_total,
        'short_test_count': short_total
    }

    return model, model_stats

def save_model(model, feature_columns, path):
    """Save trained model and feature info."""
    data = {
        'model': model,
        'features': feature_columns.tolist()
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nModel saved to {path}")

def main():
    print("=" * 50)
    print("CryptoFlow ML Signal Predictor (Enhanced)")
    print("=" * 50)
    
    # Load candle data
    df = load_candles(DB_PATH, SYMBOL, TIMEFRAME)
    
    if len(df) < 100:
        print("Not enough candles for training!")
        return
    
    # Load additional data sources
    print("\nLoading enhanced data sources...")
    depth_data = load_depth_data(DB_PATH, 'BTCUSDT')
    whale_trades = load_whale_trades(DB_PATH, 'btcusdt', threshold=5.0)
    
    # Generate training data with all features
    X, y, data_stats = generate_training_data(df, depth_data, whale_trades)

    if len(X) == 0:
        print("No training samples generated!")
        return

    # Train model
    model, model_stats = train_model(X, y)
    
    if model:
        save_model(model, X.columns, MODEL_PATH)
        print("\n✅ Enhanced training complete!")
        print(f"Total features: {len(X.columns)}")
        print("New features: CVD, Depth Walls, Whale Activity")

if __name__ == '__main__':
    main()
