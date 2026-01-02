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

def load_candles(db_path, symbol, timeframe):
    """Load candles from SQLite database."""
    conn = sqlite3.connect(db_path)
    
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
    """Find active support zones at candle index based on Swing Lows."""
    zones = []
    
    # Look back 50 candles for any valid Swing Lows
    for i in range(max(2, idx-50), idx-2):
        candle = df.iloc[i]
        
        # Fractal / Pivot Low (5-bar)
        # low[i] is lower than 2 left and 2 right
        l = df.iloc[i]['low']
        if (l < df.iloc[i-1]['low'] and 
            l < df.iloc[i-2]['low'] and 
            l < df.iloc[i+1]['low'] and 
            l < df.iloc[i+2]['low']):
            
            zone_price = l
            
            # Check if zone is still valid (not broken)
            is_broken = False
            # It's broken if a subsequent candle closes BELOW it
            for j in range(i+1, idx):
                if df.iloc[j]['close'] < zone_price * 0.9995: # slight buffer
                    is_broken = True
                    break
            
            if not is_broken:
                zones.append({
                    'price': zone_price,
                    'idx': i,
                    'strength': candle['volume'] # Simple strength proxy
                })
    
    return zones

def find_resistance_zones(df, idx):
    """Find active resistance zones at candle index based on Swing Highs."""
    zones = []
    
    for i in range(max(2, idx-50), idx-2):
        candle = df.iloc[i]
        
        # Fractal / Pivot High (5-bar)
        h = df.iloc[i]['high']
        if (h > df.iloc[i-1]['high'] and 
            h > df.iloc[i-2]['high'] and 
            h > df.iloc[i+1]['high'] and 
            h > df.iloc[i+2]['high']):
            
            zone_price = h
            
            # Check if zone is still valid (not broken)
            # It's broken if a subsequent candle closes ABOVE it
            is_broken = False
            for j in range(i+1, idx):
                if df.iloc[j]['close'] > zone_price * 1.0005:
                    is_broken = True
                    break
            
            if not is_broken:
                zones.append({
                    'price': zone_price,
                    'idx': i,
                    'strength': candle['volume']
                })
    
    return zones

def backtest_signal(df, idx, entry, sl, tp):
    """Backtest a signal and return result."""
    for j in range(idx + 1, len(df)):
        future = df.iloc[j]
        if future['low'] <= sl:
            return 'loss'
        if future['high'] >= tp:
            return 'win'
    return 'active'

def load_depth_data(db_path, symbol='BTCUSDT'):
    """Load depth heatmap snapshots from database."""
    try:
        conn = sqlite3.connect(db_path)
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
        conn = sqlite3.connect(db_path)
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
    """Generate training dataset with features and labels."""
    X = []
    y = []
    
    print("Generating training data with enhanced features...")
    
    for idx in range(10, len(df) - 10):  # Need buffer on both sides
        candle = df.iloc[idx]
        
        # Basic filters
        # REMOVED PRE-FILTERS: Allow ML to learn from red candles and negative delta too
        # We only care if it bounced from support

        
        # Find support zones
        support_zones = find_support_zones(df, idx)
        
        # Check if candle touches any support zone
        touching_support = None
        for zone in support_zones:
            if abs(candle['low'] - zone['price']) < candle['close'] * TOLERANCE:
                touching_support = zone
                break
        
        if not touching_support:
            continue
        
        # Define entry, SL, TP
        entry = candle['close']
        sl = min(candle['low'], touching_support['price']) * 0.9997
        risk = entry - sl
        
        if risk <= 0:
            continue
        
        # Find resistance for TP
        resistance_zones = find_resistance_zones(df, idx)
        
        target_resistance = None
        min_dist = float('inf')
        for zone in resistance_zones:
            if zone['price'] > entry:
                dist = zone['price'] - entry
                if dist < min_dist:
                    min_dist = dist
                    target_resistance = zone
        
        if not target_resistance or min_dist < risk * MIN_RR:
            continue
        
        tp = target_resistance['price']
        
        # Backtest the signal
        result = backtest_signal(df, idx, entry, sl, tp)
        
        if result == 'active':
            continue  # Skip signals without clear outcome
        
        # Extract features WITH new data sources
        features = extract_features(df, idx, depth_data, whale_trades)
        if features is None:
            continue
        
        # Add zone-specific features
        features['zone_distance'] = abs(candle['low'] - touching_support['price'])
        features['zone_age'] = idx - touching_support['idx']
        features['zone_strength'] = touching_support['strength']
        features['risk_reward'] = min_dist / risk
        
        X.append(features)
        y.append(1 if result == 'win' else 0)
    
    print(f"Generated {len(X)} training samples")
    print(f"Win rate: {sum(y)/len(y)*100:.1f}%" if y else "No samples")
    
    return pd.DataFrame(X), np.array(y)

def train_model(X, y):
    """Train XGBoost classifier."""
    if len(X) < 20:
        print("Not enough training samples!")
        return None
    
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
    
    print(f"\nModel Accuracy: {accuracy*100:.1f}%")
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
    
    return model

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
    X, y = generate_training_data(df, depth_data, whale_trades)
    
    if len(X) == 0:
        print("No training samples generated!")
        return
    
    # Train model
    model = train_model(X, y)
    
    if model:
        save_model(model, X.columns, MODEL_PATH)
        print("\n✅ Enhanced training complete!")
        print(f"Total features: {len(X.columns)}")
        print("New features: CVD, Depth Walls, Whale Activity")

if __name__ == '__main__':
    main()
