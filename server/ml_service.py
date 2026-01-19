"""
CryptoFlow ML Service
- Prediction API using trained model
- Scheduled retraining every 6 hours
- Backtest tracking and status reporting
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Reuse functions from trainer
from ml_trainer import (
    load_candles, extract_features, find_support_zones, 
    find_resistance_zones, generate_training_data, train_model,
    save_model, DB_PATH, MODEL_PATH, SYMBOL, TIMEFRAME, MIN_RR, TOLERANCE
)

# Service Configuration
SERVICE_PORT = 5001
RETRAIN_INTERVAL_HOURS = 6
STATUS_DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ml_status.db')

# DB connection helper with timeout to prevent corruption
DB_TIMEOUT = 30  # seconds

def get_db_connection(db_path):
    """Create DB connection with proper timeout and WAL mode"""
    conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=30000')
    return conn

# Global state
model = None
feature_columns = None
last_training = None
last_accuracy = None
last_sample_count = None
training_history = []

def init_status_db():
    """Initialize status tracking database."""
    conn = get_db_connection(STATUS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sample_count INTEGER,
            accuracy REAL,
            win_rate REAL,
            top_features TEXT,
            long_samples INTEGER DEFAULT 0,
            short_samples INTEGER DEFAULT 0,
            long_accuracy REAL DEFAULT 0,
            short_accuracy REAL DEFAULT 0,
            long_winrate REAL DEFAULT 0,
            short_winrate REAL DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            exchange TEXT,
            direction TEXT,
            setup_type TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            rr REAL,
            zone_price REAL,
            confidence REAL,
            status TEXT DEFAULT 'ACTIVE',
            closed_at TEXT,
            pnl_percent REAL,
            -- Key metrics for quick queries
            delta REAL,
            delta_pct REAL,
            volume_ratio REAL,
            cvd_slope REAL,
            imbalance_ratio REAL,
            zone_age INTEGER,
            zone_strength REAL,
            whale_intensity REAL,
            -- Full features as JSON backup
            features_json TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("Status database initialized")

def load_model_from_disk():
    """Load the trained model from disk."""
    global model, feature_columns, last_training
    
    if not os.path.exists(MODEL_PATH):
        print("No trained model found!")
        return False
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        
        model = data['model']
        feature_columns = data['features']
        
        # Get model file modification time
        mod_time = os.path.getmtime(MODEL_PATH)
        last_training = datetime.fromtimestamp(mod_time).isoformat()
        
        print(f"Model loaded successfully (trained: {last_training})")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_signal(candle_features):
    """Make a prediction for a single signal."""
    global model, feature_columns
    
    if model is None:
        return None, 0.0
    
    try:
        # Create feature vector with defaults for missing columns (e.g. zone_distance)
        input_df = pd.DataFrame([candle_features])
        
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
                
        X = input_df[feature_columns]
        
        # Get prediction and probability
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        confidence = proba[1] if pred == 1 else proba[0]
        result = 'win' if pred == 1 else 'loss'
        
        return result, float(confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

def predict_raw_candles(candles, symbol='BTCUSDT', exchange='binance'):
    """
    Smart Prediction:
    1. Reconstruct Zones (Support/Resistance)
    2. Check if current candle triggers a Setup (Zone Touch + filters)
    3. If Setup Valid -> Extract Features -> Predict
    4. Return full trade plan (Entry, SL, TP) if confident
    5. Save signal to database for tracking
    """
    global model, feature_columns
    
    if model is None or not candles:
        return {'error': 'Model not ready'}, 500
        
    try:
        # 1. Prepare Data
        df = pd.DataFrame(candles)
        
        # Rename columns to snake_case
        column_map = {
            'buyVolume': 'buy_volume',
            'sellVolume': 'sell_volume',
            'tradeCount': 'trade_count'
        }
        df = df.rename(columns=column_map)
        
        if len(df) < 50:
             return {'signal': False, 'message': 'Need more history (50+ candles)'}, 200
             
        idx = len(df) - 1
        current_candle = df.iloc[idx]
        
        # 2. Find Zones
        support_zones = find_support_zones(df, idx)
        resistance_zones = find_resistance_zones(df, idx)
        
        # 3. Check for LONG Setup (Support Zone touch)
        touching_support = None
        for zone in support_zones:
            dist = abs(current_candle['low'] - zone['price'])
            if dist < current_candle['close'] * TOLERANCE:
                touching_support = zone
                break

        # 3b. Check for SHORT Setup (Resistance Zone touch)
        touching_resistance = None
        for zone in resistance_zones:
            dist = abs(current_candle['high'] - zone['price'])
            if dist < current_candle['close'] * TOLERANCE:
                touching_resistance = zone
                break

        # Determine direction based on zone touch and candle pattern
        direction = None
        touching_zone = None

        # LONG: Touching support + green candle + positive delta
        if touching_support:
            is_green = current_candle['close'] > current_candle['open']
            has_positive_delta = current_candle.get('delta', 0) > 0
            if is_green and has_positive_delta:
                direction = 'LONG'
                touching_zone = touching_support

        # SHORT: Touching resistance + red candle + negative delta
        if touching_resistance and direction is None:
            is_red = current_candle['close'] < current_candle['open']
            has_negative_delta = current_candle.get('delta', 0) < 0
            if is_red and has_negative_delta:
                direction = 'SHORT'
                touching_zone = touching_resistance

        if direction is None:
            return {'signal': False, 'message': 'Scanning... No valid zone touch'}, 200

        # 4. Calculate Trade Params based on direction
        entry = current_candle['close']

        if direction == 'LONG':
            # LONG: SL below support, TP at resistance
            sl = min(current_candle['low'], touching_zone['price']) * 0.9995
            risk = entry - sl

            if risk <= 0:
                return {'signal': False, 'message': 'Invalid Risk'}, 200

            # Find TP (Next Resistance)
            tp = None
            min_dist = float('inf')
            for zone in resistance_zones:
                if zone['price'] > entry:
                    dist = zone['price'] - entry
                    if dist < min_dist:
                        min_dist = dist
                        tp = zone['price']

            if not tp:
                tp = entry + (risk * 2)

            rr = (tp - entry) / risk
            setup_type = 'Support Bounce'

        else:  # SHORT
            # SHORT: SL above resistance, TP at support
            sl = max(current_candle['high'], touching_zone['price']) * 1.0005
            risk = sl - entry

            if risk <= 0:
                return {'signal': False, 'message': 'Invalid Risk'}, 200

            # Find TP (Next Support)
            tp = None
            min_dist = float('inf')
            for zone in support_zones:
                if zone['price'] < entry:
                    dist = entry - zone['price']
                    if dist < min_dist:
                        min_dist = dist
                        tp = zone['price']

            if not tp:
                tp = entry - (risk * 2)

            rr = (entry - tp) / risk
            setup_type = 'Resistance Rejection'

        if rr < 1.0:
            return {'signal': False, 'message': f'Bad RR ({rr:.2f})'}, 200

        # 5. Extract Features for Model
        features = extract_features(df, idx)
        if features is None:
            return {'error': 'Feature extraction failed'}, 500

        # Add Zone Features
        if direction == 'LONG':
            features['zone_distance'] = abs(current_candle['low'] - touching_zone['price'])
        else:
            features['zone_distance'] = abs(current_candle['high'] - touching_zone['price'])
        features['zone_age'] = idx - touching_zone['idx']
        features['zone_strength'] = touching_zone['strength']
        features['risk_reward'] = rr
        features['is_short'] = 1 if direction == 'SHORT' else 0

        # Ensure all model columns exist
        input_df = pd.DataFrame([features])
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0

        X = input_df[feature_columns]

        # 6. Predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba[1] if pred == 1 else proba[0]

        # 7. Final Output Decision
        if pred == 1 and confidence > 0.6:
            # Save signal to database
            save_signal(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                setup_type=setup_type,
                entry=float(entry),
                sl=float(sl),
                tp=float(tp),
                rr=float(rr),
                zone_price=float(touching_zone['price']),
                confidence=float(confidence),
                features=features
            )

            return {
                'signal': True,
                'direction': direction,
                'setupType': setup_type,
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'rr': float(rr),
                'confidence': float(confidence),
                'zonePrice': float(touching_zone['price'])
            }, 200
        else:
            return {'signal': False, 'message': f'Weak Signal ({confidence*100:.0f}%)'}, 200

    except Exception as e:
        print(f"Raw prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, 500

def run_training():
    """Run a full training cycle."""
    global model, feature_columns, last_training, last_accuracy, last_sample_count
    
    print("\n" + "="*50)
    print(f"Starting scheduled training at {datetime.now().isoformat()}")
    print("="*50)
    
    try:
        # Load latest data
        df = load_candles(DB_PATH, SYMBOL, TIMEFRAME)
        
        if len(df) < 100:
            print("Not enough candles for training")
            return False
        
        # Generate training data
        X, y, data_stats = generate_training_data(df)

        if len(X) < 20:
            print("Not enough training samples")
            return False

        # Train model
        trained_model, model_stats = train_model(X, y)

        if trained_model is None:
            return False

        # Save model
        save_model(trained_model, X.columns, MODEL_PATH)

        # Update global state
        model = trained_model
        feature_columns = X.columns.tolist()
        last_training = datetime.now().isoformat()

        last_accuracy = model_stats['accuracy'] / 100
        last_sample_count = len(X)

        # Save to history with LONG/SHORT stats
        save_training_record(
            sample_count=len(X),
            accuracy=model_stats['accuracy'] / 100,
            win_rate=data_stats['total_winrate'] / 100,
            top_features=X.columns.tolist()[:5],
            long_samples=data_stats['long_samples'],
            short_samples=data_stats['short_samples'],
            long_accuracy=model_stats['long_accuracy'],
            short_accuracy=model_stats['short_accuracy'],
            long_winrate=data_stats['long_winrate'],
            short_winrate=data_stats['short_winrate']
        )

        print(f"Training complete! Accuracy: {model_stats['accuracy']:.1f}%")
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_training_record(sample_count, accuracy, win_rate, top_features,
                         long_samples=0, short_samples=0,
                         long_accuracy=0, short_accuracy=0,
                         long_winrate=0, short_winrate=0):
    """Save training record to database."""
    conn = get_db_connection(STATUS_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO training_history
        (timestamp, sample_count, accuracy, win_rate, top_features,
         long_samples, short_samples, long_accuracy, short_accuracy,
         long_winrate, short_winrate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        sample_count,
        accuracy,
        win_rate,
        json.dumps(top_features),
        long_samples,
        short_samples,
        long_accuracy,
        short_accuracy,
        long_winrate,
        short_winrate
    ))

    conn.commit()
    conn.close()


def save_signal(symbol, exchange, direction, setup_type, entry, sl, tp, rr, zone_price, confidence, features=None):
    """Save a signal to the database with detailed metrics."""
    try:
        conn = get_db_connection(STATUS_DB_PATH)
        cursor = conn.cursor()

        # Extract key metrics from features
        delta = features.get('delta', 0) if features else 0
        delta_pct = features.get('delta_pct', 0) if features else 0
        volume_ratio = features.get('volume_ratio', 0) if features else 0
        cvd_slope = features.get('cvd_slope', 0) if features else 0
        imbalance_ratio = features.get('imbalance_ratio', 0) if features else 0
        zone_age = features.get('zone_age', 0) if features else 0
        zone_strength = features.get('zone_strength', 0) if features else 0
        whale_intensity = features.get('whale_intensity', 0) if features else 0
        features_json = json.dumps(features) if features else None

        cursor.execute("""
            INSERT INTO signals
            (timestamp, symbol, exchange, direction, setup_type, entry, sl, tp, rr, zone_price, confidence, status,
             delta, delta_pct, volume_ratio, cvd_slope, imbalance_ratio, zone_age, zone_strength, whale_intensity, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            symbol,
            exchange,
            direction,
            setup_type,
            entry,
            sl,
            tp,
            rr,
            zone_price,
            confidence,
            delta,
            delta_pct,
            volume_ratio,
            cvd_slope,
            imbalance_ratio,
            zone_age,
            zone_strength,
            whale_intensity,
            features_json
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"Signal saved: {direction} {symbol} @ {entry} (ID: {signal_id})")
        return signal_id
    except Exception as e:
        print(f"Error saving signal: {e}")
        return None


def get_signals(limit=50, direction=None, status=None):
    """Get recent signals from the database with all details."""
    try:
        conn = get_db_connection(STATUS_DB_PATH)
        cursor = conn.cursor()

        query = "SELECT * FROM signals WHERE 1=1"
        params = []

        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = ['id', 'timestamp', 'symbol', 'exchange', 'direction', 'setup_type',
                   'entry', 'sl', 'tp', 'rr', 'zone_price', 'confidence', 'status',
                   'closed_at', 'pnl_percent',
                   'delta', 'delta_pct', 'volume_ratio', 'cvd_slope', 'imbalance_ratio',
                   'zone_age', 'zone_strength', 'whale_intensity', 'features_json']

        signals = []
        for row in rows:
            signal = dict(zip(columns, row))
            # Parse features_json if exists
            if signal.get('features_json'):
                try:
                    signal['features'] = json.loads(signal['features_json'])
                except:
                    signal['features'] = {}
            del signal['features_json']  # Remove raw JSON from response
            signals.append(signal)

        return signals
    except Exception as e:
        print(f"Error getting signals: {e}")
        return []


def get_signal_stats():
    """Get signal statistics summary."""
    try:
        conn = get_db_connection(STATUS_DB_PATH)
        cursor = conn.cursor()

        # Total counts by direction
        cursor.execute("""
            SELECT
                direction,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) as active,
                AVG(confidence) as avg_confidence,
                AVG(rr) as avg_rr
            FROM signals
            GROUP BY direction
        """)
        rows = cursor.fetchall()
        conn.close()

        stats = {'LONG': {}, 'SHORT': {}, 'total': 0}
        for row in rows:
            direction = row[0]
            stats[direction] = {
                'total': row[1],
                'wins': row[2] or 0,
                'losses': row[3] or 0,
                'active': row[4] or 0,
                'avgConfidence': row[5] or 0,
                'avgRR': row[6] or 0,
                'winRate': (row[2] / (row[2] + row[3]) * 100) if (row[2] and row[3] and row[2] + row[3] > 0) else 0
            }
            stats['total'] += row[1]

        return stats
    except Exception as e:
        print(f"Error getting signal stats: {e}")
        return {'LONG': {}, 'SHORT': {}, 'total': 0}


def get_status():
    """Get current ML service status."""
    global last_training, last_accuracy, last_sample_count

    # Get training history from DB
    history = []
    latest_stats = {}
    try:
        conn = get_db_connection(STATUS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, sample_count, accuracy, win_rate,
                   long_samples, short_samples, long_accuracy, short_accuracy,
                   long_winrate, short_winrate
            FROM training_history
            ORDER BY id DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        conn.close()

        for i, row in enumerate(rows):
            entry = {
                'timestamp': row[0],
                'samples': row[1],
                'accuracy': row[2],
                'winRate': row[3],
                'longSamples': row[4] or 0,
                'shortSamples': row[5] or 0,
                'longAccuracy': row[6] or 0,
                'shortAccuracy': row[7] or 0,
                'longWinrate': row[8] or 0,
                'shortWinrate': row[9] or 0
            }
            history.append(entry)

            # Get latest stats from most recent entry
            if i == 0:
                latest_stats = {
                    'longSamples': entry['longSamples'],
                    'shortSamples': entry['shortSamples'],
                    'longAccuracy': entry['longAccuracy'],
                    'shortAccuracy': entry['shortAccuracy'],
                    'longWinrate': entry['longWinrate'],
                    'shortWinrate': entry['shortWinrate']
                }
    except Exception as e:
        print(f"Error getting history: {e}")

    return {
        'modelLoaded': model is not None,
        'lastTraining': last_training,
        'accuracy': last_accuracy,
        'sampleCount': last_sample_count,
        'nextTraining': calculate_next_training(),
        'history': history,
        **latest_stats
    }

def calculate_next_training():
    """Calculate when next training will occur."""
    if last_training is None:
        return None
    
    try:
        last_dt = datetime.fromisoformat(last_training)
        next_dt = last_dt.replace(
            hour=(last_dt.hour + RETRAIN_INTERVAL_HOURS) % 24
        )
        if next_dt < datetime.now():
            next_dt = next_dt.replace(day=next_dt.day + 1)
        return next_dt.isoformat()
    except:
        return None

def training_scheduler():
    """Background thread for scheduled retraining."""
    while True:
        # Wait for retrain interval
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)
        
        # Run training
        run_training()

class MLAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for ML API endpoints."""
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == '/api/ml/status':
            self.send_json(get_status())
        elif parsed.path == '/api/ml/train':
            # Trigger manual training
            success = run_training()
            self.send_json({'success': success, 'status': get_status()})
        elif parsed.path == '/api/ml/signals':
            # Get signals with optional filtering
            limit = int(query_params.get('limit', [50])[0])
            direction = query_params.get('direction', [None])[0]
            status = query_params.get('status', [None])[0]
            signals = get_signals(limit=limit, direction=direction, status=status)
            stats = get_signal_stats()
            self.send_json({'signals': signals, 'stats': stats})
        else:
            self.send_error(404, 'Not found')
    
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == '/api/ml/predict':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            prediction, confidence = predict_signal(data.get('features', {}))
            
            self.send_json({
                'prediction': prediction,
                'confidence': confidence
            })
        elif parsed.path == '/api/ml/predict_raw':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            candles = data.get('candles', [])
            symbol = data.get('symbol', 'BTCUSDT')
            exchange = data.get('exchange', 'binance')
            result = predict_raw_candles(candles, symbol, exchange)

            # predict_raw_candles returns tuple (dict, status_code)
            if isinstance(result, tuple):
                response_data, status_code = result
                if status_code == 200:
                    self.send_json(response_data)
                else:
                    # Error case
                    error_msg = response_data.get('error', 'Unknown error')
                    self.send_error(status_code, error_msg)
            else:
                self.send_json(result)
        else:
            self.send_error(404, 'Not found')
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    print("="*50)
    print("CryptoFlow ML Service")
    print("="*50)
    
    # Initialize
    init_status_db()
    load_model_from_disk()
    
    # Run initial training if no model
    if model is None:
        print("No model found, running initial training...")
        run_training()
    
    # Start training scheduler
    scheduler_thread = threading.Thread(target=training_scheduler, daemon=True)
    scheduler_thread.start()
    print(f"Training scheduler started (every {RETRAIN_INTERVAL_HOURS}h)")
    
    # Start HTTP server
    # BIND TO LOCALHOST ONLY for security (Node proxy handles external access)
    server = HTTPServer(('127.0.0.1', SERVICE_PORT), MLAPIHandler)
    print(f"ML API server running on port {SERVICE_PORT}")
    print("\nEndpoints:")
    print(f"  GET  /api/ml/status  - Get ML status and history")
    print(f"  GET  /api/ml/train   - Trigger manual retraining")
    print(f"  GET  /api/ml/signals - Get signal history (?direction=LONG&limit=50)")
    print(f"  POST /api/ml/predict - Get prediction for signal")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
