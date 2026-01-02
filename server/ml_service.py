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

# Global state
model = None
feature_columns = None
last_training = None
last_accuracy = None
last_sample_count = None
training_history = []

def init_status_db():
    """Initialize status tracking database."""
    conn = sqlite3.connect(STATUS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sample_count INTEGER,
            accuracy REAL,
            win_rate REAL,
            top_features TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            candle_time INTEGER,
            confidence REAL,
            prediction TEXT,
            actual_result TEXT
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

def predict_raw_candles(candles):
    """
    Smart Prediction:
    1. Reconstruct Zones (Support/Resistance)
    2. Check if current candle triggers a Setup (Zone Touch + filters)
    3. If Setup Valid -> Extract Features -> Predict
    4. Return full trade plan (Entry, SL, TP) if confident
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
        
        # 3. Check Sequence: Is price touching a support zone? (LONG Setup)
        # Using the same logic as training
        touching_support = None
        for zone in support_zones:
            # Check proximity (using TOLERANCE from trainer)
            dist = abs(current_candle['low'] - zone['price'])
            if dist < current_candle['close'] * TOLERANCE:
                touching_support = zone
                break
                
        if not touching_support:
            return {'signal': False, 'message': 'Scanning... No Support Zone touch'}, 200
            
        # 4. Calculate Trade Params (Entry, SL, TP)
        entry = current_candle['close']
        # SL below zone or low
        sl = min(current_candle['low'], touching_support['price']) * 0.9995
        risk = entry - sl
        
        if risk <= 0:
            return {'signal': False, 'message': 'Invalid Risk'}, 200
            
        # Find TP (Next Resistance)
        tp = None
        min_dist_to_res = float('inf')
        
        for zone in resistance_zones:
            if zone['price'] > entry:
                dist = zone['price'] - entry
                if dist < min_dist_to_res:
                    min_dist_to_res = dist
                    tp = zone['price']
                    
        # Default TP if no resistance found (1:2 RR)
        if not tp:
            tp = entry + (risk * 2)
            
        rr = (tp - entry) / risk
        if rr < 1.0: # Filter bad RR
             return {'signal': False, 'message': f'Bad RR ({rr:.2f})'}, 200
             
        # 5. Extract Features for Model
        # We must manually inject the zone features that regular extract_features might miss
        features = extract_features(df, idx)
        if features is None:
            return {'error': 'Feature extraction failed'}, 500
            
        # Add Zone Features
        features['zone_distance'] = abs(current_candle['low'] - touching_support['price'])
        features['zone_age'] = idx - touching_support['idx']
        features['zone_strength'] = touching_support['strength']
        features['risk_reward'] = rr
        
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
        # Only signal if Model predicts WIN with high confidence OR if it matches our basic setup
        # User wants "sure" -> Threshold 0.6
        if pred == 1 and confidence > 0.6:
            return {
                'signal': True,
                'direction': 'LONG',
                'setupType': 'Support Bounce',
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'rr': float(rr),
                'confidence': float(confidence),
                'zonePrice': float(touching_support['price'])
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
        X, y = generate_training_data(df)
        
        if len(X) < 20:
            print("Not enough training samples")
            return False
        
        # Train model
        trained_model = train_model(X, y)
        
        if trained_model is None:
            return False
        
        # Save model
        save_model(trained_model, X.columns, MODEL_PATH)
        
        # Update global state
        model = trained_model
        feature_columns = X.columns.tolist()
        last_training = datetime.now().isoformat()
        
        # Calculate stats
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        last_accuracy = float(accuracy)
        last_sample_count = len(X)
        
        # Save to history
        save_training_record(len(X), accuracy, sum(y)/len(y), X.columns.tolist()[:5])
        
        print(f"Training complete! Accuracy: {accuracy*100:.1f}%")
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_training_record(sample_count, accuracy, win_rate, top_features):
    """Save training record to database."""
    conn = sqlite3.connect(STATUS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO training_history (timestamp, sample_count, accuracy, win_rate, top_features)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        sample_count,
        accuracy,
        win_rate,
        json.dumps(top_features)
    ))
    
    conn.commit()
    conn.close()

def get_status():
    """Get current ML service status."""
    global last_training, last_accuracy, last_sample_count
    
    # Get training history from DB
    history = []
    try:
        conn = sqlite3.connect(STATUS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, sample_count, accuracy, win_rate 
            FROM training_history 
            ORDER BY id DESC 
            LIMIT 10
        """)
        rows = cursor.fetchall()
        conn.close()
        
        history = [
            {
                'timestamp': row[0],
                'samples': row[1],
                'accuracy': row[2],
                'winRate': row[3]
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error getting history: {e}")
    
    return {
        'modelLoaded': model is not None,
        'lastTraining': last_training,
        'accuracy': last_accuracy,
        'sampleCount': last_sample_count,
        'nextTraining': calculate_next_training(),
        'history': history
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

def backtest_signals(candles):
    """
    Run prediction logic on historical candles to find past signals.
    """
    global model, feature_columns
    
    if model is None or not candles:
        return {'error': 'Model not ready'}, 500
        
    try:
        df = pd.DataFrame(candles)
        column_map = {
            'buyVolume': 'buy_volume',
            'sellVolume': 'sell_volume',
            'tradeCount': 'trade_count'
        }
        df = df.rename(columns=column_map)
        
        if len(df) < 50:
            return {'signals': []}, 200
            
        # Pre-calc features for speed
        # We need to capture TA-Lib errors if df is small, but check above handles that
        try:
            # We use the trainer's extract_features but it only returns a single row.
            # For speed, we should implement a bulk feature extraction if possible,
            # BUT our logic relies on `find_support_zones` which is point-in-time.
            # So simple loop is safest for correctness.
            pass
        except Exception as e:
            pass

        signals = []
        
        # Iterate from index 50 to end
        # Skip last candle (live) as that's handled by main loop
        for i in range(50, len(df)):
            try:
                # 1. Simulate "Live" State at index i
                # Create a mini-slice for zone detection? 
                # find_support_zones(df, idx) already handles looking ONLY at df[:idx+1] implicitly?
                # Actually find_support_zones(df, i) uses i as the "current" index.
                # It looks at pivots BEFORE i.
                
                # We need to run the EXACT same logic as predict_raw_candles but for index i
                current_candle = df.iloc[i]
                
                support_zones = find_support_zones(df, i)
                resistance_zones = find_resistance_zones(df, i)
                
                # Check for Support Touch
                touching_support = None
                for zone in support_zones:
                    dist = abs(current_candle['low'] - zone['price'])
                    if dist < current_candle['close'] * TOLERANCE:
                        touching_support = zone
                        break
                
                if not touching_support:
                    continue
                    
                # Calculate Trade Params
                entry = current_candle['close']
                sl = min(current_candle['low'], touching_support['price']) * 0.9995
                risk = entry - sl
                
                if risk <= 0: continue
                
                # TP
                tp = None
                min_dist_to_res = float('inf')
                for zone in resistance_zones:
                    if zone['price'] > entry:
                        dist = zone['price'] - entry
                        if dist < min_dist_to_res:
                            min_dist_to_res = dist
                            tp = zone['price']
                
                if not tp: tp = entry + (risk * 2)
                
                rr = (tp - entry) / risk
                if rr < 1.0: continue
                
                # Features
                features = extract_features(df, i)
                if features is None: continue
                
                features['zone_distance'] = abs(current_candle['low'] - touching_support['price'])
                features['zone_age'] = i - touching_support['idx']
                features['zone_strength'] = touching_support['strength']
                features['risk_reward'] = rr
                
                # Predict
                input_df = pd.DataFrame([features])
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                X = input_df[feature_columns]
                
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = proba[1] if pred == 1 else proba[0]
                
                if pred == 1 and confidence > 0.6:
                    signals.append({
                        'candleTime': int(current_candle['time']), # JS timestamp
                        'type': 'LONG',
                        'entry': float(entry),
                        'sl': float(sl),
                        'tp': float(tp),
                        'rr': float(rr),
                        'confidence': float(confidence),
                        'zonePrice': float(touching_support['price'])
                    })
                    
            except Exception as loop_err:
                continue

        return {'signals': signals}, 200

    except Exception as e:
        print(f"Backtest error: {e}")
        return {'error': str(e)}, 500

class MLAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for ML API endpoints."""
    
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == '/api/ml/status':
            self.send_json(get_status())
        elif parsed.path == '/api/ml/train':
            # Trigger manual training
            success = run_training()
            self.send_json({'success': success, 'status': get_status()})
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
            result = predict_raw_candles(candles)
            
            # Result is (response_dict, status_code)
            response_data, status_code = result
            
            if status_code == 200:
                self.send_json(response_data)
            else:
                self.send_error(status_code, response_data.get('error', 'Unknown error'))
                
        elif parsed.path == '/api/ml/backtest':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            candles = data.get('candles', [])
            result = backtest_signals(candles)
            
            response_data, status_code = result
            if status_code == 200:
                self.send_json(response_data)
            else:
                self.send_error(status_code, response_data.get('error', 'Unknown error'))
                
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
    print(f"  GET  /api/ml/status     - Get ML status and history")
    print(f"  GET  /api/ml/train      - Trigger manual retraining")
    print(f"  POST /api/ml/predict    - Get prediction for signal")
    print(f"  POST /api/ml/backtest   - Scan historical data for signals")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
