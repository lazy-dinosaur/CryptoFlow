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
        # Create feature vector in correct order
        X = pd.DataFrame([candle_features])[feature_columns]
        
        # Get prediction and probability
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        confidence = proba[1] if pred == 1 else proba[0]
        result = 'win' if pred == 1 else 'loss'
        
        return result, float(confidence)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

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
    print(f"  POST /api/ml/predict - Get prediction for signal")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
