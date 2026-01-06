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
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

# Reuse functions from trainer
from ml_trainer import (
    load_candles, extract_features, find_support_zones, 
    find_resistance_zones, generate_training_data, train_model,
    save_model, DB_PATH, MODEL_PATH, SYMBOL, TIMEFRAME, MIN_RR, TOLERANCE
)
# ... (imports)
try:
    from chart_renderer import ChartRenderer
    renderer = ChartRenderer()
    HAS_RENDERER = True
except ImportError as e:
    print(f"ChartRenderer not available: {e}")
    HAS_RENDERER = False

try:
    from orderflow_bot import OrderflowBot
    of_bot = OrderflowBot()
    HAS_BOT = True
except ImportError as e:
    print(f"OrderflowBot not available: {e}")
    HAS_BOT = False

# ...



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
    1. Check OrderflowBot (Absorption/Logic).
    2. If no setup, check ML Model.
    """
    global model, feature_columns
    
    if not candles:
        return {'error': 'No candles data'}, 500
        
    try:
        # 1. Orderflow Bot Check (Priority)
        if HAS_BOT:
            bot_result = of_bot.analyze(candles)
            if bot_result.get('signal'):
                return bot_result, 200

        # ... (Existing Model Logic)
        if model is None:
             return {'error': 'Model not ready'}, 500
             
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
        
        # 3. Check Sequence: Is price touching a support/resistance zone?
        # Using the same logic as training
        touching_support = None
        touching_resistance = None
        
        for zone in support_zones:
            dist = abs(current_candle['low'] - zone['price'])
            if dist < current_candle['close'] * TOLERANCE:
                touching_support = zone
                break

        for zone in resistance_zones:
            dist = abs(current_candle['high'] - zone['price'])
            if dist < current_candle['close'] * TOLERANCE:
                touching_resistance = zone
                break
                
        if not touching_support and not touching_resistance:
            return {'signal': False, 'message': 'Scanning... No Zone touch'}, 200
            
        # 4. Calculate Trade Params
        signal_type = 'LONG' if touching_support else 'SHORT'
        entry = current_candle['close']
        
        if signal_type == 'LONG':
            sl = min(current_candle['low'], touching_support['price']) * 0.9995
            risk = entry - sl
            if risk <= 0: return {'signal': False, 'message': 'Invalid Risk'}, 200
            
            # Find TP (Next Resistance)
            tp = None
            min_dist = float('inf')
            for z in resistance_zones:
                if z['price'] > entry:
                    d = z['price'] - entry
                    if d < min_dist:
                         min_dist = d
                         tp = z['price']
            if not tp: tp = entry + (risk * 2)

        else: # SHORT
            sl = max(current_candle['high'], touching_resistance['price']) * 1.0005
            risk = sl - entry
            if risk <= 0: return {'signal': False, 'message': 'Invalid Risk'}, 200
            
            # Find TP (Next Support)
            tp = None
            min_dist = float('inf')
            for z in support_zones:
                if z['price'] < entry:
                    d = entry - z['price']
                    if d < min_dist:
                         min_dist = d
                         tp = z['price']
            if not tp: tp = entry - (risk * 2)
            
        rr = abs(tp - entry) / risk
        if rr < 1.0: # Filter bad RR
             return {'signal': False, 'message': f'Bad RR ({rr:.2f})'}, 200
             
        # 5. Extract Features
        features = extract_features(df, idx)
        if features is None:
            return {'error': 'Feature extraction failed'}, 500
            
        # Add Zone Features
        target_zone = touching_support if signal_type == 'LONG' else touching_resistance
        features['zone_distance'] = abs(current_candle['low'] - target_zone['price']) if signal_type == 'LONG' else abs(current_candle['high'] - target_zone['price'])
        features['zone_age'] = idx - target_zone['idx']
        features['zone_strength'] = target_zone['strength']
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
        # XGBoost output: [prob_class0, prob_class1] (0=Loss, 1=Win)
        # We need confidence for WIN.
        confidence = proba[1]
        
        # 7. Final Output Decision
        # Threshold lowered to 0.55 for more activity.
        # Also trust the "Class 1" prediction.
        if confidence > 0.55:
            return {
                'signal': True,
                'direction': signal_type,
                'setupType': 'Zone Rejection',
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'rr': float(rr),
                'confidence': float(confidence),
                'zonePrice': float(target_zone['price'])
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
    Combines Bot and ML signals, calculates Win/Loss outcomes, and returns stats.
    """
    global model, feature_columns
    
    if model is None or not candles:
        return {'error': 'Model not ready'}, 500
        
    try:
        signals = []
        df = pd.DataFrame(candles)
        column_map = {'buyVolume': 'buy_volume', 'sellVolume': 'sell_volume', 'tradeCount': 'trade_count'}
        df = df.rename(columns=column_map)
        
        # 1. Orderflow Bot Backtest
        if HAS_BOT:
            try:
                bot_signals = of_bot.backtest(candles)
                signals.extend(bot_signals)
            except Exception as e:
                print(f"Bot backtest error: {e}")

        # 2. ML Model Backtest
        for i in range(50, len(df)):
             try:
                # Optimized: Only run ML check if no Bot signal at this candle (priority)
                # But for now, let's run parallel
                
                # Check for Zone Touch
                current_candle = df.iloc[i]
                support_zones = find_support_zones(df, i)
                resistance_zones = find_resistance_zones(df, i)
                
                touching_support = None
                touching_resistance = None
                
                for zone in support_zones:
                    if abs(current_candle['low'] - zone['price']) < current_candle['close'] * TOLERANCE:
                        touching_support = zone
                        break

                for zone in resistance_zones:
                    if abs(current_candle['high'] - zone['price']) < current_candle['close'] * TOLERANCE:
                        touching_resistance = zone
                        break
                        
                if not touching_support and not touching_resistance:
                    continue
                
                # Setup Type
                signal_type = 'LONG' if touching_support else 'SHORT'
                entry = current_candle['close']
                target_zone = touching_support if signal_type == 'LONG' else touching_resistance
                
                # Trade Params
                if signal_type == 'LONG':
                    sl = min(current_candle['low'], touching_support['price']) * 0.9995
                    risk = entry - sl
                    if risk <= 0: continue
                    tp = entry + (risk * 2) # Default 2R
                else: 
                    sl = max(current_candle['high'], touching_resistance['price']) * 1.0005
                    risk = sl - entry
                    if risk <= 0: continue
                    tp = entry - (risk * 2)

                rr = abs(tp - entry) / risk
                if rr < 1.0: continue
                
                # Predict
                features = extract_features(df, i)
                if features is None: continue
                
                features['zone_distance'] = abs(current_candle['low'] - target_zone['price']) if signal_type == 'LONG' else abs(current_candle['high'] - target_zone['price'])
                features['zone_age'] = i - target_zone['idx']
                features['zone_strength'] = target_zone['strength']
                features['risk_reward'] = rr
                
                input_df = pd.DataFrame([features])
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0.0
                X = input_df[feature_columns]
                
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = proba[1]
                
                if confidence > 0.55:
                    signals.append({
                        'candleTime': int(current_candle['time']),
                        'type': signal_type,
                        'setupType': 'Zone Rejection',
                        'entry': float(entry),
                        'sl': float(sl),
                        'tp': float(tp),
                        'rr': float(rr),
                        'confidence': float(confidence),
                        'zonePrice': float(target_zone['price']),
                        'source': 'ML Model' 
                    })
             except: continue
        
        # 3. Calculate Outcomes (Win/Loss)
        stats = {
            'total': 0, 'wins': 0, 'losses': 0, 'winRate': 0,
            'botTotal': 0, 'botWins': 0, 
            'mlTotal': 0, 'mlWins': 0
        }
        
        for sig in signals:
            sig_idx = df[df['time'] == sig['candleTime']].index[0]
            outcome = 'PENDING'
            
            # Check future candles
            for future_idx in range(sig_idx + 1, len(df)):
                future_candle = df.iloc[future_idx]
                
                if sig['type'] == 'LONG':
                    if future_candle['low'] <= sig['sl']:
                        outcome = 'LOSS'
                        break
                    if future_candle['high'] >= sig['tp']:
                        outcome = 'WIN'
                        break
                else:
                    if future_candle['high'] >= sig['sl']:
                        outcome = 'LOSS'
                        break
                    if future_candle['low'] <= sig['tp']:
                        outcome = 'WIN'
                        break
            
            sig['result'] = outcome
            
            # Update Stats
            if outcome != 'PENDING':
                stats['total'] += 1
                if outcome == 'WIN': stats['wins'] += 1
                
                if sig.get('source') == 'OrderflowBot':
                    stats['botTotal'] += 1
                    if outcome == 'WIN': stats['botWins'] += 1
                else:
                    stats['mlTotal'] += 1
                    if outcome == 'WIN': stats['mlWins'] += 1

        # Final Rates
        stats['winRate'] = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0
        stats['botWinRate'] = (stats['botWins'] / stats['botTotal'] * 100) if stats['botTotal'] > 0 else 0
        stats['mlWinRate'] = (stats['mlWins'] / stats['mlTotal'] * 100) if stats['mlTotal'] > 0 else 0

        # Sort signals by time
        signals.sort(key=lambda x: x['candleTime'])

        return {'signals': signals, 'stats': stats}, 200
        
    except Exception as e:
        print(f"Backtest error: {e}")
        return {'error': str(e)}, 500
                    



def load_env_file():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

def analyze_with_ai(data):
    """
    Send chart data to AI for analysis.
    Basic prompt engineering + Multi-TF Context.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model_name = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp")

    if not api_key:
        return {"explanation": "Server Error: Missing API Key", "action": "HOLD"}, 500

    # Extract Data
    img_data = data.get('image') # Frontend Screenshot
    context = data.get('context', {})
    symbol = context.get('symbol', 'BTCUSDT')
    price = context.get('price', 0)
    tf = context.get('timeframe', 1)
    
    # Generate Backend Multi-TF Chart
    backend_img = None
    if HAS_RENDERER:
        try:
            backend_img = renderer.render_multi_timeframe(symbol)
            if backend_img:
                print("Generated Multi-TF Context Image")
        except Exception as e:
            print(f"Rendering failed: {e}")

    prompt = f"""
    Acting as a Professional Crypto Scalper/Daytrader.
    
    **Market Context:**
    - Symbol: {symbol}
    - Current Price: {price}
    - Active Timeframe: {tf}m
    
    **Task:**
    Analyze the provided chart image(s).
    Image 1: Direct User View (Current State including indicators/heatmap).
    Image 2: Multi-Timeframe Context (15m, 5m, 1m candles) - Use this for trend confirmation.

    1. Identify key Support/Resistance levels.
    2. Analyze Price Action & Volume structure.
    3. Determine the immediate trend (Bullish/Bearish/Neutral).
    4. DECIDE: BUY, SELL, or HOLD.
    
    **Signal Rules (If BUY/SELL):**
    - Entry: Must be near current price.
    - Stop Loss: Logical invalidation point.
    - Take Profit: Realistic target (RR > 1.5).
    
    **Output Format (JSON ONLY):**
    {{
        "explanation": "Brief, punchy analysis (max 3 sentences). Why enter? What is the trend?",
        "action": "BUY" | "SELL" | "HOLD",
        "trade": {{ "entry": 123.45, "sl": 120.00, "tp": 130.00 }}  (Optional, null if HOLD)
    }}
    """
    
    # Build Content Array
    content = [{"type": "text", "text": prompt}]
    
    # Image 1 (Frontend)
    if img_data:
        content.append({
            "type": "image_url",
            "image_url": {"url": img_data}
        })
        
    # Image 2 (Backend Context)
    if backend_img:
         content.append({
            "type": "image_url",
            "image_url": {"url": backend_img}
        })

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            # Parse AI Response
            if 'choices' in result and len(result['choices']) > 0:
                ai_text = result['choices'][0]['message']['content']
                # Clean markdown code blocks if any
                ai_text = ai_text.replace('```json', '').replace('```', '').strip()
                try:
                    return json.loads(ai_text), 200
                except:
                    return {"explanation": ai_text, "action": "HOLD"}, 200
            else:
                return {"explanation": "AI returned empty response", "action": "HOLD"}, 200

    except Exception as e:
        print(f"AI Req Error: {e}")
        return {"error": str(e)}, 500
                


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
        
        elif parsed.path == '/api/ai/analyze':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            
            result, status_code = analyze_with_ai(data)
            if status_code == 200:
                self.send_json(result)
            else:
                self.send_error(status_code, result.get('error', 'AI Error'))
                
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
    load_env_file()
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
    print(f"  POST /api/ai/analyze    - AI Agent Vision Analysis")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(f"Crash at {datetime.now()}:\n")
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print("CRITICAL ERROR: See crash_log.txt")
