import os
import sys
import socket

print(f"Python: {sys.version}")

try:
    import pandas
    print("Pandas: OK")
except ImportError as e:
    print(f"Pandas Error: {e}")

try:
    import xgboost
    print("XGBoost: OK")
except ImportError as e:
    print(f"XGBoost Error: {e}")

db_path = os.path.join(os.path.dirname(__file__), 'data', 'cryptoflow.db')
print(f"DB Path: {db_path}")
if os.path.exists(db_path):
    print(f"DB Exists. Size: {os.path.getsize(db_path)} bytes")
else:
    print("DB DOES NOT EXIST!")

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 5001))
    s.close()
    print("Port 5001 binding: OK")
except Exception as e:
    print(f"Port 5001 Error: {e}")
