@echo off
REM CryptoFlow Quick Start for Windows
REM Double-click to start the server

echo Starting CryptoFlow Data Collector...
cd /d "%~dp0server"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

echo.
echo ========================================
echo   CryptoFlow Data Collector
echo   API: http://localhost:4001
echo   WS:  ws://localhost:4002
echo ========================================
echo.
echo Press Ctrl+C to stop
echo.

echo Starting ML Service...
start "CryptoFlow ML Brain" python ml_service.py
timeout /t 5

node api.js
