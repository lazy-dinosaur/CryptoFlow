@echo off
:: CryptoFlow Server - Auto-Restart Script
:: Keeps the server running permanently, restarts on crash
:: Double-click to start, close window to stop

title CryptoFlow Server
color 0A

cd /d C:\CryptoFlow\server

:start
echo.
echo ============================================
echo   CryptoFlow Server Starting...
echo   %date% %time%
echo ============================================
echo.

:: CLEANUP: Kill any lingering Node.js processes to free port 443
:: This prevents "Address already in use" errors on restart
taskkill /F /IM node.exe >nul 2>&1

:: Start the server
node api.js

echo.
echo ============================================
echo   Server stopped! Restarting in 5 seconds...
echo   Press Ctrl+C to exit
echo ============================================
timeout /t 5 /nobreak

goto start
