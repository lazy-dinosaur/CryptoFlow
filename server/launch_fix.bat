@echo off
title CryptoFlow SERVER (Port 7071)
color 0A

cd /d C:\CryptoFlow\server

echo ===================================================
echo   STARTING CRYPTOFLOW BACKEND (FIXED)
echo ===================================================

:: Kill old processes to free ports
echo [1/3] Cleaning up old processes...
taskkill /F /IM node.exe >nul 2>&1

:: Verify api.js exists
if not exist api.js (
    echo ERROR: api.js NOT FOUND!
    echo Current Dir: %CD%
    dir
    pause
    exit /b
)

:: Check file size (rough check)
for %%I in (api.js) do if %%~zI==0 (
    echo ERROR: api.js is EMPTY (0 bytes)!
    pause
    exit /b
)

echo [2/3] Starting Caddy (Background)...
cd /d C:\Caddy
start /b caddy run --config Caddyfile > caddy_debug.log 2>&1

echo [3/3] Starting Node Server...
cd /d C:\CryptoFlow\server
set PORT=7071

echo Running: node api.js
node api.js

if %errorlevel% neq 0 (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo !!! SERVER CRASHED (Exit Code: %errorlevel%) !!!
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo.
    echo Press any key to see error logs...
    pause
)

pause
