# CryptoFlow VPS Setup Script for Windows
# Run this as Administrator on Windows Server

Write-Host "🚀 CryptoFlow Windows Setup" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# Check if Node.js is installed
$nodeVersion = node --version 2>$null
if (-not $nodeVersion) {
    Write-Host "📦 Node.js not found. Installing..." -ForegroundColor Yellow
    
    # Download Node.js installer
    $nodeUrl = "https://nodejs.org/dist/v20.10.0/node-v20.10.0-x64.msi"
    $installerPath = "$env:TEMP\node-installer.msi"
    
    Write-Host "Downloading Node.js..."
    Invoke-WebRequest -Uri $nodeUrl -OutFile $installerPath
    
    Write-Host "Installing Node.js (this may take a minute)..."
    Start-Process msiexec.exe -ArgumentList "/i `"$installerPath`" /quiet /norestart" -Wait
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Host "✅ Node.js installed" -ForegroundColor Green
} else {
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
}

# Navigate to project directory
$projectDir = Split-Path -Parent $PSScriptRoot
if (Test-Path "$projectDir\server") {
    Set-Location $projectDir
} else {
    Set-Location $PSScriptRoot
}

# Install server dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Set-Location server
npm install

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Install PM2 globally
Write-Host "📦 Installing PM2..." -ForegroundColor Yellow
npm install -g pm2 pm2-windows-startup

# Start with PM2
Write-Host "🚀 Starting CryptoFlow..." -ForegroundColor Yellow
Set-Location ..
pm2 start ecosystem.config.js

# Save PM2 config
pm2 save

# Setup PM2 to start on boot
pm2-startup install

Write-Host ""
Write-Host "✅ CryptoFlow is running!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 API: http://localhost:3001" -ForegroundColor Cyan
Write-Host "📡 WebSocket: ws://localhost:3002" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  pm2 logs cryptoflow-api     - View logs"
Write-Host "  pm2 restart cryptoflow-api  - Restart service"
Write-Host "  pm2 stop cryptoflow-api     - Stop service"
Write-Host "  pm2 monit                   - Monitor dashboard"
