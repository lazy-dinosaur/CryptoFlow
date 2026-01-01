#!/bin/bash
# CryptoFlow VPS Setup Script
# Run this on your VPS to install and start the data collector

set -e

echo "🚀 CryptoFlow VPS Setup"
echo "======================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "📦 Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

echo "Node.js version: $(node -v)"

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "📦 Installing PM2..."
    sudo npm install -g pm2
fi

echo "PM2 version: $(pm2 -v)"

# Install server dependencies
echo "📦 Installing server dependencies..."
cd server
npm install

# Create logs directory
mkdir -p logs

# Start with PM2
echo "🚀 Starting CryptoFlow Data Collector..."
cd ..
pm2 start ecosystem.config.js

# Save PM2 config for auto-restart on reboot
pm2 save
pm2 startup

echo ""
echo "✅ CryptoFlow is running!"
echo ""
echo "📊 API: http://localhost:3001"
echo "📡 WebSocket: ws://localhost:3002"
echo ""
echo "Useful commands:"
echo "  pm2 logs cryptoflow-api     - View logs"
echo "  pm2 restart cryptoflow-api  - Restart service"
echo "  pm2 stop cryptoflow-api     - Stop service"
echo "  pm2 monit                   - Monitor dashboard"
