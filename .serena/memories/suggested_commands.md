# CryptoFlow - Suggested Commands

## Frontend (Vite)
```bash
npm install              # Install frontend dependencies
npm run dev              # Start Vite dev server at http://localhost:5173
npm run build            # Build to dist/
npm run preview          # Serve production build locally
```

## Backend Server (Node.js)
```bash
cd server
npm install                      # Install server dependencies
npm run start                    # Start API server (port 3000)
npm run start:collector          # Run market data collector (24/7)
npm run dev                      # API with nodemon auto-reload
npm run dev:collector            # Collector with nodemon
```

Or from root:
```bash
npm --prefix server install
npm --prefix server run start
npm --prefix server run start:collector
```

## ML & Paper Trading (Python)
```bash
python3 server/ml_service.py         # ML prediction service (port 5001)
python3 server/ml_paper_trading.py   # Paper trading service (port 5003)
```

## Backtesting
```bash
cd backtest
pip install -r requirements.txt      # Install Python dependencies
python run_backtest.py               # Run backtest with configured strategy
python ml_test_*.py                  # Run specific ML training/evaluation
```

## OCI Server Access
```bash
ssh oci                               # Connect to OCI server
ssh oci "pm2 status"                  # Check PM2 status remotely
ssh oci "pm2 logs --lines 20"         # View recent logs
```

## PM2 Process Management (Production)
```bash
pm2 start ecosystem.config.cjs              # Start all services
pm2 start ecosystem.config.cjs --only api   # Start API only
pm2 restart api                             # Restart API
pm2 restart ml                              # Restart ML
pm2 logs                                    # View all logs
pm2 logs collector                          # View collector logs
pm2 status                                  # Check status
```

## Deployment to VPS
```bash
# 1. Build locally
npm run build

# 2. Upload to VPS
scp -r dist/* Administrator@<VPS_IP>:"C:/CryptoFlow/dist/"

# Note: Frontend-only changes don't require server restart
```

## System Utilities
```bash
git status                    # Check git status
git pull                      # Pull latest changes
ls -la                        # List files
find . -name "*.js"           # Find JavaScript files
grep -r "pattern" src/        # Search in source
```
