# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Frontend (Vite)
```bash
npm install              # Install frontend dependencies
npm run dev              # Start Vite dev server at http://localhost:5173
npm run build            # Build to dist/
npm run preview          # Serve production build locally
```

### Server (Node.js)
```bash
npm --prefix server install          # Install server dependencies
npm --prefix server run start        # Start API server (port 3000)
npm --prefix server run start:collector   # Run market data collector
npm --prefix server run dev          # API with nodemon auto-reload
```

### ML & Paper Trading (Python)
```bash
python3 server/ml_service.py         # ML prediction service (port 5001)
python3 server/ml_paper_trading.py   # Paper trading service (port 5003)
```

### Backtesting
```bash
python backtest/run_backtest.py      # Run backtest with configured strategy
python backtest/ml_test_*.py         # Run specific ML training/evaluation scripts
```

## Architecture Overview

### Data Flow
```
Binance WebSocket → Collector (24/7) → SQLite → API Server (REST + WS) → Frontend
                                          ↓
                                    ML Service → Predictions → Frontend overlay
```

### Frontend (src/)
- **FootprintChart.js** - Main canvas-based chart engine (2500+ lines). Renders orderflow footprints, Bookmap-style heatmap, imbalances, big trades. All rendering is vanilla JS Canvas 2D with no framework.
- **binanceWS.js** - WebSocket client for Binance streams (trades, depth, ticker)
- **dataAggregator.js** - Converts raw trades into candles with delta, volume profile, VWAP. Supports 1m/5m/15m/30m/60m/240m timeframes
- **vpsAPI.js** - Fetches pre-aggregated candles from VPS (fast path), falls back to Binance REST
- **depthHeatmap.js** - Aggregates depth snapshots for Bookmap-style visualization

The frontend uses an event-driven pattern where services emit events (`candleUpdate`, `trade`, `statsUpdate`) consumed by UI components.

### Backend (server/)
Four independent services managed via PM2:
1. **Collector** (collector-standalone.js) - 24/7 trade collection from exchanges, aggregates to candles, stores to SQLite
2. **API** (api.js:3000) - REST endpoints `/api/candles`, `/api/symbols`, `/api/health` plus WebSocket streaming
3. **ML Service** (ml_service.py:5001) - Python prediction engine, trains every 6 hours
4. **Paper Trading** (ml_paper_trading.py:5003) - Tests 3 ML strategies simultaneously

**Database**: SQLite with tables for trades, candles (1/5/15/30/60/240/1440 min), sessions, heatmap_snapshots, ML signals.

### Backtest (backtest/)
- **engine.py** - Generic backtesting framework calculating win rate, Sharpe, profit factor, max drawdown
- **strategies/** - Pluggable strategy classes inheriting from base Strategy class
- Results output to `backtest/results/` as JSON

## Coding Conventions
- JavaScript: ES modules, 4-space indentation, semicolons
- Component files (src/components/): PascalCase (e.g., FootprintChart.js)
- Service files (src/services/): camelCase (e.g., dataAggregator.js)
- Python: snake_case filenames and functions
- Commits: Conventional Commits format (`feat:`, `fix:`, `docs:`)

## Key Files
- `ecosystem.config.cjs` - PM2 process definitions for VPS deployment
- `server/db.js` - SQLite database schema and queries
- `server/data/cryptoflow.db` - Main database (back up before destructive changes)
