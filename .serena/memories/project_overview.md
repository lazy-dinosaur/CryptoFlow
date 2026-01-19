# CryptoFlow Project Overview

## Purpose
CryptoFlow is a high-performance, professional-grade **trading visualization tool** designed to replicate institutional platforms like **Bookmap** and **ATAS**. It features a custom-built Canvas rendering engine for tick-level data, 3D imbalance bubbles, and time-aligned liquidity heatmap.

## Key Features
- **High-Performance Canvas Engine**: Zero-lag rendering of thousands of candles and ticks
- **Bookmap-Style Heatmap**: Time-aligned, gamma-corrected liquidity visualization
- **3D Imbalance Bubbles**: Large trades visualized as 3D spheres with radial gradients
- **Footprint/DOM**: Detailed volume breakdown inside candles
- **Magnifier Lens (L-Key)**: Hover to see granular tick data
- **Delta Summary**: Integrated Delta and Imbalance bars

## Tech Stack
| Layer | Technology |
|-------|------------|
| Frontend | Vanilla JavaScript, HTML5 Canvas, Vite |
| Backend | Node.js, Express, better-sqlite3 |
| ML/Backtest | Python (pandas, numpy, pyarrow) |
| Data | Binance WebSocket API |
| Deployment | Windows VPS, PM2 |
| Database | SQLite |

## Architecture / Data Flow
```
Binance WebSocket → Collector (24/7) → SQLite → API Server (REST + WS) → Frontend
                                          ↓
                                    ML Service → Predictions → Frontend overlay
```

## Project Structure
```
CryptoFlow/
├── src/                    # Frontend source
│   ├── components/         # UI components (PascalCase)
│   │   └── FootprintChart.js  # ⭐ Core chart engine (2500+ lines)
│   ├── services/           # Services (camelCase)
│   └── main.js             # App controller
├── server/                 # Backend services
│   ├── api.js              # REST API + WebSocket (port 3000)
│   ├── collector-standalone.js  # 24/7 data collector
│   ├── ml_service.py       # ML prediction (port 5001)
│   ├── ml_paper_trading.py # Paper trading (port 5003)
│   └── db.js               # SQLite schema & queries
├── backtest/               # Backtesting framework
│   ├── engine.py           # Generic backtest engine
│   ├── strategies/         # Pluggable strategy classes
│   └── ml_*.py             # ML training/evaluation scripts
├── ecosystem.config.cjs    # PM2 process definitions
└── dist/                   # Production build output
```

## Production Server (OCI)
- **Host**: `ssh oci` (Ubuntu)
- **Path**: `/home/ubuntu/projects/CryptoFlow`
- **Process Manager**: PM2
- **Services**: collector, api, ml, ml-paper-trading

## Key Files