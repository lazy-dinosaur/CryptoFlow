# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-30
**Commit:** 9aad141
**Branch:** main

## OVERVIEW

CryptoFlow: Institutional-grade crypto charting with Bookmap-style footprint/heatmap visualization. Vanilla JS Canvas frontend + Node/Express API + Python ML backtesting.

## STRUCTURE

```
CryptoFlow/
├── src/                    # Vite frontend (Canvas chart engine)
│   ├── components/         # UI: FootprintChart, MLDashboard, chart/layers
│   ├── services/           # Data: binanceWS, dataAggregator, vpsAPI
│   └── main.js             # App entry (CryptoFlowApp class)
├── server/                 # Node/Express backend
│   ├── api.js              # REST + WebSocket server (port 3000)
│   ├── collector.js        # Multi-exchange candle collector
│   ├── db.js               # SQLite schema/queries
│   └── ml_paper_trading.py # Python paper trading (port 5003)
├── backtest/               # Python ML/strategy experiments
│   ├── engine.py           # Backtesting framework
│   ├── strategies/         # Pluggable strategy classes
│   └── ml_*.py             # 40+ ML experiment scripts
└── server/data/            # SQLite databases
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Chart rendering | `src/components/chart/` | Layered Canvas architecture |
| Trade aggregation | `src/services/dataAggregator.js` | OHLCV, footprint, delta |
| Real-time data | `src/services/binanceWS.js` | WebSocket streams |
| API endpoints | `server/api.js` | `/api/candles`, `/api/symbols` |
| Database schema | `server/db.js` | candles_15/60/240/1440 tables |
| Paper trading signals | `server/ml_paper_trading.py` | Channel detection, bounce/fakeout |
| Strategy backtesting | `backtest/engine.py` | Trade, BacktestResult classes |
| Baseline strategy | `backtest/ml_channel_tiebreaker_proper.py` | Reference implementation |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| CryptoFlowApp | class | src/main.js | App orchestrator |
| DataAggregator | class | src/services/dataAggregator.js | Trade→Candle aggregation |
| BinanceWS | class | src/services/binanceWS.js | WebSocket client |
| VpsAPI | class | src/services/vpsAPI.js | Backend client |
| FootprintChart | class | src/components/FootprintChart.js | Chart wrapper |
| MLDashboard | class | src/components/MLDashboard.js | Paper trading UI |
| RenderEngine | class | src/components/chart/core/RenderEngine.js | Canvas renderer |
| ChartState | class | src/components/chart/core/ChartState.js | State management |

## DATA FLOW

```
Exchanges (Binance/Bybit/Bitget)
    ↓ REST Kline API (60s polling)
collector.js → db.js (SQLite) → api.js → Frontend
    ↓ WebSocket (real-time)
binanceWS.js → dataAggregator.js → FootprintChart.js
```

## CONVENTIONS

| Context | Rule |
|---------|------|
| JS modules | ES Modules, 4-space indent, semicolons |
| Component files | PascalCase (`FootprintChart.js`) |
| Service files | camelCase (`dataAggregator.js`) |
| Python files | snake_case (`ml_channel.py`) |
| Private methods | Underscore prefix (`_init()`) |
| Commits | Conventional: `feat:`, `fix:`, `docs:` |

## ANTI-PATTERNS (THIS PROJECT)

### CRITICAL: Lookahead Bias
```python
# CORRECT - Use previous completed candle
channel = htf_channel_map.get(htf_idx - 1)

# WRONG - Uses incomplete/future data
channel = htf_channel_map.get(htf_idx)
```
Affects all `backtest/ml_*.py` and `server/ml_paper_trading.py`.

### Array Mutation
```javascript
// WARNING: reverse() mutates original array
sortedAsks.reverse()  // src/components/OrderBook.js
```

### Loading Overlay
```javascript
// ALWAYS hide even on error to prevent UI lockup
this.elements.loadingOverlay.classList.add('hidden');
```

### Disabled Features (Validated Decisions)
| Feature | Status | Reason |
|---------|--------|--------|
| ML Filtering | DISABLED | Reduces total profit |
| FAKEOUT Strategy | DISABLED | -0.10% avg PnL after bias fix |

## UNIQUE STYLES

- **No linter/formatter** - Style enforced by convention only
- **Pure JavaScript** - No TypeScript despite Vite
- **Mixed module systems** - Frontend ESM, server CommonJS
- **No test framework** - Validation via backtest scripts

## COMMANDS

```bash
# Frontend
npm install && npm run dev        # Dev server :5173
npm run build                     # Build to dist/

# Server
npm --prefix server install
npm --prefix server run start     # API :3000
npm --prefix server run start:collector

# Python
python3 server/ml_paper_trading.py    # Paper trading :5003
python backtest/run_backtest.py       # Run backtest
```

## NOTES

- **DB Backup**: Always backup `server/data/cryptoflow.db` before destructive changes
- **Port Config**: API=3000, ML Paper=5003, Vite Dev=5173
- **Coordinate Alignment**: `CoordinateSystem.js` may not match `FootprintChart.js` - test both when changing
- **Heatmap Index**: Assumes 1:1 candle-to-snapshot mapping - breaks if data misaligned
- **Sample Size**: ML models unreliable with <10 training samples
