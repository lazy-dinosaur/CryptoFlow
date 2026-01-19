# CryptoFlow - Task Completion Checklist

## Before Committing Code

### Frontend Changes
- [ ] Test in browser at `http://localhost:5173`
- [ ] Verify Canvas rendering works correctly
- [ ] Check heatmap alignment (must be time-aligned, not index-based)
- [ ] Test zoom/pan functionality
- [ ] Verify 3D bubble rendering

### Backend Changes
- [ ] Test API endpoints manually or with curl
- [ ] Check database operations
- [ ] Verify WebSocket connections

### ML/Backtest Changes
- [ ] Run relevant backtest scripts
- [ ] Check output in `backtest/results/`
- [ ] Verify model performance metrics

## Commands to Run

### Quick Test
```bash
# Frontend
npm run dev

# Backend (separate terminal)
cd server && npm run dev
```

### Before Deployment
```bash
# Build and verify
npm run build
npm run preview

# Check for errors
npm run build 2>&1 | grep -i error
```

## DB Corruption Prevention
⚠️ **Multiple processes access cryptoflow.db simultaneously**
- collector (Node.js) - writes trades/candles
- api (Node.js) - reads/writes
- ml_paper_trading (Python) - **READ-ONLY** (fixed 2026-01-19)

**Rules:**
1. Python code accessing cryptoflow.db must use `get_readonly_connection()`
2. Node.js uses WAL mode with busy_timeout=30000
3. If corruption occurs: `sqlite3 cryptoflow.db 'PRAGMA wal_checkpoint(TRUNCATE);'`

## Important Warnings
⚠️ **Database Backup**: Always backup `server/data/cryptoflow.db` before destructive changes

⚠️ **Collector**: The collector runs 24/7 - rarely restart it. Use `pm2 restart api` for API-only changes.

⚠️ **Visual Features**: Do not modify the following without careful testing:
- Heat gradient colors
- 3D bubble rendering
- Time alignment logic
- Zoom bounds
