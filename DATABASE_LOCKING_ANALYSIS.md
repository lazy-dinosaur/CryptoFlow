# CryptoFlow Database Locking Issue - Root Cause Analysis

## Executive Summary

**Problem**: API and Collector services crash with "database is locked" errors despite WAL mode being enabled.

**Root Cause**: **Single shared database connection instance** across multiple concurrent processes without proper connection pooling or transaction isolation.

---

## Current Architecture (PROBLEMATIC)

### Database Setup (`server/db.js`)
```javascript
const db = new Database(DB_PATH);  // ❌ SINGLE INSTANCE
db.pragma('journal_mode = WAL');
db.pragma('busy_timeout = 2000');
db.pragma('synchronous = NORMAL');
db.pragma('wal_autocheckpoint = 1000');
```

**Issues:**
1. **Single connection object** exported and shared across all modules
2. **No connection pooling** - concurrent requests queue on same connection
3. **No transaction isolation** - reads/writes can block each other
4. **Synchronous API** - `better-sqlite3` blocks on I/O

### How Services Use It

#### API Server (`server/api.js`)
- **REST endpoint** `/api/candles` calls `db.getCandles()` synchronously
- **WebSocket** broadcasts call `db.getCandles()` from event handlers
- **Multiple concurrent clients** = multiple simultaneous DB reads
- **No queuing mechanism** - all requests hit DB at once

#### Collector (`server/collector.js`)
- **Backfill loop** calls `upsertCandles()` in transaction
- **Sync loop** (every 60s) calls `upsertCandles()` again
- **Runs in separate process** but shares same DB file
- **No coordination** with API writes

#### Paper Trading (`server/ml_paper_trading.py`)
- **Separate database** (`ml_paper_trading.db`) ✓ Good
- **But reads from main DB** (`cryptoflow.db`) with `readonly=True` ✓ Good
- **However**: Uses `timeout=2` seconds (too short for busy DB)

---

## Why WAL Mode Isn't Enough

WAL (Write-Ahead Logging) helps but **doesn't solve the fundamental problem**:

| Scenario | WAL Behavior | Result |
|----------|--------------|--------|
| **Multiple readers** | ✓ Can read while writing | Works fine |
| **Reader + Writer** | ✓ Reader sees old snapshot | Works fine |
| **Multiple writers** | ❌ Only ONE writer at a time | **LOCK CONTENTION** |
| **Busy timeout exceeded** | ❌ Query fails | **"database is locked"** |

**Your Problem**: Collector writes while API reads = writer blocks readers = timeout exceeded.

---

## Specific Lock Scenarios

### Scenario 1: Backfill Collision
```
Time  Collector                          API
----  ---------                          ---
T0    BEGIN TRANSACTION
T1    INSERT candles_15 (1000 rows)      [waiting for lock]
T2    INSERT candles_60 (1000 rows)      [waiting for lock]
T3    INSERT candles_240 (1000 rows)     [waiting for lock]
T4    INSERT candles_1440 (1000 rows)    [waiting for lock]
T5    COMMIT (finally!)                  [2000ms timeout exceeded]
      ❌ "database is locked"
```

**Why**: Large transaction holds write lock for 5+ seconds. API timeout (2000ms) expires.

### Scenario 2: Concurrent Sync
```
Time  Collector (sync)                   Collector (backfill)
----  ----------------                  --------------------
T0    BEGIN TRANSACTION
T1    UPDATE candles_15                  [waiting for lock]
T2    UPDATE candles_60                  [waiting for lock]
T3    UPDATE candles_240                 [waiting for lock]
T4    UPDATE candles_1440                [waiting for lock]
T5    COMMIT                             [2000ms timeout exceeded]
      ❌ "database is locked"
```

**Why**: `syncLatestCandles()` and `backfillCandles()` can run simultaneously.

### Scenario 3: WAL Checkpoint Collision
```
Time  Collector                          API
----  ---------                          ---
T0    COMMIT (triggers checkpoint)       [reading]
T1    [WAL checkpoint in progress]       [waiting for lock]
T2    [checkpoint still running]         [waiting for lock]
T3    [checkpoint complete]              [2000ms timeout exceeded]
      ❌ "database is locked"
```

**Why**: `wal_autocheckpoint = 1000` triggers after 1000 pages. Checkpoint blocks readers.

---

## Missing Configurations

### 1. Connection Pooling
**Current**: Single connection
**Needed**: Connection pool (3-5 connections)

### 2. Transaction Batching
**Current**: Each write is separate transaction
**Needed**: Batch writes into single transaction

### 3. Read-Only Connections
**Current**: All connections can write
**Needed**: Separate read-only connections for API

### 4. Busy Timeout Tuning
**Current**: `busy_timeout = 2000` (2 seconds)
**Issues**:
- Too short for large backfill transactions (5-10 seconds)
- Too long for real-time API (users see 2s lag)

### 5. WAL Checkpoint Tuning
**Current**: `wal_autocheckpoint = 1000`
**Issues**:
- Triggers too frequently during backfill
- Blocks readers during checkpoint

### 6. Synchronous Mode
**Current**: `synchronous = NORMAL`
**Issues**:
- Still syncs WAL file to disk (slower)
- Should be `FULL` for safety or `OFF` for speed

---

## Recommended Fixes (Priority Order)

### CRITICAL (Do First)
1. **Increase busy_timeout to 5000ms** (5 seconds)
   - Allows large transactions to complete
   - Still acceptable for API latency

2. **Increase wal_autocheckpoint to 5000** (5000 pages)
   - Reduces checkpoint frequency
   - Fewer lock contentions

3. **Add transaction batching in collector**
   - Wrap multiple upserts in single transaction
   - Reduces lock hold time

### HIGH (Do Next)
4. **Implement connection pooling**
   - Use `better-sqlite3` with multiple connections
   - Separate read/write connections

5. **Add read-only mode for API**
   - API uses read-only connections
   - Eliminates reader-writer conflicts

6. **Separate sync/backfill timing**
   - Don't run both simultaneously
   - Use mutex/flag to prevent overlap

### MEDIUM (Nice to Have)
7. **Async database wrapper**
   - Use `sqlite3` (async) instead of `better-sqlite3` (sync)
   - Prevents blocking event loop

8. **Database monitoring**
   - Log lock wait times
   - Alert on timeout threshold

---

## Implementation Checklist

- [ ] Increase `busy_timeout` to 5000ms in `db.js`
- [ ] Increase `wal_autocheckpoint` to 5000 in `db.js`
- [ ] Add transaction batching in `collector.js` (wrap backfill/sync in single transaction)
- [ ] Add mutex flag to prevent simultaneous backfill + sync
- [ ] Test with concurrent API requests during backfill
- [ ] Monitor lock wait times in production
- [ ] Consider connection pooling if issues persist
- [ ] Consider async database wrapper if issues persist

---

## Testing the Fix

### Before Fix
```bash
# Terminal 1: Start collector (backfill)
npm --prefix server run start:collector

# Terminal 2: Hammer API with requests
while true; do curl http://localhost:3000/api/candles?symbol=btcusdt; done

# Expected: "database is locked" errors in Terminal 2
```

### After Fix
```bash
# Same test
# Expected: No errors, API responds even during backfill
```

---

## References

- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [SQLite Busy Timeout](https://www.sqlite.org/pragma.html#pragma_busy_timeout)
- [better-sqlite3 Docs](https://github.com/WiseLibs/better-sqlite3)
- [SQLite Checkpoint](https://www.sqlite.org/wal.html#ckpt)
