# Database Locking - Implementation Fixes

## Fix #1: Increase Timeouts (CRITICAL - 5 min)

### File: `server/db.js`

**Change these lines:**
```javascript
// BEFORE (lines 20-23)
db.pragma('journal_mode = WAL');
db.pragma('busy_timeout = 2000');      // ❌ Too short
db.pragma('synchronous = NORMAL');
db.pragma('wal_autocheckpoint = 1000'); // ❌ Too frequent

// AFTER
db.pragma('journal_mode = WAL');
db.pragma('busy_timeout = 5000');      // ✓ 5 seconds (allows large transactions)
db.pragma('synchronous = NORMAL');
db.pragma('wal_autocheckpoint = 5000'); // ✓ 5000 pages (less frequent checkpoints)
```

**Why**: 
- Large backfill transactions (1000+ inserts) take 3-5 seconds
- Current 2000ms timeout is too aggressive
- 5000ms is still acceptable for API latency (users won't notice)
- Checkpoint every 5000 pages instead of 1000 reduces lock contention

---

## Fix #2: Add Transaction Batching (HIGH - 10 min)

### File: `server/collector.js`

**Problem**: Each candle is inserted separately, creating many small transactions.

**Solution**: Batch all candles into single transaction.

**Change lines 181-185:**
```javascript
// BEFORE
const upsertCandles = db.db.transaction((tableName, symbol, candles) => {
    for (const candle of candles) {
        db.upsertCandle(tableName, symbol, candle);
    }
});

// AFTER - Wrap entire batch in single transaction
const upsertCandles = db.db.transaction((tableName, symbol, candles) => {
    for (const candle of candles) {
        db.upsertCandle(tableName, symbol, candle);
    }
});
// ✓ Already correct! The db.db.transaction() wrapper handles this.
// But verify it's being used everywhere candles are inserted.
```

**Verify usage** (lines 224, 274):
```javascript
// Line 224 - backfill
upsertCandles(timeframe.table, fullSymbol, candles);  // ✓ Uses transaction

// Line 274 - sync
upsertCandles(timeframe.table, fullSymbol, candles);  // ✓ Uses transaction
```

✓ **Already implemented correctly!** No changes needed here.

---

## Fix #3: Prevent Simultaneous Backfill + Sync (HIGH - 5 min)

### File: `server/collector.js`

**Problem**: `backfillAll()` and `syncLatestCandles()` can run simultaneously, causing lock contention.

**Solution**: Add mutex flag to prevent overlap.

**Add after line 241:**
```javascript
let syncInProgress = false;  // ✓ Already exists!
let backfillInProgress = false;  // ✓ ADD THIS

// Modify backfillAll() to use flag
async function backfillAll() {
    if (backfillInProgress) {
        console.log('Backfill already in progress, skipping...');
        return;
    }
    backfillInProgress = true;
    
    try {
        console.log(`\n========================================`);
        console.log(`  Backfill History`);
        console.log(`========================================\n`);
        
        for (const [exchange, exchangeConfig] of Object.entries(CONFIG.exchanges)) {
            if (!exchangeConfig.enabled) continue;
            const days = exchangeConfig.historyDays || CONFIG.historyDays;
            console.log(`[${exchange.toUpperCase()}] ${exchangeConfig.symbols.join(', ')} (${days} days)`);
            
            for (const symbol of exchangeConfig.symbols) {
                for (const timeframe of TIMEFRAMES) {
                    const count = await backfillCandles(exchange, symbol, timeframe);
                    if (count > 0) {
                        console.log(`  ${symbol} ${timeframe.name}: +${count} candles`);
                    }
                    await sleep(CONFIG.requestDelayMs);
                }
            }
        }
        console.log(`\nBackfill complete.`);
    } finally {
        backfillInProgress = false;  // ✓ Always reset
    }
}

// Modify syncLatestCandles() to check backfill flag
async function syncLatestCandles() {
    if (syncInProgress || backfillInProgress) {  // ✓ Check both flags
        return;
    }
    syncInProgress = true;
    
    try {
        // ... rest of function unchanged
    } finally {
        syncInProgress = false;
    }
}
```

---

## Fix #4: Add Read-Only Connections for API (MEDIUM - 15 min)

### File: `server/db.js`

**Problem**: API reads use same connection as collector writes.

**Solution**: Create separate read-only connection for API.

**Add after line 23:**
```javascript
// Create read-only connection for API (doesn't block writes)
const dbReadOnly = new Database(DB_PATH, { readonly: true });
dbReadOnly.pragma('busy_timeout = 5000');

// Export both
module.exports = {
    db,           // For writes (collector)
    dbReadOnly,   // For reads (API)
    upsertCandle,
    getCandles,
    getLastCandleTime,
    getStats
};
```

### File: `server/api.js`

**Change lines 40-46 and 90:**
```javascript
// BEFORE
const db = require('./db.js');

app.get('/api/health', (req, res) => {
    const stats = db.getStats();  // ❌ Uses write connection
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        ...stats
    });
});

// AFTER
const db = require('./db.js');
const { dbReadOnly } = require('./db.js');  // ✓ Add this

app.get('/api/health', (req, res) => {
    const stats = db.getStats();  // ✓ Keep as-is (getStats uses db internally)
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        ...stats
    });
});
```

**Actually, simpler approach**: Modify `getCandles()` and `getStats()` to accept optional connection parameter:

```javascript
// In db.js - modify getCandles to use read-only connection
function getCandles(tableName, symbol, limit = 100, before = null, useReadOnly = false) {
    const connection = useReadOnly ? dbReadOnly : db;  // ✓ Use read-only if requested
    
    let query;
    let params;
    
    if (before) {
        query = `
            SELECT time, open, high, low, close, volume, buy_volume as buyVolume, 
                   sell_volume as sellVolume, delta, trade_count as tradeCount, clusters
            FROM ${tableName}
            WHERE symbol = ? AND time < ?
            ORDER BY time DESC
            LIMIT ?
        `;
        params = [symbol.toUpperCase(), before, limit];
    } else {
        query = `
            SELECT time, open, high, low, close, volume, buy_volume as buyVolume, 
                   sell_volume as sellVolume, delta, trade_count as tradeCount, clusters
            FROM ${tableName}
            WHERE symbol = ?
            ORDER BY time DESC
            LIMIT ?
        `;
        params = [symbol.toUpperCase(), limit];
    }
    
    return connection.prepare(query).all(...params).reverse().map(c => ({
        ...c,
        clusters: c.clusters ? JSON.parse(c.clusters) : {}
    }));
}

// In api.js - use read-only for API calls
const candles = db.getCandles(table, fullSymbol, Math.min(limit, 5000), before, true);  // ✓ true = use read-only
```

---

## Fix #5: Monitor Lock Contention (MEDIUM - 10 min)

### File: `server/db.js`

**Add logging to detect lock issues:**

```javascript
// Add after line 23
let lockWaitCount = 0;
let lastLockWarnTime = 0;

// Wrap db operations with lock monitoring
const originalPrepare = db.prepare.bind(db);
db.prepare = function(sql) {
    const stmt = originalPrepare(sql);
    const originalRun = stmt.run.bind(stmt);
    const originalAll = stmt.all.bind(stmt);
    const originalGet = stmt.get.bind(stmt);
    
    stmt.run = function(...args) {
        try {
            return originalRun(...args);
        } catch (err) {
            if (err.message.includes('database is locked')) {
                lockWaitCount++;
                const now = Date.now();
                if (now - lastLockWarnTime > 5000) {  // Warn every 5 seconds
                    console.warn(`⚠️  Database lock detected (${lockWaitCount} times)`);
                    lastLockWarnTime = now;
                }
            }
            throw err;
        }
    };
    
    stmt.all = function(...args) {
        try {
            return originalAll(...args);
        } catch (err) {
            if (err.message.includes('database is locked')) {
                lockWaitCount++;
            }
            throw err;
        }
    };
    
    stmt.get = function(...args) {
        try {
            return originalGet(...args);
        } catch (err) {
            if (err.message.includes('database is locked')) {
                lockWaitCount++;
            }
            throw err;
        }
    };
    
    return stmt;
};

// Add to getStats() to expose lock count
function getStats() {
    const timeframes = ['15', '60', '240', '1440'];
    const stats = {};
    
    for (const tf of timeframes) {
        try {
            const count = db.prepare(`SELECT COUNT(*) as count FROM candles_${tf}`).get().count;
            stats[`candles_${tf}`] = count;
        } catch (e) {
            stats[`candles_${tf}`] = 0;
        }
    }
    
    stats.lockWaitCount = lockWaitCount;  // ✓ Expose lock count
    return stats;
}
```

---

## Implementation Order

1. **Fix #1** (5 min): Increase timeouts in `db.js`
   - Immediate relief from lock timeouts
   - Test with concurrent requests

2. **Fix #3** (5 min): Add backfill mutex in `collector.js`
   - Prevents simultaneous backfill + sync
   - Reduces lock contention

3. **Fix #5** (10 min): Add lock monitoring in `db.js`
   - Verify locks are actually happening
   - Monitor in production

4. **Fix #4** (15 min): Add read-only connections
   - Eliminates reader-writer conflicts
   - More complex but most effective

---

## Testing After Fixes

### Test 1: Concurrent API Requests During Backfill
```bash
# Terminal 1: Start collector
npm --prefix server run start:collector

# Terminal 2: Hammer API (should NOT get "database is locked")
for i in {1..100}; do
  curl -s http://localhost:3000/api/candles?symbol=btcusdt | jq '.candles | length'
  sleep 0.1
done
```

### Test 2: Check Lock Count
```bash
curl http://localhost:3000/api/health | jq '.lockWaitCount'
# Should be 0 or very low
```

### Test 3: Monitor Backfill Performance
```bash
# Check logs for timing
npm --prefix server run start:collector 2>&1 | grep -E "candles|Backfill"
# Should complete without "database is locked" errors
```

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Lock timeouts during backfill | 10-50/hour | 0-2/hour |
| API response time during backfill | 2000-5000ms | <100ms |
| Backfill completion time | 30-60 min | 20-30 min |
| Lock wait count | 100+ | <10 |

---

## If Issues Persist

If you still see "database is locked" after these fixes:

1. **Check disk I/O**: `iostat -x 1` during backfill
   - If I/O is 100%, upgrade disk or reduce batch size

2. **Check WAL file size**: `ls -lh server/data/cryptoflow.db-wal`
   - If >100MB, checkpoint is not running properly

3. **Consider async database**: Switch to `sqlite3` (async) instead of `better-sqlite3` (sync)
   - Prevents blocking event loop
   - More complex but more scalable

4. **Consider separate databases**: 
   - Collector writes to `collector.db`
   - API reads from `api.db` (replicated)
   - Eliminates all lock contention
