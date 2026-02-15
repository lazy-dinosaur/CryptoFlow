# Database Locking Fix - Implementation Checklist

## Pre-Implementation

- [ ] Read `DATABASE_LOCKING_ANALYSIS.md` (understand the problem)
- [ ] Read `QUICK_FIX_PATCH.md` (understand the solution)
- [ ] Backup database: `cp server/data/cryptoflow.db server/data/cryptoflow.db.backup`
- [ ] Backup collector: `cp server/collector.js server/collector.js.backup`
- [ ] Backup db.js: `cp server/db.js server/db.js.backup`

## Phase 1: Quick Fixes (5 minutes)

### Fix #1: Increase Timeouts in db.js
- [ ] Open `server/db.js`
- [ ] Find line 21: `db.pragma('busy_timeout = 2000');`
- [ ] Change to: `db.pragma('busy_timeout = 5000');`
- [ ] Find line 23: `db.pragma('wal_autocheckpoint = 1000');`
- [ ] Change to: `db.pragma('wal_autocheckpoint = 5000');`
- [ ] Save file
- [ ] Verify: `grep "busy_timeout\|wal_autocheckpoint" server/db.js`

### Fix #2: Add Backfill Mutex in collector.js
- [ ] Open `server/collector.js`
- [ ] Find line 241: `let syncInProgress = false;`
- [ ] Add after it: `let backfillInProgress = false;`
- [ ] Find line 244: `if (syncInProgress) return;`
- [ ] Change to: `if (syncInProgress || backfillInProgress) return;`
- [ ] Save file
- [ ] Verify: `grep -n "backfillInProgress" server/collector.js` (should show 3 lines)

### Fix #3: Wrap backfillAll() with Mutex
- [ ] Open `server/collector.js`
- [ ] Find function `backfillAll()` (around line 285)
- [ ] Add at start: `if (backfillInProgress) { console.log('Backfill already in progress, skipping...'); return; }`
- [ ] Add: `backfillInProgress = true;`
- [ ] Wrap entire function body in `try { ... } finally { backfillInProgress = false; }`
- [ ] Save file
- [ ] Verify: `grep -A 5 "async function backfillAll" server/collector.js`

## Phase 2: Testing (10 minutes)

### Test 1: Syntax Check
- [ ] Run: `node -c server/db.js` (should output nothing if OK)
- [ ] Run: `node -c server/collector.js` (should output nothing if OK)

### Test 2: Start Services
- [ ] Terminal 1: `npm --prefix server run start` (start API)
- [ ] Wait for: "API running on..."
- [ ] Terminal 2: `npm --prefix server run start:collector` (start collector)
- [ ] Wait for: "Collector running..."

### Test 3: Verify Timeouts Applied
- [ ] Run: `curl http://localhost:3000/api/health | jq '.lockWaitCount'`
- [ ] Should return a number (0 or low)

### Test 4: Concurrent Load Test
- [ ] Terminal 3: Run load test:
  ```bash
  for i in {1..50}; do
    curl -s http://localhost:3000/api/candles?symbol=btcusdt | jq '.candles | length'
    sleep 0.1
  done
  ```
- [ ] Check for errors: Should see candle counts, NO "database is locked"
- [ ] Check API logs: Should see no errors

### Test 5: Monitor During Backfill
- [ ] Watch collector logs: Should see backfill progress
- [ ] Run load test again while backfill is running
- [ ] Verify: API responds quickly, no timeouts

### Test 6: Check Lock Count
- [ ] Run: `curl http://localhost:3000/api/health | jq '.lockWaitCount'`
- [ ] Should be 0 or very low (not 100+)

## Phase 3: Verification (5 minutes)

### Verify Changes
- [ ] `git diff server/db.js` (should show 2 lines changed)
- [ ] `git diff server/collector.js` (should show 3-4 lines changed)

### Check Logs
- [ ] API logs: No "database is locked" errors
- [ ] Collector logs: Backfill completes successfully
- [ ] No crashes or exceptions

### Performance Metrics
- [ ] Backfill time: Should be 20-30 minutes (not 30-60)
- [ ] API response time: Should be <100ms (not 2000-5000ms)
- [ ] Lock timeouts: Should be 0-2 per hour (not 10-50)

## Phase 4: Optional Enhancements (15 minutes)

### Fix #4: Add Read-Only Connections (Optional)
- [ ] See `DATABASE_LOCKING_FIXES.md` - Fix #4
- [ ] Implement if Phase 1-3 doesn't fully resolve issues

### Fix #5: Add Lock Monitoring (Optional)
- [ ] See `DATABASE_LOCKING_FIXES.md` - Fix #5
- [ ] Implement for production monitoring

## Rollback Plan (If Needed)

### Quick Rollback
```bash
# Restore from backups
cp server/db.js.backup server/db.js
cp server/collector.js.backup server/collector.js

# Or restore from git
git checkout server/db.js server/collector.js
```

### Verify Rollback
```bash
grep "busy_timeout" server/db.js  # Should show 2000
grep "backfillInProgress" server/collector.js  # Should show 0 lines
```

## Success Criteria

- [ ] No "database is locked" errors in logs
- [ ] API responds in <100ms during backfill
- [ ] Backfill completes in 20-30 minutes
- [ ] Lock wait count is 0-2 per hour
- [ ] All tests pass
- [ ] No crashes or exceptions

## Troubleshooting

### If Still Getting Lock Errors
1. Check timeouts: `grep "busy_timeout" server/db.js` (should be 5000)
2. Check mutex: `grep "backfillInProgress" server/collector.js` (should be 3+ lines)
3. Check logs: Look for "Backfill already in progress" messages
4. Increase timeout further: Try 10000ms instead of 5000ms
5. See `DATABASE_LOCKING_FIXES.md` for advanced solutions

### If Backfill is Slow
1. Check disk I/O: `iostat -x 1` (should be <50%)
2. Check CPU: `top` (should be <80%)
3. Check network: `iftop` (should be <50% of capacity)
4. Reduce batch size: Change `KLINE_LIMIT` in collector.js

### If API is Slow
1. Check database size: `ls -lh server/data/cryptoflow.db`
2. Check indexes: `sqlite3 server/data/cryptoflow.db ".indices"`
3. Check query plans: `sqlite3 server/data/cryptoflow.db "EXPLAIN QUERY PLAN SELECT ..."`
4. Consider read-only connections (Fix #4)

## Deployment

### To Production
1. Test locally first (all phases above)
2. Backup production database
3. Apply changes to production
4. Monitor logs for 1 hour
5. Check metrics: lock count, response time, backfill time
6. If issues, rollback immediately

### Monitoring
- [ ] Set up alerts for "database is locked" errors
- [ ] Monitor lock wait count in `/api/health`
- [ ] Monitor API response time
- [ ] Monitor backfill completion time

## Documentation

- [ ] Update AGENTS.md with database locking notes
- [ ] Document timeout settings in code comments
- [ ] Document mutex flags in code comments
- [ ] Add troubleshooting guide to README

## Sign-Off

- [ ] All tests passed
- [ ] No errors in logs
- [ ] Performance improved
- [ ] Ready for production
- [ ] Date: ___________
- [ ] Tested by: ___________

