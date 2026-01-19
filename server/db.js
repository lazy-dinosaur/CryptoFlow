/**
 * CryptoFlow VPS Data Collector - Database Module
 * SQLite database for persistent trade storage and candle aggregation
 */

const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

// Ensure data directory exists
const DATA_DIR = path.join(__dirname, 'data');
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

const DB_PATH = path.join(DATA_DIR, 'cryptoflow.db');
const db = new Database(DB_PATH);

// Enable WAL mode and configure for multi-process access
db.pragma('journal_mode = WAL');
db.pragma('busy_timeout = 2000');  // 2 seconds - individual writes are ms-level
db.pragma('synchronous = NORMAL');  // Balance between safety and speed
db.pragma('wal_autocheckpoint = 1000');  // Checkpoint every 1000 pages

/**
 * Initialize database schema
 */
function initSchema() {
    // Raw trades table
    db.exec(`
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            time INTEGER NOT NULL,
            is_buyer_maker INTEGER NOT NULL,
            trade_id TEXT,
            UNIQUE(symbol, trade_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, time);
    `);

    // Pre-aggregated candles for different timeframes
    const timeframes = ['1', '5', '15', '30', '60', '240', '1440'];

    for (const tf of timeframes) {
        db.exec(`
            CREATE TABLE IF NOT EXISTS candles_${tf} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                buy_volume REAL NOT NULL,
                sell_volume REAL NOT NULL,
                delta REAL NOT NULL,
                trade_count INTEGER NOT NULL,
                clusters TEXT,
                UNIQUE(symbol, time)
            );
            
            CREATE INDEX IF NOT EXISTS idx_candles_${tf}_symbol_time ON candles_${tf}(symbol, time);
        `);
    }

    // Session data (daily POC, VWAP, VAH, VAL)
    db.exec(`
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            poc REAL,
            vwap REAL,
            vah REAL,
            val REAL,
            total_volume REAL,
            UNIQUE(symbol, date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_sessions_symbol_date ON sessions(symbol, date);
    `);

    // Heatmap Snapshots (Depth History)
    db.exec(`
        CREATE TABLE IF NOT EXISTS heatmap_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            time INTEGER NOT NULL,
            bids TEXT, -- JSON
            asks TEXT, -- JSON
            max_volume REAL,
            UNIQUE(symbol, time)
        );
        CREATE INDEX IF NOT EXISTS idx_heatmap_symbol_time ON heatmap_snapshots(symbol, time);
    `);

    console.log('✅ Database schema initialized');
}

// Initialize schema BEFORE any prepared statements
initSchema();


/**
 * Insert a trade into the database
 */
const insertTradeStmt = db.prepare(`
    INSERT OR IGNORE INTO trades (symbol, price, quantity, time, is_buyer_maker, trade_id)
    VALUES (?, ?, ?, ?, ?, ?)
`);

function insertTrade(symbol, trade) {
    insertTradeStmt.run(
        symbol.toUpperCase(),
        trade.price,
        trade.quantity,
        trade.time,
        trade.isBuyerMaker ? 1 : 0,
        trade.tradeId || `${trade.time}_${trade.price}`
    );
}

/**
 * Insert multiple trades efficiently (batch)
 */
const insertTrades = db.transaction((symbol, trades) => {
    for (const trade of trades) {
        insertTrade(symbol, trade);
    }
});

/**
 * Get trades for a symbol within time range
 */
const getTradesStmt = db.prepare(`
    SELECT price, quantity, time, is_buyer_maker as isBuyerMaker
    FROM trades
    WHERE symbol = ? AND time >= ? AND time < ?
    ORDER BY time ASC
`);

function getTrades(symbol, startTime, endTime) {
    return getTradesStmt.all(symbol.toUpperCase(), startTime, endTime);
}

/**
 * Get latest trades for a symbol
 */
const getLatestTradesStmt = db.prepare(`
    SELECT price, quantity, time, is_buyer_maker as isBuyerMaker
    FROM trades
    WHERE symbol = ?
    ORDER BY time DESC
    LIMIT ?
`);

function getLatestTrades(symbol, limit = 10000) {
    return getLatestTradesStmt.all(symbol.toUpperCase(), limit).reverse();
}

/**
 * Upsert a candle
 */
function upsertCandle(tableName, symbol, candle) {
    const clustersJson = candle.clusters ? JSON.stringify(candle.clusters) : null;

    db.prepare(`
        INSERT OR REPLACE INTO ${tableName} 
        (symbol, time, open, high, low, close, volume, buy_volume, sell_volume, delta, trade_count, clusters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
        symbol.toUpperCase(),
        candle.time,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.buyVolume,
        candle.sellVolume,
        candle.delta,
        candle.tradeCount,
        clustersJson
    );
}

/**
 * Get candles for a symbol and timeframe
 */
function getCandles(tableName, symbol, limit = 100) {
    return db.prepare(`
        SELECT time, open, high, low, close, volume, buy_volume as buyVolume, 
               sell_volume as sellVolume, delta, trade_count as tradeCount, clusters
        FROM ${tableName}
        WHERE symbol = ?
        ORDER BY time DESC
        LIMIT ?
    `).all(symbol.toUpperCase(), limit).reverse().map(c => ({
        ...c,
        clusters: c.clusters ? JSON.parse(c.clusters) : {}
    }));
}

/**
 * Save session data
 */
function saveSession(symbol, date, data) {
    db.prepare(`
        INSERT OR REPLACE INTO sessions (symbol, date, poc, vwap, vah, val, total_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
        symbol.toUpperCase(),
        date,
        data.poc,
        data.vwap,
        data.vah,
        data.val,
        data.totalVolume
    );
}

/**
 * Get previous session data
 */
function getPreviousSession(symbol) {
    const today = new Date().toISOString().split('T')[0];
    return db.prepare(`
        SELECT poc, vwap, vah, val, total_volume as totalVolume
        FROM sessions
        WHERE symbol = ? AND date < ?
        ORDER BY date DESC
        LIMIT 1
    `).get(symbol.toUpperCase(), today);
}

/**
 * Clean up old trades (keep last N days)
 */
function cleanupOldData(daysToKeep = 7) {
    const cutoffTime = Date.now() - (daysToKeep * 24 * 60 * 60 * 1000);
    const result = db.prepare(`DELETE FROM trades WHERE time < ?`).run(cutoffTime);
    console.log(`🧹 Cleaned up ${result.changes} old trades`);
    return result.changes;
}

/**
 * Get database stats
 */
function getStats() {
    const tradeCount = db.prepare(`SELECT COUNT(*) as count FROM trades`).get().count;
    const symbols = db.prepare(`SELECT DISTINCT symbol FROM trades`).all().map(r => r.symbol);
    const oldestTrade = db.prepare(`SELECT MIN(time) as time FROM trades`).get().time;
    const newestTrade = db.prepare(`SELECT MAX(time) as time FROM trades`).get().time;

    return {
        tradeCount,
        symbols,
        oldestTrade: oldestTrade ? new Date(oldestTrade).toISOString() : null,
        newestTrade: newestTrade ? new Date(newestTrade).toISOString() : null
    };
}

/**
 * Insert heatmap snapshots (batch)
 */
const insertSnapshotStmt = db.prepare(`
    INSERT OR IGNORE INTO heatmap_snapshots (symbol, time, bids, asks, max_volume)
    VALUES (?, ?, ?, ?, ?)
`);

const insertSnapshots = db.transaction((symbol, snapshots) => {
    for (const snap of snapshots) {
        insertSnapshotStmt.run(
            symbol.toUpperCase(),
            snap.time,
            JSON.stringify(snap.bids),
            JSON.stringify(snap.asks),
            snap.maxVolume || 0
        );
    }
});

/**
 * Get heatmap snapshots since a specific time with DOWN-SAMPLING
 * @param {string} symbol
 * @param {number} sinceTime
 * @param {number} limit Number of RESULT snapshots to return
 * @param {number} step Step size (1 = all, 10 = every 10th)
 */
function getSnapshots(symbol, sinceTime = 0, limit = 1000, step = 1) {
    // Optimized: Use SQL modulo arithmetic to sample data directly in the database.
    // This avoids reading 150k rows into memory when we only want 2.5k.
    // We assume 'id' roughly correlates with time (which it does for sequential inserts).

    // Fallback: If step is 1, don't use modulo (get everything)
    let query;
    let params;

    if (step > 1) {
        query = `
            SELECT time, bids, asks, max_volume as maxVolume
            FROM heatmap_snapshots
            WHERE symbol = ? AND time > ? AND (id % ?) = 0
            ORDER BY time DESC
            LIMIT ?
        `;
        params = [symbol.toUpperCase(), sinceTime, step, limit];
    } else {
        query = `
            SELECT time, bids, asks, max_volume as maxVolume
            FROM heatmap_snapshots
            WHERE symbol = ? AND time > ?
            ORDER BY time DESC
            LIMIT ?
        `;
        params = [symbol.toUpperCase(), sinceTime, limit];
    }

    const rows = db.prepare(query).all(...params);

    // Parse JSON
    return rows.reverse().map(r => ({
        time: r.time,
        bids: JSON.parse(r.bids),
        asks: JSON.parse(r.asks),
        maxVolume: r.maxVolume
    }));
}


module.exports = {
    db,
    insertTrade,
    insertTrades,
    getTrades,
    getLatestTrades,
    upsertCandle,
    getCandles,
    saveSession,
    getPreviousSession,
    cleanupOldData,
    getStats,
    insertSnapshots,
    getSnapshots
};
