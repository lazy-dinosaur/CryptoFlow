/**
 * CryptoFlow Database Module
 * SQLite database for candle storage only (no trades, no heatmap)
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
db.pragma('busy_timeout = 2000');
db.pragma('synchronous = NORMAL');
db.pragma('wal_autocheckpoint = 1000');

/**
 * Initialize database schema (candles only)
 */
function initSchema() {
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

    console.log('Database schema initialized (candles only)');
}

initSchema();

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
        candle.buyVolume || 0,
        candle.sellVolume || 0,
        candle.delta || 0,
        candle.tradeCount || 0,
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
 * Get latest candle timestamp for a symbol
 */
function getLastCandleTime(tableName, symbol) {
    const row = db.prepare(`
        SELECT time
        FROM ${tableName}
        WHERE symbol = ?
        ORDER BY time DESC
        LIMIT 1
    `).get(symbol.toUpperCase());
    return row ? row.time : null;
}

/**
 * Get database stats (candle counts)
 */
function getStats() {
    const timeframes = ['1', '5', '15', '30', '60', '240', '1440'];
    const stats = {};
    
    for (const tf of timeframes) {
        try {
            const count = db.prepare(`SELECT COUNT(*) as count FROM candles_${tf}`).get().count;
            stats[`candles_${tf}`] = count;
        } catch (e) {
            stats[`candles_${tf}`] = 0;
        }
    }
    
    return stats;
}

module.exports = {
    db,
    upsertCandle,
    getCandles,
    getLastCandleTime,
    getStats
};
