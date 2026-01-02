/**
 * RESET TRADES SCRIPT
 * Running this will delete all TRADE and CANDLE data to clear "inverted logic" history.
 * IT WILL KEEP 'heatmap_snapshots' (Liquidity History).
 */

const Database = require('better-sqlite3');
const path = require('path');

const DB_PATH = path.join(__dirname, 'data', 'cryptoflow.db');

console.log('🔄 Opening Database:', DB_PATH);
const db = new Database(DB_PATH);

try {
    console.log('🗑️ Deleting all TRADES (History cleanup)...');
    db.prepare('DELETE FROM trades').run();

    console.log('🗑️ Deleting all CANDLES...');
    db.prepare('DELETE FROM candles_1').run();
    db.prepare('DELETE FROM candles_5').run();
    db.prepare('DELETE FROM candles_15').run();
    db.prepare('DELETE FROM candles_60').run();

    console.log('🧹 Optimizing Database (VACUUM)...');
    db.exec('VACUUM');

    console.log('✅ TRADES DELETED. HEATMAP PRESERVED.');
    console.log('Please restart start-server.bat now.');

} catch (err) {
    console.error('❌ Error during reset:', err.message);
}
