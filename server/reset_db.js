/**
 * Reset Database Script
 * Deletes existing cryptoflow.db and ml_paper_trading.db to start fresh
 * 
 * Usage: node reset_db.js
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, 'data');
const CRYPTOFLOW_DB = path.join(DATA_DIR, 'cryptoflow.db');
const CRYPTOFLOW_WAL = path.join(DATA_DIR, 'cryptoflow.db-wal');
const CRYPTOFLOW_SHM = path.join(DATA_DIR, 'cryptoflow.db-shm');
const PAPER_DB = path.join(DATA_DIR, 'ml_paper_trading.db');
const PAPER_WAL = path.join(DATA_DIR, 'ml_paper_trading.db-wal');
const PAPER_SHM = path.join(DATA_DIR, 'ml_paper_trading.db-shm');

console.log('========================================');
console.log('  CryptoFlow Database Reset');
console.log('========================================\n');

const filesToDelete = [
    CRYPTOFLOW_DB, CRYPTOFLOW_WAL, CRYPTOFLOW_SHM,
    PAPER_DB, PAPER_WAL, PAPER_SHM
];

let deleted = 0;
for (const file of filesToDelete) {
    if (fs.existsSync(file)) {
        fs.unlinkSync(file);
        console.log(`Deleted: ${path.basename(file)}`);
        deleted++;
    }
}

if (deleted === 0) {
    console.log('No database files found to delete.');
} else {
    console.log(`\nDeleted ${deleted} file(s).`);
}

console.log('\nDatabase reset complete.');
console.log('Run "pm2 restart collector" to recreate and populate candles.');
console.log('Run "pm2 restart paper-trading" to restart paper trading.');
