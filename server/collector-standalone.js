#!/usr/bin/env node
/**
 * CryptoFlow Standalone Data Collector
 * Runs independently from API server for continuous data collection
 *
 * Usage: node collector-standalone.js
 * Or: npm run start:collector
 */

const { start, CONFIG } = require('./collector.js');

console.log('='.repeat(50));
console.log('CryptoFlow Data Collector (Standalone)');
console.log('='.repeat(50));
console.log('Exchanges:', Object.keys(CONFIG.exchanges).filter(e => CONFIG.exchanges[e].enabled).join(', '));
console.log('Symbols:', CONFIG.symbols.join(', '));
console.log('='.repeat(50));

// Start collector
start();

// Keep process alive
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    // Don't exit - keep collecting
});

process.on('unhandledRejection', (err) => {
    console.error('Unhandled Rejection:', err);
    // Don't exit - keep collecting
});
