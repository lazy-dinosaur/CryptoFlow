/**
 * CryptoFlow VPS API Server
 * REST + WebSocket API for frontend access to stored data
 */

const express = require('express');
const cors = require('cors');
const { WebSocketServer, WebSocket } = require('ws');
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const db = require('./db.js');
// Collector runs separately - only import CONFIG for symbol info
const { CONFIG } = require('./collector.js');

// Configuration
const PORT = process.env.PORT || 443;  // HTTPS port (also handles WSS)

// SSL Certificates
const SSL_KEY = path.join(__dirname, '..', 'ssl', 'key.pem');
const SSL_CERT = path.join(__dirname, '..', 'ssl', 'cert.pem');

const app = express();
const compression = require('compression');
app.use(compression());
app.use(cors());
app.use(express.json());

// Serve static frontend files
const distPath = path.join(__dirname, '..', 'dist');
app.use(express.static(distPath));

// Serve index.html for SPA routes
app.get('/', (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
});

// ==================== REST API ====================

/**
 * Health check
 */
app.get('/api/health', (req, res) => {
    const stats = db.getStats();
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        ...stats
    });
});

/**
 * Get available symbols and exchanges
 */
app.get('/api/symbols', (req, res) => {
    res.json({
        symbols: CONFIG.symbols.map(s => s.toUpperCase()),
        tickSizes: CONFIG.tickSizes,
        exchanges: {
            binance: CONFIG.exchanges?.binance?.symbols?.map(s => s.toUpperCase()) || CONFIG.symbols.map(s => s.toUpperCase()),
            bybit: CONFIG.exchanges?.bybit?.symbols?.map(s => s.toUpperCase()) || [],
            bitget: CONFIG.exchanges?.bitget?.symbols?.map(s => s.toUpperCase()) || []
        }
    });
});

/**
 * Get candles for a symbol and timeframe
 * Query params: symbol, tf (1, 5, 15, 60), limit, exchange (binance/bybit)
 */
app.get('/api/candles', (req, res) => {
    const baseSymbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const exchange = (req.query.exchange || 'binance').toLowerCase();
    const tf = parseInt(req.query.tf || '1', 10);
    const limit = parseInt(req.query.limit || '1000', 10);

    // Build full symbol with exchange prefix
    const fullSymbol = `${exchange}:${baseSymbol}`;

    const tableMap = {
        1: 'candles_1',
        5: 'candles_5',
        15: 'candles_15',
        30: 'candles_30',
        60: 'candles_60',
        240: 'candles_240',
        1440: 'candles_1440'
    };

    const table = tableMap[tf];
    if (!table) {
        return res.status(400).json({ error: 'Invalid timeframe. Use 1, 5, 15, 30, 60, 240, or 1440' });
    }

    try {
        const candles = db.getCandles(table, fullSymbol, Math.min(limit, 5000));
        res.json({ candles, exchange, symbol: baseSymbol.toUpperCase() });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get raw trades
 * Query params: symbol, from (timestamp), to (timestamp), limit, exchange (binance/bybit/bitget)
 */
app.get('/api/trades', (req, res) => {
    const baseSymbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const exchange = (req.query.exchange || 'binance').toLowerCase();
    const limit = parseInt(req.query.limit || '10000', 10);

    const fullSymbol = `${exchange}:${baseSymbol}`;

    try {
        if (req.query.from && req.query.to) {
            const trades = db.getTrades(fullSymbol, parseInt(req.query.from), parseInt(req.query.to));
            res.json({ trades, exchange });
        } else {
            const trades = db.getLatestTrades(fullSymbol, Math.min(limit, 50000));
            res.json({ trades, exchange });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get session markers (VWAP, POC, PVOC)
 */
app.get('/api/session', (req, res) => {
    const baseSymbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const exchange = (req.query.exchange || 'binance').toLowerCase();
    const fullSymbol = `${exchange}:${baseSymbol}`;

    try {
        const previousSession = db.getPreviousSession(fullSymbol);

        res.json({
            previous: previousSession || null,
            exchange
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get heatmap snapshots
 * Query params: symbol, since (timestamp), limit, exchange
 */
app.get('/api/depth', (req, res) => {
    const baseSymbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const exchange = (req.query.exchange || 'binance').toLowerCase();
    const since = parseInt(req.query.since || '0', 10);
    const limit = parseInt(req.query.limit || '1000', 10);
    const step = parseInt(req.query.step || '1', 10);

    const fullSymbol = `${exchange}:${baseSymbol}`;

    try {
        const snapshots = db.getSnapshots(fullSymbol, since, limit, step);
        res.json({ snapshots, exchange });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Save heatmap snapshots (batch sync from client)
 */
// Deprecated/Removed for Security: /api/depth/batch
// Client does not upload data anymore. Server is autonomous.

// ==================== ML PROXY ====================

/**
 * Proxy requests to Python ML Service (Port 5001)
 * Frontend -> Node (443) -> Python (5001)
 */
app.all('/api/ml/{*path}', async (req, res) => {
    // SECURITY: Protect training endpoint
    if (req.url.includes('/train')) {
        const apiKey = req.headers['x-api-key'];
        // Hardcoded key for "Public Alpha" simplicity. 
        // In production, use process.env.ADMIN_KEY
        if (apiKey !== 'CryptoFlowMasterKey2025!') {
            return res.status(401).json({ error: 'Unauthorized: Admin Key Required for Training' });
        }
    }

    const targetUrl = `http://127.0.0.1:5001${req.url}`;

    try {
        const options = {
            method: req.method,
            headers: { 'Content-Type': 'application/json' },
        };

        if (req.method !== 'GET' && req.method !== 'HEAD') {
            options.body = JSON.stringify(req.body);
        }

        const response = await fetch(targetUrl, options);

        if (!response.ok) {
            return res.status(response.status).json({ error: `ML Service Error: ${response.statusText}` });
        }

        const data = await response.json();
        res.json(data);
    } catch (err) {
        console.error('ML Proxy Error:', err.message);
        res.status(502).json({
            error: 'ML Service Unavailable',
            details: 'The AI Brain is starting up... please wait.'
        });
    }
});

// ==================== WEBSOCKET SERVER ====================

// Create HTTPS server with SSL certificates
let server;
try {
    const sslOptions = {
        key: fs.readFileSync(SSL_KEY),
        cert: fs.readFileSync(SSL_CERT)
    };
    server = https.createServer(sslOptions, app);
    console.log('🔐 HTTPS mode enabled with SSL certificates');
} catch (err) {
    console.log('⚠️ SSL certificates not found, falling back to HTTP');
    server = http.createServer(app);
}
// WebSocket server attached to HTTPS server (same port 443 for both HTTP and WS)
const wss = new WebSocketServer({ server });

// Track subscriptions
const subscriptions = new Map(); // ws -> Set<symbol>

wss.on('connection', (ws) => {
    console.log('🔌 Client connected');
    subscriptions.set(ws, new Set());

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data);

            if (msg.type === 'subscribe') {
                const symbol = msg.symbol?.toLowerCase();
                if (symbol && CONFIG.symbols.includes(symbol)) {
                    subscriptions.get(ws).add(symbol);
                    console.log(`📡 Client subscribed to ${symbol.toUpperCase()}`);

                    // Send initial candles
                    const candles = db.getCandles('candles_1', symbol, 1000);
                    ws.send(JSON.stringify({
                        type: 'history',
                        symbol: symbol.toUpperCase(),
                        candles
                    }));
                }
            }

            if (msg.type === 'unsubscribe') {
                const symbol = msg.symbol?.toLowerCase();
                if (symbol) {
                    subscriptions.get(ws).delete(symbol);
                }
            }
        } catch (err) {
            console.error('Error processing WebSocket message:', err.message);
        }
    });

    ws.on('close', () => {
        console.log('🔌 Client disconnected');
        subscriptions.delete(ws);
    });

    ws.on('error', (err) => {
        console.error('WebSocket error:', err.message);
    });
});

/**
 * Broadcast trade to all subscribed clients
 */
function broadcastTrade(symbol, trade) {
    const msg = JSON.stringify({
        type: 'trade',
        symbol: symbol.toUpperCase(),
        trade
    });

    for (const [ws, subs] of subscriptions) {
        if (subs.has(symbol) && ws.readyState === WebSocket.OPEN) {
            ws.send(msg);
        }
    }
}

/**
 * Broadcast candle update
 */
function broadcastCandle(symbol, candle) {
    const msg = JSON.stringify({
        type: 'candle',
        symbol: symbol.toUpperCase(),
        candle
    });

    for (const [ws, subs] of subscriptions) {
        if (subs.has(symbol) && ws.readyState === WebSocket.OPEN) {
            ws.send(msg);
        }
    }
}

// ==================== STARTUP ====================

function start() {
    // Start HTTPS/HTTP server
    server.listen(PORT, () => {
        const protocol = PORT === 443 ? 'https' : 'http';
        console.log(`🌐 REST API running on ${protocol}://localhost:${PORT}`);
        console.log(`📡 WebSocket running on wss://localhost:${PORT}`);
    });

    // Note: Collector runs separately via pm2 (npm run start:collector)
    console.log('💡 Collector should be started separately: npm run start:collector');
}

// Start if run directly
if (require.main === module) {
    start();
}

module.exports = { app, server, broadcastTrade, broadcastCandle };
