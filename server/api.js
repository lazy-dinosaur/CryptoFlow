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
const { start: startCollector, CONFIG } = require('./collector.js');

// Configuration
const PORT = process.env.PORT || 443;  // HTTPS port (also handles WSS)

// SSL Certificates
const SSL_KEY = path.join(__dirname, '..', 'ssl', 'key.pem');
const SSL_CERT = path.join(__dirname, '..', 'ssl', 'cert.pem');

const app = express();
const compression = require('compression');
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '50mb' }));

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
 * Get available symbols
 */
app.get('/api/symbols', (req, res) => {
    res.json({
        symbols: CONFIG.symbols.map(s => s.toUpperCase()),
        tickSizes: CONFIG.tickSizes
    });
});

/**
 * Get candles for a symbol and timeframe
 * Query params: symbol, tf (1, 5, 15, 60), limit
 */
app.get('/api/candles', (req, res) => {
    const symbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const tf = parseInt(req.query.tf || '1', 10);
    const limit = parseInt(req.query.limit || '1000', 10);

    const tableMap = {
        1: 'candles_1',
        5: 'candles_5',
        15: 'candles_15',
        60: 'candles_60'
    };

    const table = tableMap[tf];
    if (!table) {
        return res.status(400).json({ error: 'Invalid timeframe. Use 1, 5, 15, or 60' });
    }

    try {
        const candles = db.getCandles(table, symbol, Math.min(limit, 5000));
        res.json({ candles });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get raw trades
 * Query params: symbol, from (timestamp), to (timestamp), limit
 */
app.get('/api/trades', (req, res) => {
    const symbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const limit = parseInt(req.query.limit || '10000', 10);

    try {
        if (req.query.from && req.query.to) {
            const trades = db.getTrades(symbol, parseInt(req.query.from), parseInt(req.query.to));
            res.json({ trades });
        } else {
            const trades = db.getLatestTrades(symbol, Math.min(limit, 50000));
            res.json({ trades });
        }
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get session markers (VWAP, POC, PVOC)
 */
app.get('/api/session', (req, res) => {
    const symbol = (req.query.symbol || 'btcusdt').toLowerCase();

    try {
        const previousSession = db.getPreviousSession(symbol);

        // Note: Current session data is in collector's memory
        // For full implementation, collector would expose current state
        res.json({
            previous: previousSession || null
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get heatmap snapshots
 * Query params: symbol, since (timestamp), limit
 */
app.get('/api/depth', (req, res) => {
    const symbol = (req.query.symbol || 'btcusdt').toLowerCase();
    const since = parseInt(req.query.since || '0', 10);
    const limit = parseInt(req.query.limit || '1000', 10);
    const step = parseInt(req.query.step || '1', 10);

    try {
        const snapshots = db.getSnapshots(symbol, since, limit, step);
        res.json({ snapshots });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Save heatmap snapshots (batch sync from client)
 */
// Deprecated/Removed for Security: /api/depth/batch
// Client does not upload data anymore. Server is autonomous.

// ==================== ML PROXY (DISABLED) ====================

/**
 * Proxy requests to Python ML Service (Port 5001)
 * DISABLED by request to prevent OpenRouter costs
 */
app.all(['/api/ml/*', '/api/ai/*'], async (req, res) => {
    return res.status(403).json({
        error: 'AI Service Disabled',
        details: 'AI features are currently disabled for public use.'
    });
});

// ==================== ANALYTICS ====================

const crypto = require('crypto');

/**
 * Hash IP for privacy
 */
function hashIP(ip) {
    if (!ip) return null;
    return crypto.createHash('sha256').update(ip + 'cryptoflow-salt').digest('hex').substring(0, 16);
}

/**
 * Record a pageview
 */
app.post('/api/analytics/pageview', (req, res) => {
    try {
        const clientIP = req.headers['x-forwarded-for']?.split(',')[0] || req.ip;
        db.insertPageview({
            timestamp: Date.now(),
            path: req.body.path || '/',
            referrer: req.body.referrer || req.headers['referer'] || null,
            userAgent: req.headers['user-agent'] || null,
            ipHash: hashIP(clientIP),
            sessionId: req.body.sessionId || null
        });
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Record a click event
 */
app.post('/api/analytics/click', (req, res) => {
    try {
        db.insertClick({
            timestamp: Date.now(),
            path: req.body.path || '/',
            elementId: req.body.elementId || null,
            elementClass: req.body.elementClass || null,
            elementTag: req.body.elementTag || null,
            x: req.body.x || 0,
            y: req.body.y || 0,
            sessionId: req.body.sessionId || null
        });
        res.json({ success: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/**
 * Get analytics stats (protected with simple key)
 */
app.get('/api/analytics/stats', (req, res) => {
    const key = req.query.key || req.headers['x-analytics-key'];
    if (key !== 'CryptoFlow2026!') {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    try {
        const stats = db.getAnalyticsStats();
        res.json(stats);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

/*
app.all(['/api/ml/*', '/api/ai/*'], async (req, res) => {
    // ... existing code commented out ...
*/

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
    console.log(`🚀 Starting server with PORT=${PORT}`);

    // Start HTTPS/HTTP server
    server.listen(PORT, '0.0.0.0', () => {
        const protocol = PORT === 443 ? 'https' : 'http';
        console.log(`🌐 REST API running on ${protocol}://0.0.0.0:${PORT}`);
        console.log(`📡 WebSocket running on wss://0.0.0.0:${PORT}`);

        // Start data collector after server is up (prevent blocking)
        console.log('⏳ Scheduler: Starting Data Collector in 5s...');
        setTimeout(() => {
            try {
                startCollector();
            } catch (e) {
                console.error('❌ Data Collector crashed:', e);
            }
        }, 5000);
    });
}

// Start if run directly
if (require.main === module) {
    start();
}

module.exports = { app, server, broadcastTrade, broadcastCandle };
