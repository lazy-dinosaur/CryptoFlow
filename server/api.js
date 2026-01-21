/**
 * CryptoFlow API Server
 * REST + WebSocket API for candle data access
 */

const express = require('express');
const cors = require('cors');
const { WebSocketServer, WebSocket } = require('ws');
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const db = require('./db.js');
const { CONFIG } = require('./collector.js');

const PORT = process.env.PORT || 443;

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
        exchanges: {
            binance: CONFIG.exchanges?.binance?.symbols?.map(s => s.toUpperCase()) || [],
            bybit: CONFIG.exchanges?.bybit?.symbols?.map(s => s.toUpperCase()) || [],
            bitget: CONFIG.exchanges?.bitget?.symbols?.map(s => s.toUpperCase()) || []
        }
    });
});

/**
 * Get candles for a symbol and timeframe
 * Query params: symbol, tf (15, 60, 240, 1440), limit, exchange
 */
app.get('/api/candles', (req, res) => {
    const baseSymbol = (req.query.symbol || 'btcusdt').toUpperCase();
    const exchange = (req.query.exchange || 'binance').toUpperCase();
    const tf = parseInt(req.query.tf || '15', 10);
    const limit = parseInt(req.query.limit || '1000', 10);

    const fullSymbol = `${exchange}:${baseSymbol}`;

    // Only 15m, 1h, 4h, 1d timeframes for trading
    const tableMap = {
        15: 'candles_15',
        60: 'candles_60',
        240: 'candles_240',
        1440: 'candles_1440'
    };

    const table = tableMap[tf];
    if (!table) {
        return res.status(400).json({ error: 'Invalid timeframe. Use 15, 60, 240, or 1440' });
    }

    try {
        const candles = db.getCandles(table, fullSymbol, Math.min(limit, 5000));
        res.json({ candles, exchange, symbol: baseSymbol.toUpperCase() });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ==================== WEBSOCKET SERVER ====================

let server;
try {
    const sslOptions = {
        key: fs.readFileSync(SSL_KEY),
        cert: fs.readFileSync(SSL_CERT)
    };
    server = https.createServer(sslOptions, app);
    console.log('HTTPS mode enabled');
} catch (err) {
    console.log('SSL not found, using HTTP');
    server = http.createServer(app);
}

const wss = new WebSocketServer({ server });
const subscriptions = new Map();

wss.on('connection', (ws) => {
    console.log('Client connected');
    subscriptions.set(ws, new Set());

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data);

            if (msg.type === 'subscribe') {
                const symbol = msg.symbol?.toLowerCase();
                const exchange = msg.exchange?.toLowerCase() || 'binance';
                if (symbol && CONFIG.symbols.includes(symbol)) {
                    const fullSymbol = `${exchange}:${symbol}`;
                    subscriptions.get(ws).add(fullSymbol);
                    console.log(`Subscribed: ${fullSymbol}`);

                    // Send initial candles (15m timeframe)
                    const candles = db.getCandles('candles_15', fullSymbol, 1000);
                    ws.send(JSON.stringify({
                        type: 'history',
                        symbol: symbol.toUpperCase(),
                        exchange,
                        candles
                    }));
                }
            }

            if (msg.type === 'unsubscribe') {
                const symbol = msg.symbol?.toLowerCase();
                const exchange = msg.exchange?.toLowerCase() || 'binance';
                if (symbol) {
                    subscriptions.get(ws).delete(`${exchange}:${symbol}`);
                }
            }
        } catch (err) {
            console.error('WebSocket error:', err.message);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        subscriptions.delete(ws);
    });

    ws.on('error', (err) => {
        console.error('WebSocket error:', err.message);
    });
});

/**
 * Broadcast candle update
 */
function broadcastCandle(fullSymbol, candle) {
    const msg = JSON.stringify({
        type: 'candle',
        symbol: fullSymbol,
        candle
    });

    for (const [ws, subs] of subscriptions) {
        if (subs.has(fullSymbol) && ws.readyState === WebSocket.OPEN) {
            ws.send(msg);
        }
    }
}

// ==================== STARTUP ====================

function start() {
    server.listen(PORT, () => {
        const protocol = PORT === 443 ? 'https' : 'http';
        console.log(`API running on ${protocol}://localhost:${PORT}`);
    });
}

if (require.main === module) {
    start();
}

module.exports = { app, server, broadcastCandle };
