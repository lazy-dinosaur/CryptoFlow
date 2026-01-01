/**
 * CryptoFlow VPS Data Collector
 * Runs 24/7 to collect trade data from Binance and store in SQLite
 */

const WebSocket = require('ws');
const db = require('./db.js');

// Configuration
const CONFIG = {
    symbols: ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt'],
    tickSizes: {
        btcusdt: 10,
        ethusdt: 1,
        solusdt: 0.1,
        bnbusdt: 0.1
    },
    // Cleanup old data every hour
    cleanupIntervalMs: 60 * 60 * 1000,
    daysToKeep: 7,
    // Reconnect delay
    reconnectDelayMs: 5000
};

// State for each symbol
const symbolState = new Map();

/**
 * Round price to tick size
 */
function roundToTick(price, tickSize) {
    return Math.round(price / tickSize) * tickSize;
}

/**
 * Get candle start time for a given timestamp and timeframe (minutes)
 */
function getCandleStart(timestamp, timeframeMinutes) {
    const ms = timeframeMinutes * 60 * 1000;
    return Math.floor(timestamp / ms) * ms;
}

/**
 * Initialize or get state for a symbol
 */
function getState(symbol) {
    if (!symbolState.has(symbol)) {
        symbolState.set(symbol, {
            currentCandles: new Map(), // timeframe -> current candle
            vwapSumPriceQty: 0,
            vwapSumQty: 0,
            volumeProfile: new Map(),
            tradeCount: 0,
            lastSave: Date.now()
        });
    }
    return symbolState.get(symbol);
}

/**
 * Process a trade and update candles
 */
function processTrade(symbol, trade) {
    const state = getState(symbol);
    const tickSize = CONFIG.tickSizes[symbol] || 1;
    const roundedPrice = roundToTick(trade.price, tickSize);
    const isBuy = !trade.isBuyerMaker;

    // Store raw trade
    db.insertTrade(symbol, trade);

    // Update VWAP
    state.vwapSumPriceQty += trade.price * trade.quantity;
    state.vwapSumQty += trade.quantity;

    // Update volume profile
    const vpEntry = state.volumeProfile.get(roundedPrice) || { buy: 0, sell: 0 };
    if (isBuy) {
        vpEntry.buy += trade.quantity;
    } else {
        vpEntry.sell += trade.quantity;
    }
    state.volumeProfile.set(roundedPrice, vpEntry);

    state.tradeCount++;

    // Update candles for all timeframes
    const timeframes = [
        { name: '1m', minutes: 1, table: 'candles_1' },
        { name: '5m', minutes: 5, table: 'candles_5' },
        { name: '15m', minutes: 15, table: 'candles_15' },
        { name: '1h', minutes: 60, table: 'candles_60' }
    ];

    for (const tf of timeframes) {
        updateCandle(symbol, trade, roundedPrice, isBuy, tf);
    }

    // Save candles periodically (every 10 seconds)
    if (Date.now() - state.lastSave > 10000) {
        saveAllCandles(symbol);
        state.lastSave = Date.now();
    }
}

/**
 * Update a candle for a specific timeframe
 */
function updateCandle(symbol, trade, roundedPrice, isBuy, tf) {
    const state = getState(symbol);
    const candleStart = getCandleStart(trade.time, tf.minutes);

    let candle = state.currentCandles.get(tf.name);

    // Check if we need a new candle
    if (!candle || candle.time !== candleStart) {
        // Save the old candle
        if (candle) {
            candle.clusters = Object.fromEntries(candle._clusters);
            db.upsertCandle(tf.table, symbol, candle);
        }

        // Start new candle
        candle = {
            time: candleStart,
            open: trade.price,
            high: trade.price,
            low: trade.price,
            close: trade.price,
            volume: 0,
            buyVolume: 0,
            sellVolume: 0,
            delta: 0,
            tradeCount: 0,
            _clusters: new Map()
        };
        state.currentCandles.set(tf.name, candle);
    }

    // Update OHLC
    candle.high = Math.max(candle.high, trade.price);
    candle.low = Math.min(candle.low, trade.price);
    candle.close = trade.price;

    // Update volume
    candle.volume += trade.quantity;
    candle.tradeCount++;

    if (isBuy) {
        candle.buyVolume += trade.quantity;
        candle.delta += trade.quantity;
    } else {
        candle.sellVolume += trade.quantity;
        candle.delta -= trade.quantity;
    }

    // Update cluster
    const cluster = candle._clusters.get(roundedPrice) || { bid: 0, ask: 0 };
    if (isBuy) {
        cluster.bid += trade.quantity;
    } else {
        cluster.ask += trade.quantity;
    }
    candle._clusters.set(roundedPrice, cluster);
}

/**
 * Save all current candles
 */
function saveAllCandles(symbol) {
    const state = getState(symbol);
    const timeframes = [
        { name: '1m', table: 'candles_1' },
        { name: '5m', table: 'candles_5' },
        { name: '15m', table: 'candles_15' },
        { name: '1h', table: 'candles_60' }
    ];

    for (const tf of timeframes) {
        const candle = state.currentCandles.get(tf.name);
        if (candle) {
            const candleToSave = {
                ...candle,
                clusters: candle._clusters ? Object.fromEntries(candle._clusters) : {}
            };
            db.upsertCandle(tf.table, symbol, candleToSave);
        }
    }
}

/**
 * Connect to Binance WebSocket for a symbol
 */
function connectSymbol(symbol) {
    const wsUrl = `wss://fstream.binance.com/ws/${symbol}@aggTrade`;

    console.log(`🔌 Connecting to ${symbol.toUpperCase()}...`);

    const ws = new WebSocket(wsUrl);

    ws.on('open', () => {
        console.log(`✅ Connected to ${symbol.toUpperCase()}`);
    });

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data);

            const trade = {
                price: parseFloat(msg.p),
                quantity: parseFloat(msg.q),
                time: msg.T,
                isBuyerMaker: msg.m,
                tradeId: msg.a.toString()
            };

            processTrade(symbol, trade);
        } catch (err) {
            console.error(`Error processing message for ${symbol}:`, err.message);
        }
    });

    ws.on('error', (err) => {
        console.error(`❌ WebSocket error for ${symbol}:`, err.message);
    });

    ws.on('close', () => {
        console.log(`⚠️ Disconnected from ${symbol.toUpperCase()}, reconnecting...`);
        setTimeout(() => connectSymbol(symbol), CONFIG.reconnectDelayMs);
    });

    return ws;
}

/**
 * Save session data at midnight UTC
 */
function checkSessionReset() {
    const now = new Date();
    const utcHour = now.getUTCHours();
    const utcMinute = now.getUTCMinutes();

    // Check at 00:00 UTC
    if (utcHour === 0 && utcMinute === 0) {
        const yesterday = new Date(now);
        yesterday.setUTCDate(yesterday.getUTCDate() - 1);
        const dateStr = yesterday.toISOString().split('T')[0];

        for (const symbol of CONFIG.symbols) {
            const state = getState(symbol);

            // Calculate POC from volume profile
            let poc = null;
            let maxVol = 0;
            for (const [price, vol] of state.volumeProfile) {
                const total = vol.buy + vol.sell;
                if (total > maxVol) {
                    maxVol = total;
                    poc = price;
                }
            }

            // Calculate VWAP
            const vwap = state.vwapSumQty > 0 ? state.vwapSumPriceQty / state.vwapSumQty : null;

            // Save session
            db.saveSession(symbol, dateStr, {
                poc,
                vwap,
                vah: null, // Could calculate from volume profile
                val: null,
                totalVolume: state.vwapSumQty
            });

            console.log(`📅 Saved session for ${symbol}: POC=${poc}, VWAP=${vwap?.toFixed(2)}`);

            // Reset session state
            state.vwapSumPriceQty = 0;
            state.vwapSumQty = 0;
            state.volumeProfile.clear();
        }
    }
}

/**
 * Main startup
 */
function start() {
    console.log('🚀 CryptoFlow Data Collector starting...');
    console.log(`📊 Symbols: ${CONFIG.symbols.join(', ')}`);

    // Show current stats
    const stats = db.getStats();
    console.log(`📈 Database: ${stats.tradeCount} trades`);
    if (stats.oldestTrade) {
        console.log(`   Oldest: ${stats.oldestTrade}`);
        console.log(`   Newest: ${stats.newestTrade}`);
    }

    const DepthCollector = require('./depth-collector.js');

    // ... (existing code)

    // Connect to all symbols
    for (const symbol of CONFIG.symbols) {
        connectSymbol(symbol);

        // 🔥 START DEPTH COLLECTOR (BTC ONLY)
        // We only collect depth history for BTC to save resources and allow deeper storage
        if (symbol.toLowerCase() === 'btcusdt') {
            const dc = new DepthCollector(symbol);
            dc.start();
        }
    }

    // Periodic cleanup
    setInterval(() => {
        db.cleanupOldData(CONFIG.daysToKeep);
    }, CONFIG.cleanupIntervalMs);

    // Check for session reset every minute
    setInterval(checkSessionReset, 60000);

    // Save all candles on exit
    process.on('SIGINT', () => {
        console.log('\n🛑 Shutting down...');
        for (const symbol of CONFIG.symbols) {
            saveAllCandles(symbol);
        }
        console.log('✅ All candles saved');
        process.exit(0);
    });

    console.log('✅ Collector started');
}

// Start if run directly
if (require.main === module) {
    start();
}

module.exports = { start, CONFIG };
