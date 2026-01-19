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
    // Exchange configuration
    exchanges: {
        binance: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt']
        },
        bybit: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt']  // Bybit doesn't have BNBUSDT perpetual
        },
        bitget: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt']
        }
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
 * @param {string} fullSymbol - Exchange-prefixed symbol (e.g., "binance:btcusdt")
 * @param {object} trade - Trade data
 * @param {string} baseSymbol - Base symbol without exchange prefix (e.g., "btcusdt")
 */
function processTrade(fullSymbol, trade, baseSymbol = null) {
    const symbol = baseSymbol || fullSymbol;
    const state = getState(fullSymbol);
    const tickSize = CONFIG.tickSizes[symbol] || 1;
    const roundedPrice = roundToTick(trade.price, tickSize);
    const isBuy = trade.isBuyerMaker; // INVERTED LOGIC (Matches Frontend)

    // Store raw trade (with exchange prefix)
    db.insertTrade(fullSymbol, trade);

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
        { name: '30m', minutes: 30, table: 'candles_30' },
        { name: '1h', minutes: 60, table: 'candles_60' },
        { name: '4h', minutes: 240, table: 'candles_240' },
        { name: '1d', minutes: 1440, table: 'candles_1440' }
    ];

    for (const tf of timeframes) {
        updateCandle(fullSymbol, trade, roundedPrice, isBuy, tf);
    }

    // Save candles periodically (every 10 seconds)
    if (Date.now() - state.lastSave > 10000) {
        saveAllCandles(fullSymbol);
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
        { name: '30m', table: 'candles_30' },
        { name: '1h', table: 'candles_60' },
        { name: '4h', table: 'candles_240' },
        { name: '1d', table: 'candles_1440' }
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
function connectBinance(symbol) {
    const wsUrl = `wss://fstream.binance.com/ws/${symbol}@aggTrade`;
    const fullSymbol = `binance:${symbol}`;

    console.log(`🔌 [Binance] Connecting to ${symbol.toUpperCase()}...`);

    const ws = new WebSocket(wsUrl);

    ws.on('open', () => {
        console.log(`✅ [Binance] Connected to ${symbol.toUpperCase()}`);
    });

    ws.on('message', (data) => {
        try {
            const msg = JSON.parse(data);

            const trade = {
                price: parseFloat(msg.p),
                quantity: parseFloat(msg.q),
                time: msg.T,
                isBuyerMaker: msg.m,
                tradeId: `binance_${msg.a}`
            };

            processTrade(fullSymbol, trade, symbol);
        } catch (err) {
            console.error(`[Binance] Error processing message for ${symbol}:`, err.message);
        }
    });

    ws.on('error', (err) => {
        console.error(`❌ [Binance] WebSocket error for ${symbol}:`, err.message);
    });

    ws.on('close', () => {
        console.log(`⚠️ [Binance] Disconnected from ${symbol.toUpperCase()}, reconnecting...`);
        setTimeout(() => connectBinance(symbol), CONFIG.reconnectDelayMs);
    });

    return ws;
}

/**
 * Connect to Bybit WebSocket for symbols
 * Bybit uses a single connection with multiple subscriptions
 */
function connectBybit(symbols) {
    const wsUrl = 'wss://stream.bybit.com/v5/public/linear';
    let ws = null;
    let pingInterval = null;

    function connect() {
        console.log(`🔌 [Bybit] Connecting...`);

        ws = new WebSocket(wsUrl);

        ws.on('open', () => {
            console.log(`✅ [Bybit] Connected`);

            // Subscribe to all symbols
            const args = symbols.map(s => `publicTrade.${s.toUpperCase()}`);
            ws.send(JSON.stringify({
                op: 'subscribe',
                args: args
            }));

            // Bybit requires ping every 20 seconds
            pingInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ op: 'ping' }));
                }
            }, 20000);
        });

        ws.on('message', (data) => {
            try {
                const msg = JSON.parse(data);

                // Handle pong
                if (msg.op === 'pong' || msg.ret_msg === 'pong') return;

                // Handle subscription confirmation
                if (msg.op === 'subscribe') {
                    if (msg.success) {
                        console.log(`📡 [Bybit] Subscribed to: ${symbols.map(s => s.toUpperCase()).join(', ')}`);
                    }
                    return;
                }

                // Handle trade data
                if (msg.topic && msg.topic.startsWith('publicTrade.') && msg.data) {
                    const symbol = msg.topic.replace('publicTrade.', '').toLowerCase();
                    const fullSymbol = `bybit:${symbol}`;

                    for (const t of msg.data) {
                        const trade = {
                            price: parseFloat(t.p),
                            quantity: parseFloat(t.v),
                            time: t.T,
                            isBuyerMaker: t.S === 'Sell',  // Bybit: "Buy" = buyer aggressor, "Sell" = seller aggressor
                            tradeId: `bybit_${t.i}`
                        };

                        processTrade(fullSymbol, trade, symbol);
                    }
                }
            } catch (err) {
                console.error(`[Bybit] Error processing message:`, err.message);
            }
        });

        ws.on('error', (err) => {
            console.error(`❌ [Bybit] WebSocket error:`, err.message);
        });

        ws.on('close', () => {
            console.log(`⚠️ [Bybit] Disconnected, reconnecting...`);
            if (pingInterval) clearInterval(pingInterval);
            setTimeout(connect, CONFIG.reconnectDelayMs);
        });
    }

    connect();
    return ws;
}

/**
 * Connect to Bitget WebSocket for symbols
 */
function connectBitget(symbols) {
    const wsUrl = 'wss://ws.bitget.com/v2/ws/public';
    let ws = null;
    let pingInterval = null;

    function connect() {
        console.log(`🔌 [Bitget] Connecting...`);

        ws = new WebSocket(wsUrl);

        ws.on('open', () => {
            console.log(`✅ [Bitget] Connected`);

            // Subscribe to all symbols
            const args = symbols.map(s => ({
                instType: 'USDT-FUTURES',
                channel: 'trade',
                instId: s.toUpperCase()
            }));
            ws.send(JSON.stringify({
                op: 'subscribe',
                args: args
            }));

            // Bitget requires ping every 30 seconds
            pingInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        });

        ws.on('message', (data) => {
            try {
                // Handle pong
                if (data.toString() === 'pong') return;

                const msg = JSON.parse(data);

                // Handle subscription confirmation
                if (msg.event === 'subscribe') {
                    console.log(`📡 [Bitget] Subscribed to: ${symbols.map(s => s.toUpperCase()).join(', ')}`);
                    return;
                }

                // Handle trade data (snapshot = initial, update = ongoing)
                if ((msg.action === 'snapshot' || msg.action === 'update') && msg.arg?.channel === 'trade' && msg.data) {
                    const symbol = msg.arg.instId.toLowerCase();
                    const fullSymbol = `bitget:${symbol}`;

                    for (const t of msg.data) {
                        const trade = {
                            price: parseFloat(t.price),
                            quantity: parseFloat(t.size),
                            time: parseInt(t.ts),
                            isBuyerMaker: t.side === 'sell',  // Bitget: "buy" = buyer aggressor
                            tradeId: `bitget_${t.tradeId}`
                        };

                        processTrade(fullSymbol, trade, symbol);
                    }
                }
            } catch (err) {
                console.error(`[Bitget] Error processing message:`, err.message);
            }
        });

        ws.on('error', (err) => {
            console.error(`❌ [Bitget] WebSocket error:`, err.message);
        });

        ws.on('close', () => {
            console.log(`⚠️ [Bitget] Disconnected, reconnecting...`);
            if (pingInterval) clearInterval(pingInterval);
            setTimeout(connect, CONFIG.reconnectDelayMs);
        });
    }

    connect();
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

    // Connect to Binance
    if (CONFIG.exchanges.binance.enabled) {
        console.log(`\n📊 [Binance] Symbols: ${CONFIG.exchanges.binance.symbols.join(', ')}`);
        for (const symbol of CONFIG.exchanges.binance.symbols) {
            connectBinance(symbol);

            // 🔥 START DEPTH COLLECTOR (BTC ONLY)
            if (symbol.toLowerCase() === 'btcusdt') {
                const dc = new DepthCollector(`binance:${symbol}`);
                dc.start();
            }
        }
    }

    // Connect to Bybit
    if (CONFIG.exchanges.bybit.enabled) {
        console.log(`\n📊 [Bybit] Symbols: ${CONFIG.exchanges.bybit.symbols.join(', ')}`);
        connectBybit(CONFIG.exchanges.bybit.symbols);
    }

    // Connect to Bitget
    if (CONFIG.exchanges.bitget.enabled) {
        console.log(`\n📊 [Bitget] Symbols: ${CONFIG.exchanges.bitget.symbols.join(', ')}`);
        connectBitget(CONFIG.exchanges.bitget.symbols);
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
        // Save candles for all exchange:symbol combinations
        for (const [fullSymbol] of symbolState) {
            saveAllCandles(fullSymbol);
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
