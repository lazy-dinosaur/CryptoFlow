/**
 * CryptoFlow Candle Collector
 * Fetches historical + realtime candles from futures exchanges via REST Kline API
 * No WebSocket trades, no heatmap, no delta from individual trades
 */

const db = require('./db.js');

// Configuration
const CONFIG = {
    symbols: ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt'],
    exchanges: {
        binance: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt'],
            historyDays: 730  // 2 years
        },
        bybit: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt'],
            historyDays: 730  // 2 years
        },
        bitget: {
            enabled: true,
            symbols: ['btcusdt', 'ethusdt', 'solusdt'],
            historyDays: 90,  // Bitget has limited history for small timeframes
            // 1m/5m/15m/30m/1h: ~30-60 days max, 4h/1d: 90+ days
        }
    },
    historyDays: 730,  // Default: 2 years
    pollIntervalMs: 60 * 1000,
    requestDelayMs: 200
};

// Only store timeframes needed for trading: 15m, 1h, 4h, 1d
const TIMEFRAMES = [
    { name: '15m', minutes: 15, table: 'candles_15', binance: '15m', bybit: '15', bitget: '15m' },
    { name: '1h', minutes: 60, table: 'candles_60', binance: '1h', bybit: '60', bitget: '1H' },
    { name: '4h', minutes: 240, table: 'candles_240', binance: '4h', bybit: '240', bitget: '4H' },
    { name: '1d', minutes: 1440, table: 'candles_1440', binance: '1d', bybit: 'D', bitget: '1D' }
];

const KLINE_LIMIT = 1000;

const EXCHANGE_ENDPOINTS = {
    binance: 'https://fapi.binance.com/fapi/v1/klines',
    bybit: 'https://api.bybit.com/v5/market/kline',
    bitget: 'https://api.bitget.com/api/v2/mix/market/candles'
};

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function getExchangeInterval(exchange, timeframe) {
    return timeframe[exchange];
}

function formatExchangeSymbol(exchange, symbol) {
    const upper = symbol.toUpperCase();
    // Bitget v2 API uses plain symbol without suffix
    return upper;
}

function parseKlines(exchange, rawList) {
    if (!Array.isArray(rawList)) return [];

    if (exchange === 'binance') {
        return rawList.map(k => {
            const openTime = Number(k[0]);
            const volume = parseFloat(k[5]);
            const takerBuy = parseFloat(k[9] || 0);
            const sellVolume = volume - takerBuy;
            return {
                time: openTime,
                open: parseFloat(k[1]),
                high: parseFloat(k[2]),
                low: parseFloat(k[3]),
                close: parseFloat(k[4]),
                volume,
                buyVolume: takerBuy,
                sellVolume: sellVolume,
                delta: takerBuy - sellVolume,
                tradeCount: Number(k[8] || 0),
                clusters: null
            };
        });
    }

    if (exchange === 'bybit') {
        return rawList.map(k => ({
            time: Number(k[0]),
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
            buyVolume: 0,
            sellVolume: 0,
            delta: 0,
            tradeCount: 0,
            clusters: null
        }));
    }

    if (exchange === 'bitget') {
        return rawList.map(k => ({
            time: Number(k[0]),
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
            buyVolume: 0,
            sellVolume: 0,
            delta: 0,
            tradeCount: 0,
            clusters: null
        }));
    }

    return [];
}

async function fetchKlines(exchange, symbol, timeframe, startTime, endTime, limit = KLINE_LIMIT) {
    const interval = getExchangeInterval(exchange, timeframe);
    const formattedSymbol = formatExchangeSymbol(exchange, symbol);

    let url;
    if (exchange === 'binance') {
        url = new URL(EXCHANGE_ENDPOINTS.binance);
        url.searchParams.set('symbol', formattedSymbol);
        url.searchParams.set('interval', interval);
        url.searchParams.set('startTime', String(startTime));
        url.searchParams.set('endTime', String(endTime));
        url.searchParams.set('limit', String(limit));
    } else if (exchange === 'bybit') {
        url = new URL(EXCHANGE_ENDPOINTS.bybit);
        url.searchParams.set('category', 'linear');
        url.searchParams.set('symbol', formattedSymbol);
        url.searchParams.set('interval', interval);
        url.searchParams.set('start', String(startTime));
        url.searchParams.set('end', String(endTime));
        url.searchParams.set('limit', String(limit));
    } else if (exchange === 'bitget') {
        url = new URL(EXCHANGE_ENDPOINTS.bitget);
        url.searchParams.set('productType', 'USDT-FUTURES');
        url.searchParams.set('symbol', formattedSymbol);
        url.searchParams.set('granularity', interval);
        url.searchParams.set('startTime', String(startTime));
        url.searchParams.set('endTime', String(endTime));
        url.searchParams.set('limit', String(limit));
    } else {
        return [];
    }

    const res = await fetch(url.toString());
    if (!res.ok) {
        throw new Error(`${exchange} kline HTTP ${res.status}`);
    }
    const data = await res.json();
    
    if (exchange === 'binance') {
        return parseKlines(exchange, data);
    }
    if (exchange === 'bybit') {
        const list = data?.result?.list || [];
        const candles = parseKlines(exchange, list);
        return candles.sort((a, b) => a.time - b.time);
    }
    if (exchange === 'bitget') {
        const list = data?.data || [];
        const candles = parseKlines(exchange, list);
        return candles.sort((a, b) => a.time - b.time);
    }
    return [];
}

function getLastCandleTime(tableName, symbol) {
    return db.getLastCandleTime(tableName, symbol);
}

const upsertCandles = db.db.transaction((tableName, symbol, candles) => {
    for (const candle of candles) {
        db.upsertCandle(tableName, symbol, candle);
    }
});

async function backfillCandles(exchange, symbol, timeframe) {
    const fullSymbol = `${exchange}:${symbol}`;
    const tfMs = timeframe.minutes * 60 * 1000;
    const now = Date.now();
    const endTime = Math.floor((now - tfMs) / tfMs) * tfMs;
    // Use exchange-specific history days if available
    const exchangeConfig = CONFIG.exchanges[exchange];
    const historyDays = exchangeConfig?.historyDays || CONFIG.historyDays;
    const cutoffTime = now - historyDays * 24 * 60 * 60 * 1000;
    const lastTime = getLastCandleTime(timeframe.table, fullSymbol);
    let startTime = cutoffTime;

    if (lastTime && lastTime + tfMs > startTime) {
        startTime = lastTime + tfMs;
    }

    startTime = Math.floor(startTime / tfMs) * tfMs;

    if (startTime >= endTime) {
        return 0;
    }

    let totalCandles = 0;
    let fetchStart = startTime;
    
    while (fetchStart < endTime) {
        const fetchEnd = Math.min(fetchStart + tfMs * (KLINE_LIMIT - 1), endTime);
        let candles = [];
        try {
            candles = await fetchKlines(exchange, symbol, timeframe, fetchStart, fetchEnd, KLINE_LIMIT);
        } catch (err) {
            console.error(`  [${exchange}] ${symbol} ${timeframe.name}: ${err.message}`);
            await sleep(CONFIG.requestDelayMs);
            break;
        }

        if (candles.length > 0) {
            upsertCandles(timeframe.table, fullSymbol, candles);
            totalCandles += candles.length;
            const lastCandle = candles[candles.length - 1];
            fetchStart = lastCandle.time + tfMs;
        } else {
            fetchStart += tfMs * KLINE_LIMIT;
        }

        if (candles.length < KLINE_LIMIT) {
            break;
        }
        await sleep(CONFIG.requestDelayMs);
    }
    
    return totalCandles;
}

let syncInProgress = false;
let backfillInProgress = false;

async function syncLatestCandles() {
    if (syncInProgress || backfillInProgress) return;
    syncInProgress = true;

    try {
        const now = Date.now();
        for (const [exchange, exchangeConfig] of Object.entries(CONFIG.exchanges)) {
            if (!exchangeConfig.enabled) continue;
            for (const symbol of exchangeConfig.symbols) {
                for (const timeframe of TIMEFRAMES) {
                    const tfMs = timeframe.minutes * 60 * 1000;
                    const fullSymbol = `${exchange}:${symbol}`;
                    const lastTime = getLastCandleTime(timeframe.table, fullSymbol);
                    const endTime = Math.floor((now - tfMs) / tfMs) * tfMs;

                    // If gap is too large, do backfill instead
                    if (lastTime && endTime - lastTime > tfMs * 2) {
                        await backfillCandles(exchange, symbol, timeframe);
                        continue;
                    }

                    const startTime = endTime - tfMs * 2;
                    let candles = [];
                    try {
                        candles = await fetchKlines(exchange, symbol, timeframe, startTime, endTime, 3);
                    } catch (err) {
                        // Silent fail for sync, will retry next interval
                        continue;
                    }

                    if (candles.length > 0) {
                        upsertCandles(timeframe.table, fullSymbol, candles);
                    }
                    await sleep(CONFIG.requestDelayMs);
                }
            }
        }
    } finally {
        syncInProgress = false;
    }
}

async function backfillAll() {
    backfillInProgress = true;
    try {
        console.log(`\n========================================`);
        console.log(`  Backfill History`);
        console.log(`========================================\n`);
        
        for (const [exchange, exchangeConfig] of Object.entries(CONFIG.exchanges)) {
            if (!exchangeConfig.enabled) continue;
            const days = exchangeConfig.historyDays || CONFIG.historyDays;
            console.log(`[${exchange.toUpperCase()}] ${exchangeConfig.symbols.join(', ')} (${days} days)`);
            
            for (const symbol of exchangeConfig.symbols) {
                for (const timeframe of TIMEFRAMES) {
                    const count = await backfillCandles(exchange, symbol, timeframe);
                    if (count > 0) {
                        console.log(`  ${symbol} ${timeframe.name}: +${count} candles`);
                    }
                    await sleep(CONFIG.requestDelayMs);
                }
            }
        }
        console.log(`\nBackfill complete.`);
    } finally {
        backfillInProgress = false;
    }
}

async function start() {
    console.log('========================================');
    console.log('  CryptoFlow Candle Collector');
    console.log('========================================');
    console.log(`Symbols: ${CONFIG.symbols.join(', ')}`);
    console.log(`Exchanges: ${Object.keys(CONFIG.exchanges).filter(e => CONFIG.exchanges[e].enabled).join(', ')}`);
    console.log(`History: ${CONFIG.historyDays} days`);
    console.log(`Poll interval: ${CONFIG.pollIntervalMs / 1000}s`);

    try {
        await backfillAll();
        await syncLatestCandles();
    } catch (err) {
        console.error('Initial sync failed:', err.message);
    }

    setInterval(() => {
        syncLatestCandles().catch(err => {
            console.error('Sync error:', err.message);
        });
    }, CONFIG.pollIntervalMs);

    console.log('\nCollector running. Press Ctrl+C to stop.');
}

if (require.main === module) {
    start().catch(err => {
        console.error('Collector failed to start:', err.message);
        process.exit(1);
    });
}

module.exports = { start, CONFIG };
