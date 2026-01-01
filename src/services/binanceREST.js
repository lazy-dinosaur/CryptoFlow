/**
 * Binance REST API Service
 * Fetches historical aggregate trades for footprint reconstruction
 */

const BINANCE_FUTURES_API = 'https://fapi.binance.com';

/**
 * Fetch historical aggregate trades from Binance Futures
 * @param {string} symbol - Trading pair (e.g., 'BTCUSDT')
 * @param {number} limit - Number of trades to fetch (max 1000)
 * @param {number} startTime - Start timestamp in ms (optional)
 * @param {number} endTime - End timestamp in ms (optional)
 * @returns {Promise<Array>} Array of trade objects
 */
export async function fetchHistoricalTrades(symbol, limit = 1000, startTime = null, endTime = null) {
    const params = new URLSearchParams({
        symbol: symbol.toUpperCase(),
        limit: Math.min(limit, 1000).toString()
    });

    if (startTime) params.append('startTime', startTime.toString());
    if (endTime) params.append('endTime', endTime.toString());

    const url = `${BINANCE_FUTURES_API}/fapi/v1/aggTrades?${params}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Binance API error: ${response.status}`);
        }

        const data = await response.json();

        // Transform to our trade format
        return data.map(trade => ({
            price: parseFloat(trade.p),
            quantity: parseFloat(trade.q),
            time: trade.T,
            isBuyerMaker: trade.m,
            tradeId: trade.a
        }));
    } catch (error) {
        console.error('Failed to fetch historical trades:', error);
        throw error;
    }
}

/**
 * Fetch multiple batches of historical trades
 * @param {string} symbol - Trading pair
 * @param {number} minutes - How many minutes of history to fetch
 * @param {Function} onProgress - Progress callback (0-100)
 * @returns {Promise<Array>} All trades within the time range
 */
export async function fetchTradesForPeriod(symbol, minutes = 10, onProgress = null) {
    const endTime = Date.now();
    const startTime = endTime - (minutes * 60 * 1000);

    const allTrades = [];
    let currentEndTime = endTime;
    let batchCount = 0;
    const maxBatches = 20; // Maximum batches to prevent infinite loops


    while (currentEndTime > startTime && batchCount < maxBatches) {
        try {
            const trades = await fetchHistoricalTrades(symbol, 1000, null, currentEndTime);

            if (trades.length === 0) break;

            // Filter trades within our time range
            const validTrades = trades.filter(t => t.time >= startTime);
            allTrades.push(...validTrades);

            // Update progress
            batchCount++;
            if (onProgress) {
                const progress = Math.min(100, Math.round((batchCount / maxBatches) * 100));
                onProgress(progress);
            }

            // Get oldest trade time for next batch
            const oldestTrade = trades[trades.length - 1];
            if (oldestTrade.time <= startTime) break;

            currentEndTime = oldestTrade.time - 1;

            // Rate limiting: 500ms delay between requests
            await new Promise(resolve => setTimeout(resolve, 500));

        } catch (error) {
            // On ANY error (including rate limit), stop immediately and return what we have
            console.warn('⚠️ Error fetching trades, stopping with partial data:', error.message);
            break;
        }
    }

    // Sort by time ascending
    allTrades.sort((a, b) => a.time - b.time);


    return allTrades;
}

/**
 * Fetch historical klines (OHLCV candles) - simpler alternative
 * @param {string} symbol - Trading pair
 * @param {string} interval - Timeframe (1m, 5m, 15m, 1h)
 * @param {number} limit - Number of candles
 * @returns {Promise<Array>} Array of kline objects
 */
export async function fetchHistoricalKlines(symbol, interval = '1m', limit = 100) {
    const params = new URLSearchParams({
        symbol: symbol.toUpperCase(),
        interval: interval,
        limit: limit.toString()
    });

    const url = `${BINANCE_FUTURES_API}/fapi/v1/klines?${params}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Binance API error: ${response.status}`);
        }

        const data = await response.json();

        // Transform to our candle format
        return data.map(k => ({
            time: k[0],
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
            closeTime: k[6],
            quoteVolume: parseFloat(k[7]),
            trades: k[8],
            takerBuyVolume: parseFloat(k[9]),
            takerBuyQuoteVolume: parseFloat(k[10])
        }));
    } catch (error) {
        console.error('Failed to fetch historical klines:', error);
        throw error;
    }
}

export const binanceREST = {
    fetchHistoricalTrades,
    fetchTradesForPeriod,
    fetchHistoricalKlines
};
