/**
 * Data Aggregator Service
 * Aggregates real-time trade data into footprint chart format
 */

export class DataAggregator {
    constructor(options = {}) {
        this.tickSize = options.tickSize || 1; // Price clustering size
        this.timeframe = options.timeframe || 1; // Minutes per candle
        this.maxCandles = options.maxCandles || 1500; // Full day of 1min candles (1440 + buffer)

        // Data storage
        this.candles = [];
        this.currentCandle = null;
        this.currentCandleStart = null;

        // Raw trade cache for timeframe rebuilding
        this.rawTradesCache = [];
        this.maxCachedTrades = 50000; // Keep last 50k trades

        // Running statistics
        this.cumulativeDelta = 0;
        this.totalBuyVolume = 0;
        this.totalSellVolume = 0;
        this.tradeCount = 0;
        this.tradesPerSecond = 0;
        this.tradeCountInSecond = 0;
        this.lastSecond = Math.floor(Date.now() / 1000);

        // VWAP calculation (Volume Weighted Average Price)
        this.vwapSumPriceQty = 0;  // Sum of (price * quantity)
        this.vwapSumQty = 0;       // Sum of quantity
        this.vwap = 0;             // Current VWAP

        // Session tracking (resets at 00:00 UTC)
        this.sessionStart = this._getSessionStart();
        this.previousSession = {
            poc: null,    // Previous day's POC (PVOC)
            vah: null,    // Previous day's VAH
            val: null,    // Previous day's VAL
            vwap: null    // Previous day's closing VWAP
        };

        // Volume Profile (session)
        this.volumeProfile = new Map();

        // Imbalance threshold (300% = 3x)
        this.imbalanceThreshold = 3.0;

        // Callbacks
        this.callbacks = {
            candleUpdate: [],
            candleClose: [],
            statsUpdate: []
        };
    }

    /**
     * Get session start time (00:00 UTC)
     */
    _getSessionStart() {
        const now = new Date();
        return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())).getTime();
    }

    /**
     * Check if we need to start a new session
     */
    _checkSessionReset() {
        const currentSessionStart = this._getSessionStart();
        if (currentSessionStart > this.sessionStart) {
            // New day - save previous session data
            const profile = this.getVolumeProfile();
            this.previousSession = {
                poc: profile.poc,
                vah: profile.vah,
                val: profile.val,
                vwap: this.vwap
            };

            // Reset session data
            this.sessionStart = currentSessionStart;
            this.volumeProfile.clear();
            this.vwapSumPriceQty = 0;
            this.vwapSumQty = 0;
            this.vwap = 0;

        }
    }

    /**
     * Set the timeframe for candle aggregation
     * @param {number} minutes - Timeframe in minutes
     */
    setTimeframe(minutes) {
        if (this.timeframe === minutes) return;

        this.timeframe = minutes;

        // Rebuild candles from cached trades
        this.rebuildCandles();
    }

    /**
     * Rebuild all candles from cached raw trades
     * Called when timeframe changes to re-aggregate data
     */
    rebuildCandles() {
        if (this.rawTradesCache.length === 0) {
            return;
        }


        // Reset candle state but keep volume profile and stats
        this.candles = [];
        this.currentCandle = null;
        this.currentCandleStart = null;

        // Re-process all cached trades
        const startTime = Date.now();
        for (const trade of this.rawTradesCache) {
            const { price, quantity, time, isBuyerMaker } = trade;
            const roundedPrice = this.roundToTick(price);
            const candleStart = this.getCandleStart(time);
            const isBuy = isBuyerMaker; // USER REQUEST: Invert Logic

            // Check if we need a new candle
            if (this.currentCandleStart !== candleStart) {
                if (this.currentCandle) {
                    const closedCandle = {
                        ...this.currentCandle,
                        clusters: Object.fromEntries(this.currentCandle.clusters)
                    };
                    this.candles.push(closedCandle);
                }
                this._startNewCandle(candleStart, price);
            }

            // Update current candle
            this._updateCandleSilent(roundedPrice, quantity, isBuy, price);
        }

        // Limit candles
        while (this.candles.length > this.maxCandles) {
            this.candles.shift();
        }

        const elapsed = Date.now() - startTime;

        // Emit update
        this._emit('candleUpdate', { candle: this.currentCandle, isNew: false });
    }

    /**
     * Set tick size for price clustering
     * @param {number} tickSize - Price tick size
     */
    setTickSize(tickSize) {
        this.tickSize = tickSize;
    }

    /**
     * Round price to tick size
     * @param {number} price - Raw price
     * @returns {number} - Rounded price
     */
    roundToTick(price) {
        return Math.round(price / this.tickSize) * this.tickSize;
    }

    /**
     * Get candle start time for a given timestamp
     * @param {number} timestamp - Unix timestamp in ms
     * @returns {number} - Candle start timestamp
     */
    getCandleStart(timestamp) {
        const ms = this.timeframe * 60 * 1000;
        return Math.floor(timestamp / ms) * ms;
    }

    /**
     * Process an incoming trade
     * @param {Object} trade - Trade data from WebSocket
     */
    processTrade(trade) {
        const { price, quantity, time, isBuyerMaker } = trade;
        const roundedPrice = this.roundToTick(price);
        const candleStart = this.getCandleStart(time);
        const isBuy = isBuyerMaker; // USER REQUEST: Invert Logic

        // Check for session reset (new day at 00:00 UTC)
        this._checkSessionReset();

        // Cache raw trade for timeframe rebuilding
        this.rawTradesCache.push(trade);
        if (this.rawTradesCache.length > this.maxCachedTrades) {
            this.rawTradesCache.shift();
        }

        // Update VWAP
        this.vwapSumPriceQty += price * quantity;
        this.vwapSumQty += quantity;
        if (this.vwapSumQty > 0) {
            this.vwap = this.vwapSumPriceQty / this.vwapSumQty;
        }

        // Update trades per second
        const currentSecond = Math.floor(Date.now() / 1000);
        if (currentSecond !== this.lastSecond) {
            this.tradesPerSecond = this.tradeCountInSecond;
            this.tradeCountInSecond = 0;
            this.lastSecond = currentSecond;
        }
        this.tradeCountInSecond++;
        this.tradeCount++;

        // Update cumulative stats
        if (isBuy) {
            this.totalBuyVolume += quantity;
            this.cumulativeDelta += quantity;
        } else {
            this.totalSellVolume += quantity;
            this.cumulativeDelta -= quantity;
        }

        // Update volume profile
        const existingVolume = this.volumeProfile.get(roundedPrice) || { buy: 0, sell: 0 };
        if (isBuy) {
            existingVolume.buy += quantity;
        } else {
            existingVolume.sell += quantity;
        }
        this.volumeProfile.set(roundedPrice, existingVolume);

        // Check if we need a new candle
        if (this.currentCandleStart !== candleStart) {
            if (this.currentCandle) {
                this._closeCandle();
            }
            this._startNewCandle(candleStart, price);
        }

        // Update current candle
        this._updateCandle(roundedPrice, quantity, isBuy, price);

        // Emit stats update
        this._emit('statsUpdate', this.getStats());
    }

    /**
     * Start a new candle
     * @param {number} startTime - Candle start timestamp
     * @param {number} openPrice - Opening price
     */
    _startNewCandle(startTime, openPrice) {
        this.currentCandleStart = startTime;
        this.currentCandle = {
            time: startTime,
            open: openPrice,
            high: openPrice,
            low: openPrice,
            close: openPrice,
            volume: 0,
            buyVolume: 0,
            sellVolume: 0,
            delta: 0,
            clusters: new Map(), // price -> { bid, ask, delta, imbalance }
            tradeCount: 0
        };
    }

    /**
     * Update the current candle with trade data
     * @param {number} price - Rounded price
     * @param {number} quantity - Trade quantity
     * @param {boolean} isBuy - Is buy trade
     * @param {number} rawPrice - Raw price for OHLC
     */
    _updateCandle(price, quantity, isBuy, rawPrice) {
        const candle = this.currentCandle;

        // Update OHLC
        candle.high = Math.max(candle.high, rawPrice);
        candle.low = Math.min(candle.low, rawPrice);
        candle.close = rawPrice;

        // Update volume
        candle.volume += quantity;
        candle.tradeCount++;

        if (isBuy) {
            candle.buyVolume += quantity;
            candle.delta += quantity;
        } else {
            candle.sellVolume += quantity;
            candle.delta -= quantity;
        }

        // Update cluster at price level
        const cluster = candle.clusters.get(price) || {
            bid: 0,
            ask: 0,
            delta: 0,
            imbalance: null
        };

        if (isBuy) {
            cluster.bid += quantity;
        } else {
            cluster.ask += quantity;
        }

        cluster.delta = cluster.bid - cluster.ask;

        // Check for imbalance
        if (cluster.bid > 0 && cluster.ask > 0) {
            const ratio = cluster.bid / cluster.ask;
            if (ratio >= this.imbalanceThreshold) {
                cluster.imbalance = 'buy';
            } else if (1 / ratio >= this.imbalanceThreshold) {
                cluster.imbalance = 'sell';
            } else {
                cluster.imbalance = null;
            }
        } else if (cluster.bid > 0) {
            cluster.imbalance = 'buy';
        } else if (cluster.ask > 0) {
            cluster.imbalance = 'sell';
        }

        candle.clusters.set(price, cluster);

        // Emit update
        this._emit('candleUpdate', { candle, isNew: false });
    }

    /**
     * Close the current candle
     */
    _closeCandle() {
        if (!this.currentCandle) return;

        // Convert clusters Map to object for storage
        const closedCandle = {
            ...this.currentCandle,
            clusters: Object.fromEntries(this.currentCandle.clusters)
        };

        this.candles.push(closedCandle);

        // Limit candle history
        if (this.candles.length > this.maxCandles) {
            this.candles.shift();
        }

        this._emit('candleClose', closedCandle);
    }

    /**
     * Reset current candle (e.g., on timeframe change)
     */
    resetCurrentCandle() {
        this.currentCandle = null;
        this.currentCandleStart = null;
    }

    /**
     * Process a batch of historical trades efficiently
     * @param {Array} trades - Array of trade objects
     * @param {Function} onProgress - Progress callback (0-100)
     */
    processHistoricalTrades(trades, onProgress = null) {

        const startTime = Date.now();
        const total = trades.length;

        // Cache all trades for timeframe rebuilding
        this.rawTradesCache = [...trades];
        if (this.rawTradesCache.length > this.maxCachedTrades) {
            this.rawTradesCache = this.rawTradesCache.slice(-this.maxCachedTrades);
        }

        for (let i = 0; i < trades.length; i++) {
            const trade = trades[i];

            // Process trade without emitting events (for performance)
            const { price, quantity, time, isBuyerMaker } = trade;
            const roundedPrice = this.roundToTick(price);
            const candleStart = this.getCandleStart(time);
            const isBuy = isBuyerMaker; // USER REQUEST: Invert Logic

            // Update cumulative stats
            if (isBuy) {
                this.totalBuyVolume += quantity;
                this.cumulativeDelta += quantity;
            } else {
                this.totalSellVolume += quantity;
                this.cumulativeDelta -= quantity;
            }
            this.tradeCount++;

            // Update volume profile
            const existingVolume = this.volumeProfile.get(roundedPrice) || { buy: 0, sell: 0 };
            if (isBuy) {
                existingVolume.buy += quantity;
            } else {
                existingVolume.sell += quantity;
            }
            this.volumeProfile.set(roundedPrice, existingVolume);

            // Check if we need a new candle
            if (this.currentCandleStart !== candleStart) {
                if (this.currentCandle) {
                    // Close current candle
                    const closedCandle = {
                        ...this.currentCandle,
                        clusters: Object.fromEntries(this.currentCandle.clusters)
                    };
                    this.candles.push(closedCandle);
                }
                this._startNewCandle(candleStart, price);
            }

            // Update current candle
            this._updateCandleSilent(roundedPrice, quantity, isBuy, price);

            // Report progress every 1000 trades
            if (onProgress && i % 1000 === 0) {
                onProgress(Math.round((i / total) * 100));
            }
        }

        // Limit candles
        while (this.candles.length > this.maxCandles) {
            this.candles.shift();
        }

        const elapsed = Date.now() - startTime;

        // Emit final update
        this._emit('candleUpdate', { candle: this.currentCandle, isNew: false });
        this._emit('statsUpdate', this.getStats());
    }

    /**
     * Import pre-aggregated candles from VPS (much faster than raw trades)
     * @param {Array} candles - Array of candle objects from VPS API
     */
    importCandles(candles) {
        if (!candles || candles.length === 0) {
            return;
        }

        const startTime = Date.now();

        // Clear existing candles and start fresh
        this.candles = [];

        for (const c of candles) {
            // Parse clusters if they're a string (JSON)
            let clusters = c.clusters;
            if (typeof clusters === 'string') {
                try {
                    clusters = JSON.parse(clusters);
                } catch {
                    clusters = {};
                }
            }

            // Convert clusters object to Map
            const clusterMap = new Map();
            if (clusters && typeof clusters === 'object') {
                for (const [price, data] of Object.entries(clusters)) {
                    clusterMap.set(parseFloat(price), data);
                }
            }

            // Create normalized candle
            const candle = {
                time: c.time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
                volume: c.volume || 0,
                buyVolume: c.buyVolume || 0,
                sellVolume: c.sellVolume || 0,
                delta: c.delta || 0,
                tradeCount: c.tradeCount || 0,
                clusters: clusterMap
            };

            this.candles.push(candle);

            // Update cumulative stats
            this.totalBuyVolume += candle.buyVolume;
            this.totalSellVolume += candle.sellVolume;
            this.cumulativeDelta += candle.delta;
            this.tradeCount += candle.tradeCount;

            // Update volume profile with {buy, sell} format (matching processTrade)
            for (const [price, cluster] of clusterMap) {
                const existing = this.volumeProfile.get(price) || { buy: 0, sell: 0 };
                existing.buy += (cluster.bid || 0);  // bid = buy volume
                existing.sell += (cluster.ask || 0); // ask = sell volume
                this.volumeProfile.set(price, existing);
            }
        }

        // Set the last candle as current for live updates
        if (this.candles.length > 0) {
            const lastCandle = this.candles[this.candles.length - 1];
            this.currentCandle = {
                ...lastCandle,
                clusters: new Map(lastCandle.clusters)
            };
            this.currentCandleStart = lastCandle.time;

            // Pop last candle since it becomes currentCandle
            this.candles.pop();
        }

        const elapsed = Date.now() - startTime;

        // Emit updates
        if (this.currentCandle) {
            this._emit('candleUpdate', { candle: this.currentCandle, isNew: false });
        }
        this._emit('statsUpdate', this.getStats());
    }

    /**
     * Prepend historical candles (for pagination - loading older data)
     * @param {Array} candles - Array of older candle objects to prepend
     * @returns {number} Number of candles added
     */
    prependCandles(candles) {
        if (!candles || candles.length === 0) {
            return 0;
        }

        // Get oldest existing candle time to avoid duplicates
        const oldestExistingTime = this.candles.length > 0 ? this.candles[0].time : Infinity;
        
        // Filter out candles that already exist (by time)
        const newCandles = candles.filter(c => c.time < oldestExistingTime);
        
        if (newCandles.length === 0) {
            return 0;
        }

        // Process and prepend candles
        const processedCandles = newCandles.map(c => {
            // Parse clusters if they're a string (JSON)
            let clusters = c.clusters;
            if (typeof clusters === 'string') {
                try {
                    clusters = JSON.parse(clusters);
                } catch {
                    clusters = {};
                }
            }

            // Convert clusters object to Map
            const clusterMap = new Map();
            if (clusters && typeof clusters === 'object') {
                for (const [price, data] of Object.entries(clusters)) {
                    clusterMap.set(parseFloat(price), data);
                }
            }

            return {
                time: c.time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
                volume: c.volume || 0,
                buyVolume: c.buyVolume || 0,
                sellVolume: c.sellVolume || 0,
                delta: c.delta || 0,
                tradeCount: c.tradeCount || 0,
                clusters: clusterMap
            };
        });

        // Prepend to existing candles array
        this.candles = [...processedCandles, ...this.candles];

        // Update cumulative stats with new candles
        for (const candle of processedCandles) {
            this.totalBuyVolume += candle.buyVolume;
            this.totalSellVolume += candle.sellVolume;
            this.cumulativeDelta += candle.delta;
            this.tradeCount += candle.tradeCount;

            // Update volume profile
            for (const [price, cluster] of candle.clusters) {
                const existing = this.volumeProfile.get(price) || { buy: 0, sell: 0 };
                existing.buy += (cluster.bid || 0);
                existing.sell += (cluster.ask || 0);
                this.volumeProfile.set(price, existing);
            }
        }

        // Emit history update event
        this._emit('historyPrepended', { count: processedCandles.length });
        this._emit('statsUpdate', this.getStats());

        return processedCandles.length;
    }

    /**
     * Get the oldest candle timestamp
     */
    getOldestTime() {
        if (this.candles.length > 0) {
            return this.candles[0].time;
        }
        return null;
    }

    /**
     * Update candle without emitting events (for batch processing)
     */
    _updateCandleSilent(price, quantity, isBuy, rawPrice) {
        const candle = this.currentCandle;
        if (!candle) return;

        // Update OHLC
        candle.high = Math.max(candle.high, rawPrice);
        candle.low = Math.min(candle.low, rawPrice);
        candle.close = rawPrice;

        // Update volume
        candle.volume += quantity;
        candle.tradeCount++;

        if (isBuy) {
            candle.buyVolume += quantity;
            candle.delta += quantity;
        } else {
            candle.sellVolume += quantity;
            candle.delta -= quantity;
        }

        // Update cluster at price level
        const cluster = candle.clusters.get(price) || {
            bid: 0,
            ask: 0,
            delta: 0,
            imbalance: null
        };

        if (isBuy) {
            cluster.bid += quantity;
        } else {
            cluster.ask += quantity;
        }

        cluster.delta = cluster.bid - cluster.ask;

        // Check for imbalance
        if (cluster.bid > 0 && cluster.ask > 0) {
            const ratio = cluster.bid / cluster.ask;
            if (ratio >= this.imbalanceThreshold) {
                cluster.imbalance = 'buy';
            } else if (1 / ratio >= this.imbalanceThreshold) {
                cluster.imbalance = 'sell';
            } else {
                cluster.imbalance = null;
            }
        } else if (cluster.bid > 0) {
            cluster.imbalance = 'buy';
        } else if (cluster.ask > 0) {
            cluster.imbalance = 'sell';
        }

        candle.clusters.set(price, cluster);
    }

    /**
     * Reset all data
     */
    reset() {
        this.candles = [];
        this.currentCandle = null;
        this.currentCandleStart = null;
        this.cumulativeDelta = 0;
        this.totalBuyVolume = 0;
        this.totalSellVolume = 0;
        this.tradeCount = 0;
        this.volumeProfile.clear();
        this.rawTradesCache = []; // Clear trade cache
    }

    /**
     * Get current statistics
     * @returns {Object} - Stats object
     */
    getStats() {
        return {
            cumulativeDelta: this.cumulativeDelta,
            totalBuyVolume: this.totalBuyVolume,
            totalSellVolume: this.totalSellVolume,
            tradeCount: this.tradeCount,
            tradesPerSecond: this.tradesPerSecond,
            currentDelta: this.currentCandle?.delta || 0
        };
    }

    /**
     * Get session markers for chart overlay
     * @returns {Object} - Session markers (VWAP, POC, PVOC, VAH, VAL)
     */
    getSessionMarkers() {
        const currentProfile = this.getVolumeProfile();

        return {
            // Current session
            vwap: this.vwap,
            poc: currentProfile.poc,
            vah: currentProfile.vah,
            val: currentProfile.val,

            // Previous session (PVOC etc)
            pvoc: this.previousSession.poc,
            pvah: this.previousSession.vah,
            pval: this.previousSession.val,
            pvwap: this.previousSession.vwap
        };
    }

    /**
     * Get all candles including current
     * @returns {Array} - Array of candles
     */
    getCandles() {
        // Convert all candles to have clusters as Objects (not Maps)
        // FootprintChart._renderCandle uses Object.keys() which doesn't work on Maps
        const allCandles = this.candles.map(candle => ({
            ...candle,
            clusters: candle.clusters instanceof Map
                ? Object.fromEntries(candle.clusters)
                : candle.clusters
        }));

        if (this.currentCandle) {
            allCandles.push({
                ...this.currentCandle,
                clusters: this.currentCandle.clusters instanceof Map
                    ? Object.fromEntries(this.currentCandle.clusters)
                    : this.currentCandle.clusters
            });
        }
        return allCandles;
    }

    /**
     * Get volume profile data
     * @returns {Object} - Volume profile with POC and value area
     */
    getVolumeProfile() {
        if (this.volumeProfile.size === 0) {
            return { levels: [], poc: null, vah: null, val: null };
        }

        // Convert to array and sort by price
        const levels = Array.from(this.volumeProfile.entries())
            .map(([price, vol]) => ({
                price,
                buy: vol.buy,
                sell: vol.sell,
                total: vol.buy + vol.sell
            }))
            .sort((a, b) => b.price - a.price);

        // Find POC (Point of Control - highest volume)
        let poc = levels[0];
        for (const level of levels) {
            if (level.total > poc.total) {
                poc = level;
            }
        }

        // Calculate Value Area (70% of total volume)
        const totalVolume = levels.reduce((sum, l) => sum + l.total, 0);
        const valueAreaVolume = totalVolume * 0.7;

        // Start from POC and expand
        let vaVolume = poc.total;
        let vahIndex = levels.indexOf(poc);
        let valIndex = vahIndex;

        while (vaVolume < valueAreaVolume && (vahIndex > 0 || valIndex < levels.length - 1)) {
            const upperVol = vahIndex > 0 ? levels[vahIndex - 1].total : 0;
            const lowerVol = valIndex < levels.length - 1 ? levels[valIndex + 1].total : 0;

            if (upperVol >= lowerVol && vahIndex > 0) {
                vahIndex--;
                vaVolume += upperVol;
            } else if (valIndex < levels.length - 1) {
                valIndex++;
                vaVolume += lowerVol;
            } else {
                break;
            }
        }

        return {
            levels,
            poc: poc.price,
            vah: levels[vahIndex]?.price,
            val: levels[valIndex]?.price,
            totalVolume
        };
    }

    /**
     * Register event callback
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        }
        return this;
    }

    /**
     * Emit event
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    _emit(event, data) {
        if (this.callbacks[event]) {
            for (const cb of this.callbacks[event]) {
                cb(data);
            }
        }
    }
}

// Singleton instance
export const dataAggregator = new DataAggregator();
