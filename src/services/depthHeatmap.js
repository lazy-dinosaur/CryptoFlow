/**
 * Depth Heatmap Service
 * Stores historical order book depth to visualize liquidity changes over time.
 */

export class DepthHeatmapStore {
    constructor(options = {}) {
        // Configuration
        this.maxSnapshots = options.maxSnapshots || 2500; // 40h history with step=60
        this.snapshotInterval = options.snapshotInterval || 1000; // 1 second

        // Detect environment - use VPS URL when running locally
        const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        this.baseUrl = isLocalHost ? 'http://134.185.107.33:3000' : '';

        // Filter dust (Binance futures quantities can be < 1.0 frequently)
        this.minVolumeForHeatmap = options.minVolume ?? 0.01;

        // State
        this.currentBids = new Map();
        this.currentAsks = new Map();
        this.snapshots = []; // Array of { time, bids: [], asks: [] }
        this.lastSnapshotTime = 0;

        // Statistics for relative coloring
        this.maxVolumeInHistory = 5000;

        // Persistence
        this.pendingSnapshots = []; // Snapshots to upload
        this._loadHistory();
        this._startSync();

    }

    /**
     * Load history from server
     */
    async _loadHistory() {
        try {
            // Load smart history: 24h+ coverage (step=60 -> 1 min resolution)
            const symbol = (this.symbol || 'btcusdt').toLowerCase();
            const qs = new URLSearchParams({
                symbol,
                limit: String(this.maxSnapshots),
                step: '60'
            });

            const res = await fetch(`${this.baseUrl}/api/depth?${qs.toString()}`);
            const data = await res.json();

            if (data.snapshots && data.snapshots.length > 0) {
                this.snapshots = data.snapshots;
                // Recalculate max volume
                this.maxVolumeInHistory = Math.max(
                    ...this.snapshots.map(s => s.maxVolume || 0),
                    10
                );
            }
        } catch (err) {
            // History is optional (live heatmap will still work)
            console.warn('🔥 Heatmap history unavailable:', err?.message || err);
        }
    }

    /**
     * Start periodic sync to fetch latest snapshots from server
     * This ensures we get full-depth data (500 levels per side) instead of limited WebSocket data
     */
    _startSync() {
        // Fetch latest snapshot from server every second
        setInterval(async () => {
            if (!this.symbol) return;

            try {
                const qs = new URLSearchParams({
                    symbol: this.symbol,
                    limit: '1',  // Just the latest snapshot
                    step: '1'   // No downsampling
                });

                const res = await fetch(`${this.baseUrl}/api/depth?${qs.toString()}`);
                const data = await res.json();

                if (data.snapshots && data.snapshots.length > 0) {
                    const latestFromServer = data.snapshots[data.snapshots.length - 1];

                    // Only append if newer than what we have
                    const lastLocal = this.snapshots[this.snapshots.length - 1];
                    if (!lastLocal || latestFromServer.time > lastLocal.time) {
                        this.snapshots.push(latestFromServer);

                        // Update max volume
                        if (latestFromServer.maxVolume > this.maxVolumeInHistory) {
                            this.maxVolumeInHistory = latestFromServer.maxVolume;
                        }

                        // Limit history size
                        if (this.snapshots.length > this.maxSnapshots) {
                            this.snapshots.shift();
                        }
                    }
                }
            } catch (err) {
                // Silent fail - WebSocket data is fallback
            }
        }, 5000); // Every 5 seconds (reduced from 1s to prevent flickering)
    }

    /**
     * Set current symbol
     */
    async setSymbol(symbol) {
        this.symbol = symbol?.toLowerCase();

        // Adjust dust filter by symbol (units differ per market)
        const s = this.symbol;
        if (s === 'btcusdt') this.minVolumeForHeatmap = 0.01;
        else if (s === 'ethusdt') this.minVolumeForHeatmap = 0.1;
        else this.minVolumeForHeatmap = 1.0;

        await this._loadHistory(); // Wait for history to load
    }

    /**
     * Process depth update from WebSocket
     * @param {Object} depth - Depth update object
     */
    addDepthUpdate(depth) {
        // 1. Update current state (live order book)
        this._updateCurrentState(depth);

        // 2. Take snapshot if interval exceeded
        const now = Date.now();
        if (now - this.lastSnapshotTime >= this.snapshotInterval) {
            this._takeSnapshot(now);
            this.lastSnapshotTime = now;
        }
    }

    /**
     * Update internal order book state
     */
    _updateCurrentState(depth) {
        // Update bids
        for (const { price, quantity } of depth.bids) {
            if (quantity === 0) {
                this.currentBids.delete(price);
            } else {
                this.currentBids.set(price, quantity);
            }
        }

        // Update asks
        for (const { price, quantity } of depth.asks) {
            if (quantity === 0) {
                this.currentAsks.delete(price);
            } else {
                this.currentAsks.set(price, quantity);
            }
        }
    }

    /**
     * Convert Map to simplified array for storage
     * Only storing relevant levels to save memory
     */
    _takeSnapshot(time) {
        // Filter and sort bids (highest price first)
        const bids = Array.from(this.currentBids.entries())
            .filter(([_, qty]) => qty >= this.minVolumeForHeatmap)
            .sort((a, b) => b[0] - a[0])
            .slice(0, 500) // Top 500 levels for better coverage
            .map(([price, qty]) => ({ p: price, q: qty }));

        // Filter and sort asks (lowest price first)
        const asks = Array.from(this.currentAsks.entries())
            .filter(([_, qty]) => qty >= this.minVolumeForHeatmap)
            .sort((a, b) => a[0] - b[0])
            .slice(0, 500) // Top 500 levels for better coverage
            .map(([price, qty]) => ({ p: price, q: qty }));

        // Update stats
        const snapshotMax = Math.max(
            ...bids.map(b => b.q),
            ...asks.map(a => a.q),
            0
        );
        this.maxVolumeInHistory = Math.max(this.maxVolumeInHistory, snapshotMax);

        const snapshot = {
            time,
            bids,
            asks,
            maxVolume: snapshotMax
        };

        // Store snapshot
        this.snapshots.push(snapshot);
        this.pendingSnapshots.push(snapshot); // Queue for sync

        // Limit history size
        if (this.snapshots.length > this.maxSnapshots) {
            this.snapshots.shift();
        }
    }

    /**
     * Get heatmap data for rendering
     * @returns {Object} Data ready for chart
     */
    getHeatmapData() {
        return {
            snapshots: this.snapshots,
            maxVolume: this.maxVolumeInHistory
        };
    }

    /**
     * Return the strongest visible bid/ask liquidity walls from the *current* book.
     * Used for the L-toggle overlay.
     */
    getTopWalls(options = {}) {
        const {
            countPerSide = 3,
            aroundPrice = null,
            maxDistancePct = 0.5,
            minQty = this.minVolumeForHeatmap,
        } = options;

        const price = Number(aroundPrice);
        const hasAnchor = Number.isFinite(price) && price > 0;

        const within = (p) => {
            if (!hasAnchor) return true;
            const distPct = (Math.abs(p - price) / price) * 100;
            return distPct <= maxDistancePct;
        };

        const pick = (map, type) => {
            const arr = Array.from(map.entries())
                .map(([p, q]) => ({ price: p, volume: q, type }))
                .filter((x) => Number.isFinite(x.price) && Number.isFinite(x.volume) && x.volume >= minQty)
                .filter((x) => within(x.price))
                .sort((a, b) => b.volume - a.volume)
                .slice(0, countPerSide);

            // For nicer display: bids desc, asks asc
            if (type === 'bid') arr.sort((a, b) => b.price - a.price);
            if (type === 'ask') arr.sort((a, b) => a.price - b.price);
            return arr;
        };

        return [...pick(this.currentBids, 'bid'), ...pick(this.currentAsks, 'ask')];
    }

    /**
     * Reset history
     */
    reset() {
        this.currentBids.clear();
        this.currentAsks.clear();
        this.lastSnapshotTime = 0;
        this.maxVolumeInHistory = 5000; // Start high to avoid flash
    }
}

// Singleton instance
export const depthHeatmap = new DepthHeatmapStore();
