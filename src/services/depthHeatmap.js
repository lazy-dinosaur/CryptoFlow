/**
 * Depth Heatmap Service
 * Stores historical order book depth to visualize liquidity changes over time.
 */

export class DepthHeatmapStore {
    constructor(options = {}) {
        // Configuration
        this.maxSnapshots = options.maxSnapshots || 2500; // 40h history with step=60
        this.snapshotInterval = options.snapshotInterval || 1000; // 1 second

        // Filter dust (Binance futures quantities can be < 1.0 frequently)
        this.minVolumeForHeatmap = options.minVolume ?? 0.01;

        // State
        this.currentBids = new Map();
        this.currentAsks = new Map();
        this.snapshots = []; // Array of { time, bids: [], asks: [] }
        this.lastSnapshotTime = 0;

        // Statistics for relative coloring
        this.maxVolumeInHistory = 5000;

        // ========== CRATER SYSTEM ==========
        // Tracks historical max volume at each price level
        this.craters = new Map(); // price -> { maxVolume, lastSeen, decayedVolume }
        this.craterDecayRate = 0.98; // Faster decay (was 0.995)
        this.craterMinVolume = 15; // Higher threshold - only significant walls (was 5)
        this.maxCraters = 5; // Only show top 5 craters to reduce noise
        this.lastCraterUpdate = Date.now();

        // ========== SMOOTHING SYSTEM ==========
        // Rolling average of wall volumes to reduce jumping
        this.wallHistory = []; // Last N wall snapshots
        this.wallHistorySize = 5; // Average over 5 samples
        this.smoothedWalls = []; // Current smoothed wall values

        // ========== DYNAMIC THRESHOLD SYSTEM ==========
        // Calculate threshold as % of recent total orderbook volume
        this.recentVolumeRatio = 0.10; // Wall must be > 10% of recent avg volume
        this.minVolumeFloor = 5; // Absolute minimum in BTC
        this.recentTotalVolumes = []; // Track recent orderbook totals
        this.recentVolumeSamples = 10; // Average over 10 samples

        // ========== ZONE CLUSTERING ==========
        // Cluster nearby price levels into zones
        this.zoneMergeThreshold = 0.001; // Merge if within 0.1% of each other

        // Persistence
        this.pendingSnapshots = []; // Snapshots to upload
        this._loadHistory();
        this._startSync();
        this._startCraterDecay();
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

            const res = await fetch(`/api/depth?${qs.toString()}`);
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

                const res = await fetch(`/api/depth?${qs.toString()}`);
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
     * Start crater decay timer - craters slowly fade over time
     */
    _startCraterDecay() {
        setInterval(() => {
            const now = Date.now();
            const elapsed = (now - this.lastCraterUpdate) / 1000; // seconds
            this.lastCraterUpdate = now;

            // Decay all craters
            const decay = Math.pow(this.craterDecayRate, elapsed);
            for (const [price, crater] of this.craters.entries()) {
                crater.decayedVolume *= decay;

                // Remove if too small
                if (crater.decayedVolume < this.craterMinVolume * 0.5) {
                    this.craters.delete(price);
                }
            }
        }, 1000);
    }

    /**
     * Update craters when we see volume at a price level
     * Called during wall detection
     */
    _updateCraters(walls) {
        const now = Date.now();

        for (const wall of walls) {
            const price = wall.price;
            const volume = wall.volume;

            if (volume < this.craterMinVolume) continue;

            const existing = this.craters.get(price);
            if (existing) {
                // Update if new volume is higher
                if (volume > existing.maxVolume) {
                    existing.maxVolume = volume;
                    existing.decayedVolume = volume;
                }
                existing.lastSeen = now;
            } else {
                // Create new crater
                this.craters.set(price, {
                    maxVolume: volume,
                    decayedVolume: volume,
                    lastSeen: now
                });
            }
        }
    }

    /**
     * Get smoothed walls with crater indicators
     * Returns walls that are stable (averaged) + crater markers for historical max
     */
    getSmoothedWalls(options = {}) {
        // Get current raw walls
        const rawWalls = this.getTopWalls(options);

        // Update craters with current wall data
        this._updateCraters(rawWalls);

        // Calculate dynamic threshold from recent orderbook volume
        const totalVolume = rawWalls.reduce((sum, w) => sum + w.volume, 0);
        this.recentTotalVolumes.push(totalVolume);
        while (this.recentTotalVolumes.length > this.recentVolumeSamples) {
            this.recentTotalVolumes.shift();
        }
        const avgTotalVolume = this.recentTotalVolumes.reduce((a, b) => a + b, 0) / this.recentTotalVolumes.length;
        const dynamicThreshold = Math.max(this.minVolumeFloor, avgTotalVolume * this.recentVolumeRatio);

        // Add to rolling history
        this.wallHistory.push({
            time: Date.now(),
            walls: rawWalls
        });

        // Limit history size
        while (this.wallHistory.length > this.wallHistorySize) {
            this.wallHistory.shift();
        }

        // Calculate smoothed walls (average volume per price across history)
        const priceVolumes = new Map(); // price -> [volumes]
        const priceTypes = new Map(); // price -> type

        for (const snapshot of this.wallHistory) {
            for (const wall of snapshot.walls) {
                if (!priceVolumes.has(wall.price)) {
                    priceVolumes.set(wall.price, []);
                    priceTypes.set(wall.price, wall.type);
                }
                priceVolumes.get(wall.price).push(wall.volume);
            }
        }

        // Build filtered walls (apply dynamic threshold + persistence)
        let filteredWalls = [];
        for (const [price, volumes] of priceVolumes.entries()) {
            const persistenceRatio = volumes.length / this.wallHistorySize;
            if (persistenceRatio < 0.6) continue;

            const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

            // Apply dynamic threshold
            if (avgVolume < dynamicThreshold) continue;

            filteredWalls.push({
                price,
                volume: avgVolume,
                type: priceTypes.get(price),
                persistence: persistenceRatio
            });
        }

        // Cluster nearby prices into zones
        filteredWalls.sort((a, b) => a.price - b.price);
        const zones = this._clusterWallsIntoZones(filteredWalls);

        // Sort by volume
        zones.sort((a, b) => b.volume - a.volume);

        this.smoothedWalls = zones;
        return zones;
    }

    /**
     * Cluster nearby walls into zones (price ranges)
     */
    _clusterWallsIntoZones(walls) {
        if (walls.length === 0) return [];

        const zones = [];
        let currentZone = {
            priceMin: walls[0].price,
            priceMax: walls[0].price,
            volume: walls[0].volume,
            type: walls[0].type,
            count: 1
        };

        for (let i = 1; i < walls.length; i++) {
            const wall = walls[i];
            const avgZonePrice = (currentZone.priceMin + currentZone.priceMax) / 2;
            const distanceRatio = Math.abs(wall.price - avgZonePrice) / avgZonePrice;

            if (distanceRatio <= this.zoneMergeThreshold && wall.type === currentZone.type) {
                // Merge into current zone
                currentZone.priceMax = Math.max(currentZone.priceMax, wall.price);
                currentZone.priceMin = Math.min(currentZone.priceMin, wall.price);
                currentZone.volume += wall.volume;
                currentZone.count++;
            } else {
                // Finalize current zone, start new one
                zones.push(this._finalizeZone(currentZone));
                currentZone = {
                    priceMin: wall.price,
                    priceMax: wall.price,
                    volume: wall.volume,
                    type: wall.type,
                    count: 1
                };
            }
        }

        // Don't forget last zone
        zones.push(this._finalizeZone(currentZone));
        return zones;
    }

    _finalizeZone(zone) {
        return {
            price: (zone.priceMin + zone.priceMax) / 2, // Center price
            priceMin: zone.priceMin,
            priceMax: zone.priceMax,
            volume: zone.volume,
            type: zone.type,
            isZone: zone.priceMin !== zone.priceMax // True if clustered
        };
    }

    /**
     * Get crater data for visualization
     * Returns historical max volume markers that slowly fade
     */
    getCraters(options = {}) {
        const { aroundPrice = null, maxDistancePct = 1.0 } = options;
        const price = Number(aroundPrice);
        const hasAnchor = Number.isFinite(price) && price > 0;

        const craterList = [];
        for (const [craterPrice, crater] of this.craters.entries()) {
            // Distance filter
            if (hasAnchor) {
                const distPct = (Math.abs(craterPrice - price) / price) * 100;
                if (distPct > maxDistancePct) continue;
            }

            // Only show if still significant
            if (crater.decayedVolume >= this.craterMinVolume) {
                craterList.push({
                    price: craterPrice,
                    maxVolume: crater.maxVolume,
                    currentVolume: crater.decayedVolume,
                    intensity: crater.decayedVolume / crater.maxVolume, // 0-1 fade
                    age: Date.now() - crater.lastSeen
                });
            }
        }

        // Sort by max volume (most significant first) and limit to top N
        craterList.sort((a, b) => b.maxVolume - a.maxVolume);

        return craterList.slice(0, this.maxCraters);
    }

    /**
     * Reset history
     */
    reset() {
        this.currentBids.clear();
        this.currentAsks.clear();
        this.lastSnapshotTime = 0;
        this.maxVolumeInHistory = 5000; // Start high to avoid flash
        this.craters.clear();
        this.wallHistory = [];
        this.smoothedWalls = [];
    }
}

// Singleton instance
export const depthHeatmap = new DepthHeatmapStore();
