/**
 * Depth Collector
 * Maintains a local order book and saves snapshots for Heatmap logic
 */

const WebSocket = require("ws");
const db = require("./db.js");

class DepthCollector {
  constructor(symbol) {
    // Handle exchange:symbol format (e.g., "binance:btcusdt")
    if (symbol.includes(":")) {
      const [exchange, baseSymbol] = symbol.toLowerCase().split(":");
      this.fullSymbol = symbol.toLowerCase(); // For DB storage
      this.symbol = baseSymbol; // For API calls
      this.exchange = exchange;
    } else {
      this.symbol = symbol.toLowerCase();
      this.fullSymbol = symbol.toLowerCase();
      this.exchange = "binance";
    }

    this.ws = null;
    this.bids = new Map();
    this.asks = new Map();
    this.lastUpdateId = 0;
    this.buffer = [];
    this.synced = false;

    // Heatmap config
    this.snapshotInterval = 1000; // Save every 1s
    this.lastSnapshotTime = 0;
    this.maxStoredLevels = 100;
    this.minVolumeForHeatmap = 0.01; // Filter dust
  }

  async start() {
    console.log(`🔥 Starting Depth Collector for ${this.symbol.toUpperCase()}`);
    await this._initSnapshot();
    this._connectWS();
    this._startLoop();
  }

  /**
   * Fetch initial snapshot from REST API
   */
  async _initSnapshot() {
    try {
      const url = `https://fapi.binance.com/fapi/v1/depth?symbol=${this.symbol.toUpperCase()}&limit=1000`;
      const res = await fetch(url);
      const data = await res.json();

      this.lastUpdateId = data.lastUpdateId;

      // Populate maps
      this.bids.clear();
      this.asks.clear();

      for (const [p, q] of data.bids)
        this.bids.set(parseFloat(p), parseFloat(q));
      for (const [p, q] of data.asks)
        this.asks.set(parseFloat(p), parseFloat(q));

      console.log(
        `🔥 Snapshot loaded for ${this.symbol.toUpperCase()} (ID: ${this.lastUpdateId})`,
      );
      this.synced = true;

      // Process buffer
      this._processBuffer();
    } catch (err) {
      console.error(
        `🔥 Failed to fetch snapshot for ${this.symbol}:`,
        err.message,
      );
      setTimeout(() => this._initSnapshot(), 5000);
    }
  }

  /**
   * Connect to WebSocket Diff Stream
   */
  _connectWS() {
    const wsUrl = `wss://fstream.binance.com/ws/${this.symbol}@depth`;
    this.ws = new WebSocket(wsUrl);

    this.ws.on("open", () => {
      console.log(`🔥 Depth WS connected for ${this.symbol.toUpperCase()}`);
    });

    this.ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data);
        this._handleDepthUpdate(msg);
      } catch (err) {
        console.error(`🔥 WS Error for ${this.symbol}:`, err);
      }
    });

    this.ws.on("error", (err) =>
      console.error(`🔥 WS Error ${this.symbol}:`, err.message),
    );

    this.ws.on("close", () => {
      console.log(`🔥 Depth WS closed for ${this.symbol}, reconnecting...`);
      this.synced = false;
      setTimeout(() => this.start(), 5000); // Restart full sequence
    });
  }

  _handleDepthUpdate(depth) {
    // Validation: u is Final Update ID, U is First Update ID
    const finalUpdateId = depth.u;
    const firstUpdateId = depth.U;

    if (!this.synced) {
      this.buffer.push(depth);
      return;
    }

    // Drop old events
    if (finalUpdateId <= this.lastUpdateId) return;

    // Verify continuity (if strictly needed, but eager mode is fine for heatmap)
    // Strictly: firstUpdateId should be <= lastUpdateId + 1

    this._applyUpdate(depth);
    this.lastUpdateId = finalUpdateId;
  }

  _processBuffer() {
    for (const depth of this.buffer) {
      if (depth.u > this.lastUpdateId) {
        this._applyUpdate(depth);
        this.lastUpdateId = depth.u;
      }
    }
    this.buffer = [];
  }

  _applyUpdate(depth) {
    // Bids
    for (const [pStr, qStr] of depth.b) {
      const p = parseFloat(pStr);
      const q = parseFloat(qStr);
      if (q === 0) this.bids.delete(p);
      else this.bids.set(p, q);
    }
    // Asks
    for (const [pStr, qStr] of depth.a) {
      const p = parseFloat(pStr);
      const q = parseFloat(qStr);
      if (q === 0) this.asks.delete(p);
      else this.asks.set(p, q);
    }
  }

  _startLoop() {
    setInterval(() => {
      if (!this.synced) return;
      this._saveSnapshot();
    }, this.snapshotInterval);
  }

  _saveSnapshot() {
    try {
      const now = Date.now();

      // --- SMART FILTERING STRATEGY ---
      // 1. Context: Keep 50 levels closest to price (Spread area)
      // 2. Walls: Scan deep book (e.g. 5000 levels) and keep top 450 by Volume (Whales)

      const CONTEXT_LEVELS = 50;
      const WALL_LEVELS = 450;
      const SCAN_DEPTH = 5000;

      const processSide = (map, isAsk) => {
        // Convert to array
        let levels = Array.from(map.entries()).map(([p, q]) => ({ p, q }));

        // Sort by price distance (0 = closest)
        // Bids: Descending (Higher is closer). Asks: Ascending (Lower is closer).
        if (isAsk) levels.sort((a, b) => a.p - b.p);
        else levels.sort((a, b) => b.p - a.p);

        // 1. Get Near Context
        const context = levels.slice(0, CONTEXT_LEVELS);

        // 2. Get Deep Walls from the REST
        const deepLevels = levels.slice(CONTEXT_LEVELS, SCAN_DEPTH);
        // Sort by Volume Descending to find walls
        deepLevels.sort((a, b) => b.q - a.q);
        const walls = deepLevels.slice(0, WALL_LEVELS);

        // Merge and Re-Sort by Price for Storage
        // Bids: Descending. Asks: Ascending.
        // Note: We use Set to avoid duplicates if overlaps occur (unlikely due to slicing)
        const result = [...context, ...walls];
        if (isAsk) result.sort((a, b) => a.p - b.p);
        else result.sort((a, b) => b.p - a.p);

        return result;
      };

      const bids = processSide(this.bids, false);
      const asks = processSide(this.asks, true);

      const maxVolume = Math.max(
        ...bids.map((b) => b.q),
        ...asks.map((a) => a.q),
        0,
      );

      // Db Insert (use fullSymbol for exchange:symbol format)
      db.insertSnapshots(this.fullSymbol, [
        {
          time: now,
          bids,
          asks,
          maxVolume,
        },
      ]);
    } catch (err) {
      console.error(`🔥 Save snapshot failed for ${this.symbol}:`, err.message);
    }
  }
}

module.exports = DepthCollector;
