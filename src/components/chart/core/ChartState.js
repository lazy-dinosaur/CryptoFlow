/**
 * ChartState
 * Single source of truth for the chart's data and view configuration.
 * Decouples state from rendering logic.
 */
export class ChartState {
    constructor() {
        // Data
        this.candles = [];
        this.heatmapData = [];
        this.nakedLiquidityLevels = [];
        this.bigTrades = [];
        this.sessionMarkers = [];
        this.currentPrice = null;

        // View State
        this.pixelRatio = window.devicePixelRatio || 1;
        this.width = 0;
        this.height = 0;
        this.zoomX = 10; // Pixels per candle
        this.zoomY = 2;  // Pixels per price tick
        this.offsetX = 0;
        this.offsetY = 0;

        // Configuration
        this.tickSize = 0.1;
        this.minZoomX = 2;
        this.maxZoomX = 100;
        this.autoScroll = true;

        // Settings / Flags
        this.showHeatmap = true;
        this.showDelta = true;
        this.showImbalances = true;
        this.showBigTrades = true;
        this.showCrosshair = true;
        this.showNakedLiquidity = true; // 'L' key
        this.showML = true;             // 'M' key

        // Interaction State
        this.crosshairX = null;
        this.crosshairY = null;

        // Heatmap specific configs
        this.heatmapIntensityThreshold = 0.02;
        this.heatmapOpacity = 0.5;

        // EventEmitter-like capability
        this.listeners = new Set();
    }

    setData(candles) {
        this.candles = candles || [];
        this.notify();
    }

    setHeatmapData(data) {
        this.heatmapData = data || [];
        this.notify(); // Depending on performance we might not want to notify on every update
    }

    updateDimensions(width, height) {
        this.width = width;
        this.height = height;
        this.notify();
    }

    subscribe(callback) {
        this.listeners.add(callback);
        return () => this.listeners.delete(callback);
    }

    notify() {
        this.listeners.forEach(cb => cb(this));
    }
}
