/**
 * FootprintChart.js - VISUAL POLISH
 * - Modern Colors (Emerald/Rose)
 * - Improved Typography (Bold, Readable)
 * - Center Gap for Footprint Columns
 * - Subtler Imbalance Highlights
 */

export class FootprintChart {
    constructor(options = {}) {
        let containerId;
        if (typeof options === 'string') {
            containerId = options;
        } else {
            containerId = options.containerId;
        }

        // STATE
        this.candles = [];
        this.heatmapData = [];
        this.nakedLiquidityLevels = [];
        this.bigTrades = [];
        this.currentPrice = null;

        // View
        this.width = 0;
        this.height = 0;
        this.zoomX = 10;
        this.zoomY = 2;
        this.offsetX = 0;
        this.offsetY = 0;
        this.initialCenterDone = false;

        // Config
        this.tickSize = 0.1;
        this.minZoomX = 0.5; // Allow deeper zoom out (squeeze)
        this.maxZoomX = 150;
        this.minZoomY = 0.1;
        this.maxZoomY = 50;
        this.deltaRowHeight = 24; // Slightly taller for readability
        this.cvdPanelHeight = 60; // New CVD Panel

        // Flags
        this.showHeatmap = true;
        this.showDelta = true;
        this.showCVD = true;
        this.showSessionProfile = true; // NEW
        this.showBigTrades = true;
        this.showCrosshair = true;
        this.showML = true;
        this.showImbalances = true;
        this.showLens = false;

        // Profile Data
        this.sessionProfile = []; // { price, vol }
        this.pocPrice = 0;
        this.vahPrice = 0; // Value Area High
        this.valPrice = 0; // Value Area Low
        this.maxProfileVol = 0;

        // Heatmap controls
        this.heatmapHistoryPercent = 60;

        // Big trade auto-filter
        this.autoFilter = false;
        this._recentTradeSizes = [];
        this._recentTradeMax = 5000;

        // Thresholds
        // Heatmap defaults (Bookmap style)
        this.heatmapIntensityThreshold = 0.02; // Default 2% to filter noise
        this.heatmapOpacity = 0.8;
        this.maxVolumeInHistory = 5000; // Start high to avoid "Flash of Red" on load
        this.showBigTrades = true;
        this.bigTradeThreshold = 20.0; // User requested 20 default
        this.bigTradeScale = 1.0; // Visual size multiplier
        this.imbalanceThreshold = 3.0;

        // Interaction
        this.crosshairX = null;
        this.crosshairY = null;
        this.hoveredCandle = null;
        this.isDragging = false;
        this.isDraggingTime = false;
        this.isDraggingPrice = false;
        this.lastX = 0;
        this.lastY = 0;
        this.dragStartX = 0;
        this.dragStartY = 0;

        this.mlPrediction = null; // { prediction: 'win', confidence: 0.85 }
        this.mlHistory = new Map(); // timestamp -> signal

        // Drawing System
        this.drawingManager = null;
        this.currentSymbol = 'btcusdt';

        // Crater System (historical max volume markers)
        this.craters = [];

        // PRO COLORS
        this.colors = {
            bg: '#0e1012',            // Slightly lighter dark for better contrast
            grid: '#1f2226',          // Subtle grid
            text: '#a0a0a0',          // Muted text
            textBright: '#ffffff',    // Bright text for active values
            candleUp: '#10b981',      // Emerald 500
            candleDown: '#ef4444',    // Red 500
            wickUp: '#10b981',
            wickDown: '#ef4444',

            clusterText: '#ffffff',

            // Subtler Imbalance (Background)
            imbalanceBuy: 'rgba(16, 185, 129, 0.25)',
            imbalanceSell: 'rgba(239, 68, 68, 0.25)',

            // Bars (Volume)
            barBuy: 'rgba(16, 185, 129, 0.6)',
            barSell: 'rgba(239, 68, 68, 0.6)'
        };

        // Container
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Chart container not found:', containerId);
            this.isReady = false;
            return;
        }
        this.isReady = true;

        // Canvas
        this.canvas = document.createElement('canvas');
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d', { alpha: false });

        // Heatmap buffer
        this.heatmapBuffer = document.createElement('canvas');
        this.heatmapBufferCtx = this.heatmapBuffer.getContext('2d', { alpha: true });

        this.animationFrame = null;
        this._initEvents();
        this._initResizeObserver();
        this.requestDraw();
    }

    // PUBLIC API
    updateCandles(candles) {
        const oldCandles = this.candles || [];
        this.candles = candles || [];

        // PRE-CALCULATE CVD
        let runningDelta = 0;
        for (const c of this.candles) {
            const d = c.delta || 0;
            runningDelta += d;
            c.cvd = runningDelta;
        }

        // PRE-CALCULATE SESSION PROFILE
        this._calculateSessionProfile();

        // Auto-Center ONLY on actual dataset change (Symbol/TF switch)
        // NOT when candles naturally roll forward (which changes first candle time)
        // Heuristic: Only reset if we went from 0 to N candles (fresh load)
        // OR if the total number of candles doubled (massive reload)
        const isNewDataset = (oldCandles.length === 0 && this.candles.length > 0) ||
            (this.candles.length > oldCandles.length * 2);

        if (isNewDataset) {
            // ALWAYS set current price from last candle when loading new data
            if (this.candles.length > 0) {
                this.currentPrice = this.candles[this.candles.length - 1].close;
            }
            this.resetView();
        }

        this.requestDraw();
    }

    updateSessionMarkers(markers) {
        this.sessionMarkers = markers || [];
        this.requestDraw();
    }

    updateDepthHeatmap(data) {
        this.heatmapData = data ? (data.snapshots || data) : [];

        if (data && typeof data.maxVolume === 'number' && data.maxVolume > 0) {
            this.maxVolumeInHistory = data.maxVolume;
        } else {
            // Fallback: derive max volume (slower)
            let maxVol = 0;
            for (const s of this.heatmapData) {
                if (s.bids) for (const l of s.bids) maxVol = Math.max(maxVol, l.q || 0);
                if (s.asks) for (const l of s.asks) maxVol = Math.max(maxVol, l.q || 0);
            }
            this.maxVolumeInHistory = maxVol || 10;
        }

        this.requestDraw();
    }

    updatePrice(price) {
        this.currentPrice = price;

        // Try to complete any pending centering
        this._completePendingCenter();

        // Fallback auto-center on first price if not done yet
        const chartHeight = this.height - this._deltaH();
        if (!this.initialCenterDone && price > 0 && chartHeight > 100) {
            this.initialCenterDone = true;
            const priceInPixels = price / this.tickSize * this.zoomY;
            this.offsetY = chartHeight / 2 - priceInPixels;
            console.log('Auto-centered at', price, 'chartHeight', chartHeight);
        }
        this.requestDraw();
    }

    addBigTrade(trade) {
        if (!this.candles || this.candles.length === 0) return false;
        const qty = parseFloat(trade.quantity);
        if (qty < this.bigTradeThreshold) return false;

        this.bigTrades.push({
            candleIndex: this.candles.length - 1,
            volume: qty,
            price: parseFloat(trade.price),
            side: trade.isBuyerMaker ? 'sell' : 'buy'
        });
        if (this.bigTrades.length > 500) this.bigTrades.shift();
        this.requestDraw();
        return true;
    }

    setTickSize(size) { this.tickSize = size; this.requestDraw(); }
    setZoom(level) { this.zoomX = level; this.requestDraw(); }
    setBigTradeThreshold(val) { this.bigTradeThreshold = val; this.requestDraw(); }
    setBigTradeScale(val) { this.bigTradeScale = val; this.requestDraw(); } // New setter

    setMLPrediction(result) {
        this.mlPrediction = result; // { prediction: 'win', confidence: 0.85 }
        this.requestDraw();
    }

    setMLHistory(signals) {
        this.mlHistory.clear();
        if (signals && Array.isArray(signals)) {
            for (const s of signals) {
                this.mlHistory.set(s.candleTime, s);
            }
        }
        this.requestDraw();
    }

    setHeatmapIntensityThreshold(val) { this.heatmapIntensityThreshold = val; this.requestDraw(); }
    setHeatmapHistoryPercent(val) {
        this.heatmapHistoryPercent = Math.max(0, Math.min(100, val));
        this.requestDraw();
    }

    setAISignal(signal) {
        this.aiSignal = signal;
        console.log("AI Signal Set:", signal);
        this.requestDraw();
    }

    setLiquidityLevels(levels) {
        this.nakedLiquidityLevels = levels || [];
        this.requestDraw();
    }

    setCraters(craters) {
        this.craters = craters || [];
        this.requestDraw();
    }

    setDrawingManager(manager) {
        this.drawingManager = manager;

        // Listen for drawing changes
        manager.on('drawingsChange', () => this.requestDraw());
        manager.on('pendingChange', () => this.requestDraw());
    }

    setCurrentSymbol(symbol) {
        this.currentSymbol = symbol;
    }

    resetView() {
        this.offsetX = 0; // Jump to latest time (horizontal: show latest candles)
        this.initialCenterDone = false;

        // Calculate chart area height (excluding bottom panels like Delta, CVD, Time axis)
        const deltaH = this._deltaH();
        const chartHeight = this.height - deltaH;

        // Only center vertically if we have a valid chart height AND price
        if (this.currentPrice && chartHeight > 100) {
            // Center the current price in the middle of the CHART AREA
            const priceInPixels = this.currentPrice / this.tickSize * this.zoomY;
            this.offsetY = chartHeight / 2 - priceInPixels;
            this.initialCenterDone = true;
            console.log('resetView centered:', {
                price: this.currentPrice,
                chartHeight,
                deltaH,
                priceInPixels,
                offsetY: this.offsetY
            });
        } else {
            // Defer centering until height is available
            this._pendingCenter = true;
            console.log('resetView deferred - chartHeight:', chartHeight, 'price:', this.currentPrice);
        }

        this.requestDraw();
    }

    // Called from ResizeObserver or when price updates to complete deferred centering
    _completePendingCenter() {
        if (!this._pendingCenter) return;

        const deltaH = this._deltaH();
        const chartHeight = this.height - deltaH;

        if (!this.currentPrice || chartHeight <= 100) return;

        const priceInPixels = this.currentPrice / this.tickSize * this.zoomY;
        this.offsetY = chartHeight / 2 - priceInPixels;
        this.initialCenterDone = true;
        this._pendingCenter = false;
        console.log('Deferred center completed:', { price: this.currentPrice, offsetY: this.offsetY });
        this.requestDraw();
    }

    /**
     * Track all trades (used for auto-threshold suggestions)
     */
    trackTrade(trade) {
        const qty = Number(trade?.quantity);
        if (!Number.isFinite(qty) || qty <= 0) return;

        this._recentTradeSizes.push(qty);
        if (this._recentTradeSizes.length > this._recentTradeMax) {
            this._recentTradeSizes.splice(0, this._recentTradeSizes.length - this._recentTradeMax);
        }

        // Absorption Tracking
        // Check if trade hit any tracked wall
        if (this._wallHistory) {
            const price = Number(trade.price);
            const isBuyer = trade.isBuyerMaker === false; // Buyer = Taker Buy (Green)
            // If Buyer -> hits Ask wall. If Seller -> hits Bid wall.
            const targetType = isBuyer ? 'ask' : 'bid';

            for (const entry of this._wallHistory.values()) {
                // Only check active/qualified walls of correct type
                if (entry.zone.type !== targetType) continue;

                // Check if price is within wall zone (with small buffer)
                const zoneMin = entry.zone.priceMin - 5;
                const zoneMax = entry.zone.priceMax + 5;

                if (price >= zoneMin && price <= zoneMax) {
                    if (!entry.absorbed) entry.absorbed = 0;
                    entry.absorbed += qty;
                    entry.lastAbsorbTime = Date.now();
                }
            }
        }
    }

    /**
     * Auto-adaptive big trade threshold based on recent trade size distribution.
     * Uses the 99th percentile by default.
     */
    _calculateDynamicThreshold(percentile = 0.99) {
        const arr = this._recentTradeSizes;
        if (!arr || arr.length < 200) return this.bigTradeThreshold;

        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(sorted.length * percentile)));
        const p = sorted[idx];

        // Clamp for sanity
        const next = Math.max(0.01, Math.min(10000, p));
        this.setBigTradeThreshold(next);
        return next;
    }

    _calculateSessionProfile() {
        if (!this.candles || this.candles.length === 0) {
            this.sessionProfile = [];
            return;
        }

        const map = new Map();
        let maxVol = 0;
        let poc = 0;

        for (const c of this.candles) {
            if (!c.clusters) continue;

            // Handle both Map and Object formats
            let entries = [];
            if (c.clusters instanceof Map) entries = c.clusters.entries();
            else entries = Object.entries(c.clusters).map(([p, d]) => [parseFloat(p), d]);

            for (const [price, data] of entries) {
                const vol = (data.bid || 0) + (data.ask || 0);
                const current = map.get(price) || 0;
                const next = current + vol;
                map.set(price, next);

                if (next > maxVol) {
                    maxVol = next;
                    poc = price;
                }
            }
        }

        // Convert to array for rendering
        this.sessionProfile = Array.from(map.entries())
            .map(([price, vol]) => ({ price, vol }))
            .sort((a, b) => b.price - a.price); // Sort by price descending
        this.maxProfileVol = maxVol;
        this.pocPrice = poc;

        // Calculate Value Area (70% of volume) - VAH and VAL
        if (this.sessionProfile.length > 0) {
            const totalVolume = this.sessionProfile.reduce((sum, level) => sum + level.vol, 0);
            const valueAreaVolume = totalVolume * 0.7;

            // Find POC index
            let pocIndex = this.sessionProfile.findIndex(l => Math.abs(l.price - poc) < 0.0001);
            if (pocIndex < 0) pocIndex = 0;

            // Expand from POC to capture 70% volume
            let vaVolume = this.sessionProfile[pocIndex].vol;
            let vahIndex = pocIndex;
            let valIndex = pocIndex;

            while (vaVolume < valueAreaVolume && (vahIndex > 0 || valIndex < this.sessionProfile.length - 1)) {
                const upperVol = vahIndex > 0 ? this.sessionProfile[vahIndex - 1].vol : 0;
                const lowerVol = valIndex < this.sessionProfile.length - 1 ? this.sessionProfile[valIndex + 1].vol : 0;

                if (upperVol >= lowerVol && vahIndex > 0) {
                    vahIndex--;
                    vaVolume += upperVol;
                } else if (valIndex < this.sessionProfile.length - 1) {
                    valIndex++;
                    vaVolume += lowerVol;
                } else {
                    break;
                }
            }

            this.vahPrice = this.sessionProfile[vahIndex]?.price || poc;
            this.valPrice = this.sessionProfile[valIndex]?.price || poc;
        } else {
            this.vahPrice = poc;
            this.valPrice = poc;
        }
    }

    toggleHeatmap() { this.showHeatmap = !this.showHeatmap; this.requestDraw(); return this.showHeatmap; }
    toggleBigTrades() { this.showBigTrades = !this.showBigTrades; this.requestDraw(); return this.showBigTrades; }
    toggleCrosshair() { this.showCrosshair = !this.showCrosshair; this.requestDraw(); return this.showCrosshair; }
    toggleDelta() { this.showDelta = !this.showDelta; this.requestDraw(); return this.showDelta; }
    toggleImbalances() { this.showImbalances = !this.showImbalances; this.requestDraw(); return this.showImbalances; }

    /**
     * Calculate visible price range for VolumeProfile alignment
     */
    _calculatePriceRange() {
        const height = this.height - this._deltaH();
        return {
            max: this._getPriceFromY(0),
            min: this._getPriceFromY(height)
        };
    }

    // COORDS
    _deltaH() {
        let h = 30; // Reserve fixed 30px for Time Axis
        if (this.showDelta) h += this.deltaRowHeight;
        if (this.showCVD) h += this.cvdPanelHeight;
        return h;
    }
    _getX(index) { return Math.floor((index * this.zoomX) + this.offsetX) + 0.5; }
    _getY(price) { return Math.floor(this.height - ((price / this.tickSize * this.zoomY) + this.offsetY) - this._deltaH()) + 0.5; }
    _getPriceFromY(y) { return (this.height - y - this.offsetY - this._deltaH()) / this.zoomY * this.tickSize; }
    _getCandleAtX(x) {
        const index = Math.floor((x - this.offsetX) / this.zoomX);
        return (index >= 0 && index < this.candles.length) ? this.candles[index] : null;
    }

    // RENDER
    requestDraw() {
        if (!this.isReady || this.animationFrame) return;
        this.animationFrame = requestAnimationFrame(() => {
            this._render();
            this.animationFrame = null;
        });
    }

    _render() {
        if (!this.isReady || !this.ctx) return;

        // SAFETY: Auto-recover from NaN state (which causes blank chart)
        if (!Number.isFinite(this.zoomX) || this.zoomX <= 0) this.zoomX = 10;
        if (!Number.isFinite(this.zoomY) || this.zoomY <= 0) this.zoomY = 2;
        if (!Number.isFinite(this.offsetX)) this.offsetX = 0;
        if (!Number.isFinite(this.offsetY)) this.offsetY = 0;
        if (!Number.isFinite(this.cvdPanelHeight)) this.cvdPanelHeight = 60;

        const ctx = this.ctx;
        const { width, height } = this;
        if (width <= 0 || height <= 0) return;

        try {
            ctx.fillStyle = this.colors.bg;
            ctx.fillRect(0, 0, width, height);

            if (this.showHeatmap) this._drawHeatmap(ctx);
            this._drawGrid(ctx); // Grid behind profile?

            // Draw Profile BEHIND candles? or On Top?
            // Usually Profile is subtle background on the right.
            if (this.showSessionProfile) this._drawSessionProfile(ctx);

            this._drawCandles(ctx);
            this._drawSessionHighlights(ctx); // London/NY open markers

            // Draw Bottom Panels
            if (this.showDelta) this._drawDeltaSummary(ctx);
            if (this.showCVD) this._drawCVD(ctx);

            this._drawPriceLine(ctx);
            if (this.showBigTrades) this._drawBigTrades(ctx);
            if (this.showCrosshair && this.crosshairX) this._drawCrosshair(ctx);

            // Draw AI Signal
            if (this.aiSignal) this._drawAISignal(ctx);

            if (this.showLens) this._drawLens(ctx);

            // Wall Attack (Always On or toggle? User asked for it, lets keep it always on or tied to ML)
            // Let's tie it to 'showML' flag for now, or just always show if heavy wall nearby.
            if (this.showML) {
                this._drawLiquidityZones(ctx); // Draw zones as rectangles
                this._drawCraters(ctx); // Crater markers (historical max volume)
                this._drawWallAttack(ctx); // Existing HUD
                // this._drawHistoricalSignals(ctx); // Disabled: Bot signal labels
                this._drawTradePlan(ctx);  // NEW Trade Lines
            }

            // User Drawings (always on top)
            if (this.drawingManager) {
                this._drawDrawings(ctx);
            }

        } catch (e) {
            console.error('Chart Render Error:', e);
            // Fallback: Draw Error Text
            ctx.fillStyle = 'red';
            ctx.font = '14px sans-serif';
            ctx.fillText('Render Error: ' + e.message, 10, 20);
        }
    }

    /**
     * Draw session highlights - vertical lines at London Open (09:00 UTC) and NY Open (14:00 UTC)
     */
    _drawSessionHighlights(ctx) {
        if (!this.candles || this.candles.length === 0) return;

        const { height } = this;
        const deltaH = this._deltaH();
        const chartHeight = height - deltaH;

        ctx.save();
        ctx.setLineDash([5, 3]);
        ctx.lineWidth = 1;
        ctx.font = 'bold 10px sans-serif';

        const sessions = [
            { hour: 9, label: 'LDN', color: 'rgba(66, 165, 245, 0.6)' },   // London Open
            { hour: 14, label: 'NY', color: 'rgba(255, 167, 38, 0.6)' }    // NY Open
        ];

        for (let i = 0; i < this.candles.length; i++) {
            const candle = this.candles[i];
            const date = new Date(candle.time);
            const hourUTC = date.getUTCHours();
            const minuteUTC = date.getUTCMinutes();

            for (const session of sessions) {
                // Only mark the first candle of the hour
                if (hourUTC === session.hour && minuteUTC < 5) {
                    const x = this._getX(i) + this.zoomX / 2;

                    ctx.strokeStyle = session.color;
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, chartHeight);
                    ctx.stroke();

                    // Draw label at top
                    ctx.fillStyle = session.color;
                    ctx.fillText(session.label, x + 4, 12);
                }
            }
        }

        ctx.setLineDash([]);
        ctx.restore();
    }

    _drawSessionProfile(ctx) {
        if (!this.sessionProfile || this.sessionProfile.length === 0) return;

        const { width, height, maxProfileVol, pocPrice } = this;
        const deltaH = this._deltaH();
        const profileWidth = 100; // Max width in pixels
        const startX = width - profileWidth; // Right align

        ctx.save();
        ctx.globalAlpha = 0.3; // Subtle

        for (const { price, vol } of this.sessionProfile) {
            const y = this._getY(price);
            if (y < 0 || y > height - deltaH) continue;

            const barH = Math.max(1, this.zoomY); // Match candle row height
            // Don't draw if too dense? logic check:
            // if (this.zoomY < 0.5) ... maybe skip or generic block

            const barW = (vol / maxProfileVol) * profileWidth;

            // Color: POC is Yellow, others are Grey/Blue
            if (Math.abs(price - pocPrice) < 0.0001) {
                ctx.fillStyle = '#fbbf24'; // Amber/Gold POC
                ctx.globalAlpha = 0.8; // PoC stands out
            } else {
                ctx.fillStyle = '#64748b'; // Slate 500
                ctx.globalAlpha = 0.2;
            }

            // Draw from Right Edge
            ctx.fillRect(width - barW, y - barH / 2, barW, barH);
        }

        // Draw POC Line extending
        const pocY = this._getY(pocPrice);
        if (pocY > 0 && pocY < height - deltaH) {
            ctx.globalAlpha = 0.8;
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(width - profileWidth, pocY);
            ctx.lineTo(width, pocY);
            ctx.stroke();

            // POC Label
            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = '#fbbf24';
            ctx.textAlign = 'right';
            ctx.fillText('POC', width - profileWidth - 4, pocY + 3);
        }

        // Draw VAH Line (Value Area High) - dashed cyan
        const vahY = this._getY(this.vahPrice);
        if (this.vahPrice && vahY > 0 && vahY < height - deltaH) {
            ctx.globalAlpha = 0.6;
            ctx.strokeStyle = '#06b6d4'; // Cyan
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(width - profileWidth, vahY);
            ctx.lineTo(width, vahY);
            ctx.stroke();
            ctx.setLineDash([]);

            // VAH Label
            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = '#06b6d4';
            ctx.textAlign = 'right';
            ctx.fillText('VAH', width - profileWidth - 4, vahY + 3);
        }

        // Draw VAL Line (Value Area Low) - dashed magenta
        const valY = this._getY(this.valPrice);
        if (this.valPrice && valY > 0 && valY < height - deltaH) {
            ctx.globalAlpha = 0.6;
            ctx.strokeStyle = '#d946ef'; // Magenta
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(width - profileWidth, valY);
            ctx.lineTo(width, valY);
            ctx.stroke();
            ctx.setLineDash([]);

            // VAL Label
            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = '#d946ef';
            ctx.textAlign = 'right';
            ctx.fillText('VAL', width - profileWidth - 4, valY + 3);
        }

        // Draw Value Area shading
        if (this.vahPrice && this.valPrice && vahY > 0 && valY < height - deltaH) {
            ctx.globalAlpha = 0.05;
            ctx.fillStyle = '#fbbf24';
            ctx.fillRect(width - profileWidth, vahY, profileWidth, valY - vahY);
        }

        ctx.restore();
    }

    _drawHeatmap(ctx) {
        if (!this.heatmapData || this.heatmapData.length === 0) return;
        const { width, height } = this;
        const deltaH = this._deltaH();

        if (this.heatmapBuffer.width !== width || this.heatmapBuffer.height !== height) {
            this.heatmapBuffer.width = width;
            this.heatmapBuffer.height = height;
        }

        const bufCtx = this.heatmapBufferCtx;
        bufCtx.clearRect(0, 0, width, height);

        // OPTIMIZATION: Create a time-indexed lookup for heatmap snapshots to avoid O(N²) search
        // We assume heatmapData is sorted by time. We can just use a find helper.

        const startIdx = Math.floor(-this.offsetX / this.zoomX);
        const endIdx = startIdx + Math.ceil(width / this.zoomX) + 1;

        // Start search pointer
        let hPtr = 0;
        const hLen = this.heatmapData.length;
        const totalCandles = this.candles.length;
        const isLiveCandle = (i) => i === totalCandles - 1; // The rightmost forming candle

        // Use GLOBAL maxVolumeInHistory for consistent rendering at all zoom levels
        // This prevents the heatmap from becoming too dense when zoomed in
        // or too sparse when zoomed out
        const minFloor = (this.liquidityThreshold || 10) * 20; // e.g. 200 BTC minimum scale
        const visibleMaxVol = Math.max(this.maxVolumeInHistory || 5000, minFloor);

        for (let i = startIdx; i <= endIdx; i++) {
            // Find candle
            if (i < 0 || i >= totalCandles) continue;
            const candle = this.candles[i];
            if (!candle) continue;

            const cTime = candle.time;

            // Find snapshot (Cached index optimization if sequential?)
            // We'll stick to robust search since we fixed the logic
            while (hPtr < hLen - 1 && this.heatmapData[hPtr].time < cTime) {
                hPtr++;
            }

            // Snapshot selection logic 
            // If cTime is much newer than last snapshot, utilize Live Merging logic
            // Check if we are in the "Live Window" (last 2 mins of history edge)
            const lastSnapTime = hLen > 0 ? this.heatmapData[hLen - 1].time : 0;
            const timeBuffer = 120000;

            let levels = [];
            let snapshot = null;

            if (cTime >= lastSnapTime - timeBuffer && hLen > 0) {
                // Live Merging: Last 60 snaps
                const limit = 60;
                const endPtr = hPtr;
                const startPtr = Math.max(0, endPtr - limit + 1);

                const seen = new Set();
                for (let k = startPtr; k <= endPtr; k++) {
                    const snap = this.heatmapData[k];
                    if (snap.bids) {
                        for (const b of snap.bids) {
                            const key = b.p.toFixed(1); // Dedup by price
                            if (!seen.has(key)) { seen.add(key); levels.push(b); }
                        }
                    }
                    if (snap.asks) {
                        for (const a of snap.asks) {
                            const key = a.p.toFixed(1);
                            if (!seen.has(key)) { seen.add(key); levels.push(a); }
                        }
                    }
                }
                snapshot = this.heatmapData[endPtr]; // Just for reference
            } else {
                // Historical: Single snapshot
                snapshot = this.heatmapData[hPtr];
                if (snapshot) {
                    if (snapshot.bids) levels.push(...snapshot.bids);
                    if (snapshot.asks) levels.push(...snapshot.asks);
                }
            }

            if (levels.length === 0) continue;

            // DRAW COLUMN (Heatmap)
            const x = this._getX(i);
            const w = this.zoomX;
            const h = Math.max(1, this.zoomY);



            for (const lev of levels) {
                if (!lev.p || !lev.q) continue;
                const y = this._getY(lev.p);

                // Optimization: Skip off-screen
                if (y + h < -200 || y > height + 200) continue;

                this._drawHeatmapRect(bufCtx, x, y - h / 2, w + 1, h, lev.q, visibleMaxVol);
            }


        }

        ctx.drawImage(this.heatmapBuffer, 0, 0);
    }

    _drawHeatmapRect(ctx, x, y, w, h, q, maxVol) {
        // Auto-Contrast Ratio
        const ratio = q / maxVol;
        if (ratio < this.heatmapIntensityThreshold) return;

        // Color mapping
        const visRatio = Math.pow(ratio, 0.4);
        let color;
        const alpha = this.heatmapOpacity;
        if (visRatio < 0.2) color = `rgba(0, 50, 200, ${alpha * 0.4})`;
        else if (visRatio < 0.4) color = `rgba(0, 150, 255, ${alpha * 0.6})`;
        else if (visRatio < 0.6) color = `rgba(0, 255, 200, ${alpha * 0.8})`;
        else if (visRatio < 0.8) color = `rgba(255, 255, 0, ${alpha * 0.9})`;
        else color = `rgba(255, 50, 0, ${alpha})`;

        ctx.fillStyle = color;
        ctx.fillRect(x, y, w, h);
    }

    _drawGrid(ctx) {
        const { width, height, tickSize, offsetX, zoomX } = this;
        const chartHeight = height - this._deltaH();

        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();

        // Horizontal Grid (Price)
        const minPrice = this._getPriceFromY(chartHeight);
        const maxPrice = this._getPriceFromY(0);
        const step = tickSize * 10;
        const startPrice = Math.floor(minPrice / step) * step;

        for (let p = startPrice; p <= maxPrice; p += step) {
            const y = this._getY(p);
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
        }

        // Vertical Grid (Time) & Labels
        const labelWidth = 50;
        const candleStep = Math.max(1, Math.ceil(labelWidth / zoomX));
        const startIdx = Math.floor(-offsetX / zoomX);
        const endIdx = startIdx + (width / zoomX) + 1;

        for (let i = Math.floor(startIdx / candleStep) * candleStep; i <= endIdx; i += candleStep) {
            if (i < 0 || i >= this.candles.length) continue;
            const x = this._getX(i);

            // Grid Line
            ctx.moveTo(x, 0);
            ctx.lineTo(x, chartHeight);

            // Labels
            const candle = this.candles[i];
            if (candle && candle.time) {
                const d = new Date(candle.time);
                const tStr = `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
                const dStr = `${d.getDate().toString().padStart(2, '0')}.${(d.getMonth() + 1).toString().padStart(2, '0')}`;

                ctx.textAlign = 'center';
                ctx.fillStyle = this.colors.textBright;
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(tStr, x, chartHeight + 14); // Below chart

                ctx.fillStyle = '#888';
                ctx.font = '10px sans-serif';
                ctx.fillText(dStr, x, chartHeight + 26);
            }
        }
        ctx.stroke();

        // Price Labels (Right Axis)
        ctx.fillStyle = this.colors.text;
        ctx.font = '10px "Roboto Mono", monospace';
        ctx.textAlign = 'right';
        for (let p = startPrice; p <= maxPrice; p += step) {
            const y = this._getY(p);
            if (y > 10 && y < chartHeight - 10) ctx.fillText(p.toFixed(2), width - 5, y);
        }
    }

    _drawCandles(ctx) {
        const { candles, zoomX, zoomY, offsetX, width, height } = this;
        if (!candles || candles.length === 0) return;

        const startIdx = Math.max(0, Math.floor(-offsetX / zoomX) - 1);
        const endIdx = Math.min(candles.length - 1, Math.ceil((width - offsetX) / zoomX) + 1);
        const candleWidth = Math.max(1, zoomX - 4); // More spacing between candles

        // GRADUATED DETAIL LEVELS
        // Level 1: Standard Candles (Zoom < 45)
        // Level 2: Graphic Footprint (Zoom 45 - 80)
        // Level 3: Full Text Footprint (Zoom > 80)
        const showGraphics = zoomX > 45;
        const showText = zoomX > 80;

        ctx.lineWidth = 1;

        for (let i = startIdx; i <= endIdx; i++) {
            const c = candles[i];
            if (!c) continue;

            const x = this._getX(i);
            const bull = c.close >= c.open;

            // Common Wick (Always drawn)
            ctx.strokeStyle = bull ? this.colors.wickUp : this.colors.wickDown;
            const centerX = Math.floor(x + candleWidth / 2) + 0.5;

            ctx.beginPath();
            ctx.moveTo(centerX, this._getY(c.high));
            ctx.lineTo(centerX, this._getY(c.low));
            ctx.stroke();

            // LEVEL 1: STANDARD SOLID CANDLE
            if (!showGraphics) {
                const color = bull ? this.colors.candleUp : this.colors.candleDown;
                const openY = this._getY(c.open);
                const closeY = this._getY(c.close);
                ctx.fillStyle = color;

                const topY = Math.min(openY, closeY);
                const h = Math.abs(openY - closeY) || 1;

                // Solid fill for clear "Candle" recognition
                ctx.fillRect(x, topY, candleWidth, h);

                // Border for crispness
                ctx.strokeStyle = bull ? '#000' : '#000'; // Slight outline for pop? Or simple color
                // Actually, pure solid is best for readability at low zoom.

            } else {
                // LEVEL 2 & 3: FOOTPRINT (Hollow Body + Internals)
                const clusters = c.clusters;
                if (!clusters) continue; // Fallback?

                let entries = [];
                if (clusters instanceof Map) {
                    entries = Array.from(clusters.entries());
                } else {
                    entries = Object.entries(clusters).map(([p, data]) => [parseFloat(p), data]);
                }

                // Defensive: ignore malformed cluster entries
                entries = entries.filter(([price, data]) => Number.isFinite(price) && data && typeof data === 'object');

                let maxVol = 0;
                for (const [, data] of entries) {
                    maxVol = Math.max(maxVol, (data.bid || 0), (data.ask || 0));
                }

                const gap = 2; // Cleaner gap
                const halfWidth = (candleWidth - (gap * 2)) / 2;

                for (const [price, data] of entries) {
                    const y = this._getY(price);
                    const rowH = Math.max(1, zoomY);
                    const drawY = y - rowH / 2;

                    if (drawY + rowH < 0 || drawY > height - this.deltaRowHeight) continue;

                    // Imbalance (Subtle Background)
                    if (this.showImbalances && data.imbalance) {
                        ctx.fillStyle = data.imbalance === 'buy' ? this.colors.imbalanceBuy : this.colors.imbalanceSell;
                        ctx.fillRect(x, drawY, candleWidth, rowH);
                    }

                    // VOLUME BARS (Graphic Level)
                    // Bid Left (Red/Sell)
                    if (data.bid > 0) {
                        const len = (data.bid / maxVol) * halfWidth;
                        ctx.fillStyle = this.colors.barSell;
                        ctx.fillRect(centerX - gap - len, drawY + 1, len, rowH - 1);
                    }
                    // Ask Right (Green/Buy)
                    if (data.ask > 0) {
                        const len = (data.ask / maxVol) * halfWidth;
                        ctx.fillStyle = this.colors.barBuy;
                        ctx.fillRect(centerX + gap, drawY + 1, len, rowH - 1);
                    }

                    // TEXT (Full Detail Level)
                    if (showText && rowH > 11) {
                        ctx.fillStyle = this.colors.clusterText;
                        ctx.font = 'bold 11px "Roboto Mono", monospace';

                        // Left Text (Bid/Red)
                        ctx.textAlign = 'right';
                        if (data.bid > 0) ctx.fillText(data.bid.toFixed(0), centerX - gap - 4, y + 4);

                        // Right Text (Ask/Green)
                        ctx.textAlign = 'left';
                        if (data.ask > 0) ctx.fillText(data.ask.toFixed(0), centerX + gap + 4, y + 4);
                    }
                }

                // CANDLE BORDER (The "Container")
                // This makes it look like a "Candle" even when full of data
                const openY = this._getY(c.open);
                const closeY = this._getY(c.close);

                ctx.strokeStyle = bull ? this.colors.candleUp : this.colors.candleDown;
                ctx.lineWidth = 2; // Thicker border for better visibility
                ctx.strokeRect(x, Math.min(openY, closeY), candleWidth, Math.abs(openY - closeY) || 1);
                ctx.lineWidth = 1; // Reset
            }
        }
    }

    _drawDeltaSummary(ctx) {
        const { candles, zoomX, offsetX, width, height } = this;
        const panelHeight = this.deltaRowHeight;
        const rowTop = height - panelHeight - this.cvdPanelHeight; // Shift up above CVD

        // Darker footer background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, rowTop, width, panelHeight);

        // Top Border
        ctx.strokeStyle = '#333';
        ctx.beginPath();
        ctx.moveTo(0, rowTop);
        ctx.lineTo(width, rowTop);
        ctx.stroke();

        const startIdx = Math.max(0, Math.floor(-offsetX / zoomX) - 1);
        const endIdx = Math.min(candles.length - 1, Math.ceil((width - offsetX) / zoomX) + 1);
        const candleWidth = Math.max(1, zoomX - 2);

        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';

        for (let i = startIdx; i <= endIdx; i++) {
            const c = candles[i];
            if (!c) continue;

            const x = this._getX(i);
            const centerX = x + candleWidth / 2;
            const delta = c.delta || 0;
            const isPos = delta >= 0;

            // Subtle Delta Bar
            ctx.fillStyle = isPos ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
            ctx.fillRect(x, rowTop + 2, candleWidth, panelHeight - 4);

            // Value
            if (zoomX > 30) {
                ctx.fillStyle = isPos ? this.colors.candleUp : this.colors.candleDown;
                ctx.fillText(delta.toFixed(0), centerX, rowTop + 15);
            } else {
                ctx.fillStyle = isPos ? this.colors.candleUp : this.colors.candleDown;
                ctx.fillRect(centerX - 1.5, rowTop + 10, 3, 3);
            }
        }
    }

    _drawCVD(ctx) {
        const { candles, width, height, zoomX, offsetX } = this;
        const panelH = this.cvdPanelHeight;
        const topY = height - panelH;

        // Background
        ctx.fillStyle = '#0e1012';
        ctx.fillRect(0, topY, width, panelH);

        // Border Top
        ctx.strokeStyle = '#333';
        ctx.beginPath();
        ctx.moveTo(0, topY);
        ctx.lineTo(width, topY);
        ctx.stroke();

        if (!candles || candles.length === 0) return;

        // Auto-Scale CVD logic
        // We only care about visible range min/max to scale appropriately? -> Better: Global or Loaded-Range scaling
        // Let's use Loaded-Range for stability
        let minCVD = Infinity, maxCVD = -Infinity;

        const startIdx = Math.max(0, Math.floor(-offsetX / zoomX) - 1);
        const endIdx = Math.min(candles.length - 1, Math.ceil((width - offsetX) / zoomX) + 1);

        // Find min/max in visible area for better local contrast
        for (let i = startIdx; i <= endIdx; i++) {
            const val = candles[i]?.cvd;
            if (val !== undefined) {
                if (val < minCVD) minCVD = val;
                if (val > maxCVD) maxCVD = val;
            }
        }

        if (minCVD === Infinity) return; // No data
        if (maxCVD === minCVD) { maxCVD += 100; minCVD -= 100; } // avoid div/0

        const range = maxCVD - minCVD;
        const padding = 5;
        const plotH = panelH - (padding * 2);

        // Helper to map CVD value to Y
        const getCvdY = (val) => {
            const norm = (val - minCVD) / range; // 0..1
            // 0 is bottom, 1 is top. 
            // Screen Y: TopY + padding + (1-norm)*plotH
            return topY + padding + (1 - norm) * plotH;
        };

        // Draw Line
        ctx.strokeStyle = '#3b82f6'; // Blue 500
        ctx.lineWidth = 2;
        ctx.beginPath();

        let first = true;
        for (let i = startIdx; i <= endIdx; i++) {
            const x = this._getX(i) + (zoomX / 2); // Center of candle
            const val = candles[i]?.cvd;
            if (val === undefined) continue;

            const y = getCvdY(val);
            if (first) { ctx.moveTo(x, y); first = false; }
            else { ctx.lineTo(x, y); }
        }
        ctx.stroke();

        // Draw Zero Line if visible? Usually CVD is huge number, so Zero is far off.
        // Instead draw Start Line or just text label.

        // Label Last Value
        const lastCandle = candles[Math.min(candles.length - 1, endIdx)];
        if (lastCandle && lastCandle.cvd !== undefined) {
            const lastY = getCvdY(lastCandle.cvd);
            ctx.fillStyle = '#3b82f6';
            ctx.textAlign = 'right';
            ctx.fillText(`CVD: ${lastCandle.cvd.toFixed(0)}`, width - 5, topY + 12);
        }
    }

    _drawLens(ctx) {
        if (!this.lastX || !this.lastY) return;

        const mx = this.lastX;
        const my = this.lastY;

        // --- CONFIGURATION ---
        const lensSize = 260;
        const lensX = 20; // Fixed Left
        const lensY = 80; // Fixed Top (below header)

        // Lens Zoom Settings (minimum 85px for text)
        const lensZoomX = Math.max(this.zoomX * 1.0, 85);
        const lensZoomY = Math.max(this.zoomY * 1.0, 15);

        // --- 1. DRAW SOURCE INDICATOR (On Main Chart) ---
        // How big is the lens viewport in "World Units"?
        // WidthWorld = lensSize / lensZoomX
        // HeightWorld = lensSize / lensZoomY
        //
        // So on the Main Chart (current zoom), the source rect size is:
        const srcW = (lensSize / lensZoomX) * this.zoomX;
        const srcH = (lensSize / lensZoomY) * this.zoomY;

        ctx.save();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 2]);
        ctx.shadowColor = '#000';
        ctx.shadowBlur = 2;
        ctx.strokeRect(mx - srcW / 2, my - srcH / 2, srcW, srcH);

        // Crosshair center of source
        ctx.beginPath();
        ctx.moveTo(mx - 5, my); ctx.lineTo(mx + 5, my);
        ctx.moveTo(mx, my - 5); ctx.lineTo(mx, my + 5);
        ctx.stroke();
        ctx.restore();


        // --- 2. PREPARE LENS RENDERING ---
        // Logic: Map User Mouse Point (World Coords) -> Center of Lens Box (Lens Coords)

        // World Point under mouse:
        const worldInd = (mx - this.offsetX) / this.zoomX;
        const priceUnits = (this.height - my - this.offsetY - this._deltaH()) / this.zoomY;

        // Calculate Offset for Lens Context
        // We want: worldPoint * lensZoom + newOffset = CenterOfLens
        // So: newOffset = CenterOfLens - (worldPoint * lensZoom)

        const lensCenterX = lensX + lensSize / 2;
        const lensCenterY = lensY + lensSize / 2;

        const lensOffsetX = lensCenterX - (worldInd * lensZoomX);
        const lensOffsetY = (lensCenterY - this._deltaH()) - (priceUnits * lensZoomY); // Inverse Y logic from mouse calc?

        // Re-calcing OffsetY properly:
        // ScreenY = Height - OffsetY - DeltaH - (PriceUnits * ZoomY)
        // We want ScreenY to be lensCenterY.
        // lensCenterY = LensHeight (which is effectively this.height in the clipped ctx?)
        // Wait, 'this.height' is global.
        // Let's stick to the simpler translation:
        // LensOffsetY needs to shift the drawing so that 'worldInd' is at 'lensCenterX'

        // Y-Axis is inverted in _getY: height - offsetY - ...
        // We need: this.height - lensOffsetY - this._deltaH() - (priceUnits * lensZoomY) = lensCenterY
        // => lensOffsetY = this.height - lensCenterY - this._deltaH() - (priceUnits * lensZoomY)

        const calcLensOffsetY = this.height - lensCenterY - this._deltaH() - (priceUnits * lensZoomY);


        // --- 3. CLIP & RENDER LENS ---
        ctx.save();

        // Draw Lens Background (Fixed Box)
        ctx.fillStyle = '#0e1012';
        ctx.fillRect(lensX, lensY, lensSize, lensSize);

        // Clip to Box
        ctx.beginPath();
        ctx.rect(lensX, lensY, lensSize, lensSize);
        ctx.clip();

        // Swap State
        const saveZx = this.zoomX;
        const saveZy = this.zoomY;
        const saveOx = this.offsetX;
        const saveOy = this.offsetY;

        this.zoomX = lensZoomX;
        this.zoomY = lensZoomY;
        this.offsetX = lensOffsetX;
        this.offsetY = calcLensOffsetY;

        // Render
        if (this.showHeatmap) this._drawHeatmap(ctx);
        this._drawGrid(ctx); // Grid lines might look weird without bounds check, but OK
        this._drawCandles(ctx);
        if (this.showBigTrades) this._drawBigTrades(ctx);

        // Restore State
        this.zoomX = saveZx;
        this.zoomY = saveZy;
        this.offsetX = saveOx;
        this.offsetY = saveOy;

        ctx.restore();


        // --- 4. DATA HEADER OVERLAY (Inside Lens) ---
        // Draw a header bar inside the lens to show exact Time/Price/Delta
        if (this.hoveredCandle) {
            ctx.save();
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            ctx.fillRect(lensX, lensY, lensSize, 20);
            ctx.fillStyle = '#ccc';
            ctx.font = '10px Roboto Mono';
            ctx.textAlign = 'left';
            const c = this.hoveredCandle;
            const d = new Date(c.time);
            const timeStr = `${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
            const delta = c.delta || 0;
            ctx.fillText(`${timeStr} | $${this.currentPrice?.toFixed(1) || ''} | Δ ${delta.toFixed(0)}`, lensX + 5, lensY + 14);
            ctx.restore();
        }

        // --- 5. LENS BORDER ---
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.strokeRect(lensX, lensY, lensSize, lensSize);

        // Center Crosshair inside Lens
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lensCenterX - 10, lensCenterY); ctx.lineTo(lensCenterX + 10, lensCenterY);
        ctx.moveTo(lensCenterX, lensCenterY - 10); ctx.lineTo(lensCenterX, lensCenterY + 10);
        ctx.stroke();

    }

    _drawPriceLine(ctx) {
        if (!this.currentPrice) return;
        const y = this._getY(this.currentPrice);
        if (y < 0 || y > this.height - this.deltaRowHeight) return;

        ctx.strokeStyle = '#fff';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(this.width, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#10b981'; // Active price color
        ctx.fillRect(this.width - 60, y - 10, 60, 20);
        ctx.fillStyle = '#000';
        ctx.textAlign = 'right';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(this.currentPrice.toFixed(2), this.width - 8, y + 4);
    }

    _drawBigTrades(ctx) {
        for (const t of this.bigTrades) {
            const x = this._getX(t.candleIndex);
            const y = this._getY(t.price);
            const deltaH = this._deltaH();
            if (x < 0 || x > this.width || y < 0 || y > this.height - deltaH) continue;

            // Radius based on volume relative to threshold, MULTIPLIED by user scale
            // Base size: 3px, Max: 20px
            // Logarithmic scale for better distribution
            const relVol = t.q / this.bigTradeThreshold;
            const baseRadius = 3 + Math.log(relVol) * 3;
            const radius = Math.max(2, Math.min(30, baseRadius * this.bigTradeScale)); // Apply Scale

            // Validate all values are finite before creating gradient
            if (!isFinite(x) || !isFinite(y) || !isFinite(radius) || radius <= 0) continue;

            // 3D Sphere Effect (Radial Gradient)
            const gradient = ctx.createRadialGradient(
                x - radius * 0.3, y - radius * 0.3, radius * 0.1, // Highlight offset
                x, y, radius // Base circle
            );

            if (t.side === 'buy') {
                gradient.addColorStop(0, '#86efac'); // bright green center
                gradient.addColorStop(1, '#16a34a'); // darker green edge
                ctx.strokeStyle = '#dcfce7';     // nearly white outline
            } else {
                gradient.addColorStop(0, '#fca5a5'); // bright red center
                gradient.addColorStop(1, '#dc2626'); // darker red edge
                ctx.strokeStyle = '#fee2e2';     // nearly white outline
            }

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.stroke();

            // Optional: Volume Label inside bigger bubbles
            if (radius > 8) {
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 9px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(t.volume.toFixed(0), x, y);
            }
        }
    }




    /**
     * ADVANCED LIQUIDITY INTELLIGENCE SYSTEM
     * Features:
     * - Wall Absorption Tracking (size changes over time)
     * - Spoofing Detection (sudden wall disappearance)
     * - Wall Clustering (group nearby walls into zones)
     * - Delta Integration (order flow direction)
     * - Momentum Factor (recent trades weighted more)
     * - Modern Glassmorphism UI
     */

    /**
     * Draw crater markers - fading indicators where volume WAS high historically
     */
    _drawCraters(ctx) {
        if (!this.craters || this.craters.length === 0) return;

        const { width } = this;
        ctx.save();

        for (const crater of this.craters) {
            const y = this._getY(crater.price);

            // Skip if off-screen
            if (y < 0 || y > this.height) continue;

            // Color based on intensity (fades over time)
            const intensity = crater.intensity || 0.5;
            const alpha = Math.max(0.1, intensity * 0.4);

            // Purple/magenta color for craters (distinct from walls)
            ctx.strokeStyle = `rgba(156, 39, 176, ${alpha})`;
            ctx.fillStyle = `rgba(156, 39, 176, ${alpha * 0.3})`;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);

            // Draw dashed line across chart
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width - 120, y);
            ctx.stroke();

            // Draw crater label on the right
            if (intensity > 0.3) {
                const labelX = width - 115;
                ctx.setLineDash([]);
                ctx.font = 'bold 9px monospace';
                ctx.fillStyle = `rgba(156, 39, 176, ${Math.max(0.5, intensity)})`;
                ctx.fillText(`⬤ ${crater.maxVolume.toFixed(1)}`, labelX, y + 3);
            }
        }

        ctx.setLineDash([]);
        ctx.restore();
    }

    /**
     * Draw liquidity zones as colored rectangles
     * Green = Support (bid walls), Red = Resistance (ask walls)
     */
    _drawLiquidityZones(ctx) {
        if (!this.nakedLiquidityLevels || this.nakedLiquidityLevels.length === 0) return;

        const { width } = this;
        ctx.save();

        for (const zone of this.nakedLiquidityLevels) {
            // Get Y coordinates
            const yMin = zone.priceMin ? this._getY(zone.priceMax) : this._getY(zone.price * 1.0005);
            const yMax = zone.priceMin ? this._getY(zone.priceMin) : this._getY(zone.price * 0.9995);
            const height = Math.max(yMax - yMin, 4); // Minimum 4px height

            // Skip if off-screen
            if (yMax < 0 || yMin > this.height) continue;

            // Color based on type: bid = green (support), ask = red (resistance)
            const isBid = zone.type === 'bid';
            const baseColor = isBid ? '0, 200, 83' : '244, 67, 54';

            // Draw filled rectangle with transparency
            ctx.fillStyle = `rgba(${baseColor}, 0.15)`;
            ctx.fillRect(0, yMin, width - 120, height);

            // Draw border
            ctx.strokeStyle = `rgba(${baseColor}, 0.6)`;
            ctx.lineWidth = 1;
            ctx.strokeRect(0, yMin, width - 120, height);

            // Draw volume label
            ctx.fillStyle = `rgba(${baseColor}, 0.9)`;
            ctx.font = 'bold 10px monospace';
            const label = `${zone.volume.toFixed(1)} BTC`;
            const labelX = width - 115;
            const labelY = yMin + height / 2 + 3;
            ctx.fillText(label, labelX, labelY);
        }

        ctx.restore();
    }

    _drawWallAttack(ctx) {
        if (!this.heatmapData || !this.currentPrice || !this.candles.length) return;

        const now = Date.now();
        const snapshot = this.heatmapData[this.heatmapData.length - 1];
        if (!snapshot) return;

        // Initialize wall tracker if needed
        if (!this._wallTracker) {
            this._wallTracker = new Map(); // price -> wall data
            this._spoofAlerts = [];
        }

        // Reset wall zones for hover detection
        this._wallZones = [];

        // ===============================
        // 1. SCAN & CLUSTER WALLS
        // ===============================
        // Much lower threshold to detect more walls (10 BTC max)
        const wallThreshold = Math.min(10, Math.max(3, this.maxVolumeInHistory * 0.01));
        const viewMin = this._getPriceFromY(this.height - this._deltaH());
        const viewMax = this._getPriceFromY(0);
        const CLUSTER_RANGE = 100; // $100 clustering

        // Debug: Log threshold
        if (!this._lastThresholdLog || Date.now() - this._lastThresholdLog > 5000) {
            console.log(`[Liquidity] Threshold: ${wallThreshold.toFixed(1)} BTC, MaxVol: ${this.maxVolumeInHistory?.toFixed(1)}`);
            this._lastThresholdLog = Date.now();
        }

        // Collect all significant walls
        const rawWalls = [];
        const scanWalls = (levels, type) => {
            if (!levels) return;
            for (const l of levels) {
                if (l.p < viewMin || l.p > viewMax) continue;
                if (l.q >= wallThreshold) {
                    rawWalls.push({ price: l.p, size: l.q, type });
                }
            }
        };
        scanWalls(snapshot.bids, 'bid');
        scanWalls(snapshot.asks, 'ask');

        // Cluster nearby walls into zones
        const zones = [];
        const usedWalls = new Set();

        for (const wall of rawWalls) {
            if (usedWalls.has(wall.price)) continue;

            // Find all walls within cluster range
            const cluster = rawWalls.filter(w =>
                !usedWalls.has(w.price) &&
                w.type === wall.type &&
                Math.abs(w.price - wall.price) <= CLUSTER_RANGE
            );

            if (cluster.length > 0) {
                const totalSize = cluster.reduce((sum, w) => sum + w.size, 0);
                const avgPrice = cluster.reduce((sum, w) => sum + w.price * w.size, 0) / totalSize;
                const minPrice = Math.min(...cluster.map(w => w.price));
                const maxPrice = Math.max(...cluster.map(w => w.price));

                zones.push({
                    priceMin: minPrice,
                    priceMax: maxPrice,
                    avgPrice: avgPrice,
                    totalSize: totalSize,
                    wallCount: cluster.length,
                    type: wall.type
                });

                cluster.forEach(w => usedWalls.add(w.price));
            }
        }

        // ===============================
        // WALL PERSISTENCE SCORING SYSTEM
        // Instead of just "exists now or not", we track:
        // - How often the wall was seen in the last 30 seconds
        // - persistence = samples_seen / total_samples
        // - Only show walls with >50% persistence (stable walls)
        // ===============================
        const wallNow = Date.now();
        const TRACKING_WINDOW = 30000; // 30 seconds
        const SAMPLE_INTERVAL = 1000;  // Sample every 1 second
        const MIN_PERSISTENCE = 0.5;   // 50% presence required
        const FADEOUT_TIME = 15000;    // 15 seconds fade-out after dropping below threshold

        if (!this._wallHistory) this._wallHistory = new Map();

        // Update wall history with current zones
        const currentZoneKeys = new Set();
        for (const zone of zones) {
            const key = `${zone.type}_${Math.round(zone.avgPrice / 200) * 200}`; // $200 buckets
            currentZoneKeys.add(key);

            if (!this._wallHistory.has(key)) {
                // New wall - start tracking
                this._wallHistory.set(key, {
                    firstSeen: wallNow,
                    lastSeen: wallNow,
                    zone: zone,
                    samples: [wallNow], // Array of timestamps when seen
                    qualified: false,   // Has met persistence threshold
                    qualifiedTime: null
                });
            } else {
                // Update existing - add sample if enough time passed
                const entry = this._wallHistory.get(key);
                entry.lastSeen = wallNow;
                entry.zone = zone;

                // Add sample if > SAMPLE_INTERVAL since last sample
                const lastSample = entry.samples[entry.samples.length - 1];
                if (wallNow - lastSample >= SAMPLE_INTERVAL) {
                    entry.samples.push(wallNow);
                }

                // Clean old samples outside tracking window
                entry.samples = entry.samples.filter(t => wallNow - t < TRACKING_WINDOW);
            }
        }

        // Calculate persistence and build validated zones list
        const validatedZones = [];
        for (const [key, entry] of this._wallHistory) {
            const timeSinceLastSeen = wallNow - entry.lastSeen;
            const trackingDuration = Math.min(TRACKING_WINDOW, wallNow - entry.firstSeen);

            // Remove very old entries
            if (timeSinceLastSeen > FADEOUT_TIME && !entry.qualified) {
                this._wallHistory.delete(key);
                continue;
            }

            // Calculate persistence score
            const expectedSamples = Math.max(1, Math.floor(trackingDuration / SAMPLE_INTERVAL));
            const actualSamples = entry.samples.length;
            let persistence = Math.min(1, actualSamples / expectedSamples);

            // Size-Weighted Persistence Bonus
            // Boost persistence for very large walls (they are reliable even if flickering)
            const sizeBonus = Math.min(0.3, entry.zone.totalSize / 200); // Up to +30% for 60+ BTC walls
            persistence = Math.min(1, persistence + sizeBonus);

            // Check if meets persistence threshold
            const isActive = currentZoneKeys.has(key);

            if (persistence >= MIN_PERSISTENCE) {
                if (!entry.qualified) {
                    entry.qualified = true;
                    entry.qualifiedTime = wallNow;
                }
            }

            // Only show qualified walls
            if (!entry.qualified) continue;

            // SNAP-TO-GRID & DEAD BAND: Stabilize AVG Price for display
            // We store a 'displayPrice' in the history entry.
            // We ONLY update it if the new average moves > $50 away.
            // This prevents the marker from "chasing" minor fluctuations.

            if (!entry.displayPrice) {
                // Initialize snapped to 10
                entry.displayPrice = Math.round(entry.zone.avgPrice / 10) * 10;
            } else {
                // Only update if drift is significant (> $50)
                const drift = Math.abs(entry.zone.avgPrice - entry.displayPrice);
                if (drift > 50) {
                    // Snap to new position
                    entry.displayPrice = Math.round(entry.zone.avgPrice / 10) * 10;
                }
            }

            const displayZone = { ...entry.zone, avgPrice: entry.displayPrice };

            // If qualified but now inactive/low persistence, start fade-out
            let opacity = 1.0;
            if (!isActive || persistence < MIN_PERSISTENCE) {
                const timeSinceFail = timeSinceLastSeen;

                // INTELLIGENT SMOOTHING: Dynamic Fadeout Time based on Wall Size
                // Large walls (e.g., 50+ BTC) should have "stickier" memory to prevent flickering.
                // Base: 15s. Bonus: 1s per 2 BTC size. Max: 60s.
                // Examples: 
                // 10 BTC -> 15 + 5 = 20s
                // 50 BTC -> 15 + 25 = 40s
                // 100 BTC -> 60s (capped)
                const sizeBonusSec = (entry.zone.totalSize || 0) / 2;
                const dynamicFadeTime = Math.min(60000, FADEOUT_TIME + (sizeBonusSec * 1000));

                opacity = Math.max(0.3, 1 - (timeSinceFail / dynamicFadeTime));

                // Remove if fully faded
                if (opacity <= 0.3 && timeSinceFail > dynamicFadeTime) {
                    this._wallHistory.delete(key);
                    continue;
                }
            }

            validatedZones.push({
                ...displayZone, // Use stabilized zone for display
                rawPrice: entry.zone.avgPrice, // Keep raw price for logic if needed
                opacity: opacity,
                persistence: persistence,
                isActive: isActive,
                ageSeconds: Math.round((wallNow - entry.firstSeen) / 1000),
                absorbed: entry.absorbed || 0
            });
        }

        // Find nearest zone to current price
        let activeZone = null;
        let minDist = Infinity;
        for (const zone of zones) {
            const dist = Math.abs(zone.avgPrice - this.currentPrice);
            if (dist < minDist) {
                minDist = dist;
                activeZone = zone;
            }
        }

        // ===============================
        // 2. WALL ABSORPTION TRACKING
        // ===============================
        if (activeZone) {
            const zoneKey = `${activeZone.type}_${Math.round(activeZone.avgPrice / 10) * 10}`;
            let tracked = this._wallTracker.get(zoneKey);

            if (!tracked) {
                // New zone - start tracking
                tracked = {
                    initialSize: activeZone.totalSize,
                    currentSize: activeZone.totalSize,
                    firstSeen: wallNow,
                    lastUpdate: now,
                    history: [{ time: now, size: activeZone.totalSize }],
                    absorbed: 0,
                    absorptionRate: 0,
                    isSpoof: false,
                    spoofTime: null
                };
                this._wallTracker.set(zoneKey, tracked);
            } else {
                // Update existing tracking
                const prevSize = tracked.currentSize;
                tracked.currentSize = activeZone.totalSize;
                tracked.lastUpdate = now;

                // Track history (keep last 60 seconds)
                tracked.history.push({ time: now, size: activeZone.totalSize });
                tracked.history = tracked.history.filter(h => now - h.time < 60000);

                // Calculate absorption
                tracked.absorbed = Math.max(0, tracked.initialSize - tracked.currentSize);

                // Calculate absorption rate (BTC/min)
                const timeElapsed = (now - tracked.firstSeen) / 60000; // minutes
                tracked.absorptionRate = timeElapsed > 0 ? tracked.absorbed / timeElapsed : 0;

                // SPOOFING DETECTION: >50% drop in <2 seconds
                if (prevSize > 0 && tracked.currentSize < prevSize * 0.5) {
                    const timeDiff = now - (tracked.history[tracked.history.length - 2]?.time || now);
                    if (timeDiff < 2000) {
                        tracked.isSpoof = true;
                        tracked.spoofTime = now;
                        this._spoofAlerts.push({ time: now, price: activeZone.avgPrice, type: activeZone.type });
                    }
                }

                // Clear spoof flag after 10 seconds
                if (tracked.isSpoof && now - tracked.spoofTime > 10000) {
                    tracked.isSpoof = false;
                }
            }

            activeZone.tracked = tracked;
        }

        // Clean old spoof alerts
        this._spoofAlerts = this._spoofAlerts.filter(a => now - a.time < 30000);

        // ===============================
        // DRAW WALL MARKERS ON PRICE AXIS
        // ===============================
        const markerX = this.width - 55;
        const markerWidth = 50;
        const markerHeight = 16;

        // Use VALIDATED zones (confirmed + fade-out)
        for (const zone of validatedZones) {
            const zoneY = this._getY(zone.avgPrice);
            if (zoneY < 0 || zoneY > this.height - this._deltaH()) continue;

            const isAsk = zone.type === 'ask';
            const baseColor = isAsk ? '#ef4444' : '#10b981';
            const opacity = zone.opacity || 1.0;
            const persistence = zone.persistence || 0;
            const ageSeconds = zone.ageSeconds || 0;

            // Size-based marker height (bigger wall = taller marker)
            const sizeNormalized = Math.min(1, zone.totalSize / 50); // Cap at 50 BTC
            const dynamicHeight = 14 + sizeNormalized * 12; // 14-26px based on size

            // Price distance factor (closer to current price = more prominent)
            const distFromPrice = Math.abs(zone.avgPrice - this.currentPrice);
            const distFactor = Math.max(0.5, 1 - distFromPrice / 2000); // Fade if >$2000 away

            // Store zone bounds for click detection
            zone.markerY = zoneY;
            zone.markerX = markerX;
            zone.markerWidth = markerWidth;
            zone.markerHeight = dynamicHeight;
            this._wallZones.push(zone);

            ctx.globalAlpha = opacity * distFactor;

            // Draw main marker pill
            ctx.fillStyle = baseColor + '30';
            ctx.strokeStyle = baseColor;
            ctx.lineWidth = zone.isActive ? 2 : 1;
            ctx.beginPath();
            ctx.roundRect(markerX, zoneY - dynamicHeight / 2, markerWidth, dynamicHeight, 4);
            ctx.fill();
            ctx.stroke();

            // Draw Wall Health Bar (Physical representation of wall state)
            // Replaces simple persistence bar with a "Health Meter"
            const barWidth = markerWidth - 6;
            const barHeight = 4; // Slightly thicker
            const barY = zoneY + dynamicHeight / 2 - barHeight - 2;
            const barX = markerX + 3;

            // 1. Calculate Health & Pressure
            // Base Health = Persistence (how stable it is)
            // Damage = Absorption (trades eating the wall)

            // Allow health to exceed 100% visually if it grew larger than initial seen
            // But for the bar, we cap at 1.0 (full)
            let healthRatio = persistence;

            // Calculate "Damage" from absorption
            // If we absorbed 10 BTC of a 50 BTC wall, that's 20% damage
            const damageRatio = Math.min(1.0, zone.absorbed / (zone.totalSize || 1));
            const remainingHealth = Math.max(0, healthRatio - damageRatio);

            // 2. Draw Background (Empty / Void)
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(barX, barY, barWidth, barHeight);

            // 3. Draw Remaining Health (Green/Yellow)
            const healthColor = remainingHealth > 0.7 ? '#22c55e' : remainingHealth > 0.3 ? '#eab308' : '#ef4444';
            ctx.fillStyle = healthColor;
            ctx.fillRect(barX, barY, barWidth * remainingHealth, barHeight);

            // 4. Draw Damage (Red Flash) - Representing "eaten" wall
            if (damageRatio > 0.01) {
                const damageX = barX + (barWidth * remainingHealth);
                ctx.fillStyle = '#b91c1c'; // Deep Red
                ctx.fillRect(damageX, barY, barWidth * damageRatio, barHeight);
            }

            // Draw size text
            ctx.font = 'bold 9px monospace';
            ctx.fillStyle = baseColor;
            ctx.textAlign = 'center';
            const sizeText = zone.totalSize >= 100 ? `${zone.totalSize.toFixed(0)}` : zone.totalSize.toFixed(1);

            // If heavily damaged, show size in RED
            if (damageRatio > 0.5) ctx.fillStyle = '#ef4444';

            ctx.fillText(sizeText, markerX + markerWidth / 2, zoneY - 2);

            // Draw age in smaller text
            ctx.font = '7px sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.6)';
            const ageText = ageSeconds >= 60 ? `${Math.floor(ageSeconds / 60)}m` : `${ageSeconds}s`;

            // Show Absorption text separate from Age
            if (zone.absorbed > 0.1) {
                // Flash if high absorption (>50% of wall size)
                if (zone.absorbed > zone.totalSize * 0.5) {
                    const flash = (Math.sin(Date.now() / 150) + 1) / 2; // 0..1
                    ctx.fillStyle = `rgba(255, 100, 100, ${0.5 + flash * 0.5})`;
                } else {
                    ctx.fillStyle = '#ffaaaa';
                }
                const absText = zone.absorbed >= 1000 ? `${(zone.absorbed / 1000).toFixed(1)}K` : zone.absorbed.toFixed(0);

                // Show below marker
                ctx.textAlign = 'center';
                ctx.fillText(`-${absText}`, markerX + markerWidth / 2, zoneY + dynamicHeight / 2 + 8);
            } else {
                ctx.fillText(ageText, markerX + markerWidth / 2, zoneY + dynamicHeight / 2 + 8);
            }

            // Draw status icon
            const icon = zone.isActive ? '🛡️' : '👻';
            ctx.font = '9px sans-serif'; // Bigger icon
            ctx.fillText(icon, markerX - 6, zoneY + 3); // Position to left of pill

            ctx.globalAlpha = 1.0; // Reset
        }

        // Debug: Log zones found
        if (!this._lastZoneLog || Date.now() - this._lastZoneLog > 5000) {
            const activeCount = validatedZones.filter(z => z.isActive).length;
            const avgPersistence = validatedZones.length > 0
                ? (validatedZones.reduce((s, z) => s + z.persistence, 0) / validatedZones.length * 100).toFixed(0)
                : 0;
            console.log(`[Liquidity] Raw: ${zones.length}, Qualified: ${validatedZones.length} (${activeCount} active), Avg Persistence: ${avgPersistence}%`);
            this._lastZoneLog = Date.now();
        }

        // CLICK-BASED SELECTION: HUD only shows when user clicks on a marker
        // (No more auto-jumping between walls!)
        let activeZoneForHUD = null;

        if (this._selectedWallPrice) {
            // Find the zone matching selected price (in validated zones)
            activeZoneForHUD = validatedZones.find(z => Math.abs(z.avgPrice - this._selectedWallPrice) < 100);

            // If selected wall no longer exists (faded out), clear selection
            if (!activeZoneForHUD) {
                this._selectedWallPrice = null;
            }
        }

        if (!activeZoneForHUD) {
            // Show hint to click on markers
            if (validatedZones.length > 0) {
                ctx.font = 'bold 10px sans-serif';
                ctx.fillStyle = 'rgba(255,255,255,0.5)';
                ctx.textAlign = 'right';
                ctx.fillText(`🛡️ ${validatedZones.length} walls - click marker for details`, this.width - 60, 20);
            }
            return;
        }

        // Use selected zone
        activeZone = activeZoneForHUD;

        // ===============================
        // 3. CALCULATE DELTA & MOMENTUM
        // ===============================
        let netDelta = 0;
        let attackVol = 0;
        let momentumVol = 0;

        if (activeZone) {
            const tick = this.tickSize;
            const lookback = 10; // candles

            for (let i = this.candles.length - 1; i >= Math.max(0, this.candles.length - lookback); i--) {
                const c = this.candles[i];
                if (!c || !c.clusters) continue;

                const age = this.candles.length - 1 - i;
                const timeWeight = Math.exp(-age / 5); // Exponential decay

                // Delta calculation
                const candleDelta = (c.buyVolume || 0) - (c.sellVolume || 0);
                netDelta += candleDelta * timeWeight;

                // Attack volume at wall price
                let clusters = c.clusters instanceof Map
                    ? Array.from(c.clusters.entries())
                    : Object.entries(c.clusters).map(([p, d]) => [parseFloat(p), d]);

                for (const [p, vol] of clusters) {
                    if (p >= activeZone.priceMin - tick && p <= activeZone.priceMax + tick) {
                        const volAtPrice = activeZone.type === 'ask'
                            ? (vol.bid || 0)
                            : (vol.ask || 0);
                        attackVol += volAtPrice;
                        momentumVol += volAtPrice * timeWeight;
                    }
                }
            }
        }

        // ===============================
        // 4. CALCULATE BREAK PROBABILITY
        // ===============================
        let probability = 0;
        let confidenceLevel = 'LOW';

        if (activeZone) {
            const tracked = activeZone.tracked;

            // Base probability: Attack / Wall
            const baseProb = Math.min(1, attackVol / activeZone.totalSize);

            // Absorption factor (wall getting weaker)
            const absorptionFactor = tracked ? Math.min(0.3, (tracked.absorbed / tracked.initialSize) * 0.5) : 0;

            // Delta factor (momentum in right direction)
            let deltaFactor = 0;
            if (activeZone.type === 'ask' && netDelta > 0) deltaFactor = Math.min(0.2, netDelta / 100 * 0.01);
            if (activeZone.type === 'bid' && netDelta < 0) deltaFactor = Math.min(0.2, Math.abs(netDelta) / 100 * 0.01);

            // Spoof penalty
            const spoofPenalty = tracked?.isSpoof ? -0.3 : 0;

            // Momentum factor
            const momentumFactor = attackVol > 0 ? Math.min(0.15, (momentumVol / attackVol - 0.5) * 0.3) : 0;

            probability = Math.max(0, Math.min(100, (baseProb + absorptionFactor + deltaFactor + momentumFactor + spoofPenalty) * 100));

            // Confidence level
            if (probability > 70 && tracked && tracked.absorptionRate > 5) confidenceLevel = 'HIGH';
            else if (probability > 40) confidenceLevel = 'MED';
            else confidenceLevel = 'LOW';
        }

        // ===============================
        // 5. STICKY DISPLAY (prevent flickering)
        // ===============================
        const STICKY_DURATION = 3000;

        if (activeZone) {
            if (!this._stickyZone ||
                !this._stickyZoneTime ||
                (now - this._stickyZoneTime > STICKY_DURATION) ||
                (Math.abs(activeZone.avgPrice - this._stickyZone.avgPrice) < 50)) {
                this._stickyZone = activeZone;
                this._stickyZoneTime = now;
                this._stickyProbability = probability;
                this._stickyDelta = netDelta;
                this._stickyConfidence = confidenceLevel;
            }
        } else if (this._stickyZone && (now - this._stickyZoneTime < STICKY_DURATION)) {
            activeZone = this._stickyZone;
            probability = this._stickyProbability;
            netDelta = this._stickyDelta;
            confidenceLevel = this._stickyConfidence;
        }

        // ===============================
        // 6. DRAW MODERN HUD
        // ===============================
        if (!activeZone) {
            ctx.font = 'bold 11px sans-serif';
            ctx.fillStyle = 'rgba(255,255,255,0.4)';
            ctx.textAlign = 'right';
            ctx.fillText('🔍 SCANNING LIQUIDITY...', this.width - 10, 20);
            return;
        }

        const tracked = activeZone.tracked;

        // FIXED POSITION: Top-right corner (prevents flickering)
        const cardWidth = 240;
        const cardHeight = 130;
        const x = this.width - 310; // Left of price axis
        const y = 30; // Fixed at top

        // Wall Y for connection line
        const wallY = this._getY(activeZone.avgPrice);

        // Card Background (Glassmorphism)
        const isAsk = activeZone.type === 'ask';
        const baseColor = isAsk ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)';
        const borderColor = isAsk ? '#ef4444' : '#10b981';

        ctx.fillStyle = baseColor;
        ctx.strokeStyle = tracked?.isSpoof ? '#f59e0b' : borderColor;
        ctx.lineWidth = tracked?.isSpoof ? 3 : 2;
        ctx.beginPath();
        ctx.roundRect(x, y - 20, cardWidth, cardHeight, 12);
        ctx.fill();
        ctx.stroke();

        // Inner gradient overlay
        const gradient = ctx.createLinearGradient(x, y, x, y + cardHeight);
        gradient.addColorStop(0, 'rgba(0,0,0,0.6)');
        gradient.addColorStop(1, 'rgba(0,0,0,0.8)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x + 2, y - 18, cardWidth - 4, cardHeight - 4, 10);
        ctx.fill();

        // Title Row
        ctx.font = 'bold 13px "Inter", sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'left';
        const title = activeZone.wallCount > 1
            ? `🛡️ ${isAsk ? 'RESISTANCE' : 'SUPPORT'} ZONE`
            : `🛡️ ${isAsk ? 'RESISTANCE' : 'SUPPORT'}`;
        ctx.fillText(title, x + 12, y);

        // Spoof Warning
        if (tracked?.isSpoof) {
            ctx.fillStyle = '#f59e0b';
            ctx.textAlign = 'right';
            ctx.fillText('⚠️ SPOOF RISK', x + cardWidth - 12, y);
        }

        // Price Range
        ctx.font = '10px monospace';
        ctx.fillStyle = '#888';
        ctx.textAlign = 'left';
        const priceText = activeZone.wallCount > 1
            ? `$${activeZone.priceMin.toFixed(0)} - $${activeZone.priceMax.toFixed(0)}`
            : `$${activeZone.avgPrice.toFixed(1)}`;
        ctx.fillText(priceText, x + 12, y + 14);

        // Wall Stats Row
        ctx.font = '11px "Inter", sans-serif';
        ctx.fillStyle = '#aaa';
        const wallText = `WALL: ${activeZone.totalSize.toFixed(0)} BTC`;
        const absorbedText = tracked ? `ABSORBED: ${tracked.absorbed.toFixed(0)}` : '';
        ctx.fillText(wallText, x + 12, y + 32);
        ctx.textAlign = 'right';
        ctx.fillText(absorbedText, x + cardWidth - 12, y + 32);

        // Absorption Progress Bar
        if (tracked && tracked.initialSize > 0) {
            const barX = x + 12;
            const barY = y + 40;
            const barWidth = cardWidth - 24;
            const barHeight = 8;
            const absorptionRatio = Math.min(1, tracked.absorbed / tracked.initialSize);

            // Background
            ctx.fillStyle = 'rgba(255,255,255,0.1)';
            ctx.beginPath();
            ctx.roundRect(barX, barY, barWidth, barHeight, 4);
            ctx.fill();

            // Progress
            const progressGradient = ctx.createLinearGradient(barX, 0, barX + barWidth * absorptionRatio, 0);
            progressGradient.addColorStop(0, '#10b981');
            progressGradient.addColorStop(0.5, '#fbbf24');
            progressGradient.addColorStop(1, '#ef4444');
            ctx.fillStyle = progressGradient;
            ctx.beginPath();
            ctx.roundRect(barX, barY, barWidth * absorptionRatio, barHeight, 4);
            ctx.fill();

            // Percentage
            ctx.font = 'bold 10px monospace';
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'right';
            ctx.fillText(`${(absorptionRatio * 100).toFixed(0)}%`, barX + barWidth, barY + barHeight + 12);
        }

        // Stats Row 2
        const statsY = y + 68;
        ctx.font = '10px "Inter", sans-serif';
        ctx.textAlign = 'left';

        // Absorption Rate
        ctx.fillStyle = '#6ee7b7';
        ctx.fillText(`⚡ ${tracked?.absorptionRate.toFixed(1) || '0.0'} BTC/min`, x + 12, statsY);

        // Delta
        const deltaColor = netDelta > 0 ? '#10b981' : netDelta < 0 ? '#ef4444' : '#888';
        const deltaArrow = netDelta > 0 ? '↑' : netDelta < 0 ? '↓' : '→';
        ctx.fillStyle = deltaColor;
        ctx.fillText(`📊 ${deltaArrow}${Math.abs(netDelta).toFixed(0)} Δ`, x + 100, statsY);

        // Time at wall
        const timeAtWall = tracked ? Math.round((now - tracked.firstSeen) / 1000) : 0;
        const mins = Math.floor(timeAtWall / 60);
        const secs = timeAtWall % 60;
        ctx.fillStyle = '#888';
        ctx.textAlign = 'right';
        ctx.fillText(`⏱️ ${mins}:${secs.toString().padStart(2, '0')}`, x + cardWidth - 12, statsY);

        // Break Probability Section
        const probY = y + 90;

        // Probability background bar
        ctx.fillStyle = 'rgba(255,255,255,0.05)';
        ctx.beginPath();
        ctx.roundRect(x + 12, probY, cardWidth - 24, 24, 6);
        ctx.fill();

        // Probability fill
        let probColor;
        if (probability < 30) probColor = '#ef4444';
        else if (probability < 60) probColor = '#fbbf24';
        else probColor = '#10b981';

        ctx.fillStyle = probColor + '40';
        ctx.beginPath();
        ctx.roundRect(x + 12, probY, (cardWidth - 24) * (probability / 100), 24, 6);
        ctx.fill();

        // Probability text
        ctx.font = 'bold 14px "Inter", sans-serif';
        ctx.fillStyle = probColor;
        ctx.textAlign = 'center';
        ctx.fillText(`${probability.toFixed(0)}% BREAK`, x + cardWidth / 2 - 20, probY + 16);

        // Confidence badge
        ctx.font = 'bold 9px monospace';
        ctx.fillStyle = confidenceLevel === 'HIGH' ? '#10b981' : confidenceLevel === 'MED' ? '#fbbf24' : '#888';
        ctx.textAlign = 'right';
        ctx.fillText(confidenceLevel, x + cardWidth - 16, probY + 16);

        // Draw zone markers on chart
        if (activeZone.wallCount > 1) {
            const zoneTop = this._getY(activeZone.priceMax);
            const zoneBottom = this._getY(activeZone.priceMin);

            ctx.fillStyle = isAsk ? 'rgba(239, 68, 68, 0.05)' : 'rgba(16, 185, 129, 0.05)';
            ctx.fillRect(0, zoneTop, x - 10, zoneBottom - zoneTop);

            ctx.strokeStyle = borderColor + '40';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(0, zoneTop);
            ctx.lineTo(x - 10, zoneTop);
            ctx.moveTo(0, zoneBottom);
            ctx.lineTo(x - 10, zoneBottom);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw connection line from HUD to wall
        ctx.strokeStyle = borderColor + '60';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + cardWidth, y + cardHeight / 2);
        ctx.lineTo(this.width - 55, wallY); // To price axis marker area
        ctx.stroke();
        ctx.setLineDash([]);
    }

    _drawCrosshair(ctx) {
        if (this.showLens) return;

        const { crosshairX, crosshairY, width, height } = this;
        if (!crosshairX || !crosshairY) return;

        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(crosshairX, 0);
        ctx.lineTo(crosshairX, height - this._deltaH());
        ctx.moveTo(0, crosshairY);
        ctx.lineTo(width, crosshairY);
        ctx.stroke();
        ctx.setLineDash([]);

        const price = this._getPriceFromY(crosshairY);
        ctx.fillStyle = '#222';
        ctx.fillRect(width - 55, crosshairY - 10, 50, 20);
        ctx.fillStyle = '#ddd';
        ctx.textAlign = 'right';
        ctx.fillText(price.toFixed(2), width - 8, crosshairY + 4);
    }

    _initEvents() {
        this.canvas.addEventListener('wheel', this._onWheel.bind(this), { passive: false });
        this.canvas.addEventListener('mousedown', this._onMouseDown.bind(this));
        window.addEventListener('mousemove', this._onMouseMove.bind(this));
        window.addEventListener('mouseup', this._onMouseUp.bind(this));
        window.addEventListener('keydown', this._onKeyDown.bind(this));
        this.canvas.addEventListener('dblclick', this._onDoubleClick.bind(this));
    }

    _onDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        if (x > this.width - 50) {
            this.zoomY = 2;
            if (this.currentPrice) {
                this.offsetY = (this.height / 2) - (this.currentPrice / this.tickSize * this.zoomY);
            }
            this.requestDraw();
        }
    }

    _onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const k = e.key.toUpperCase();
        let handled = true;

        switch (k) {
            case 'H': this.toggleHeatmap(); break;
            case 'B': this.toggleBigTrades(); break;
            case 'C': this.toggleCrosshair(); break;
            case 'D': this.toggleDelta(); break;
            case 'I': this.toggleImbalances(); break;
            case 'L': this.showLens = !this.showLens; this.requestDraw(); break;
            case 'M': window.dispatchEvent(new CustomEvent('toggle-ml-dashboard')); break;
            case '+': case '=': this.zoomX = Math.min(this.maxZoomX, this.zoomX * 1.1); break;
            case '-': case '_': this.zoomX = Math.max(this.minZoomX, this.zoomX / 1.1); break;
            case 'ARROWLEFT': this.offsetX += 50; break;
            case 'ARROWRIGHT': this.offsetX -= 50; break;
            case 'ARROWUP': this.offsetY += 50; break;
            case 'ARROWDOWN': this.offsetY -= 50; break;
            case 'ESCAPE': this._selectedWallPrice = null; this.requestDraw(); break;
            default: handled = false;
        }
        if (handled) { e.preventDefault(); this.requestDraw(); }
    }

    _onWheel(e) {
        e.preventDefault();
        const factor = 1.1;
        const dir = e.deltaY > 0 ? 1 / factor : factor;

        // Smart Zoom:
        // Ctrl+Wheel = Price Zoom (Y)
        // Normal Wheel = Time Zoom (X)

        if (e.ctrlKey) {
            this.zoomY = Math.max(this.minZoomY, Math.min(this.maxZoomY, this.zoomY * dir));
            // Keep center centered?
            if (this.currentPrice) {
                this.offsetY = (this.height / 2) - (this.currentPrice / this.tickSize * this.zoomY);
            }
        } else {
            this.zoomX = Math.max(this.minZoomX, Math.min(this.maxZoomX, this.zoomX * dir));
        }

        this.requestDraw();
    }

    _onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const { width, height } = this;

        // DRAWING MODE: Handle drawing tool clicks
        if (this.drawingManager && this.drawingManager.getActiveTool() !== 'select') {
            const tool = this.drawingManager.getActiveTool();
            const price = this._getPriceFromY(y);
            const candle = this._getCandleAtX(x);
            const time = candle ? candle.time : Date.now();

            if (tool === 'horizontal') {
                // Single click creates horizontal line
                this.drawingManager.addDrawing({
                    type: 'horizontal',
                    points: [{ price, time }],
                    color: '#ffd700',
                    symbol: this.currentSymbol
                });
                this.drawingManager.setActiveTool('select');
                return;
            } else if (['trendline', 'rectangle', 'fibonacci'].includes(tool)) {
                // Two-click drawings
                if (!this.drawingManager.pendingDrawing) {
                    this.drawingManager.startPendingDrawing(tool, { price, time });
                } else {
                    this.drawingManager.updatePendingDrawing({ price, time });
                    this.drawingManager.finalizePendingDrawing(this.currentSymbol);
                    this.drawingManager.setActiveTool('select');
                }
                return;
            }
        }

        // Check if clicking on a wall marker
        if (this._wallZones && this._wallZones.length > 0) {
            for (const zone of this._wallZones) {
                if (zone.markerX && zone.markerY && zone.markerWidth && zone.markerHeight) {
                    if (x >= zone.markerX && x <= zone.markerX + zone.markerWidth &&
                        y >= zone.markerY - zone.markerHeight / 2 && y <= zone.markerY + zone.markerHeight / 2) {
                        // Toggle selection: click same = deselect, click other = select
                        if (this._selectedWallPrice && Math.abs(this._selectedWallPrice - zone.avgPrice) < 100) {
                            this._selectedWallPrice = null; // Deselect
                        } else {
                            this._selectedWallPrice = zone.avgPrice; // Select
                        }
                        this.requestDraw();
                        return; // Don't start drag
                    }
                }
            }
        }

        if (x > width - 50) {
            this.isDraggingPrice = true;
            this.dragStartY = y;
            this.initialZoomY = this.zoomY;
            this.initialOffsetY = this.offsetY;
        } else if (y > height - 50) { // Time Axis Area (50px)
            this.isDraggingTime = true;
            this.dragStartX = x;
            this.initialZoomX = this.zoomX;
            this.initialOffsetX = this.offsetX;
        } else {
            this.isDragging = true;
        }

        this.lastX = x;
        this.lastY = y;
    }

    _onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const { width, height } = this;

        this.crosshairX = x;
        this.crosshairY = y;

        const index = Math.floor((x - this.offsetX) / this.zoomX);
        if (index >= 0 && index < this.candles.length) {
            this.hoveredCandle = this.candles[index];
        } else {
            this.hoveredCandle = null;
        }

        if (this.isDraggingPrice || x > width - 50) {
            this.canvas.style.cursor = 'ns-resize';
        } else if (this.isDraggingTime || y > height - 50) {
            this.canvas.style.cursor = 'col-resize';
        } else {
            this.canvas.style.cursor = this.isDragging ? 'grabbing' : 'crosshair';
        }

        // Update pending drawing preview
        if (this.drawingManager && this.drawingManager.pendingDrawing) {
            const price = this._getPriceFromY(y);
            const candle = this._getCandleAtX(x);
            const time = candle ? candle.time : Date.now();
            this.drawingManager.updatePendingDrawing({ price, time });
        }

        if (this.isDraggingPrice) {
            const dy = y - this.dragStartY;
            const sensitivity = 0.005;
            const factor = 1 - (dy * sensitivity);

            const newZoomY = Math.max(this.minZoomY, Math.min(this.maxZoomY, this.initialZoomY * factor));

            const valAtPivot = (this.height - this.dragStartY - this.initialOffsetY - this.deltaRowHeight) / this.initialZoomY;

            this.zoomY = newZoomY;
            this.offsetY = (this.height - this.dragStartY - this.deltaRowHeight) - (valAtPivot * newZoomY);

        } else if (this.isDraggingTime) {
            const dx = x - this.dragStartX;
            const sensitivity = 0.005;
            const factor = 1 + (dx * sensitivity);

            const newZoomX = Math.max(this.minZoomX, Math.min(this.maxZoomX, this.initialZoomX * factor));

            const valAtPivot = (this.dragStartX - this.initialOffsetX) / this.initialZoomX;

            this.zoomX = newZoomX;
            this.offsetX = this.dragStartX - (valAtPivot * newZoomX);

        } else if (this.isDragging) {
            this.offsetX += x - this.lastX;
            this.offsetY -= y - this.lastY;
        }

        // Check hover over wall markers
        if (this._wallZones && this._wallZones.length > 0) {
            let foundZone = null;
            for (const zone of this._wallZones) {
                if (zone.markerX && zone.markerY && zone.markerWidth && zone.markerHeight) {
                    if (x >= zone.markerX && x <= zone.markerX + zone.markerWidth &&
                        y >= zone.markerY - zone.markerHeight / 2 && y <= zone.markerY + zone.markerHeight / 2) {
                        foundZone = zone;
                        break;
                    }
                }
            }
            // Store by price to survive redraws (zones get recreated each frame)
            const newHoveredPrice = foundZone ? foundZone.avgPrice : null;
            if (newHoveredPrice !== this._hoveredWallPrice) {
                this._hoveredWallPrice = newHoveredPrice;
                this.requestDraw();
            }
        }

        // Check ML Signal Hover
        if (this.hoveredCandle) {
            const signal = this.mlHistory.get(this.hoveredCandle.time);
            if (signal !== this.hoveredSignal) {
                this.hoveredSignal = signal;
                this.requestDraw();
            }
        } else if (this.hoveredSignal) {
            this.hoveredSignal = null;
            this.requestDraw();
        }

        if (this.isDragging || this.isDraggingPrice || this.isDraggingTime || this.showLens) {
            this.lastX = x;
            this.lastY = y;
            this.requestDraw();
        } else {
            if (this.showCrosshair) this.requestDraw();
        }
    }

    _onMouseUp() {
        this.isDragging = false;
        this.isDraggingPrice = false;
        this.isDraggingTime = false;
        this.canvas.style.cursor = 'crosshair';
    }

    _initResizeObserver() {
        new ResizeObserver(entries => {
            const rect = entries[0].contentRect;
            const prevHeight = this.height;
            this.width = rect.width;
            this.height = rect.height;
            this.canvas.width = this.width;
            this.canvas.height = this.height;

            // Complete any pending centering when height becomes available
            const chartHeight = this.height - this._deltaH();
            if (chartHeight > 100) {
                this._completePendingCenter();
            }

            // Also try auto-center if first time getting valid size
            if (prevHeight <= 100 && this.height > 100 && this.currentPrice && !this.initialCenterDone) {
                const priceInPixels = this.currentPrice / this.tickSize * this.zoomY;
                this.offsetY = chartHeight / 2 - priceInPixels;
                this.initialCenterDone = true;
                console.log('ResizeObserver centered at', this.currentPrice, 'chartHeight', chartHeight);
            }

            this.requestDraw();
        }).observe(this.container);
    }

    _drawHistoricalSignals(ctx) {
        if (this.mlHistory.size === 0) return;

        const { width, height } = this;
        const startIdx = Math.floor(-this.offsetX / this.zoomX);
        const endIdx = startIdx + Math.ceil(width / this.zoomX) + 1;

        const actualStart = Math.max(0, startIdx);
        const actualEnd = Math.min(this.candles.length - 1, endIdx);

        ctx.save();
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';

        for (let i = actualStart; i <= actualEnd; i++) {
            const candle = this.candles[i];
            if (!candle) continue;

            const signal = this.mlHistory.get(candle.time);
            if (!signal) continue;

            const x = this._getX(i) + (this.zoomX / 2); // Center of candle

            // Draw Signal Marker (Long)
            // Arrow below Low
            const y = this._getY(candle.low) + 20;

            // Determine Style based on Source
            const isBot = signal.source === 'OrderflowBot';
            const color = isBot ? '#ffd700' : '#00e676'; // Gold for Bot, Green for ML

            ctx.fillStyle = color;
            ctx.strokeStyle = color;

            if (isBot) {
                // Draw SHIELD (🛡️) shape
                const shieldY = y + 5;
                ctx.beginPath();
                ctx.moveTo(x - 6, shieldY);
                ctx.lineTo(x + 6, shieldY);
                ctx.lineTo(x + 6, shieldY + 6);
                ctx.bezierCurveTo(x + 6, shieldY + 12, x, shieldY + 16, x, shieldY + 16);
                ctx.bezierCurveTo(x, shieldY + 16, x - 6, shieldY + 12, x - 6, shieldY + 6);
                ctx.closePath();
                ctx.fill();

                // Label
                ctx.font = 'bold 10px sans-serif';
                ctx.fillText('BOT', x, y + 25);
            } else {
                // Draw ARROW (ML)
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x - 5, y + 8);
                ctx.lineTo(x + 5, y + 8);
                ctx.fill();

                // Label
                ctx.font = 'bold 10px sans-serif';
                ctx.fillText('ML', x, y + 22);
            }

            // Draw Setup Type (e.g. "Absorption")
            if (signal.setupType) {
                const labelY = y + 35;
                ctx.fillStyle = '#ffffff';
                ctx.font = '9px monospace';
                // Truncate if too long
                let typeText = signal.setupType.replace('Absorption', 'Abs.');
                if (typeText.length > 15) typeText = typeText.substring(0, 15) + '..';
                ctx.fillText(typeText, x, labelY);
            }

            // Draw Hover Details
            if (this.hoveredSignal === signal) {
                this._drawSignalDetails(ctx, signal, i);
            }
        }

        ctx.restore();
    }

    _drawSignalDetails(ctx, signal, startIndex) {
        // Calculate Outcome
        let outcome = 'RUNNING';
        let exitIndex = this.candles.length - 1;

        for (let k = startIndex + 1; k < this.candles.length; k++) {
            const h = this.candles[k].high;
            const l = this.candles[k].low;

            if (l <= signal.sl) {
                outcome = 'LOSS';
                exitIndex = k;
                break;
            }
            if (h >= signal.tp) {
                outcome = 'WIN';
                exitIndex = k;
                break;
            }
        }

        // Colors
        const isWin = outcome === 'WIN';
        const isLoss = outcome === 'LOSS';
        const mainColor = isWin ? '#00e676' : (isLoss ? '#ef4444' : '#00bcd4');
        const bgColor = isWin ? 'rgba(0, 230, 118, 0.1)' : (isLoss ? 'rgba(239, 68, 68, 0.1)' : 'rgba(0, 188, 212, 0.1)');

        // Draw Background Box
        const startX = this._getX(startIndex) + this.zoomX / 2;
        const endX = this._getX(exitIndex) + this.zoomX / 2;
        const entryY = this._getY(signal.entry);
        const targetY = this._getY(isLoss ? signal.sl : signal.tp);

        ctx.fillStyle = bgColor;
        ctx.fillRect(startX, Math.min(entryY, targetY), endX - startX, Math.abs(entryY - targetY));

        // Draw Lines
        this.drawLevel(ctx, signal.entry, mainColor, 'ENTRY', startX, endX, false);
        this.drawLevel(ctx, signal.tp, isWin ? '#00e676' : '#555', `TP: ${signal.tp.toFixed(1)}`, startX, endX, true);
        this.drawLevel(ctx, signal.sl, isLoss ? '#ef4444' : '#555', `SL: ${signal.sl.toFixed(1)}`, startX, endX, true);

        // Result Label
        if (outcome !== 'RUNNING') {
            ctx.font = 'bold 12px sans-serif';
            ctx.fillStyle = mainColor;
            ctx.fillText(outcome, endX + 25, (entryY + targetY) / 2);
        }
    }

    drawLevel(ctx, price, color, label, x1, x2, dashed) {
        const y = this._getY(price);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        if (dashed) ctx.setLineDash([4, 4]);
        else ctx.setLineDash([]);

        ctx.beginPath();
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.stroke();

        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = color;
        ctx.font = '10px sans-serif';
        ctx.fillText(label, x1 + 5, y - 4);
    }

    _drawTradePlan(ctx) {
        if (!this.mlPrediction || !this.mlPrediction.signal) return;

        const { entry, sl, tp, confidence, rr } = this.mlPrediction;
        const width = this.width;

        ctx.save();
        ctx.font = 'bold 12px "Inter", sans-serif';

        // Helper to draw line
        const drawLevel = (price, color, label, isDashed = false) => {
            const y = this._getY(price);
            if (y < 0 || y > this.height) return;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash(isDashed ? [6, 4] : []);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();

            // Label
            ctx.fillStyle = color;
            ctx.fillText(label, width - 120, y - 5);
        };

        // 1. ENTRY (Dashed Green)
        drawLevel(entry, '#00e676', `ENTRY: ${entry.toFixed(2)}`, true);

        // 2. STOP LOSS (Red)
        const lossPct = ((entry - sl) / entry * 100).toFixed(2);
        drawLevel(sl, '#ff1744', `SL: ${sl.toFixed(2)} (-${lossPct}%)`);

        // 3. TARGET (Green)
        const gainPct = ((tp - entry) / entry * 100).toFixed(2);
        drawLevel(tp, '#00bcd4', `TP: ${tp.toFixed(2)} (+${gainPct}%) R:${rr.toFixed(1)}`);

        // Setup Box
        const startY = this._getY(entry);
        if (startY > 50) {
            ctx.fillStyle = 'rgba(15, 20, 25, 0.9)';
            ctx.strokeStyle = '#00e676';
            ctx.lineWidth = 1;
            ctx.setLineDash([]);
            // Box near right edge
            const boxX = width - 180;
            const boxY = startY - 80;

            ctx.beginPath();
            ctx.roundRect(boxX, boxY, 170, 70, 8);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#fff';
            ctx.textAlign = 'left';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText('🤖 AI TRADE SETUP', boxX + 10, boxY + 20);

            ctx.font = '11px monospace';
            ctx.fillStyle = '#aaa';
            ctx.fillText(`Type: Support Bounce`, boxX + 10, boxY + 38);

            const confColor = confidence > 0.8 ? '#00e676' : '#ffd700';
            ctx.fillStyle = confColor;
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`Confidence: ${(confidence * 100).toFixed(0)}%`, boxX + 10, boxY + 58);
        }

        ctx.restore();
    }

    _drawAISignal(ctx) {
        if (!this.aiSignal) return;
        const { entry, sl, tp, type } = this.aiSignal;
        const width = this.width;

        ctx.save();
        ctx.font = 'bold 12px sans-serif';
        ctx.setLineDash([5, 3]);

        // ENTRY
        const yEntry = this._getY(entry);
        ctx.strokeStyle = '#ba68c8'; // AI Purple Light
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, yEntry);
        ctx.lineTo(width, yEntry);
        ctx.stroke();

        ctx.fillStyle = '#ba68c8';
        ctx.textAlign = 'right';
        ctx.fillText(`✨ AI+ENTRY ${type || 'TRADE'} @ ${entry}`, width - 80, yEntry - 8);

        ctx.lineWidth = 1;

        // SL
        if (sl) {
            const ySL = this._getY(sl);
            ctx.strokeStyle = '#ef5350';
            ctx.fillStyle = '#ef5350';
            ctx.beginPath();
            ctx.moveTo(0, ySL);
            ctx.lineTo(width, ySL);
            ctx.stroke();
            ctx.fillText(`SL @ ${sl}`, width - 80, ySL - 4);
        }

        // TP
        if (tp) {
            const yTP = this._getY(tp);
            ctx.strokeStyle = '#66bb6a';
            ctx.fillStyle = '#66bb6a';
            ctx.beginPath();
            ctx.moveTo(0, yTP);
            ctx.lineTo(width, yTP);
            ctx.stroke();
            ctx.fillText(`TP @ ${tp}`, width - 80, yTP - 4);
        }

        ctx.restore();
    }

    _drawDrawings(ctx) {
        if (!this.drawingManager) return;

        const { width, height } = this;
        const drawings = this.drawingManager.getDrawings(this.currentSymbol);
        const pending = this.drawingManager.pendingDrawing;
        const selected = this.drawingManager.selectedDrawing;

        ctx.save();
        ctx.lineWidth = 1.5;

        // Draw saved drawings
        for (const d of drawings) {
            this._renderDrawing(ctx, d, d.id === selected?.id);
        }

        // Draw pending (in-progress) drawing
        if (pending) {
            this._renderDrawing(ctx, pending, false, true);
        }

        ctx.restore();
    }

    _renderDrawing(ctx, drawing, isSelected, isPending = false) {
        const color = isPending ? '#ffffff' : (drawing.color || '#ffd700');
        ctx.strokeStyle = color;
        ctx.fillStyle = color;

        if (isSelected) {
            ctx.lineWidth = 2.5;
            ctx.setLineDash([]);
        } else if (isPending) {
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 5]);
        } else {
            ctx.lineWidth = 1.5;
            ctx.setLineDash([]);
        }

        switch (drawing.type) {
            case 'horizontal':
                this._drawHorizontalLine(ctx, drawing, isSelected);
                break;
            case 'trendline':
                this._drawTrendline(ctx, drawing, isSelected);
                break;
            case 'rectangle':
                this._drawRectangle(ctx, drawing, isSelected);
                break;
            case 'fibonacci':
                this._drawFibonacci(ctx, drawing, isSelected);
                break;
        }

        ctx.setLineDash([]);
    }

    _drawHorizontalLine(ctx, drawing, isSelected) {
        const { width } = this;
        const price = drawing.points[0]?.price;
        if (!price) return;

        const y = this._getY(price);

        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        // Price label
        ctx.font = 'bold 11px monospace';
        ctx.fillText(price.toFixed(2), width - 80, y - 4);

        // Selection handles
        if (isSelected) {
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath();
            ctx.arc(50, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    _drawTrendline(ctx, drawing, isSelected) {
        if (drawing.points.length < 2) return;

        const p1 = drawing.points[0];
        const p2 = drawing.points[1];

        // Find candle index for time
        const x1 = this._getXForTime(p1.time);
        const y1 = this._getY(p1.price);
        const x2 = this._getXForTime(p2.time);
        const y2 = this._getY(p2.price);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

        if (isSelected) {
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath();
            ctx.arc(x1, y1, 5, 0, Math.PI * 2);
            ctx.arc(x2, y2, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    _drawRectangle(ctx, drawing, isSelected) {
        if (drawing.points.length < 2) return;

        const p1 = drawing.points[0];
        const p2 = drawing.points[1];

        const x1 = this._getXForTime(p1.time);
        const y1 = this._getY(p1.price);
        const x2 = this._getXForTime(p2.time);
        const y2 = this._getY(p2.price);

        const minX = Math.min(x1, x2);
        const minY = Math.min(y1, y2);
        const w = Math.abs(x2 - x1);
        const h = Math.abs(y2 - y1);

        // Fill
        ctx.globalAlpha = 0.15;
        ctx.fillRect(minX, minY, w, h);
        ctx.globalAlpha = 1.0;

        // Stroke
        ctx.strokeRect(minX, minY, w, h);

        if (isSelected) {
            ctx.fillStyle = ctx.strokeStyle;
            ctx.fillRect(x1 - 4, y1 - 4, 8, 8);
            ctx.fillRect(x2 - 4, y2 - 4, 8, 8);
        }
    }

    _drawFibonacci(ctx, drawing, isSelected) {
        if (drawing.points.length < 2) return;

        const p1 = drawing.points[0];
        const p2 = drawing.points[1];
        const { width } = this;

        const y1 = this._getY(p1.price);
        const y2 = this._getY(p2.price);
        const priceRange = p2.price - p1.price;

        const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
        ctx.font = '10px monospace';

        for (const level of levels) {
            const price = p1.price + (priceRange * level);
            const y = this._getY(price);

            ctx.globalAlpha = level === 0 || level === 1 ? 1 : 0.6;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();

            ctx.fillText(`${(level * 100).toFixed(1)}% - ${price.toFixed(2)}`, 10, y - 4);
        }
        ctx.globalAlpha = 1;
    }

    _getXForTime(time) {
        // Find candle with this time
        const idx = this.candles.findIndex(c => c.time === time);
        if (idx >= 0) {
            return this._getX(idx) + this.zoomX / 2;
        }
        // Fallback: estimate based on first candle
        if (this.candles.length > 0) {
            const firstTime = this.candles[0].time;
            const candleWidth = 60000; // 1 min candles
            const estimatedIdx = (time - firstTime) / candleWidth;
            return this._getX(estimatedIdx) + this.zoomX / 2;
        }
        return 0;
    }
}

