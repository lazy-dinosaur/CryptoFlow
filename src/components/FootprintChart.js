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
        this.maxProfileVol = 0;

        // Heatmap controls
        this.heatmapHistoryPercent = 60;

        // Big trade auto-filter
        this.autoFilter = false;
        this._recentTradeSizes = [];
        this._recentTradeMax = 5000;

        // Thresholds
        // Heatmap defaults (Bookmap style)
        this.heatmapIntensityThreshold = 0.005; // Show more detail (was 0.02)
        this.heatmapOpacity = 0.8; // More solid/vibrant (was 0.5)
        this.maxVolumeInHistory = 10;
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
        if (!this.initialCenterDone && price > 0 && this.height > 0) {
            this.initialCenterDone = true;
            this.offsetY = (this.height / 2) - (price / this.tickSize * this.zoomY);
            console.log('Auto-centered at', price);
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
    setHeatmapIntensityThreshold(val) { this.heatmapIntensityThreshold = val; this.requestDraw(); }
    setHeatmapHistoryPercent(val) {
        this.heatmapHistoryPercent = Math.max(0, Math.min(100, val));
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
        this.sessionProfile = Array.from(map.entries()).map(([price, vol]) => ({ price, vol }));
        this.maxProfileVol = maxVol;
        this.pocPrice = poc;
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
        let h = 0;
        if (this.showDelta) h += this.deltaRowHeight;
        if (this.showCVD) h += this.cvdPanelHeight;
        return h;
    }
    _getX(index) { return Math.floor((index * this.zoomX) + this.offsetX) + 0.5; }
    _getY(price) { return Math.floor(this.height - ((price / this.tickSize * this.zoomY) + this.offsetY) - this._deltaH()) + 0.5; }
    _getPriceFromY(y) { return (this.height - y - this.offsetY - this._deltaH()) / this.zoomY * this.tickSize; }

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

            // Draw Bottom Panels
            if (this.showDelta) this._drawDeltaSummary(ctx);
            if (this.showCVD) this._drawCVD(ctx);

            this._drawPriceLine(ctx);
            if (this.showBigTrades) this._drawBigTrades(ctx);
            if (this.showCrosshair && this.crosshairX) this._drawCrosshair(ctx);
            if (this.showLens) this._drawLens(ctx);

            // Wall Attack (Always On or toggle? User asked for it, lets keep it always on or tied to ML)
            // Let's tie it to 'showML' flag for now, or just always show if heavy wall nearby.
            if (this.showML) this._drawWallAttack(ctx);

        } catch (e) {
            console.error('Chart Render Error:', e);
            // Fallback: Draw Error Text
            ctx.fillStyle = 'red';
            ctx.font = '14px sans-serif';
            ctx.fillText('Render Error: ' + e.message, 10, 20);
        }
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

        for (let i = startIdx; i <= endIdx; i++) {
            // Find candle
            if (i < 0 || i >= this.candles.length) continue;
            const candle = this.candles[i];
            if (!candle) continue;

            const cTime = candle.time;

            // Find best matching snapshot: The one closest to candle.time (or simply the last one before next candle)
            // Since both arrays are time-sorted, we can advance hPtr.

            let snapshot = null;

            // Limit search window to reasonable proximity (e.g. within 2x timeframe)
            // But simplify: just find the snapshot with time >= cTime and < next_candle_time
            // Or just closest.

            // 1. Advance hPtr until we reach/pass cTime
            while (hPtr < hLen - 1 && this.heatmapData[hPtr].time < cTime) {
                hPtr++;
            }

            // Now heatmapData[hPtr] is likely >= cTime. 
            // We want the snapshot representing this candle's duration. 
            // Ideally, we'd average them? For performance, let's take the one at 'Close' (latest in duration)
            // Actually, if we just take the one at 'Open' (hPtr), it's stable.
            // Let's check the next one too.

            // Stabilize: Use the snapshot closest to the candle time.
            snapshot = this.heatmapData[hPtr];

            // Fallback for live edge: If no snapshot found (future?), use last available?
            if (!snapshot && hLen > 0) snapshot = this.heatmapData[hLen - 1];

            if (!snapshot) continue;

            // DRAW COLUMN
            const x = this._getX(i);
            const w = this.zoomX;

            let levels = [];
            if (snapshot.bids) for (const b of snapshot.bids) levels.push({ p: b.p, q: b.q });
            if (snapshot.asks) for (const a of snapshot.asks) levels.push({ p: a.p, q: a.q });

            for (const lev of levels) {
                if (!lev.p || !lev.q) continue;
                const y = this._getY(lev.p);
                const h = Math.max(1, (this.zoomY));

                if (y + h < 0 || y > height - deltaH) continue;

                // 1. Noise Filter: Ignore < 2% of max volume (cleans up "Blue Rain")
                // 1. Threshold Filter (Controlled by Slider)
                const ratio = lev.q / this.maxVolumeInHistory;
                if (ratio < this.heatmapIntensityThreshold) continue; // Slider Control Only

                // BOOKMAP STYLE: Gamma Corrected Opacity
                const visRatio = Math.pow(ratio, 0.4);

                let color;
                const alpha = this.heatmapOpacity; // Default 1.0 or user setting?

                if (visRatio < 0.2) {
                    // Very Low (Dark Blue) - Make it more transparent
                    color = `rgba(0, 50, 200, ${alpha * 0.4})`;
                } else if (visRatio < 0.4) {
                    color = `rgba(0, 150, 255, ${alpha * 0.6})`;
                } else if (visRatio < 0.6) {
                    color = `rgba(0, 255, 100, ${alpha * 0.8})`;
                } else if (visRatio < 0.8) {
                    color = `rgba(255, 255, 0, ${alpha})`;
                } else {
                    color = `rgba(255, 0, 0, ${alpha})`; // MAX Heat
                }

                bufCtx.fillStyle = color;
                // +1 width to prevent vertical gaps between candles
                bufCtx.fillRect(x, y - h / 2, w + 1, h);
            }
        }
        ctx.drawImage(this.heatmapBuffer, 0, 0);
    }

    _drawGrid(ctx) {
        const { width, height, tickSize, offsetX, zoomX } = this;
        const chartHeight = height - this._deltaH();

        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();

        const minPrice = this._getPriceFromY(chartHeight);
        const maxPrice = this._getPriceFromY(0);
        const step = tickSize * 10;
        const startPrice = Math.floor(minPrice / step) * step;

        for (let p = startPrice; p <= maxPrice; p += step) {
            const y = this._getY(p);
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
        }

        // Calculate step to avoid overlap. Time (30px) + Padding.
        const labelWidth = 50;
        const pixelStep = labelWidth;
        const candleStep = Math.max(1, Math.ceil(pixelStep / zoomX));

        const startIdx = Math.floor(-offsetX / zoomX);
        const endIdx = startIdx + (width / zoomX);

        for (let i = Math.floor(startIdx / candleStep) * candleStep; i <= endIdx; i += candleStep) {
            if (i < 0 || i >= this.candles.length) continue;
            const x = this._getX(i);
            if (x < -50 || x > width + 50) continue;

            const candle = this.candles[i];
            if (candle && candle.time) {
                const d = new Date(candle.time);
                const tStr = `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
                const dStr = `${d.getDate().toString().padStart(2, '0')}.${(d.getMonth() + 1).toString().padStart(2, '0')}`; // DD.MM

                // Background Box REMOVED for cleaner look
                // ctx.fillStyle = this.colors.bg;
                // ctx.fillRect(x - 22, chartHeight - 32, 44, 28);

                // Time (Top)
                ctx.textAlign = 'center';
                ctx.fillStyle = this.colors.textBright;
                ctx.font = 'bold 11px sans-serif';
                ctx.fillText(tStr, x, chartHeight - 18);

                // Date (Bottom)
                ctx.fillStyle = '#888'; // Muted grey for date
                ctx.font = '10px sans-serif';
                ctx.fillText(dStr, x, chartHeight - 6);
            }
        }
        ctx.stroke();

        ctx.fillStyle = this.colors.text;
        ctx.font = '10px "Roboto Mono", monospace'; // Clean font
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


    _drawWallAttack(ctx) {
        if (!this.heatmapData || !this.currentPrice || !this.candles.length) return;

        // 1. FIND THE WALL
        // Look usage latest snapshot
        const snapshot = this.heatmapData[this.heatmapData.length - 1];
        if (!snapshot) return;

        // Define "Wall" as > 10% of max volume (Lowered for visibility)
        const wallThreshold = Math.max(5, this.maxVolumeInHistory * 0.1);

        // Search visible range
        const viewMin = this._getPriceFromY(this.height - this._deltaH());
        const viewMax = this._getPriceFromY(0);

        let nearestWall = null;
        let minDist = Infinity;

        const scan = (levels, type) => {
            if (!levels) return;
            for (const l of levels) {
                if (l.p < viewMin || l.p > viewMax) continue; // Skip off-screen
                if (l.q >= wallThreshold) {
                    const dist = Math.abs(l.p - this.currentPrice);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestWall = { ...l, type };
                    }
                }
            }
        };
        scan(snapshot.bids, 'bid');
        scan(snapshot.asks, 'ask');

        if (!nearestWall) {
            // Draw "Scanning" status so user knows it's on
            ctx.font = 'bold 11px "Inter", sans-serif';
            ctx.fillStyle = '#666';
            ctx.textAlign = 'right';
            ctx.fillText('SCANNING FOR WALLS...', this.width - 20, 30);
            return;
        }

        // 2. CALCULATE ATTACK STATS
        // How much volume happened at this price in the last few candles?
        let attackVol = 0;
        const targetPrice = nearestWall.p;
        const tick = this.tickSize;

        // Look back last 5 candles
        for (let i = this.candles.length - 1; i >= Math.max(0, this.candles.length - 5); i--) {
            const c = this.candles[i];
            if (!c || !c.clusters) continue;

            // Extract cluster
            let clusters = c.clusters instanceof Map ? Array.from(c.clusters.entries()) : Object.entries(c.clusters).map(([p, d]) => [parseFloat(p), d]);

            for (const [p, vol] of clusters) {
                if (Math.abs(p - targetPrice) < tick * 1.5) { // Close enough
                    // If attacking ASK wall (Resistance), use BID volume (Real Buys in inverted logic)
                    if (nearestWall.type === 'ask') attackVol += (vol.bid || 0);
                    // If attacking BID wall (Support), use ASK volume (Real Sells in inverted logic)
                    else attackVol += (vol.ask || 0);
                }
            }
        }

        // 3. PROBABILITY MATH
        // Simple: Attack / Wall
        // Advanced: Add momentum factor?
        const ratio = Math.min(1, attackVol / nearestWall.q);
        const probability = ratio * 100;

        // 4. DRAW HUD
        const y = this._getY(nearestWall.p);
        const x = this.width - 220; // Right aligned

        if (y < 0 || y > this.height - this._deltaH()) return;

        // Box
        ctx.fillStyle = 'rgba(10,10,10, 0.85)';
        ctx.strokeStyle = nearestWall.type === 'ask' ? '#ef4444' : '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(x, y - 25, 200, 50, 8);
        ctx.fill();
        ctx.stroke();

        // Title
        ctx.font = 'bold 12px "Inter", sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'left';
        ctx.fillText(`${nearestWall.type === 'ask' ? '🛡️ RESISTANCE' : '🛡️ SUPPORT'}`, x + 10, y - 5);

        // Stats
        ctx.font = '11px monospace';
        ctx.fillStyle = '#aaa';
        ctx.fillText(`Wall: ${nearestWall.q.toFixed(0)} | Hit: ${attackVol.toFixed(0)}`, x + 10, y + 15);

        // Probability
        ctx.textAlign = 'right';
        ctx.font = 'bold 16px sans-serif';

        // Color based on prob
        let probColor = '#ef4444'; // Low
        if (probability > 40) probColor = '#fbbf24'; // Med
        if (probability > 75) probColor = '#10b981'; // High

        ctx.fillStyle = probColor;
        ctx.fillText(`${probability.toFixed(0)}%`, x + 190, y + 5);

        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#666';
        ctx.fillText('BREAK CHANCE', x + 190, y + 18);
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
            this.width = rect.width;
            this.height = rect.height;
            this.canvas.width = this.width;
            this.canvas.height = this.height;
            this.requestDraw();
        }).observe(this.container);
    }
}
