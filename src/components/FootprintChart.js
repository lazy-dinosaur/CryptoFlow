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

        // Flags
        this.showHeatmap = true;
        this.showDelta = true;
        this.showBigTrades = true;
        this.showCrosshair = true;
        this.showNakedLiquidity = false;
        this.showML = true;
        this.showImbalances = true;
        this.showLens = false;

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
        this.bigTradeThreshold = 5.0;
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
    _deltaH() { return this.showDelta ? this.deltaRowHeight : 0; }
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
        const ctx = this.ctx;
        const { width, height } = this;
        if (width <= 0 || height <= 0) return;

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, width, height);

        if (this.showHeatmap) this._drawHeatmap(ctx);
        this._drawGrid(ctx);
        this._drawCandles(ctx);
        if (this.showDelta) this._drawDeltaSummary(ctx);
        this._drawPriceLine(ctx);
        if (this.showBigTrades) this._drawBigTrades(ctx);
        if (this.showNakedLiquidity && !this.showLens) this._drawLiquidity(ctx);
        if (this.showCrosshair && this.crosshairX) this._drawCrosshair(ctx);
        if (this.showLens) this._drawLens(ctx);
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

                const ratio = lev.q / this.maxVolumeInHistory; // 0 to 1
                if (ratio < this.heatmapIntensityThreshold) continue; // Slider

                // BOOKMAP STYLE: Gamma Corrected Opacity
                const visRatio = Math.pow(ratio, 0.4);

                let color;
                const alpha = this.heatmapOpacity;

                if (visRatio < 0.15) {
                    if (visRatio < 0.05) continue;
                    const localAlpha = alpha * (visRatio / 0.15);
                    color = `rgba(0, 0, 50, ${localAlpha})`;
                } else if (visRatio < 0.3) {
                    color = `rgba(0, 0, 150, ${alpha})`;
                } else if (visRatio < 0.45) {
                    color = `rgba(0, 150, 255, ${alpha})`;
                } else if (visRatio < 0.6) {
                    color = `rgba(0, 255, 0, ${alpha})`;
                } else if (visRatio < 0.75) {
                    color = `rgba(255, 255, 0, ${alpha})`;
                } else if (visRatio < 0.9) {
                    color = `rgba(255, 69, 0, ${alpha})`;
                } else {
                    color = `rgba(255, 255, 255, ${alpha})`;
                }

                bufCtx.fillStyle = color;
                bufCtx.fillRect(x, y - h / 2, w, h);
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

                // Background Box (Taller for stacked text)
                ctx.fillStyle = this.colors.bg;
                ctx.fillRect(x - 22, chartHeight - 32, 44, 28);

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
                    // Bid Left
                    if (data.bid > 0) {
                        const len = (data.bid / maxVol) * halfWidth;
                        ctx.fillStyle = this.colors.barSell;
                        ctx.fillRect(centerX - gap - len, drawY + 1, len, rowH - 1);
                    }
                    // Ask Right
                    if (data.ask > 0) {
                        const len = (data.ask / maxVol) * halfWidth;
                        ctx.fillStyle = this.colors.barBuy;
                        ctx.fillRect(centerX + gap, drawY + 1, len, rowH - 1);
                    }

                    // TEXT (Full Detail Level)
                    if (showText && rowH > 11) {
                        ctx.fillStyle = this.colors.clusterText;
                        ctx.font = 'bold 11px "Roboto Mono", monospace';

                        ctx.textAlign = 'right';
                        if (data.bid > 0) ctx.fillText(data.bid.toFixed(0), centerX - gap - 4, y + 4);

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
        const rowTop = height - this.deltaRowHeight;

        // Darker footer background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, rowTop, width, this.deltaRowHeight);

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
            ctx.fillRect(x, rowTop + 2, candleWidth, this.deltaRowHeight - 4);

            // Value
            if (zoomX > 30) {
                ctx.fillStyle = isPos ? this.colors.candleUp : this.colors.candleDown;
                ctx.fillText(delta.toFixed(0), centerX, height - 7);
            } else {
                ctx.fillStyle = isPos ? this.colors.candleUp : this.colors.candleDown;
                ctx.fillRect(centerX - 1.5, height - 12, 3, 3);
            }
        }
    }

    _drawLens(ctx) {
        if (!this.hoveredCandle) return;
        const c = this.hoveredCandle;

        const lensW = 280;
        const lensH = 320;
        let lx = this.lastX + 20;
        let ly = this.lastY - lensH / 2;
        if (lx + lensW > this.width) lx = this.lastX - lensW - 20;
        if (ly < 0) ly = 10;
        if (ly + lensH > this.height) ly = this.height - lensH - 10;

        // Modern Lens Background
        ctx.fillStyle = 'rgba(20, 25, 30, 0.98)';
        ctx.fillRect(lx, ly, lensW, lensH);

        // Border
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.strokeRect(lx, ly, lensW, lensH);

        // Header info
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px "Roboto Mono", monospace';
        ctx.textAlign = 'left';

        const d = new Date(c.time);
        const timeStr = `${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`;
        // Header
        ctx.fillStyle = '#ccc';
        ctx.fillText(`${timeStr}`, lx + 12, ly + 22);

        const delta = c.delta || 0;
        const bull = c.close >= c.open;
        ctx.fillStyle = delta >= 0 ? this.colors.candleUp : this.colors.candleDown;
        ctx.textAlign = 'right';
        ctx.fillText(`Δ ${delta.toFixed(2)}`, lx + lensW - 12, ly + 22);

        // Separator
        ctx.strokeStyle = '#333';
        ctx.beginPath(); ctx.moveTo(lx, ly + 35); ctx.lineTo(lx + lensW, ly + 35); ctx.stroke();

        const clusters = c.clusters;
        if (!clusters) return;

        let entries = [];
        if (clusters instanceof Map) {
            entries = Array.from(clusters.entries());
        } else {
            entries = Object.entries(clusters).map(([p, data]) => [parseFloat(p), data]);
        }

        // Defensive: ignore malformed cluster entries
        entries = entries.filter(([price, data]) => Number.isFinite(price) && data && typeof data === 'object');

        entries.sort((a, b) => b[0] - a[0]);

        const lensRowH = 15;
        const startY = ly + 45;
        const centerX = lx + lensW / 2;

        let maxVol = 0;
        let pocIndex = 0;
        entries.forEach((e, idx) => {
            const v = (e[1].bid || 0) + (e[1].ask || 0);
            if (v > maxVol) { maxVol = v; pocIndex = idx; }
        });

        let startRow = Math.max(0, pocIndex - 8);
        // Ensure we cover enough range
        let endRow = Math.min(entries.length, startRow + 17);

        let drawY = startY;
        const candleColor = bull ? this.colors.candleUp : this.colors.candleDown;

        for (let i = startRow; i < endRow; i++) {
            const [price, data] = entries[i];

            // --- CONTEXT: WICK & BODY ---
            // Determine if this row is part of the candle body or wick
            // Note: Since prices are floats, we use a small epsilon or just ranges
            const p = price;
            const topBody = Math.max(c.open, c.close);
            const botBody = Math.min(c.open, c.close);
            const eps = this.tickSize / 2;

            // 1. WICK (Vertical Line)
            // If price is between Low and High
            if (p <= c.high && p >= c.low) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.1)'; // Subtle wick guide
                ctx.fillRect(centerX - 1, drawY, 2, lensRowH);
            }

            // 2. BODY (Colored Background)
            // If price is strictly inside the body range
            // We use a slight overlap logic for smoother look
            if (p <= topBody && p >= botBody) {
                ctx.fillStyle = bull ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
                ctx.fillRect(lx + 2, drawY, lensW - 4, lensRowH);
            }

            // 3. BODY EDGES (Open/Close Lines)
            // Check proximity to Open
            if (Math.abs(p - c.open) < eps || Math.abs(p - c.close) < eps) {
                ctx.strokeStyle = candleColor;
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 2]); // Dashed border for Body limits
                ctx.beginPath();
                ctx.moveTo(lx + 2, drawY + (p === topBody ? 0 : lensRowH)); // Draw at top or bottom? 
                // Simplification for list view: Draw border at Top of row if Price ~ TopBody
                // Draw border at Bottom of row if Price ~ BotBody. 
                // Since sorted descending: TopBody is first.

                // Let's just draw Top/Bottom lines of the rect for this row if it matches
                if (Math.abs(p - topBody) < eps) {
                    ctx.beginPath(); ctx.moveTo(lx + 2, drawY); ctx.lineTo(lx + lensW - 2, drawY); ctx.stroke();
                }
                if (Math.abs(p - botBody) < eps) {
                    ctx.beginPath(); ctx.moveTo(lx + 2, drawY + lensRowH); ctx.lineTo(lx + lensW - 2, drawY + lensRowH); ctx.stroke();
                }
                ctx.setLineDash([]);
            }
            // ----------------------------

            if (data.imbalance) {
                // Overlay imbalance highlight stronger if needed, or keep subtle
                ctx.fillStyle = data.imbalance === 'buy' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';
                ctx.fillRect(lx + 1, drawY, lensW - 2, lensRowH);
            }

            const maxBarW = (lensW / 2) - 50;

            // Bid
            if (data.bid > 0) {
                const w = Math.min(maxBarW, (data.bid / maxVol) * maxBarW * 2.5);
                ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';
                ctx.fillRect(centerX - w - 2, drawY + 2, w, lensRowH - 4);

                ctx.fillStyle = '#ddd';
                ctx.textAlign = 'right';
                ctx.font = 'bold 11px "Roboto Mono", monospace';
                ctx.fillText(data.bid.toFixed(0), centerX - 6, drawY + 11);
            }

            // Ask
            if (data.ask > 0) {
                const w = Math.min(maxBarW, (data.ask / maxVol) * maxBarW * 2.5);
                ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
                ctx.fillRect(centerX + 2, drawY + 2, w, lensRowH - 4);

                ctx.fillStyle = '#ddd';
                ctx.textAlign = 'left';
                ctx.font = 'bold 11px "Roboto Mono", monospace';
                ctx.fillText(data.ask.toFixed(0), centerX + 6, drawY + 11);
            }

            // Price (Left side)
            ctx.fillStyle = '#666';
            ctx.textAlign = 'left';
            ctx.font = '10px sans-serif';
            ctx.fillText(price.toFixed(1), lx + 6, drawY + 11);

            drawY += lensRowH;
        }
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
            if (x < 0 || x > this.width || y < 0 || y > this.height) continue;

            const r = Math.min(25, Math.max(4, Math.sqrt(t.volume) * 0.5));

            // 3D BUBBLE EFFECT (Radial Gradient)
            const grad = ctx.createRadialGradient(x - r / 3, y - r / 3, r / 10, x, y, r);

            if (t.side === 'buy') {
                grad.addColorStop(0, '#86efac'); // bright green center
                grad.addColorStop(1, '#16a34a'); // darker green edge
                ctx.strokeStyle = '#dcfce7';     // nearly white outline
            } else {
                grad.addColorStop(0, '#fca5a5'); // bright red center
                grad.addColorStop(1, '#dc2626'); // darker red edge
                ctx.strokeStyle = '#fee2e2';     // nearly white outline
            }

            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            ctx.lineWidth = 1;
            ctx.stroke();

            // Optional: Volume Label inside bigger bubbles
            if (r > 8) {
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 9px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(t.volume.toFixed(0), x, y);
            }
        }
    }

    setLiquidityLevels(levels) {
        this.nakedLiquidityLevels = Array.isArray(levels) ? levels : [];
        this.requestDraw();
    }

    _drawLiquidity(ctx) {
        if (!this.showNakedLiquidity) return;

        const levels = this.nakedLiquidityLevels || [];
        if (levels.length === 0) return;

        const deltaH = this._deltaH();

        ctx.font = '10px "Roboto Mono", monospace';
        ctx.textBaseline = 'middle';

        for (const lev of levels) {
            if (!lev || !Number.isFinite(lev.price)) continue;

            const y = this._getY(lev.price);
            if (y < 0 || y > this.height - deltaH) continue;

            const isBid = lev.type === 'bid';
            const lineColor = isBid ? '#10b981' : '#ef4444';

            // line
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 1;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.width, y);
            ctx.stroke();
            ctx.setLineDash([]);

            // label
            // label
            const vol = Number(lev.volume);
            if (Number.isFinite(vol) && vol > 0) {
                const text = `${isBid ? '' : ''}${vol.toFixed(2)}`; // Minimal label
                // Subtle, small text
                ctx.font = '9px "Roboto Mono", monospace';
                const w = ctx.measureText(text).width + 6;

                ctx.globalAlpha = 0.6; // More transparent background
                ctx.fillStyle = '#0a0a0a';
                ctx.fillRect(this.width - w - 4, y - 7, w, 14);
                ctx.globalAlpha = 1;

                ctx.fillStyle = lineColor;
                ctx.textAlign = 'right';
                ctx.fillText(text, this.width - 6, y);
            }
        }
    }

    _drawCrosshair(ctx) {
        if (this.showLens) return;

        const { crosshairX, crosshairY, width, height } = this;
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
            case 'L': this.showNakedLiquidity = !this.showNakedLiquidity; this.requestDraw(); break;
            case 'X': this.showLens = !this.showLens; this.requestDraw(); break;
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

        this.lastX = e.clientX;
        this.lastY = e.clientY;
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
            this.offsetX += e.clientX - this.lastX;
            this.offsetY -= e.clientY - this.lastY;
        }

        if (this.isDragging || this.isDraggingPrice || this.isDraggingTime || this.showLens) {
            this.lastX = e.clientX;
            this.lastY = e.clientY;
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
