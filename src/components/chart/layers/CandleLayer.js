/**
 * CandleLayer
 * Responsible for rendering the main price action: Candles and Footprint Clusters.
 */
export class CandleLayer {
    render(ctx, state, coords) {
        if (!state.candles || state.candles.length === 0) return;

        const { candles, width, height, zoomX, offsetX, tickSize, colors } = state;

        // Optimization: Find visible range
        // Since Coords maps index -> X:
        // x = idx * zoom + offset
        // 0 = idx * zoom + offset => idx = -offset / zoom
        // width = idx * zoom + offset => idx = (width - offset) / zoom

        const startIdx = Math.floor(-offsetX / zoomX);
        const endIdx = Math.ceil((width - offsetX) / zoomX);

        // Clamp to data range
        const first = Math.max(0, startIdx - 1); // 1 candle buffer
        const last = Math.min(candles.length - 1, endIdx + 1);

        // Calculate global max volume for heatmap mode relative scaling
        let globalMaxVol = 0;
        if (state.heatmapMode) {
            // We ideally should pre-calculate this in ChartState or pass it in
            // Traversing all candles every frame is expensive if excessive.
            // But for visible range + heuristic? 
            // FootprintChart calculated it per frame.
            for (let i = first; i <= last; i++) {
                const candle = candles[i];
                if (!candle.clusters) continue;
                for (const price in candle.clusters) {
                    const c = candle.clusters[price];
                    globalMaxVol = Math.max(globalMaxVol, (c.bid || 0) + (c.ask || 0));
                }
            }
        }

        const candleWidth = this._calculateCandleWidth(zoomX);

        for (let i = first; i <= last; i++) {
            this._renderCandle(ctx, candles[i], i, candleWidth, globalMaxVol, state, coords);
        }
    }

    _calculateCandleWidth(zoomX) {
        // Gap of 20% or min 1px
        const gap = Math.max(1, zoomX * 0.2);
        return Math.max(1, zoomX - gap);
    }

    _renderCandle(ctx, candle, index, candleWidth, globalMaxVol, state, coords) {
        const x = coords.getX(index); // This is left edge?
        // CoordinateSystem.getX returns center? In FootprintChart it returned Left edge usually.
        // FootprintChart: x = i * (width + gap)
        // Coords: x = i * zoomX + offset
        // This effectively represents the "start" of the slot.
        // The candle should be centered or fill it.

        const clusters = candle.clusters || {};
        const clusterPrices = Object.keys(clusters).map(Number).sort((a, b) => b - a);

        const isBullish = candle.close >= candle.open;

        // LOD 0: Simple OHLC (Zoom < 0.4)
        if (state.zoomX < 5) { // Threshold for footprint visibility
            this._renderSimpleCandle(ctx, x, candle, candleWidth, isBullish, state, coords);
            return;
        }

        // Full Footprint Render
        if (clusterPrices.length === 0) {
            this._renderSimpleCandle(ctx, x, candle, candleWidth, isBullish, state, coords);
            return;
        }

        // Find range
        const high = Math.max(candle.high, ...clusterPrices);
        const low = Math.min(candle.low, ...clusterPrices);
        const topY = coords.getY(high + state.tickSize); // +tick because Y is inverted and we want top of box
        const bottomY = coords.getY(low);

        // Wait, coords.getY(price) returns Y for that specific price level (bottom line? center?).
        // If Logic: y = height - (price...
        // Let's assume coords.getY returns the CENTER or BOTTOM of the row.
        // Need to be consistent. 
        // FootprintChart: rowHeight = zoomY. y = coord(price).
        // Let's assume coords.getY(price) is the geometric Y coordinate for that price.

        const rowHeight = state.zoomY;

        // Draw Wick
        const centerX = x + candleWidth / 2;
        ctx.strokeStyle = isBullish ? 'rgba(0, 200, 83, 0.6)' : 'rgba(255, 23, 68, 0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        // High to Low
        ctx.moveTo(centerX, coords.getY(high));
        ctx.lineTo(centerX, coords.getY(low));
        ctx.stroke();

        // Draw Body Box (Open - Close)
        const bodyTop = coords.getY(Math.max(candle.open, candle.close));
        const bodyBot = coords.getY(Math.min(candle.open, candle.close));
        const bodyH = Math.abs(bodyBot - bodyTop) || 1;

        // Fill Body
        ctx.fillStyle = state.colors.candleBody || '#1e222d';
        // Note: Footprint chart logic fills only the "Body" range background

        if (bodyH > 0) {
            // We need to account for inverted Y. Top Y < Bottom Y.
            // coords.getY(Max) returns smaller Y (higher on screen).
            ctx.fillRect(x, bodyTop, candleWidth, bodyH);
            ctx.strokeRect(x, bodyTop, candleWidth, bodyH);
        }

        // Render Clusters
        // Max Volume for this candle calculation
        let maxVol = 0;
        for (const p of clusterPrices) {
            const c = clusters[p];
            maxVol = Math.max(maxVol, c.bid || 0, c.ask || 0);
        }

        const halfWidth = (candleWidth - 4) / 2; // Padding 2px

        for (const price of clusterPrices) {
            const cluster = clusters[price];
            const y = coords.getY(price); // This is y for the price level

            // Adjust y to draw box. Assume y is bottom-left? Or center? 
            // Standard: Y is top-left of the row usually?
            // Let's center it: y - rowHeight/2
            const drawY = y - rowHeight / 2;

            // 1. Backgrounds (Imbalances, POC)
            if (state.showImbalances) {
                if (cluster.isStackedBuy) {
                    ctx.fillStyle = 'rgba(0, 255, 106, 0.25)';
                    ctx.fillRect(x + 1, drawY, candleWidth - 2, rowHeight);
                } else if (cluster.isStackedSell) {
                    // ...
                }
            }

            // 2. Heatmap Mode vs Delta Mode
            if (state.heatmapMode && globalMaxVol > 0) {
                // ... Heatmap logic inside candle ...
            } else {
                // Bars
                // Left (Bid/Sell)
                if (cluster.bid > 0) {
                    const w = (cluster.bid / maxVol) * halfWidth;
                    ctx.fillStyle = 'rgba(255, 51, 102, 0.6)'; // Red
                    ctx.fillRect(centerX - 2 - w, drawY + 1, w, rowHeight - 2);
                }

                // Right (Ask/Buy)
                if (cluster.ask > 0) {
                    const w = (cluster.ask / maxVol) * halfWidth;
                    ctx.fillStyle = 'rgba(0, 255, 106, 0.6)'; // Green
                    ctx.fillRect(centerX + 2, drawY + 1, w, rowHeight - 2);
                }

                // Text
                if (state.zoomX > 20) { // Text visible
                    // Render text
                    ctx.fillStyle = '#fff';
                    ctx.font = '10px monospace';
                    // ...
                }
            }
        }
    }

    _renderSimpleCandle(ctx, x, candle, width, isBullish, state, coords) {
        const open = coords.getY(candle.open);
        const close = coords.getY(candle.close);
        const high = coords.getY(candle.high);
        const low = coords.getY(candle.low);

        const centerX = x + width / 2;

        ctx.strokeStyle = isBullish ? '#00e676' : '#ff1744';
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(centerX, high);
        ctx.lineTo(centerX, low);
        ctx.stroke();

        ctx.fillStyle = isBullish ? '#00e676' : '#ff1744';
        const h = Math.abs(close - open) || 1;
        const y = Math.min(open, close);
        ctx.fillRect(x, y, width, h);
    }
}
