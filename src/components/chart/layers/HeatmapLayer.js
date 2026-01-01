/**
 * HeatmapLayer
 * Responsible for rendering the order book depth history.
 * Implements double-buffering to prevent flickering.
 */
export class HeatmapLayer {
    constructor(state, coords) {
        this.state = state;
        this.coords = coords;

        // Off-screen buffer for double buffering
        this.buffer = document.createElement('canvas');
        this.bufferCtx = this.buffer.getContext('2d', { alpha: true }); // Enable alpha for transparency

        // Cache state
        this.lastWidth = 0;
        this.lastHeight = 0;
        this.lastTransform = null;
    }

    render(ctx, state, coords) {
        if (!state.showHeatmap) return;

        // Ensure buffer is sized correctly
        // IMPORTANT: The buffer should be the size of the VIEWPORT, not the entire world.
        // But for heatmap we might want to cache the whole thing? 
        // No, caching the whole 24h history is impossible.
        // We render what is visible.

        const width = state.width * state.pixelRatio;
        const height = state.height * state.pixelRatio;

        if (width <= 0 || height <= 0) return;

        if (this.buffer.width !== width || this.buffer.height !== height) {
            this.buffer.width = width;
            this.buffer.height = height;
            // When resized, we must redraw the buffer
            this._redrawBuffer(width, height);
        } else {
            // Check if we need to redraw buffer (e.g. data changed or zoom changed substantially)
            // Ideally RenderEngine calls render every frame on interaction.
            // But Heatmap calculation is expensive.
            // For now, let's redraw every frame but OPTIMIZE inside.
            // Actually, we should redraw the buffer every frame because the CoordinateSystem changes
            // (panning changes the X/Y mapping).
            // A static background cache only works if we just blit it with an offset, 
            // but zooming changes scaling.

            // Re-implementing the optimization from FootprintChart:
            // It seems it was redrawing to buffer then blitting.
            this._redrawBuffer(width, height);
        }

        // Blit buffer to main context
        // Since we are drawing 1:1 to the buffer in _redrawBuffer using the current coords,
        // we just drawImage(0,0).
        ctx.drawImage(this.buffer, 0, 0, width / state.pixelRatio, height / state.pixelRatio);
    }

    _redrawBuffer(width, height) {
        const ctx = this.bufferCtx;
        const { heatmapData, maxVolumeInHistory, heatmapIntensityThreshold, heatmapOpacity } = this.state;

        ctx.clearRect(0, 0, width, height); // Clear to transparent, not solid

        if (!heatmapData || heatmapData.length === 0) return;

        // Calculate time range for visible area
        // We can optimize by only iterating visible snapshots
        // Using coords.getIndexFromX to find start/end indices in the *candle* array won't work 
        // directly for heatmap snapshots unless they are 1:1 aligned.
        // They are usually aligned by time.

        // Brute force for now (optimize later like passing 'visibleRange'):
        // Iterate all snapshots.

        // Optimization: Temporal sampling? 
        // The original efficient logic was:
        // const pixelWidth = this.lastCandleWidth * this.transform.k; // effective width

        const effectiveZoomX = this.state.zoomX;
        const skip = effectiveZoomX < 5 ? 2 : 1; // Simple LOD

        for (let i = 0; i < heatmapData.length; i += skip) {
            const snapshot = heatmapData[i];

            // X coordinate (Time)
            // We need to map time -> X. 
            // CoordinateSystem currently maps Index -> X.
            // We need to know which candle index this snapshot corresponds to.
            // Assuming 1:1 mapping for now (snapshot I corresponds to Candle I)
            // TODO: This is a weak assumption. In reality we match by Time.
            // Let's assume for now the data is pre-aligned or we use time.

            // The existing FootprintChart logic used `snapshot.index` or just `i` if aligned.
            // Let's use `i` for now as `heatmapData` mirrors `candles`.

            const x = this.coords.getX(i);
            const w = effectiveZoomX; // Fill the candle width

            // Optimization: Frustum culling
            if (x + w < 0 || x > this.state.width) continue;

            // Handle both data formats: levels[] or bids[]/asks[]
            let levels = snapshot.levels;
            if (!levels) {
                // Convert bids/asks format to levels format
                levels = [];
                if (snapshot.bids) {
                    for (const bid of snapshot.bids) {
                        levels.push({ p: bid.p || bid.price, q: bid.q || bid.quantity });
                    }
                }
                if (snapshot.asks) {
                    for (const ask of snapshot.asks) {
                        levels.push({ p: ask.p || ask.price, q: ask.q || ask.quantity });
                    }
                }
            }

            for (const level of levels) {
                const price = level.p || level.price;
                const qty = level.q || level.quantity;
                if (!price || !qty) continue;

                const y = this.coords.getY(price);
                const h = Math.abs(this.coords.getY(price + this.state.tickSize) - y) || 1;

                // Y-culling
                if (y + h < 0 || y > this.state.height) continue;

                // Color calculation
                const ratio = qty / maxVolumeInHistory;
                if (ratio < heatmapIntensityThreshold) continue;

                ctx.fillStyle = this._getColor(ratio, heatmapOpacity);
                ctx.fillRect(x, y, w, h);
            }
        }
    }

    _getColor(ratio, baseOpacity) {
        // Vibrant Heatmap Colors
        let alpha = Math.min(1, (ratio + 0.1) * 2.5 * baseOpacity);
        let hue, saturation, lightness;

        if (ratio < 0.05) { // Low volume - Deep Blue
            hue = 240;
            saturation = 80;
            lightness = 30 + (ratio * 1000);
        } else if (ratio < 0.15) { // Medium Low - Cyan
            hue = 200 - ((ratio - 0.05) * 400);
            saturation = 90;
            lightness = 50;
        } else if (ratio < 0.3) { // Medium - Green
            hue = 160 - ((ratio - 0.15) * 400);
            saturation = 100;
            lightness = 50;
            alpha = Math.min(1, alpha + 0.2);
        } else { // High - Yellow to Red
            hue = 60 - ((ratio - 0.3) * 85);
            saturation = 100;
            lightness = 50 + (ratio * 20);
            alpha = 1;
        }

        return `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
    }
}
