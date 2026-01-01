/**
 * GridLayer
 * Renders the background grid lines and price/time axes visuals.
 */
export class GridLayer {
    render(ctx, state, coords) {
        const { width, height, tickSize, offsetX, zoomX, zoomY } = state;

        ctx.strokeStyle = '#2a2a2a'; // Dark grey
        ctx.lineWidth = 1;
        ctx.beginPath();

        // Horizontal Grid (Price)
        // Find visible price range
        const minPrice = coords.getPriceFromY(height);
        const maxPrice = coords.getPriceFromY(0);

        // Determine grid step (e.g. every 10 ticks)
        // Adjust step based on zoomY to prevent clutter
        const step = tickSize * 10;

        // Snap to nearest step
        const startPrice = Math.floor(minPrice / step) * step;

        for (let p = startPrice; p <= maxPrice; p += step) {
            const y = coords.getY(p);
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
        }

        // Vertical Grid (Time/Candles)
        // Draw every 10th candle?
        const startIdx = Math.floor(-offsetX / zoomX);
        const endIdx = startIdx + (width / zoomX);
        const candleStep = 10;

        for (let i = Math.floor(startIdx / candleStep) * candleStep; i <= endIdx; i += candleStep) {
            const x = coords.getX(i);
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
        }

        ctx.stroke();

        // -------------------------
        // Draw Axis Labels
        // -------------------------
        ctx.fillStyle = '#888888';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        // Price Axis (Right)
        for (let p = startPrice; p <= maxPrice; p += step) {
            const y = coords.getY(p);
            // Label
            if (y > 0 && y < height) {
                // Ensure price is formatted nicely (e.g. 2 decimals)
                ctx.fillText(p.toFixed(2), width - 5, y);
            }
        }

        // Time Axis (Bottom)
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        for (let i = Math.floor(startIdx / candleStep) * candleStep; i <= endIdx; i += candleStep) {
            // Bounds check - skip negative or out-of-range indices
            if (i < 0 || i >= state.candles.length) continue;

            const x = coords.getX(i);
            if (x > 0 && x < width) {
                const candle = state.candles[i];
                if (candle && candle.openTime) {
                    const date = new Date(candle.openTime);
                    // Validate date is valid (not NaN)
                    if (!isNaN(date.getTime())) {
                        const timeStr = date.getHours().toString().padStart(2, '0') + ':' +
                            date.getMinutes().toString().padStart(2, '0');
                        ctx.fillText(timeStr, x, height - 15);
                    }
                }
            }
        }
    }
}
