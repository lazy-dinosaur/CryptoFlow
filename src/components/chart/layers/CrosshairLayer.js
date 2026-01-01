/**
 * CrosshairLayer
 * Renders the mouse crosshair, price label, and time label.
 */
export class CrosshairLayer {
    render(ctx, state, coords) {
        if (!state.showCrosshair || !state.crosshairX || !state.crosshairY) return;

        const { width, height, colors } = state;
        const x = state.crosshairX;
        const y = state.crosshairY;

        // Lines
        ctx.strokeStyle = '#666'; // or state.colors.crosshair
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 6]);

        ctx.beginPath();
        // Horizontal
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        // Vertical
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);

        // Labels
        ctx.fillStyle = '#1e222d'; // Label bg
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        // Price Label (Right Axis)
        const price = coords.getPriceFromY(y);
        const priceText = price.toFixed(2); // Todo: dynamic precision
        const textW = ctx.measureText(priceText).width + 10;

        // Draw Price Box on right edge
        ctx.fillRect(width - textW, y - 10, textW, 20);
        ctx.fillStyle = '#fff';
        ctx.fillText(priceText, width - 5, y);

        // Time Label (Bottom Axis)
        // Need time from X
        // const idx = coords.getIndexFromX(x);
        // const time = state.candles[Math.floor(idx)]?.time;
        // If we can get time, render it.
        // For now skip time label if we don't have easy lookup, or implement:
        const idx = Math.round(coords.getIndexFromX(x));
        if (idx >= 0 && idx < state.candles.length) {
            const timeDate = new Date(state.candles[idx].time);
            const timeText = timeDate.toLocaleTimeString(); // Simple time

            ctx.textAlign = 'center';
            ctx.fillStyle = '#1e222d';
            ctx.fillRect(x - 30, height - 20, 60, 20);
            ctx.fillStyle = '#fff';
            ctx.fillText(timeText, x, height - 10);
        }
    }
}
