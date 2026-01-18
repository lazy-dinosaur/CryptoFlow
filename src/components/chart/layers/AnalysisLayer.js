/**
 * AnalysisLayer
 * Renders analytical overlays like Naked Liquidity Lines, Big Trades, and Stacked Imbalances.
 */
export class AnalysisLayer {
    render(ctx, state, coords) {
        // 1. Naked Liquidity Lines
        this._renderNakedLiquidity(ctx, state, coords);

        // 2. Big Trades (Bubbles)
        this._renderBigTrades(ctx, state, coords);

        // 3. Current Price Line
        this._renderCurrentPriceLine(ctx, state, coords);

        // 4. Trade Signals (Paper Trading entries)
        this._renderTradeSignals(ctx, state, coords);
    }

    _renderNakedLiquidity(ctx, state, coords) {
        if (!state.showNakedLiquidity) return; // Check L-key toggle
        if (!state.nakedLiquidityLevels || state.nakedLiquidityLevels.length === 0) return;

        const { width, zoomX, offsetX } = state;

        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        // Calculate visible time range to optimize? 
        // For lines extending to right, we just need to know if they are vertically visible 
        // and if their start time is before the right edge of screen.

        for (const level of state.nakedLiquidityLevels) {
            const y = coords.getY(level.price);

            // Vertical culling
            if (y < 0 || y > state.height) continue;

            const startX = coords.getX(level.candleIndex);

            // If line starts off-screen right, skip
            if (startX > width) continue;

            const lineStartX = Math.max(0, startX);
            const lineEndX = width; // Extends to infinity (right edge)

            ctx.beginPath();
            ctx.strokeStyle = level.type === 'bid' ? 'rgba(0, 255, 106, 0.7)' : 'rgba(255, 51, 102, 0.7)';
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.moveTo(lineStartX, y);
            ctx.lineTo(lineEndX, y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label (Volume)
            // Draw label at the right edge or near start if visible?
            // "Standard" is often right edge.

            ctx.fillStyle = level.type === 'bid' ? '#00ff6a' : '#ff3366';
            // Background for text
            const text = `${this._formatVolume(level.volume)}`;
            const textWidth = ctx.measureText(text).width;

            // Draw label at right edge minus padding
            const labelX = width - 5;

            ctx.globalAlpha = 0.8;
            ctx.fillStyle = '#000';
            ctx.fillRect(labelX - textWidth - 4, y - 7, textWidth + 8, 14);
            ctx.globalAlpha = 1.0;

            ctx.fillStyle = level.type === 'bid' ? '#00ff6a' : '#ff3366';
            ctx.fillText(text, labelX, y);
        }
    }

    _renderBigTrades(ctx, state, coords) {
        if (!state.showBigTrades || !state.bigTrades) return;

        for (const trade of state.bigTrades) {
            // Check if visible
            // We need trade.candleIndex or trade.time
            // If we only have trade.time, we need a helper to get index.
            // Assuming trade has .index or we map it.
            // If based on time, we might need coordinate system to handle time->x.

            // For now assume trade has `candleIndex` computed during data loading or similar.
            // Or we iterate candles and find big trades inside them.

            if (trade.candleIndex === undefined) continue;

            const x = coords.getX(trade.candleIndex);
            const y = coords.getY(trade.price);

            // Culling
            if (x < 0 || x > state.width || y < 0 || y > state.height) continue;

            const radius = Math.min(20, Math.max(3, Math.sqrt(trade.volume) * 0.5));

            ctx.beginPath();
            ctx.fillStyle = trade.side === 'buy' ? 'rgba(0, 230, 118, 0.3)' : 'rgba(255, 23, 68, 0.3)';
            ctx.strokeStyle = trade.side === 'buy' ? '#00e676' : '#ff1744';
            ctx.lineWidth = 1;
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }
    }

    _formatVolume(val) {
        if (val >= 1000000) return (val / 1000000).toFixed(1) + 'M';
        if (val >= 1000) return (val / 1000).toFixed(0) + 'K';
        return val.toString();
    }

    _renderCurrentPriceLine(ctx, state, coords) {
        if (!state.currentPrice) return;

        const y = coords.getY(state.currentPrice);
        const { width } = state;

        if (y < 0 || y > state.height) return; // Off screen

        ctx.beginPath();
        ctx.strokeStyle = '#ffffff'; // White for current price
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]); // Dotted
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label on Right Axis
        const text = state.currentPrice.toFixed(2);
        ctx.font = '11px Inter, sans-serif';
        const textW = ctx.measureText(text).width + 10;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(width - textW, y - 10, textW, 20);

        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#000000';
        ctx.fillText(text, width - 5, y);
    }

    _renderTradeSignals(ctx, state, coords) {
        if (!state.tradeSignals || state.tradeSignals.length === 0) return;
        if (!state.candles || state.candles.length === 0) return;

        const { width, height } = state;

        for (const signal of state.tradeSignals) {
            // Find candle index by timestamp (timestamp is in milliseconds)
            const signalTime = typeof signal.timestamp === 'number'
                ? signal.timestamp
                : new Date(signal.timestamp).getTime();
            let candleIdx = -1;

            // Find the candle that CONTAINS the signal time
            // (last candle where candleTime <= signalTime)
            for (let i = state.candles.length - 1; i >= 0; i--) {
                if (state.candles[i].time <= signalTime) {
                    candleIdx = i;
                    break;
                }
            }

            // Fallback: if signal is before all candles, use first candle
            if (candleIdx < 0 && state.candles.length > 0) {
                candleIdx = 0;
            }

            if (candleIdx < 0) continue;

            const x = coords.getX(candleIdx);
            const y = coords.getY(signal.entry_price);

            // Culling
            if (x < 0 || x > width || y < 0 || y > height) continue;

            const isLong = signal.direction === 'LONG';
            const color = isLong ? '#00e676' : '#ff1744';
            const bgColor = isLong ? 'rgba(0, 230, 118, 0.2)' : 'rgba(255, 23, 68, 0.2)';

            // Draw entry marker (triangle)
            ctx.beginPath();
            ctx.fillStyle = color;
            if (isLong) {
                // Up triangle for LONG
                ctx.moveTo(x, y - 12);
                ctx.lineTo(x - 8, y + 4);
                ctx.lineTo(x + 8, y + 4);
            } else {
                // Down triangle for SHORT
                ctx.moveTo(x, y + 12);
                ctx.lineTo(x - 8, y - 4);
                ctx.lineTo(x + 8, y - 4);
            }
            ctx.closePath();
            ctx.fill();

            // Draw setup type label
            const setupType = signal.setup_type || 'SIGNAL';
            ctx.font = 'bold 9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';

            const labelY = isLong ? y - 16 : y + 20;
            const textWidth = ctx.measureText(setupType).width;

            // Label background
            ctx.fillStyle = bgColor;
            ctx.fillRect(x - textWidth/2 - 4, labelY - 10, textWidth + 8, 12);

            // Label text
            ctx.fillStyle = color;
            ctx.fillText(setupType, x, labelY);

            // Draw entry price line (dotted horizontal)
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.5;
            ctx.moveTo(x, y);
            ctx.lineTo(width - 60, y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1.0;
        }
    }
}
