/**
 * CoordinateSystem
 * Handles all mathematical transformations between data space (Price, Time/Index)
 * and screen space (Pixels).
 */
export class CoordinateSystem {
    constructor(state) {
        this.state = state;
    }

    /**
     * Get X pixel coordinate for a candle index
     * @param {number} index - The index of the candle
     * @returns {number} Pixel X coordinate
     */
    getX(index) {
        // center screen x + (index relative to data center?) * zoom
        // Or simpler: (index * zoomX) + offsetX
        // Existing logic from FootprintChart:
        // x = (index * this.candleWidth) + this.offsetX
        // where candleWidth is essentially zoomX
        return (index * this.state.zoomX) + this.state.offsetX;
    }

    /**
     * Get Y pixel coordinate for a price
     * @param {number} price 
     * @returns {number} Pixel Y coordinate
     */
    getY(price) {
        // Existing logic:
        // y = this.canvas.height / 2 - (price - this.viewCenterPrice) / this.tickSize * this.pixelsPerTick
        // Let's use the offset model which is more robust:
        // y = (price * pixelsPerUnit) + offsetY 
        // Typically charts use inverted Y (0 is top).
        // Standard formula: height - ((price - minPrice) * scaleY)

        // Using the logic derived from offset:
        // y = height - (price / tickSize * zoomY) + offsetY
        // NOTE: We need to align this with how FootprintChart currently works or standardise it.
        // FootprintChart typical logic:
        // y = height - ( (price - basePrice) / tickSize * zoomY )

        // Let's stick to a generic offset model that matches the Zoom logic:
        // Y = (Price / TickSize * ZoomY) + OffsetY
        // Since Canvas Y is inverted (0 at top), we typically invert the result or the scale.
        // Let's try:
        return this.state.height - ((price / this.state.tickSize * this.state.zoomY) + this.state.offsetY);
    }

    /**
     * Get Candle Index from X pixel
     * @param {number} x - Pixel X
     * @returns {number} Floating point index
     */
    getIndexFromX(x) {
        return (x - this.state.offsetX) / this.state.zoomX;
    }

    /**
     * Get Price from Y pixel
     * @param {number} y - Pixel Y
     * @returns {number} Price
     */
    getPriceFromY(y) {
        // Inverse of getY
        // y = height - (val + offsetY)
        // val + offsetY = height - y
        // val = height - y - offsetY
        // price / tick * zoom = val

        const value = this.state.height - y - this.state.offsetY;
        return (value / this.state.zoomY) * this.state.tickSize;
    }

    /**
     * Check if a price range is visible
     */
    isVisible(price, timeIndex) {
        const x = this.getX(timeIndex);
        const y = this.getY(price);
        return x >= -this.state.zoomX && x <= this.state.width + this.state.zoomX &&
            y >= -this.state.zoomY && y <= this.state.height + this.state.zoomY;
    }
}
