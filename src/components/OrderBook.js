/**
 * Order Book Component
 * Real-time DOM (Depth of Market) ladder
 */

export class OrderBook {
    constructor(options = {}) {
        this.asksContainer = document.getElementById(options.asksId || 'orderbookAsks');
        this.bidsContainer = document.getElementById(options.bidsId || 'orderbookBids');
        this.spreadElement = document.getElementById(options.spreadId || 'orderbookSpread');
        this.bidRatioElement = document.getElementById(options.bidRatioId || 'bidRatio');
        this.askRatioElement = document.getElementById(options.askRatioId || 'askRatio');
        this.ratioFillElement = document.getElementById(options.ratioFillId || 'ratioFill');

        // Settings
        this.levels = options.levels || 15;
        this.precision = options.precision || 2;
        this.grouping = options.grouping || 1;
        this.largeOrderThreshold = options.largeOrderThreshold || 10;

        // Data
        this.bids = new Map();
        this.asks = new Map();
        this.lastPrice = null;

        // Animation state
        this.previousBids = new Map();
        this.previousAsks = new Map();
    }

    /**
     * Update order book with depth data
     * @param {Object} depthData - Depth update from WebSocket
     */
    update(depthData) {
        // Update bids
        for (const { price, quantity } of depthData.bids) {
            if (quantity === 0) {
                this.bids.delete(price);
            } else {
                this.bids.set(price, quantity);
            }
        }

        // Update asks
        for (const { price, quantity } of depthData.asks) {
            if (quantity === 0) {
                this.asks.delete(price);
            } else {
                this.asks.set(price, quantity);
            }
        }

        this._render();
    }

    /**
     * Update last trade price
     * @param {number} price
     */
    updatePrice(price) {
        this.lastPrice = price;
    }

    /**
     * Set price precision
     * @param {number} precision
     */
    setPrecision(precision) {
        this.precision = precision;
    }

    /**
     * Render the order book
     */
    _render() {
        // Sort and get top N levels
        const sortedBids = Array.from(this.bids.entries())
            .sort((a, b) => b[0] - a[0])
            .slice(0, this.levels);

        const sortedAsks = Array.from(this.asks.entries())
            .sort((a, b) => a[0] - b[0])
            .slice(0, this.levels);

        // Calculate totals for depth visualization
        const bidTotal = sortedBids.reduce((sum, [, qty]) => sum + qty, 0);
        const askTotal = sortedAsks.reduce((sum, [, qty]) => sum + qty, 0);
        const maxTotal = Math.max(bidTotal, askTotal) || 1;

        // Extract best prices BEFORE corrupting array with reverse()
        const bestBid = sortedBids.length > 0 ? sortedBids[0][0] : 0;
        const bestAsk = sortedAsks.length > 0 ? sortedAsks[0][0] : 0;

        // Update ratio display
        const totalVolume = bidTotal + askTotal;
        if (totalVolume > 0) {
            const bidPercent = Math.round((bidTotal / totalVolume) * 100);
            const askPercent = 100 - bidPercent;
            this.bidRatioElement.textContent = `${bidPercent}%`;
            this.askRatioElement.textContent = `${askPercent}%`;
            this.ratioFillElement.style.width = `${bidPercent}%`;
        }

        // Render asks (reversed so lowest price is at bottom)
        // NOTE: reverse() mutates the array!
        this._renderSide(this.asksContainer, sortedAsks.reverse(), 'ask', maxTotal);

        // Render bids
        this._renderSide(this.bidsContainer, sortedBids, 'bid', maxTotal);

        // Update spread with clean values
        this._renderSpread(bestBid, bestAsk);

        // Save for flash animation
        this.previousBids = new Map(sortedBids);
        this.previousAsks = new Map(sortedAsks);
    }

    /**
     * Render one side of the order book
     * @param {HTMLElement} container
     * @param {Array} orders
     * @param {string} side - 'bid' or 'ask'
     * @param {number} maxTotal
     */
    _renderSide(container, orders, side, maxTotal) {
        let html = '';
        let cumulative = 0;
        const previousMap = side === 'bid' ? this.previousBids : this.previousAsks;

        for (const [price, quantity] of orders) {
            cumulative += quantity;
            const depthWidth = (cumulative / maxTotal) * 100;
            const isLargeOrder = quantity >= this.largeOrderThreshold;

            // Check for flash animation
            const prevQty = previousMap.get(price);
            let flashClass = '';
            if (prevQty !== undefined && quantity !== prevQty) {
                flashClass = quantity > prevQty ? 'flash-buy' : 'flash-sell';
            }

            html += `
        <div class="orderbook-row ${side} ${flashClass} ${isLargeOrder ? 'large-order' : ''}">
          <span class="depth-bar" style="width: ${depthWidth}%"></span>
          <span class="price">${this._formatPrice(price)}</span>
          <span class="size">${this._formatQuantity(quantity)}</span>
          <span class="total">${this._formatQuantity(cumulative)}</span>
        </div>
      `;
        }

        container.innerHTML = html;
    }

    /**
     * Render spread display
     * @param {number} bestBid
     * @param {number} bestAsk
     */
    _renderSpread(bestBid, bestAsk) {
        if (!bestBid || !bestAsk) return;
        const spread = bestAsk - bestBid;
        const spreadPercent = bestBid > 0 ? ((spread / bestBid) * 100).toFixed(4) : 0;

        const spreadPriceEl = this.spreadElement.querySelector('.spread-price');
        const spreadValueEl = this.spreadElement.querySelector('.spread-value');

        if (spreadPriceEl && this.lastPrice) {
            spreadPriceEl.textContent = this._formatPrice(this.lastPrice);
        }

        if (spreadValueEl) {
            spreadValueEl.textContent = `Spread: ${this._formatPrice(spread)} (${spreadPercent}%)`;
        }
    }

    /**
     * Format price
     * @param {number} price
     * @returns {string}
     */
    _formatPrice(price) {
        if (price >= 10000) {
            return price.toFixed(this.precision > 0 ? Math.min(this.precision, 1) : 0);
        } else if (price >= 100) {
            return price.toFixed(Math.min(this.precision, 2));
        } else if (price >= 1) {
            return price.toFixed(Math.min(this.precision, 4));
        } else {
            return price.toFixed(Math.min(this.precision, 6));
        }
    }

    /**
     * Format quantity
     * @param {number} qty
     * @returns {string}
     */
    _formatQuantity(qty) {
        if (qty >= 1000000) {
            return (qty / 1000000).toFixed(2) + 'M';
        } else if (qty >= 1000) {
            return (qty / 1000).toFixed(2) + 'K';
        } else if (qty >= 1) {
            return qty.toFixed(3);
        } else {
            return qty.toFixed(5);
        }
    }

    /**
     * Clear order book
     */
    clear() {
        this.bids.clear();
        this.asks.clear();
        this.asksContainer.innerHTML = '';
        this.bidsContainer.innerHTML = '';
    }
}
