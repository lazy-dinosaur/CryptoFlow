/**
 * Volume Profile Component
 * Displays cumulative volume at each price level
 */

export class VolumeProfile {
    constructor(containerId, canvasId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Settings
        this.barHeight = 3;
        this.padding = { top: 10, right: 5, bottom: 10, left: 5 };

        // Data
        this.data = null;
        this.priceRange = null;

        // Colors
        this.colors = {
            background: '#1a2332',
            buy: '#10b981',
            sell: '#ef4444',
            poc: '#fbbf24',
            valueArea: 'rgba(59, 130, 246, 0.3)',
            text: '#94a3b8'
        };

        // Bind methods
        this._handleResize = this._handleResize.bind(this);

        // Initialize
        this._init();
    }

    /**
     * Initialize the component
     */
    _init() {
        this._setupCanvas();
        window.addEventListener('resize', this._handleResize);
        this._startRenderLoop();
    }

    /**
     * Setup canvas dimensions
     */
    _setupCanvas() {
        const rect = this.container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        // Reset transform before scaling to avoid compounding on resize
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);

        this.width = rect.width;
        this.height = rect.height;
    }

    /**
     * Handle resize
     */
    _handleResize() {
        this._setupCanvas();
    }

    /**
     * Start render loop
     */
    _startRenderLoop() {
        const render = () => {
            this._render();
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    /**
     * Update volume profile data
     * @param {Object} data - Volume profile data from aggregator
     * @param {Object} priceRange - Current price range from footprint chart
     */
    update(data, priceRange) {
        this.data = data;
        this.priceRange = priceRange;
    }

    /**
     * Main render function
     */
    _render() {
        const ctx = this.ctx;

        // Clear
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, this.width, this.height);

        if (!this.data || !this.data.levels || this.data.levels.length === 0 || !this.priceRange) {
            this._renderEmptyState();
            return;
        }

        // Draw value area background
        this._renderValueArea();

        // Draw volume bars
        this._renderVolumeBars();

        // Draw POC line
        this._renderPOC();
    }

    /**
     * Render empty state
     */
    _renderEmptyState() {
        const ctx = this.ctx;
        ctx.fillStyle = this.colors.text;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.save();
        ctx.translate(this.width / 2, this.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Volume Profile', 0, 0);
        ctx.restore();
    }

    /**
     * Render value area background
     */
    _renderValueArea() {
        if (!this.data.vah || !this.data.val) return;

        const ctx = this.ctx;
        const vahY = this._priceToY(this.data.vah);
        const valY = this._priceToY(this.data.val);

        ctx.fillStyle = this.colors.valueArea;
        ctx.fillRect(0, Math.min(vahY, valY), this.width, Math.abs(valY - vahY));
    }

    /**
     * Render volume bars
     */
    _renderVolumeBars() {
        const ctx = this.ctx;
        const maxWidth = this.width - this.padding.left - this.padding.right;
        const maxVolume = this.data.levels.reduce((max, l) => Math.max(max, l.total), 0);

        for (const level of this.data.levels) {
            const y = this._priceToY(level.price);

            // Skip if out of view
            if (y < 0 || y > this.height) continue;

            const totalWidth = (level.total / maxVolume) * maxWidth;
            const buyWidth = level.total > 0 ? (level.buy / level.total) * totalWidth : 0;
            const sellWidth = totalWidth - buyWidth;

            // Draw from right to left
            const startX = this.width - this.padding.right;

            // Sell volume (red)
            if (sellWidth > 0) {
                ctx.fillStyle = this.colors.sell;
                ctx.fillRect(startX - sellWidth, y - this.barHeight / 2, sellWidth, this.barHeight);
            }

            // Buy volume (green)
            if (buyWidth > 0) {
                ctx.fillStyle = this.colors.buy;
                ctx.fillRect(startX - sellWidth - buyWidth, y - this.barHeight / 2, buyWidth, this.barHeight);
            }
        }
    }

    /**
     * Render POC (Point of Control)
     */
    _renderPOC() {
        if (!this.data.poc) return;

        const ctx = this.ctx;
        const y = this._priceToY(this.data.poc);

        // POC line
        ctx.strokeStyle = this.colors.poc;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(this.width, y);
        ctx.stroke();

        // POC marker
        ctx.fillStyle = this.colors.poc;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(8, y - 4);
        ctx.lineTo(8, y + 4);
        ctx.closePath();
        ctx.fill();
    }

    /**
     * Convert price to Y coordinate
     * @param {number} price
     * @returns {number}
     */
    _priceToY(price) {
        if (!this.priceRange) return 0;

        const { minPrice, maxPrice } = this.priceRange;
        const chartHeight = this.height - 50; // Match footprint chart
        const ratio = (price - minPrice) / (maxPrice - minPrice);
        return chartHeight - (ratio * chartHeight);
    }

    /**
     * Cleanup
     */
    destroy() {
        window.removeEventListener('resize', this._handleResize);
    }
}
