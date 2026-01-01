/**
 * RenderEngine
 * Manages the drawing loop and delegates rendering to layers.
 */
export class RenderEngine {
    constructor(canvas, state, coordinateSystem) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false }); // Optimize for speed
        this.ctx.imageSmoothingEnabled = false;

        this.state = state;
        this.coords = coordinateSystem;
        this.layers = [];

        this.animationFrameId = null;
        this.isDirty = true;
    }

    addLayer(layer) {
        this.layers.push(layer);
        // Sort by z-index if layer has it? For now order of addition matters.
    }

    /**
     * Marks the chart as needing a redraw.
     */
    requestDraw() {
        this.isDirty = true;
        if (!this.animationFrameId) {
            this.animationFrameId = requestAnimationFrame(() => this.draw());
        }
    }

    draw() {
        this.animationFrameId = null;

        if (!this.isDirty) return;
        this.isDirty = false;

        const { width, height, pixelRatio } = this.state;

        // Resize if needed
        const displayWidth = Math.floor(width * pixelRatio);
        const displayHeight = Math.floor(height * pixelRatio);

        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.ctx.scale(pixelRatio, pixelRatio);
        }

        // Clear canvas
        this.ctx.fillStyle = '#121212'; // Base background
        this.ctx.fillRect(0, 0, width, height);

        // Render layers
        this.ctx.save();
        // Global clip or transform could go here

        for (const layer of this.layers) {
            this.ctx.save();
            layer.render(this.ctx, this.state, this.coords);
            this.ctx.restore();
        }

        this.ctx.restore();
    }
}
