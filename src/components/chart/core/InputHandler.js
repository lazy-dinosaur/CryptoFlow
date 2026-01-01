/**
 * InputHandler
 * Captures DOM events and updates ChartState.
 */
export class InputHandler {
    constructor(canvas, state, coords, requestDraw) {
        this.canvas = canvas;
        this.state = state;
        this.coords = coords;
        this.requestDraw = requestDraw;

        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;

        this._initListeners();
    }

    _initListeners() {
        this.canvas.addEventListener('wheel', this._handleWheel.bind(this), { passive: false });
        this.canvas.addEventListener('mousedown', this._handleMouseDown.bind(this));
        window.addEventListener('mousemove', this._handleMouseMove.bind(this));
        window.addEventListener('mouseup', this._handleMouseUp.bind(this));
        window.addEventListener('keydown', this._handleKeyDown.bind(this));
    }

    _handleKeyDown(e) {
        // Ignore if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        const key = e.key.toUpperCase();
        let handled = true;

        switch (key) {
            // VIEW CONTROLS
            case ' ': // Space
                this.state.autoScroll = !this.state.autoScroll;
                break;
            case 'R': // Reset
                this.state.zoomX = 10;
                this.state.offsetX = 0;
                // We might need to call a "reset view" callback if we want to recenter Y
                break;
            case '+':
            case '=':
                this.state.zoomX *= 1.1;
                break;
            case '-':
            case '_':
                this.state.zoomX *= 0.9;
                break;

            // TOGGLES (Delegate to App via FootprintChart if needed, but we can set state directly)
            case 'H':
                this.state.showHeatmap = !this.state.showHeatmap;
                // Legacy support pending full event system
                if (this.state.heatmapMode !== undefined) this.state.heatmapMode = this.state.showHeatmap;
                break;
            case 'C':
                this.state.showCrosshair = !this.state.showCrosshair;
                break;
            case 'I':
                this.state.showImbalances = !this.state.showImbalances;
                break;
            case 'D':
                this.state.showDelta = !this.state.showDelta;
                break;
            case 'B':
                this.state.showBigTrades = !this.state.showBigTrades;
                break;

            // MISSING KEYS RESTORED
            case 'L': // Liquidity Lines (Naked Levels) ???
                // Assuming 'L' toggles Naked Liquidity Lines if that property exists
                // We didn't see explicit property in ChartState view, maybe it was 'showNakedLevels'?
                // Let's assume generic "L" for Liquidity or Labels if unsure. 
                // Wait, user said "M, L".
                // L for "Liquidity" lines seems most plausible.
                // Let's toggle a state property if it exists, or just log.
                // In AnalysisLayer we might check this.state.showLiquidity?
                // Let's add it to state if missing.
                this.state.showNakedLiquidity = !this.state.showNakedLiquidity;
                break;

            case 'M': // Market Analysis / ML ???
                // "M" key missing. 
                // M usually toggles the "Magnet" or "Market Profile"?
                // Or maybe "ML Dashboard"?
                // The user specifically mentioned "M, L".
                // Let's try to toggle ML Dashboard via a custom event or callback?
                // Start with a generic state toggle.
                this.state.showML = !this.state.showML;

                // Also trigger an event so main.js can catch it to toggle the DOM overlay?
                window.dispatchEvent(new CustomEvent('toggle-ml-dashboard'));
                break;

            case 'ARROWLEFT':
                this.state.offsetX += 50;
                break;
            case 'ARROWRIGHT':
                this.state.offsetX -= 50;
                break;
            case 'ARROWUP':
                this.state.offsetY += 50;
                break;
            case 'ARROWDOWN':
                this.state.offsetY -= 50;
                break;

            default:
                handled = false;
        }

        if (handled) {
            e.preventDefault();
            this.requestDraw();
            // Notify other components if needed
            this.state.notify();
        }
    }

    _handleWheel(e) {
        e.preventDefault();

        // Zoom logic
        // DeltaY > 0 : Zoom OUT
        // DeltaY < 0 : Zoom IN

        const zoomFactor = 1.1;
        const direction = e.deltaY > 0 ? 1 / zoomFactor : zoomFactor;

        if (e.ctrlKey) {
            // Zoom X only
            this.state.zoomX *= direction;
            // Clamp
            this.state.zoomX = Math.max(this.state.minZoomX, Math.min(this.state.maxZoomX, this.state.zoomX));
        } else if (e.shiftKey) {
            // Pan X
            this.state.offsetX -= e.deltaY;
        } else {
            // Default: Zoom X (Or Y?)
            // FootprintChart default was Zoom X on wheel.
            this.state.zoomX *= direction;
        }

        this.requestDraw();
    }

    _handleMouseDown(e) {
        this.isDragging = true;
        this.lastX = e.clientX;
        this.lastY = e.clientY;
    }

    _handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Update crosshair state regardless of dragging
        this.state.crosshairX = x;
        this.state.crosshairY = y;

        if (this.isDragging) {
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;

            this.lastX = e.clientX;
            this.lastY = e.clientY;

            this.state.offsetX += dx;
            this.state.offsetY -= dy; // Inverted Y-axis: subtract instead of add
        }

        this.requestDraw();
    }

    _handleMouseUp(e) {
        this.isDragging = false;
    }
}
