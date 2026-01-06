/**
 * Drawing Manager
 * Handles all chart drawings (horizontal lines, trendlines, rectangles, fibonacci)
 * Persists to localStorage
 */

class DrawingManager {
    constructor() {
        this.drawings = [];
        this.selectedDrawing = null;
        this.activeTool = 'select'; // select, horizontal, trendline, rectangle, fibonacci
        this.pendingDrawing = null;
        this.listeners = {};

        this._loadFromStorage();
    }

    // Event System
    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => cb(data));
        }
    }

    // Tool Management
    setActiveTool(tool) {
        this.activeTool = tool;
        this.pendingDrawing = null;
        this.emit('toolChange', tool);
    }

    getActiveTool() {
        return this.activeTool;
    }

    // Drawing CRUD
    addDrawing(drawing) {
        const newDrawing = {
            id: this._generateId(),
            createdAt: Date.now(),
            ...drawing
        };
        this.drawings.push(newDrawing);
        this._saveToStorage();
        this.emit('drawingsChange', this.drawings);
        return newDrawing;
    }

    updateDrawing(id, updates) {
        const idx = this.drawings.findIndex(d => d.id === id);
        if (idx !== -1) {
            this.drawings[idx] = { ...this.drawings[idx], ...updates };
            this._saveToStorage();
            this.emit('drawingsChange', this.drawings);
        }
    }

    deleteDrawing(id) {
        this.drawings = this.drawings.filter(d => d.id !== id);
        if (this.selectedDrawing?.id === id) {
            this.selectedDrawing = null;
        }
        this._saveToStorage();
        this.emit('drawingsChange', this.drawings);
    }

    deleteAllDrawings(symbol = null) {
        if (symbol) {
            this.drawings = this.drawings.filter(d => d.symbol !== symbol);
        } else {
            this.drawings = [];
        }
        this.selectedDrawing = null;
        this._saveToStorage();
        this.emit('drawingsChange', this.drawings);
    }

    getDrawings(symbol = null, timeframe = null) {
        return this.drawings.filter(d => {
            if (symbol && d.symbol !== symbol) return false;
            // Drawings are shown across timeframes by default
            return true;
        });
    }

    getDrawingById(id) {
        return this.drawings.find(d => d.id === id);
    }

    // Selection
    selectDrawing(id) {
        this.selectedDrawing = this.getDrawingById(id) || null;
        this.emit('selectionChange', this.selectedDrawing);
    }

    clearSelection() {
        this.selectedDrawing = null;
        this.emit('selectionChange', null);
    }

    // Pending Drawing (for multi-click drawings)
    startPendingDrawing(type, point) {
        this.pendingDrawing = {
            type,
            points: [point],
            color: this._getDefaultColor(type)
        };
        this.emit('pendingChange', this.pendingDrawing);
    }

    updatePendingDrawing(point) {
        if (this.pendingDrawing) {
            if (this.pendingDrawing.points.length === 1) {
                this.pendingDrawing.points.push(point);
            } else {
                this.pendingDrawing.points[1] = point;
            }
            this.emit('pendingChange', this.pendingDrawing);
        }
    }

    finalizePendingDrawing(symbol) {
        if (this.pendingDrawing) {
            const drawing = this.addDrawing({
                ...this.pendingDrawing,
                symbol
            });
            this.pendingDrawing = null;
            this.emit('pendingChange', null);
            return drawing;
        }
        return null;
    }

    cancelPendingDrawing() {
        this.pendingDrawing = null;
        this.emit('pendingChange', null);
    }

    // Storage
    _saveToStorage() {
        try {
            localStorage.setItem('cryptoflow_drawings', JSON.stringify(this.drawings));
        } catch (e) {
            console.warn('Failed to save drawings:', e);
        }
    }

    _loadFromStorage() {
        try {
            const saved = localStorage.getItem('cryptoflow_drawings');
            if (saved) {
                this.drawings = JSON.parse(saved);
            }
        } catch (e) {
            console.warn('Failed to load drawings:', e);
            this.drawings = [];
        }
    }

    // Helpers
    _generateId() {
        return 'draw_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    _getDefaultColor(type) {
        const colors = {
            horizontal: '#ffd700',
            trendline: '#00e676',
            rectangle: '#2196f3',
            fibonacci: '#e91e63'
        };
        return colors[type] || '#ffffff';
    }
}

export const drawingManager = new DrawingManager();
