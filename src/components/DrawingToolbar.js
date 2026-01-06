/**
 * Drawing Toolbar Component
 * Floating toolbar for chart drawing tools
 */

export class DrawingToolbar {
    constructor(drawingManager) {
        this.manager = drawingManager;
        this.container = null;
        this._init();
    }

    _init() {
        this.container = document.createElement('div');
        this.container.id = 'drawing-toolbar';
        this.container.innerHTML = `
            <div class="drawing-toolbar">
                <button class="dt-btn active" data-tool="select" title="Select (V)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M7 2l12 11.2-5.8.5 3.3 7.3-2.2 1-3.2-7.4L7 18.5V2z"/>
                    </svg>
                </button>
                <button class="dt-btn" data-tool="horizontal" title="Horizontal Line (H)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="3" y1="12" x2="21" y2="12"/>
                    </svg>
                </button>
                <button class="dt-btn" data-tool="trendline" title="Trendline (T)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="3" y1="18" x2="21" y2="6"/>
                    </svg>
                </button>
                <button class="dt-btn" data-tool="rectangle" title="Rectangle (R)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="6" width="18" height="12" rx="1"/>
                    </svg>
                </button>
                <button class="dt-btn" data-tool="fibonacci" title="Fibonacci (F)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="3" y1="4" x2="21" y2="4"/>
                        <line x1="3" y1="9" x2="21" y2="9"/>
                        <line x1="3" y1="13" x2="21" y2="13"/>
                        <line x1="3" y1="16" x2="21" y2="16"/>
                        <line x1="3" y1="20" x2="21" y2="20"/>
                    </svg>
                </button>
                <div class="dt-divider"></div>
                <button class="dt-btn dt-delete" data-action="deleteAll" title="Delete All">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"/>
                    </svg>
                </button>
                <div class="dt-divider"></div>
                <button class="dt-btn dt-replay" data-action="replay" title="Replay Mode (P)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </button>
            </div>
        `;

        this._addStyles();
        this._setupEventListeners();
        document.body.appendChild(this.container);

        // Listen for tool changes
        this.manager.on('toolChange', (tool) => this._updateActiveButton(tool));
    }

    _addStyles() {
        if (document.getElementById('drawing-toolbar-styles')) return;

        const style = document.createElement('style');
        style.id = 'drawing-toolbar-styles';
        style.textContent = `
            #drawing-toolbar {
                position: fixed;
                left: 10px;
                top: 50%;
                transform: translateY(-50%);
                z-index: 1000;
            }
            
            .drawing-toolbar {
                display: flex;
                flex-direction: column;
                gap: 4px;
                background: rgba(20, 25, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px;
                backdrop-filter: blur(10px);
            }
            
            .dt-btn {
                width: 36px;
                height: 36px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                color: #888;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            
            .dt-btn:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #fff;
            }
            
            .dt-btn.active {
                background: rgba(0, 230, 118, 0.2);
                border-color: rgba(0, 230, 118, 0.5);
                color: #00e676;
            }
            
            .dt-divider {
                height: 1px;
                background: rgba(255, 255, 255, 0.1);
                margin: 4px 0;
            }
            
            .dt-btn.dt-delete:hover {
                background: rgba(239, 68, 68, 0.2);
                color: #ef4444;
            }
        `;
        document.head.appendChild(style);
    }

    _setupEventListeners() {
        this.container.querySelectorAll('.dt-btn[data-tool]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tool = btn.dataset.tool;
                this.manager.setActiveTool(tool);
            });
        });

        this.container.querySelector('[data-action="deleteAll"]')?.addEventListener('click', () => {
            if (confirm('Delete all drawings?')) {
                this.manager.deleteAllDrawings();
            }
        });

        // Replay button
        this.container.querySelector('[data-action="replay"]')?.addEventListener('click', () => {
            window.dispatchEvent(new CustomEvent('start-replay'));
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            const shortcuts = {
                'v': 'select',
                'h': 'horizontal',
                't': 'trendline',
                'r': 'rectangle',
                'f': 'fibonacci'
            };

            if (shortcuts[e.key.toLowerCase()]) {
                e.preventDefault();
                this.manager.setActiveTool(shortcuts[e.key.toLowerCase()]);
            }

            // Replay shortcut (P key)
            if (e.key.toLowerCase() === 'p') {
                e.preventDefault();
                window.dispatchEvent(new CustomEvent('start-replay'));
            }

            // Delete selected drawing
            if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.manager.selectedDrawing) {
                    this.manager.deleteDrawing(this.manager.selectedDrawing.id);
                }
            }

            // Escape to cancel pending
            if (e.key === 'Escape') {
                this.manager.cancelPendingDrawing();
                this.manager.setActiveTool('select');
            }
        });
    }

    _updateActiveButton(tool) {
        this.container.querySelectorAll('.dt-btn[data-tool]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tool === tool);
        });
    }
}
