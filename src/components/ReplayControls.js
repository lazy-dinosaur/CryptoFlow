/**
 * Replay Controls Component
 * Bottom bar UI for replay mode
 */

export class ReplayControls {
    constructor(replayManager, dataAggregator) {
        this.manager = replayManager;
        this.dataAggregator = dataAggregator;
        this.container = null;
        this.onCandleUpdate = null; // Callback to update chart
        this._init();
    }

    setOnCandleUpdate(callback) {
        this.onCandleUpdate = callback;
    }

    _init() {
        this.container = document.createElement('div');
        this.container.id = 'replay-controls';
        this.container.style.display = 'none';
        this.container.innerHTML = `
            <div class="replay-bar">
                <button class="replay-btn replay-play" id="replayPlayBtn" title="Play/Pause (Space)">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </button>
                <div class="replay-slider-container">
                    <input type="range" id="replaySlider" class="replay-slider" min="0" max="100" value="0">
                    <span class="replay-progress" id="replayProgress">0 / 0</span>
                </div>
                <div class="replay-speed">
                    <button class="replay-speed-btn" data-speed="0.5">0.5x</button>
                    <button class="replay-speed-btn active" data-speed="1">1x</button>
                    <button class="replay-speed-btn" data-speed="2">2x</button>
                    <button class="replay-speed-btn" data-speed="5">5x</button>
                </div>
                <button class="replay-btn replay-exit" id="replayExitBtn" title="Exit Replay">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            </div>
        `;

        this._addStyles();
        this._setupEventListeners();
        document.body.appendChild(this.container);

        // Listen to replay events
        this.manager.on('start', () => this.show());
        this.manager.on('stop', () => this.hide());
        this.manager.on('play', () => this._updatePlayButton(true));
        this.manager.on('pause', () => this._updatePlayButton(false));
        this.manager.on('tick', (candles) => this._onTick(candles));
        this.manager.on('speedChange', (speed) => this._updateSpeedButtons(speed));
    }

    _addStyles() {
        if (document.getElementById('replay-controls-styles')) return;

        const style = document.createElement('style');
        style.id = 'replay-controls-styles';
        style.textContent = `
            #replay-controls {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                z-index: 1000;
            }
            
            .replay-bar {
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(10, 15, 20, 0.95);
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                padding: 10px 20px;
                backdrop-filter: blur(10px);
            }
            
            .replay-btn {
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 50%;
                color: #fff;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            
            .replay-btn:hover {
                background: rgba(0, 230, 118, 0.3);
            }
            
            .replay-play.playing {
                background: rgba(0, 230, 118, 0.3);
            }
            
            .replay-exit:hover {
                background: rgba(239, 68, 68, 0.3);
            }
            
            .replay-slider-container {
                flex: 1;
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .replay-slider {
                flex: 1;
                height: 6px;
                -webkit-appearance: none;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                outline: none;
            }
            
            .replay-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background: #00e676;
                border-radius: 50%;
                cursor: pointer;
            }
            
            .replay-progress {
                color: #888;
                font-size: 12px;
                min-width: 80px;
            }
            
            .replay-speed {
                display: flex;
                gap: 4px;
            }
            
            .replay-speed-btn {
                padding: 6px 10px;
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 4px;
                color: #888;
                font-size: 11px;
                cursor: pointer;
                transition: all 0.15s ease;
            }
            
            .replay-speed-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                color: #fff;
            }
            
            .replay-speed-btn.active {
                background: rgba(0, 230, 118, 0.3);
                color: #00e676;
            }
        `;
        document.head.appendChild(style);
    }

    _setupEventListeners() {
        // Play/Pause
        this.container.querySelector('#replayPlayBtn').addEventListener('click', () => {
            this.manager.toggle();
        });

        // Slider
        const slider = this.container.querySelector('#replaySlider');
        slider.addEventListener('input', (e) => {
            const progress = parseInt(e.target.value) / 100;
            const index = Math.floor(progress * (this.manager.fullHistory.length - 1));
            this.manager.seekTo(index);
        });

        // Speed buttons
        this.container.querySelectorAll('.replay-speed-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const speed = parseFloat(btn.dataset.speed);
                this.manager.setSpeed(speed);
            });
        });

        // Exit
        this.container.querySelector('#replayExitBtn').addEventListener('click', () => {
            this.manager.stop();
        });

        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (!this.manager.isActive) return;
            if (e.target.tagName === 'INPUT') return;

            if (e.code === 'Space') {
                e.preventDefault();
                this.manager.toggle();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                this.manager.stepForward();
            } else if (e.key === 'Escape') {
                this.manager.stop();
            }
        });
    }

    show() {
        this.container.style.display = 'block';
    }

    hide() {
        this.container.style.display = 'none';
    }

    _onTick(candles) {
        // Update slider
        const progress = this.manager.getProgress() * 100;
        this.container.querySelector('#replaySlider').value = progress;

        // Update progress text
        this.container.querySelector('#replayProgress').textContent =
            `${this.manager.currentIndex + 1} / ${this.manager.fullHistory.length}`;

        // Update chart
        if (this.onCandleUpdate) {
            this.onCandleUpdate(candles);
        }
    }

    _updatePlayButton(isPlaying) {
        const btn = this.container.querySelector('#replayPlayBtn');
        btn.classList.toggle('playing', isPlaying);

        // Change icon
        btn.innerHTML = isPlaying
            ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="4" width="4" height="16"/>
                <rect x="14" y="4" width="4" height="16"/>
               </svg>`
            : `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
               </svg>`;
    }

    _updateSpeedButtons(speed) {
        this.container.querySelectorAll('.replay-speed-btn').forEach(btn => {
            btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
        });
    }

    /**
     * Start replay with current candles
     */
    startReplay() {
        const candles = this.dataAggregator.getCandles();
        if (this.manager.start(candles, 50)) {
            // Success - emit event to pause live data
            window.dispatchEvent(new CustomEvent('replay:active', { detail: true }));
        }
    }
}
