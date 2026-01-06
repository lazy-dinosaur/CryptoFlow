/**
 * Replay Manager
 * Allows playback of historical candle data for practice trading
 */

class ReplayManager {
    constructor() {
        this.isActive = false;
        this.isPaused = true;
        this.speed = 1; // 0.5x, 1x, 2x, 5x
        this.currentIndex = 0;
        this.fullHistory = []; // All candles
        this.visibleHistory = []; // Candles shown so far
        this.playInterval = null;
        this.listeners = {};
    }

    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => cb(data));
        }
    }

    /**
     * Start replay mode with given candles
     */
    start(candles, startIndex = 50) {
        if (!candles || candles.length < 50) {
            console.warn('Not enough candles for replay');
            return false;
        }

        this.fullHistory = [...candles];
        this.currentIndex = Math.min(startIndex, candles.length - 1);
        this.visibleHistory = this.fullHistory.slice(0, this.currentIndex + 1);
        this.isActive = true;
        this.isPaused = true;

        this.emit('start', {
            total: this.fullHistory.length,
            current: this.currentIndex
        });
        this.emit('tick', this.visibleHistory);

        return true;
    }

    /**
     * Stop replay and return to live
     */
    stop() {
        this._clearInterval();
        this.isActive = false;
        this.isPaused = true;
        this.currentIndex = 0;
        this.fullHistory = [];
        this.visibleHistory = [];
        this.emit('stop');
    }

    /**
     * Play/Resume
     */
    play() {
        if (!this.isActive) return;
        this.isPaused = false;
        this._startInterval();
        this.emit('play');
    }

    /**
     * Pause
     */
    pause() {
        this.isPaused = true;
        this._clearInterval();
        this.emit('pause');
    }

    /**
     * Toggle play/pause
     */
    toggle() {
        if (this.isPaused) {
            this.play();
        } else {
            this.pause();
        }
    }

    /**
     * Set playback speed (0.5, 1, 2, 5)
     */
    setSpeed(speed) {
        this.speed = speed;
        if (!this.isPaused) {
            this._clearInterval();
            this._startInterval();
        }
        this.emit('speedChange', speed);
    }

    /**
     * Jump to specific index
     */
    seekTo(index) {
        if (!this.isActive) return;
        this.currentIndex = Math.max(0, Math.min(index, this.fullHistory.length - 1));
        this.visibleHistory = this.fullHistory.slice(0, this.currentIndex + 1);
        this.emit('tick', this.visibleHistory);
        this.emit('seek', this.currentIndex);
    }

    /**
     * Step forward one candle
     */
    stepForward() {
        if (!this.isActive || this.currentIndex >= this.fullHistory.length - 1) return;
        this.currentIndex++;
        this.visibleHistory.push(this.fullHistory[this.currentIndex]);
        this.emit('tick', this.visibleHistory);
    }

    /**
     * Get progress (0-1)
     */
    getProgress() {
        if (this.fullHistory.length === 0) return 0;
        return this.currentIndex / (this.fullHistory.length - 1);
    }

    _startInterval() {
        this._clearInterval();
        const baseMs = 1000; // 1 second per candle at 1x
        const intervalMs = baseMs / this.speed;

        this.playInterval = setInterval(() => {
            if (this.currentIndex >= this.fullHistory.length - 1) {
                this.pause();
                this.emit('end');
                return;
            }
            this.stepForward();
        }, intervalMs);
    }

    _clearInterval() {
        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }
}

export const replayManager = new ReplayManager();
