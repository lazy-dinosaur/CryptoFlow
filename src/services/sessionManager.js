/**
 * Session Manager
 * Handles automatic session resets (e.g. at 00:00 UTC)
 */

class SessionManager {
    constructor(dataAggregator) {
        this.dataAggregator = dataAggregator;
        this.checkInterval = null;
        this.lastResetDate = new Date().getUTCDate(); // Track day change
        this.enabled = true;
    }

    /**
     * Start the session monitoring
     */
    start() {
        if (this.checkInterval) return;


        // Check every minute
        this.checkInterval = setInterval(() => {
            this._checkSession();
        }, 60 * 1000);
    }

    /**
     * Stop monitoring
     */
    stop() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
            this.checkInterval = null;
        }
    }

    /**
     * Enable/Disable auto reset
     */
    setEnabled(enabled) {
        this.enabled = enabled;
    }

    /**
     * Check if a new session started (day changed in UTC)
     */
    _checkSession() {
        if (!this.enabled) return;

        const now = new Date();
        const currentDay = now.getUTCDate();

        // If day changed
        if (currentDay !== this.lastResetDate) {

            // Trigger reset
            this.dataAggregator.reset();

            // Update reference
            this.lastResetDate = currentDay;
        }
    }
}

export const sessionManager = (aggregator) => new SessionManager(aggregator);
