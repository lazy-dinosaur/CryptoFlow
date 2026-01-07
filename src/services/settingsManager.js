/**
 * Settings Manager
 * Handles saving and loading user preferences to localStorage
 */

const STORAGE_KEY = 'cryptoflow_settings_v1';

const DEFAULT_SETTINGS = {
    zoomLevel: 1.0,
    soundEnabled: true,
    showCrosshair: true,
    showImbalances: true,
    showDelta: true,

    // Most users expect heatmap visible by default
    showHeatmap: true,

    showBigTrades: true,

    // UI sliders
    heatmapIntensityThreshold: 0.005, // 0.5% - lower to show more (was 0.02)
    heatmapHistoryPercent: 60,

    tickSizes: {
        btcusdt: 10,
        ethusdt: 1,
        solusdt: 0.1,
        bnbusdt: 0.1
    },
    bigTradeThresholds: {
        btcusdt: 5.0,
        ethusdt: 50.0,
        solusdt: 500.0,
        bnbusdt: 100.0
    }
};

class SettingsManager {
    constructor() {
        this.settings = this._loadSettings();
    }

    /**
     * Load settings from localStorage
     */
    _loadSettings() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
            }
        } catch (e) {
            console.error('Failed to load settings:', e);
        }
        return { ...DEFAULT_SETTINGS };
    }

    /**
     * Save current settings to localStorage
     */
    saveSettings() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(this.settings));
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
    }

    /**
     * Get a setting value
     * @param {string} key 
     */
    get(key) {
        return this.settings[key];
    }

    /**
     * Set a setting value and save
     * @param {string} key 
     * @param {*} value 
     */
    set(key, value) {
        this.settings[key] = value;
        this.saveSettings();
    }

    /**
     * Get tick size for symbol
     */
    getTickSize(symbol) {
        return this.settings.tickSizes[symbol.toLowerCase()] || 1;
    }

    /**
     * Set tick size for symbol
     */
    setTickSize(symbol, size) {
        this.settings.tickSizes[symbol.toLowerCase()] = size;
        this.saveSettings();
    }

    /**
     * Get big trade threshold for symbol
     */
    getBigTradeThreshold(symbol) {
        return this.settings.bigTradeThresholds[symbol.toLowerCase()] || 100;
    }

    /**
     * Set big trade threshold for symbol
     */
    setBigTradeThreshold(symbol, val) {
        this.settings.bigTradeThresholds[symbol.toLowerCase()] = val;
        this.saveSettings();
    }

    getHeatmapIntensityThreshold() {
        return this.settings.heatmapIntensityThreshold;
    }

    setHeatmapIntensityThreshold(val) {
        this.settings.heatmapIntensityThreshold = val;
        this.saveSettings();
    }

    getHeatmapHistoryPercent() {
        return this.settings.heatmapHistoryPercent;
    }

    setHeatmapHistoryPercent(val) {
        this.settings.heatmapHistoryPercent = val;
        this.saveSettings();
    }
}

export const settingsManager = new SettingsManager();
