/**
 * CryptoFlow - Order Flow Analysis Platform
 * Main application entry point
 */

import { binanceWS } from './services/binanceWS.js';
import { dataAggregator } from './services/dataAggregator.js';
import { fetchTradesForPeriod } from './services/binanceREST.js';
import { settingsManager } from './services/settingsManager.js';
import { audioService } from './services/audioService.js';
import { sessionManager } from './services/sessionManager.js';
import { depthHeatmap } from './services/depthHeatmap.js';
import { vpsAPI } from './services/vpsAPI.js';
import { FootprintChart } from './components/FootprintChart.js';
import { VolumeProfile } from './components/VolumeProfile.js';
import { OrderBook } from './components/OrderBook.js';
import { MLDashboard } from './components/MLDashboard.js';


class CryptoFlowApp {
    constructor() {
        // State
        this.currentSymbol = 'btcusdt';
        this.currentTimeframe = 1; // minutes
        this.currentPrice = null;
        this.priceChangePercent = 0;
        this.isLoadingHistory = false;

        // Components
        this.footprintChart = null;
        this.volumeProfile = null;
        this.orderBook = null;

        // DOM elements
        this.elements = {
            symbolSelect: document.getElementById('symbolSelect'),
            connectionStatus: document.getElementById('connectionStatus'),
            priceDisplay: document.getElementById('priceDisplay'),
            timeframeBtns: document.querySelectorAll('.tf-btn'),
            toggleDelta: document.getElementById('toggleDelta'),
            toggleImbalances: document.getElementById('toggleImbalances'),
            toggleCrosshair: document.getElementById('toggleCrosshair'),
            toggleHeatmap: document.getElementById('toggleHeatmap'),
            toggleBigTrades: document.getElementById('toggleBigTrades'),
            toggleSound: document.getElementById('toggleSound'), // NEW
            tickSizeSelect: document.getElementById('tickSizeSelect'),
            volume24h: document.getElementById('volume24h'),
            cvdValue: document.getElementById('cvdValue'),
            deltaValue: document.getElementById('deltaValue'),
            tradesPerSec: document.getElementById('tradesPerSec'),
            pocValue: document.getElementById('pocValue'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.querySelector('.loading-text'),
            helpOverlay: document.getElementById('helpOverlay'),
            helpClose: document.getElementById('helpClose'),
            toggleML: document.getElementById('toggleML')
        };

        // Tick sizes for different symbols
        this.tickSizes = {
            btcusdt: 10,
            ethusdt: 1,
            solusdt: 0.1,
            bnbusdt: 0.1
        };

        // Big trade thresholds (whale markers)
        this.bigTradeThresholds = {
            btcusdt: 20.0,   // 20 BTC = whale (User Default)
            ethusdt: 50.0,   // 50 ETH = whale
            solusdt: 500.0,  // 500 SOL = whale
            bnbusdt: 100.0   // 100 BNB = whale
        };

        // Historical data settings
        this.historyMinutes = 10; // Load 10 minutes of history

        // Initialize Services
        sessionManager(dataAggregator).start();

        // Initialize
        this._init();
    }

    /**
     * Initialize the application
     */
    _init() {

        // Initialize components with saved settings
        this._initComponents();

        // Setup event listeners
        this._setupEventListeners();

        // Setup WebSocket handlers
        this._setupWebSocket();

        // Start with default symbol (this will load history)
        this._switchSymbol(this.currentSymbol);
    }

    /**
     * Initialize UI components
     */
    _initComponents() {
        // Footprint Chart
        // Footprint Chart
        this.footprintChart = new FootprintChart({
            containerId: 'footprintContainer',
            width: window.innerWidth - 320, // Subtract sidebar width
            height: window.innerHeight
        });

        this.depthHeatmap = depthHeatmap;

        // Apply saved tick size
        const savedTickSize = settingsManager.getTickSize(this.currentSymbol) || this.tickSizes[this.currentSymbol];
        this.footprintChart.setTickSize(savedTickSize);

        // Apply default heatmap settings
        this.footprintChart.setHeatmapIntensityThreshold(0.05);
        this.footprintChart.setHeatmapHistoryPercent(60);

        // Apply saved zoom
        const savedZoom = settingsManager.get('zoomLevel');
        if (savedZoom) this.footprintChart.setZoom(savedZoom);

        // Update settings when zoom changes
        this.footprintChart.onZoomChange = (level) => {
            settingsManager.set('zoomLevel', level);
        };

        // Apply toggles from settings
        const showDelta = settingsManager.get('showDelta') !== false;
        this.footprintChart.showDelta = showDelta;
        this.elements.toggleDelta.classList.toggle('active', showDelta);

        const showImbalances = settingsManager.get('showImbalances') !== false;
        this.footprintChart.showImbalances = showImbalances;
        this.elements.toggleImbalances.classList.toggle('active', showImbalances);

        const showCrosshair = settingsManager.get('showCrosshair') !== false;
        this.footprintChart.showCrosshair = showCrosshair;
        this.elements.toggleCrosshair.classList.toggle('active', showCrosshair);

        const showHeatmap = settingsManager.get('showHeatmap') === true;
        this.footprintChart.showHeatmap = showHeatmap;
        this.elements.toggleHeatmap.classList.toggle('active', showHeatmap);

        const showBigTrades = settingsManager.get('showBigTrades') !== false;
        this.footprintChart.showBigTrades = showBigTrades;
        this.elements.toggleBigTrades.classList.toggle('active', showBigTrades);

        // Sound State
        const soundEnabled = settingsManager.get('soundEnabled') !== false; // Default true
        audioService.setEnabled(soundEnabled);
        if (this.elements.toggleSound) this.elements.toggleSound.classList.toggle('active', soundEnabled);

        // Volume Profile
        this.volumeProfile = new VolumeProfile('volumeProfileContainer', 'volumeProfileCanvas');

        // Order Book
        this.orderBook = new OrderBook({
            asksId: 'orderbookAsks',
            bidsId: 'orderbookBids',
            spreadId: 'orderbookSpread',
            bidRatioId: 'bidRatio',
            askRatioId: 'askRatio',
            ratioFillId: 'ratioFill',
            levels: 15,
            precision: 2,
            largeOrderThreshold: 5
        });

        // ML Dashboard
        this.mlDashboard = new MLDashboard('mlDashboard');

    }

    /**
     * Setup DOM event listeners
     */
    _setupEventListeners() {
        // Symbol selector
        this.elements.symbolSelect.addEventListener('change', (e) => {
            this._switchSymbol(e.target.value);
        });

        // Timeframe buttons
        this.elements.timeframeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this._switchTimeframe(parseInt(btn.dataset.tf, 10));
                this.elements.timeframeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Toggle buttons
        this.elements.toggleDelta.addEventListener('click', () => {
            const active = this.footprintChart.toggleDelta();
            this.elements.toggleDelta.classList.toggle('active', active);
            settingsManager.set('showDelta', active);
        });

        this.elements.toggleImbalances.addEventListener('click', () => {
            const active = this.footprintChart.toggleImbalances();
            this.elements.toggleImbalances.classList.toggle('active', active);
            settingsManager.set('showImbalances', active);
        });

        this.elements.toggleCrosshair.addEventListener('click', () => {
            const active = this.footprintChart.toggleCrosshair();
            this.elements.toggleCrosshair.classList.toggle('active', active);
            settingsManager.set('showCrosshair', active);
        });

        this.elements.toggleHeatmap.addEventListener('click', () => {
            const active = this.footprintChart.toggleHeatmap();
            this.elements.toggleHeatmap.classList.toggle('active', active);
            settingsManager.set('showHeatmap', active);
        });

        this.elements.toggleBigTrades.addEventListener('click', () => {
            const active = this.footprintChart.toggleBigTrades();
            this.elements.toggleBigTrades.classList.toggle('active', active);
            settingsManager.set('showBigTrades', active);
        });

        // Toggle Sound
        this.elements.toggleSound.addEventListener('click', () => {
            const enabled = !audioService.enabled;
            audioService.setEnabled(enabled);
            this.elements.toggleSound.classList.toggle('active', enabled);
            settingsManager.set('soundEnabled', enabled);
        });



        // Listen for 'M' key event from chart
        window.addEventListener('toggle-ml-dashboard', () => {
            const btn = this.elements.toggleML;
            if (btn) {
                // Simulate click
                this.elements.toggleML.click(); // Logic not yet implemented for click? Check if toggleML logic exists
                // Wait, we haven't implemented toggleML listener loop yet in main.js view? 
                // Looking at file, I see toggleBigTrades at 212. I need to add toggleML logic.
            }
        });

        // Toggle ML Dashboard Button logic
        if (this.elements.toggleML) {
            this.elements.toggleML.addEventListener('click', () => {
                const active = this.mlDashboard.toggle();
                this.elements.toggleML.classList.toggle('active', active);
                // Enable/Disable Wall Attack on chart
                this.footprintChart.showML = active;
                this.footprintChart.requestDraw();
            });
        }


        // Whale threshold slider
        const whaleSlider = document.getElementById('whaleThresholdSlider');
        const whaleValue = document.getElementById('whaleThresholdValue');
        if (whaleSlider && whaleValue) {
            // Initialize from persisted settings or Default 1.0
            const persisted = settingsManager.getBigTradeThreshold(this.currentSymbol);
            const val = (typeof persisted === 'number') ? persisted : 1.0;

            whaleSlider.value = String(val);
            whaleValue.textContent = String(val);
            this.footprintChart.setBigTradeThreshold(val);

            // Heatmap intensity slider
            const heatmapSlider = document.getElementById('heatmapIntensitySlider');
            const heatmapValue = document.getElementById('heatmapIntensityValue');
            if (heatmapSlider && heatmapValue) {
                // Initialize from persisted settings or Default 0.05
                const persistedIntensity = settingsManager.getHeatmapIntensityThreshold();
                const intensityVal = (typeof persistedIntensity === 'number') ? persistedIntensity : 0.05;

                heatmapSlider.value = String(intensityVal);
                heatmapValue.textContent = Math.round(intensityVal * 100) + '%';
                this.footprintChart.setHeatmapIntensityThreshold(intensityVal);

                heatmapSlider.addEventListener('input', (e) => {
                    const val = parseFloat(e.target.value);
                    heatmapValue.textContent = Math.round(val * 100) + '%';
                    this.footprintChart.setHeatmapIntensityThreshold(val);
                    settingsManager.setHeatmapIntensityThreshold(val);
                });
            }

            // Heatmap history slider
            const historySlider = document.getElementById('heatmapHistorySlider');
            const historyValue = document.getElementById('heatmapHistoryValue');
            if (historySlider && historyValue) {
                historySlider.addEventListener('input', (e) => {
                    const val = parseInt(e.target.value, 10);
                    historyValue.textContent = val + '%';
                    this.footprintChart.setHeatmapHistoryPercent(val);
                    settingsManager.setHeatmapHistoryPercent(val);
                });
            }

            whaleSlider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                whaleValue.textContent = val.toFixed(1);
                this.footprintChart.setBigTradeThreshold(val);
                settingsManager.setBigTradeThreshold(this.currentSymbol, val);
            });
        }
        // Auto Filter Toggle
        const toggleAutoFilter = document.getElementById('toggleAutoFilter');
        if (toggleAutoFilter) {
            toggleAutoFilter.addEventListener('click', () => {
                this.footprintChart.autoFilter = !this.footprintChart.autoFilter;
                toggleAutoFilter.classList.toggle('active', this.footprintChart.autoFilter);

                if (this.footprintChart.autoFilter) {
                    const next = this.footprintChart._calculateDynamicThreshold(0.99);

                    // Sync UI
                    const whaleSliderEl = document.getElementById('whaleThresholdSlider');
                    const whaleValueEl = document.getElementById('whaleThresholdValue');
                    if (whaleSliderEl) whaleSliderEl.value = String(next.toFixed(1));
                    if (whaleValueEl) whaleValueEl.textContent = String(next.toFixed(1));
                }
            });
        }

        // Tick size selector
        this.elements.tickSizeSelect.addEventListener('change', (e) => {
            const tickSize = parseFloat(e.target.value);
            this.footprintChart.setTickSize(tickSize);
            dataAggregator.setTickSize(tickSize);

            // Save settings
            settingsManager.setTickSize(this.currentSymbol, tickSize);

            dataAggregator.reset();
        });

        // Help overlay
        this.elements.helpClose.addEventListener('click', () => {
            this.elements.helpOverlay.classList.add('hidden');
        });

        this.elements.helpOverlay.addEventListener('click', (e) => {
            if (e.target === this.elements.helpOverlay) {
                this.elements.helpOverlay.classList.add('hidden');
            }
        });

        // Global keyboard shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            if (e.key === '?' || (e.shiftKey && e.key === '/')) {
                e.preventDefault();
                this.elements.helpOverlay.classList.toggle('hidden');
            }
            if (e.key === 'Escape') {
                this.elements.helpOverlay.classList.add('hidden');
            }
        });
    }

    /**
     * Setup WebSocket event handlers
     */
    _setupWebSocket() {
        // Connection status
        binanceWS.on('connect', ({ symbol }) => {
            this.elements.connectionStatus.classList.add('connected');
            this.elements.connectionStatus.querySelector('.status-text').textContent = 'Connected';
            // Hide loading overlay
            this.elements.loadingOverlay.classList.add('hidden');
        });

        binanceWS.on('disconnect', () => {
            this.elements.connectionStatus.classList.remove('connected');
            this.elements.connectionStatus.querySelector('.status-text').textContent = 'Disconnected';
        });

        binanceWS.on('error', ({ error }) => {
            console.error('WebSocket error:', error);
        });

        // Trade data
        binanceWS.on('trade', (trade) => {
            // Update current price
            this.currentPrice = trade.price;
            this.footprintChart.updatePrice(trade.price);
            this.orderBook.updatePrice(trade.price);

            // Track trades for auto-filter
            this.footprintChart.trackTrade(trade);

            // Track big trades
            const isBigTrade = this.footprintChart.addBigTrade(trade);
            if (isBigTrade) {
                audioService.playPing();
            }

            // Process trade in aggregator
            dataAggregator.processTrade(trade);
        });

        // Depth data (throttled for performance)
        this._lastHeatmapUpdate = 0;
        binanceWS.on('depth', (depth) => {
            if (this.currentSymbol && depth.symbol.toLowerCase() !== this.currentSymbol) return;
            this.orderBook.update(depth);
            this.depthHeatmap.addDepthUpdate(depth);

            // Throttle chart updates to max 1/sec (1000ms) to reduce flickering and load
            const now = Date.now();
            if (now - this._lastHeatmapUpdate >= 1000) {
                this.footprintChart.updateDepthHeatmap(this.depthHeatmap.getHeatmapData());

                // L-toggle overlay: show strongest current book walls near price
                const walls = this.depthHeatmap.getTopWalls({
                    aroundPrice: this.currentPrice,
                    countPerSide: 3,
                    maxDistancePct: 0.6,
                });
                this.footprintChart.setLiquidityLevels(walls);

                this._lastHeatmapUpdate = now;
            }
        });

        // 24h ticker
        binanceWS.on('ticker', (ticker) => {
            this.priceChangePercent = ticker.priceChangePercent;
            this._updatePriceDisplay();
            this._updateVolume24h(ticker.quoteVolume);
        });

        // Data aggregator events
        dataAggregator.on('candleUpdate', () => {
            this._updateCharts();

            // Update Market Analysis panel
            if (this.marketAnalysis && this.footprintChart) {
                this.marketAnalysis.analyze({
                    candles: this.footprintChart.candles,
                    heatmapData: this.footprintChart.heatmapData,
                    bigTrades: this.footprintChart.bigTrades
                });
            }
        });

        dataAggregator.on('statsUpdate', (stats) => {
            this._updateStats(stats);
        });
    }

    /**
     * Switch to a different trading symbol
     * @param {string} symbol
     */
    async _switchSymbol(symbol) {
        symbol = symbol.toLowerCase();

        this.currentSymbol = symbol;

        // Show loading overlay
        this.elements.loadingOverlay.classList.remove('hidden');
        this._updateLoadingText('Loading historical data...');

        // Update tick size
        // Priority: Saved setting > Default for symbol > Global default
        const savedTickSize = settingsManager.getTickSize(symbol);
        const tickSize = savedTickSize || this.tickSizes[symbol] || 1;

        this.footprintChart.setTickSize(tickSize);
        dataAggregator.setTickSize(tickSize);

        // Update tick size selector UI
        if (this.elements.tickSizeSelect) {
            this.elements.tickSizeSelect.value = tickSize;
        }

        // Update big trade threshold (whale marker)
        const bigTradeThreshold = this.bigTradeThresholds[symbol] || 5.0;
        this.footprintChart.setBigTradeThreshold(bigTradeThreshold);

        // Reset data
        dataAggregator.reset();
        this.orderBook.clear();
        this.depthHeatmap.reset();
        // Load heatmap history in background (non-blocking)
        this.depthHeatmap.setSymbol(symbol).then(() => {
            this.footprintChart.updateDepthHeatmap(this.depthHeatmap.getHeatmapData());
        });

        // Update precision for order book
        if (symbol === 'btcusdt') {
            this.orderBook.setPrecision(1);
        } else {
            this.orderBook.setPrecision(2);
        }

        // Load historical data - try VPS candles first (faster), then Binance trades fallback
        try {
            this._updateLoadingText('Connecting to VPS...');

            // Try to get pre-aggregated candles from VPS (much faster than raw trades)
            let vpsAvailable = false;

            try {
                // Load ALL available candles (max 1500)
                const candlesNeeded = 1500;

                try {
                    const vpsCandles = await vpsAPI.getCandles(symbol, this.currentTimeframe, candlesNeeded);

                    if (vpsCandles && vpsCandles.length > 0) {
                        this._updateLoadingText(`✅ VPS: Importing ${vpsCandles.length} candles...`);
                        await new Promise(r => setTimeout(r, 500)); // Show success briefly

                        dataAggregator.importCandles(vpsCandles);
                        vpsAvailable = true;

                        // Update charts immediately
                        this._updateCharts();
                    } else {
                        throw new Error('VPS returned 0 candles');
                    }
                } catch (apiErr) {
                    console.error('VPS API fetch failed:', apiErr);
                    this._updateLoadingText(`⚠️ VPS Error: ${apiErr.message}`);
                    await new Promise(r => setTimeout(r, 1000)); // Show error
                    throw apiErr;
                }
            } catch (vpsError) {
                this._updateLoadingText('⚠️ VPS failed. Trying Binance...');
            }

            // Fallback to Binance raw trades if VPS didn't work
            if (!vpsAvailable) {
                this._updateLoadingText('Fetching historical trades from Binance...');
                const trades = await fetchTradesForPeriod(
                    symbol.toUpperCase(),
                    this.historyMinutes,
                    (progress) => {
                        this._updateLoadingText(`Fetching trades... ${progress}%`);
                    }
                );

                if (trades.length > 0) {
                    this._updateLoadingText(`Processing ${trades.length} trades...`);
                    await new Promise(r => setTimeout(r, 50));

                    dataAggregator.processHistoricalTrades(trades, (progress) => {
                        this._updateLoadingText(`Processing trades... ${progress}%`);
                    });

                    this._updateCharts();
                } else {
                }
            }
        } catch (error) {
            console.error('Failed to load historical data:', error);
            // Continue anyway with live data
        }

        // ALWAYS hide loading overlay after processing
        this.elements.loadingOverlay.classList.add('hidden');

        // Connect WebSocket
        this._updateLoadingText('Connecting to live feed...');
        binanceWS.subscribe(symbol);
    }

    /**
     * Update loading text
     */
    _updateLoadingText(text) {
        if (this.elements.loadingText) {
            this.elements.loadingText.textContent = text;
        }
    }

    /**
     * Switch timeframe
     * @param {number} minutes
     */
    async _switchTimeframe(minutes) {
        if (this.currentTimeframe === minutes) return;

        this.currentTimeframe = minutes;

        // Show loading
        this.elements.loadingOverlay.classList.remove('hidden');
        this._updateLoadingText(`Loading ${minutes}m candles...`);

        try {
            // Load ALL available candles (max 1500)
            const candlesNeeded = 1500;

            // Fetch candles for new timeframe from VPS
            const vpsCandles = await vpsAPI.getCandles(this.currentSymbol, minutes, candlesNeeded);

            if (vpsCandles && vpsCandles.length > 0) {
                this._updateLoadingText(`Importing ${vpsCandles.length} candles...`);

                // Reset and import new candles
                dataAggregator.reset();
                dataAggregator.timeframe = minutes;
                dataAggregator.importCandles(vpsCandles);

            } else {
                // Just update timeframe for new trades
                dataAggregator.timeframe = minutes;
                dataAggregator.reset();
            }
        } catch (error) {
            console.error(`Failed to load ${minutes}m candles:`, error);
            // Fallback: just change timeframe for new trades
            dataAggregator.timeframe = minutes;
        }

        // Hide loading and update charts
        this.elements.loadingOverlay.classList.add('hidden');
        this._updateCharts();
    }

    /**
     * Update footprint chart and volume profile
     */
    _updateCharts() {
        const candles = dataAggregator.getCandles();
        const volumeProfile = dataAggregator.getVolumeProfile();
        const sessionMarkers = dataAggregator.getSessionMarkers();

        this.footprintChart.updateCandles(candles);
        this.footprintChart.updateSessionMarkers(sessionMarkers);

        // Get price range from footprint chart for volume profile alignment
        const priceRange = this.footprintChart._calculatePriceRange();
        this.volumeProfile.update(volumeProfile, priceRange);

        // Update POC in footer
        if (volumeProfile.poc) {
            this.elements.pocValue.textContent = this._formatPrice(volumeProfile.poc);
        }
    }

    /**
     * Update price display
     */
    _updatePriceDisplay() {
        const priceEl = this.elements.priceDisplay.querySelector('.price');
        const changeEl = this.elements.priceDisplay.querySelector('.price-change');

        if (this.currentPrice) {
            priceEl.textContent = this._formatPrice(this.currentPrice);
        }

        changeEl.textContent = `${this.priceChangePercent >= 0 ? '+' : ''}${this.priceChangePercent.toFixed(2)}%`;
        changeEl.classList.remove('positive', 'negative');
        changeEl.classList.add(this.priceChangePercent >= 0 ? 'positive' : 'negative');
    }

    /**
     * Update 24h volume display
     * @param {number} volume
     */
    _updateVolume24h(volume) {
        if (volume >= 1e9) {
            this.elements.volume24h.textContent = (volume / 1e9).toFixed(2) + 'B';
        } else if (volume >= 1e6) {
            this.elements.volume24h.textContent = (volume / 1e6).toFixed(2) + 'M';
        } else if (volume >= 1e3) {
            this.elements.volume24h.textContent = (volume / 1e3).toFixed(2) + 'K';
        } else {
            this.elements.volume24h.textContent = volume.toFixed(2);
        }
    }

    /**
     * Update footer statistics
     * @param {Object} stats
     */
    _updateStats(stats) {
        // CVD
        const cvd = stats.cumulativeDelta;
        this.elements.cvdValue.textContent = this._formatVolume(cvd);
        this.elements.cvdValue.classList.remove('positive', 'negative');
        this.elements.cvdValue.classList.add(cvd >= 0 ? 'positive' : 'negative');

        // Current Delta
        const delta = stats.currentDelta;
        this.elements.deltaValue.textContent = this._formatVolume(delta);
        this.elements.deltaValue.classList.remove('positive', 'negative');
        this.elements.deltaValue.classList.add(delta >= 0 ? 'positive' : 'negative');

        // Trades per second
        this.elements.tradesPerSec.textContent = stats.tradesPerSecond.toString();
    }

    /**
     * Format price based on value
     * @param {number} price
     * @returns {string}
     */
    _formatPrice(price) {
        if (price >= 10000) {
            return price.toFixed(1);
        } else if (price >= 100) {
            return price.toFixed(2);
        } else if (price >= 1) {
            return price.toFixed(4);
        } else {
            return price.toFixed(6);
        }
    }

    /**
     * Format volume
     * @param {number} vol
     * @returns {string}
     */
    _formatVolume(vol) {
        const sign = vol >= 0 ? '+' : '';
        const absVol = Math.abs(vol);

        if (absVol >= 1000) {
            return sign + (vol / 1000).toFixed(2) + 'K';
        } else if (absVol >= 1) {
            return sign + vol.toFixed(2);
        } else {
            return sign + vol.toFixed(4);
        }
    }
}

// Start the application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CryptoFlowApp();
});
