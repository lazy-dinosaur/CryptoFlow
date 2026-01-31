/**
 * CryptoFlow - Order Flow Analysis Platform
 * Main application entry point
 */

import { binanceWS } from './services/binanceWS.js';
import { dataAggregator } from './services/dataAggregator.js';
import { fetchTradesForPeriod } from './services/binanceREST.js';
import { settingsManager } from './services/settingsManager.js';
import { sessionManager } from './services/sessionManager.js';
import { vpsAPI } from './services/vpsAPI.js';
import { FootprintChart } from './components/FootprintChart.js';
import { OrderBook } from './components/OrderBook.js';
import { MLDashboard } from './components/MLDashboard.js';


class CryptoFlowApp {
    constructor() {
        // State
        this.currentSymbol = 'btcusdt';
        this.currentTimeframe = 15; // minutes (15m, 60m, 240m, 1440m available)
        this.currentPrice = null;
        this.priceChangePercent = 0;
        this.isLoadingHistory = false;

        // Components
        this.footprintChart = null;
        this.orderBook = null;

        // Current exchange
        this.currentExchange = 'binance';

        // DOM elements
        this.elements = {
            exchangeSelect: document.getElementById('exchangeSelect'),
            symbolSelect: document.getElementById('symbolSelect'),
            connectionStatus: document.getElementById('connectionStatus'),
            priceDisplay: document.getElementById('priceDisplay'),
            timeframeBtns: document.querySelectorAll('.tf-btn'),
            volume24h: document.getElementById('volume24h'),
            cvdValue: document.getElementById('cvdValue'),
            deltaValue: document.getElementById('deltaValue'),
            tradesPerSec: document.getElementById('tradesPerSec'),
            pocValue: document.getElementById('pocValue'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            loadingText: document.querySelector('.loading-text'),
            helpOverlay: document.getElementById('helpOverlay'),
            helpClose: document.getElementById('helpClose'),
            toggleML: document.getElementById('toggleML'),
            mobileMenuBtn: document.getElementById('mobileMenuBtn'),
            mobileSidebarBtn: document.getElementById('mobileSidebarBtn'),
            navGroup: document.getElementById('navGroup'),
            orderBookSection: document.querySelector('.orderbook-section')
        };

        // Tick sizes for different symbols
        this.tickSizes = {
            btcusdt: 10,
            ethusdt: 1,
            solusdt: 0.1,
            bnbusdt: 0.1
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
        // Chart
        this.footprintChart = new FootprintChart({
            containerId: 'footprintContainer',
            width: window.innerWidth - 320, // Subtract sidebar width
            height: window.innerHeight
        });

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

        // Paper Trading Dashboard
        this.mlDashboard = new MLDashboard('mlDashboard');
        this.mlDashboard.setFootprintChart(this.footprintChart); // Connect chart for channel testing

        // Setup infinite scroll - load more history when user scrolls left
        this.footprintChart.onNeedMoreHistory = async (oldestTime) => {
            await this._loadMoreHistory(oldestTime);
        };

        // Setup Channel Fetching from Paper Trading API
        this._startChannelPolling();
    }

    /**
     * Start polling for paper trading channel data
     */
    _startChannelPolling() {
        const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        const paperApiUrl = isLocalHost ? 'http://134.185.107.33:5003' : `${window.location.protocol}//${window.location.hostname}:5003`;

        const fetchChannel = async () => {
            try {
                const response = await fetch(`${paperApiUrl}/status`);
                if (response.ok) {
                    const data = await response.json();
                    if (this.footprintChart) {
                        // Set channel data
                        if (data.channel) {
                            this.footprintChart.setChannel(data.channel);
                        }
                        // Set trade signals for chart markers
                        if (data.recent_signals) {
                            this.footprintChart.setTradeSignals(data.recent_signals);
                        }
                    }
                }
            } catch (e) {
                // Silent fail - paper trading service might not be running
            }
        };

        // Initial fetch
        fetchChannel();

        // Poll every 30 seconds
        setInterval(fetchChannel, 30000);
    }

    /**
     * Setup DOM event listeners
     */
    _setupEventListeners() {
        // Modal Logic
        const modal = document.getElementById('settingsModal');
        const openBtn = document.getElementById('settingsBtn');
        const closeBtn = document.getElementById('closeSettingsBtn');

        if (openBtn) openBtn.onclick = () => modal.style.display = "block";
        if (closeBtn) closeBtn.onclick = () => modal.style.display = "none";
        window.onclick = (e) => {
            if (e.target == modal) modal.style.display = "none";
        };

        // Exchange selector
        if (this.elements.exchangeSelect) {
            this.elements.exchangeSelect.addEventListener('change', (e) => {
                this._switchExchange(e.target.value);
            });
        }

        // Symbol selector
        this.elements.symbolSelect.addEventListener('change', (e) => {
            this._switchSymbol(e.target.value);
        });

        // Timeframe buttons (Keep as is)
        this.elements.timeframeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this._switchTimeframe(parseInt(btn.dataset.tf, 10));
                this.elements.timeframeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        if (this.elements.toggleML) {
            const initial = settingsManager.get('showML') !== false;
            this.elements.toggleML.checked = initial;
            this.mlDashboard.setVisible(initial);
            this.elements.toggleML.addEventListener('change', (e) => {
                const active = e.target.checked;
                this.mlDashboard.setVisible(active);
                settingsManager.set('showML', active);
            });
        }

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
                if (this.elements.navGroup) this.elements.navGroup.classList.remove('active');
                if (this.elements.orderBookSection) this.elements.orderBookSection.classList.remove('active');
            }
        });

        if (this.elements.mobileMenuBtn) {
            this.elements.mobileMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.elements.navGroup.classList.toggle('active');
                if (this.elements.orderBookSection) this.elements.orderBookSection.classList.remove('active');
            });
        }

        if (this.elements.mobileSidebarBtn) {
            this.elements.mobileSidebarBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.elements.orderBookSection.classList.toggle('active');
                if (this.elements.navGroup) this.elements.navGroup.classList.remove('active');
            });
        }

        document.addEventListener('click', (e) => {
            if (this.elements.navGroup && 
                this.elements.navGroup.classList.contains('active') && 
                !this.elements.navGroup.contains(e.target) && 
                !this.elements.mobileMenuBtn.contains(e.target)) {
                this.elements.navGroup.classList.remove('active');
            }
            
            if (this.elements.orderBookSection && 
                this.elements.orderBookSection.classList.contains('active') && 
                !this.elements.orderBookSection.contains(e.target) && 
                !this.elements.mobileSidebarBtn.contains(e.target)) {
                this.elements.orderBookSection.classList.remove('active');
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

            // Process trade in aggregator
            dataAggregator.processTrade(trade);
        });

        // Depth data (throttled for performance)
        binanceWS.on('depth', (depth) => {
            if (this.currentSymbol && depth.symbol.toLowerCase() !== this.currentSymbol) return;
            this.orderBook.update(depth);
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
        });

        dataAggregator.on('statsUpdate', (stats) => {
            this._updateStats(stats);
        });
    }

    /**
     * Switch to a different exchange
     * @param {string} exchange
     */
    async _switchExchange(exchange) {
        exchange = exchange.toLowerCase();
        this.currentExchange = exchange;
        vpsAPI.setExchange(exchange);

        // Update symbol dropdown based on exchange
        const symbolsForExchange = {
            binance: ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt'],
            bybit: ['btcusdt', 'ethusdt', 'solusdt'],
            bitget: ['btcusdt', 'ethusdt', 'solusdt']
        };

        const symbols = symbolsForExchange[exchange] || symbolsForExchange.binance;

        // Update dropdown options
        if (this.elements.symbolSelect) {
            this.elements.symbolSelect.innerHTML = symbols.map(s =>
                `<option value="${s}">${s.toUpperCase()}</option>`
            ).join('');
        }

        // If current symbol not available on new exchange, switch to first available
        if (!symbols.includes(this.currentSymbol)) {
            this.currentSymbol = symbols[0];
        }

        // Reload data for current symbol on new exchange
        await this._switchSymbol(this.currentSymbol);
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

        // Update tick size for aggregation
        const tickSize = this.tickSizes[symbol] || 1;
        dataAggregator.setTickSize(tickSize);

        // Reset data
        dataAggregator.reset();
        this.orderBook.clear();
        this.footprintChart.setChannel(null);
        this.footprintChart.setTradeSignals([]);

        // Update precision for order book
        if (symbol === 'btcusdt') {
            this.orderBook.setPrecision(1);
        } else {
            this.orderBook.setPrecision(2);
        }

        // Load historical data - try VPS candles first (faster), then direct exchange fallback
        const exchangeName = (this.currentExchange || 'binance').charAt(0).toUpperCase() + (this.currentExchange || 'binance').slice(1);
        try {
            this._updateLoadingText(`Connecting to VPS (${exchangeName})...`);

            // Try to get pre-aggregated candles from VPS (much faster than raw trades)
            let vpsAvailable = false;

            try {
                // Initial load - get recent candles first, then load more via pagination
                const candlesNeeded = 2000;

                try {
                    const result = await vpsAPI.getCandles(symbol, this.currentTimeframe, candlesNeeded);
                    const vpsCandles = result.candles;

                    if (vpsCandles && vpsCandles.length > 0) {
                        this._updateLoadingText(`✅ ${exchangeName}: Importing ${vpsCandles.length} candles...`);
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

                // Scale history based on timeframe
                let minutesToLoad = this.historyMinutes;
                if (this.currentTimeframe === 15) minutesToLoad = 120;
                if (this.currentTimeframe === 60) minutesToLoad = 240;
                if (this.currentTimeframe === 240) minutesToLoad = 480;
                if (this.currentTimeframe === 1440) minutesToLoad = 1440;

                const trades = await fetchTradesForPeriod(
                    symbol.toUpperCase(),
                    minutesToLoad,
                    (progress) => {
                        this._updateLoadingText(`Fetching trades... ${progress}%`);
                    }
                );

                if (trades.length > 0) {
                    this._updateLoadingText(`Processing ${trades.length} trades...`);
                    await new Promise(r => setTimeout(r, 50));

                    dataAggregator.timeframe = this.currentTimeframe; // Ensure correct mode
                    dataAggregator.processHistoricalTrades(trades, (progress) => {
                        this._updateLoadingText(`Processing trades... ${progress}%`);
                    });

                    this._updateCharts();
                } else {
                    this._updateLoadingText('⚠️ No trades found.');
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
            // Initial load - get recent candles, more will be loaded via pagination
            const candlesNeeded = 2000;

            // Fetch candles for new timeframe from VPS
            const result = await vpsAPI.getCandles(this.currentSymbol, minutes, candlesNeeded);
            const vpsCandles = result.candles;

            if (vpsCandles && vpsCandles.length > 0) {
                this._updateLoadingText(`Importing ${vpsCandles.length} candles...`);

                // Reset and import new candles
                dataAggregator.reset();
                dataAggregator.setTimeframe(minutes); // Update Aggregator Mode
                dataAggregator.importCandles(vpsCandles);

            } else {
                throw new Error('VPS returned 0 candles');
            }
        } catch (error) {
            console.error(`Failed to load ${minutes}m candles from VPS:`, error);

            // FALLBACK: Load from Binance Raw Trades
            this._updateLoadingText(`VPS Failed. Fetching raw trades...`);

            // Calculate history needed (fallback for when VPS fails)
            let historyMins = 120; // Default for 15m
            if (minutes === 60) historyMins = 240;
            if (minutes === 240) historyMins = 480;
            if (minutes === 1440) historyMins = 1440;

            try {
                const trades = await fetchTradesForPeriod(
                    this.currentSymbol.toUpperCase(),
                    historyMins,
                    (p) => this._updateLoadingText(`Loading trades... ${p}%`)
                );

                if (trades && trades.length > 0) {
                    dataAggregator.reset();
                    dataAggregator.setTimeframe(minutes);
                    dataAggregator.processHistoricalTrades(trades, (p) => this._updateLoadingText(`Processing... ${p}%`));
                } else {
                    // Just switch mode empty
                    dataAggregator.setTimeframe(minutes);
                }
            } catch (binanceError) {
                console.error('Binance fallback failed:', binanceError);
                // Just switch mode empty
                dataAggregator.setTimeframe(minutes);
            }
        }

        // Hide loading and update charts
        this.elements.loadingOverlay.classList.add('hidden');
        this._updateCharts();
        this.footprintChart.resetView(); // Auto-center on new timeframe
    }

    /**
     * Update chart data
     */
    _updateCharts() {
        const candles = dataAggregator.getCandles();
        this.footprintChart.updateCandles(candles);
        const volumeProfile = dataAggregator.getVolumeProfile();
        if (volumeProfile.poc && this.elements.pocValue) {
            this.elements.pocValue.textContent = this._formatPrice(volumeProfile.poc);
        }
    }

    /**
     * Load more historical candles (infinite scroll pagination)
     * @param {number} oldestTime - Timestamp of oldest currently loaded candle
     */
    async _loadMoreHistory(oldestTime) {
        try {
            console.log(`Loading more history before ${new Date(oldestTime).toISOString()}`);
            
            const result = await vpsAPI.loadMoreCandles(
                this.currentSymbol,
                this.currentTimeframe,
                oldestTime,
                2000,  // Load 2000 candles at a time
                this.currentExchange
            );

            if (result.candles && result.candles.length > 0) {
                // Prepend historical candles to dataAggregator
                const added = dataAggregator.prependCandles(result.candles);
                console.log(`Added ${added} historical candles`);

                if (added > 0) {
                    this.footprintChart.shiftVisibleRange(added);
                    this._updateCharts();
                }
            }

            // Signal that loading is complete
            this.footprintChart.historyLoadComplete();

            // If no more data, disable further loading
            if (!result.hasMore) {
                console.log('All historical data loaded');
            }
        } catch (err) {
            console.error('Failed to load more history:', err);
            this.footprintChart.historyLoadComplete();
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
