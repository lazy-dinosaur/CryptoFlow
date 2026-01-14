/**
 * ML Dashboard Component
 * Displays ML training status, accuracy, and prediction confidence
 */

export class MLDashboard {
    constructor(containerId) {
        this.container = document.getElementById(containerId);

        // Detect environment - use VPS URL when running locally
        const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        const baseUrl = isLocalHost ? 'http://134.185.107.33:3000' : '';
        this.apiUrl = `${baseUrl}/api/ml`;

        this.status = null;
        this.expanded = false;
        this.signals = [];
        this.signalStats = {};
        this.signalFilter = 'ALL'; // ALL, LONG, SHORT

        this._init();
        this._startPolling();
    }

    get isVisible() {
        return !this.container.classList.contains('hidden');
    }

    setVisible(visible) {
        if (visible) {
            this.container.classList.remove('hidden');
        } else {
            this.container.classList.add('hidden');
        }
    }

    toggle() {
        this.setVisible(!this.isVisible);
        return this.isVisible;
    }

    _init() {
        // Create a dedicated container for the dashboard
        this.container = document.createElement('div');
        this.container.id = 'ml-dashboard-overlay-root';
        document.body.appendChild(this.container);

        this.container.innerHTML = `
            <div class="ml-dashboard">
                <div class="ml-header" id="mlHeader">
                    <span class="ml-icon">🤖</span>
                    <span class="ml-title">ML Signal AI</span>
                    <span class="ml-badge" id="mlBadge">--</span>
                    <span class="ml-expand">▼</span>
                </div>
                <div class="ml-content" id="mlContent" style="display: none;">
                    <div class="ml-stats" id="mlStats">
                        Loading...
                    </div>
                </div>
            </div>
        `;

        this._addStyles();

        // Toggle expand
        const header = this.container.querySelector('#mlHeader');
        if (header) {
            header.addEventListener('click', () => {
                this.expanded = !this.expanded;
                const content = this.container.querySelector('#mlContent');
                const icon = this.container.querySelector('.ml-expand');
                if (content) content.style.display = this.expanded ? 'block' : 'none';
                if (icon) icon.textContent = this.expanded ? '▲' : '▼';
            });
        }

        console.log('MLDashboard initialized (Dynamic Overlay Mode)');
        this.setVisible(true);
    }

    _addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .ml-dashboard {
                position: fixed;
                top: 120px; /* Moved down to avoid headers */
                left: 20px;
                background: rgba(15, 20, 25, 0.95);
                border: 1px solid rgba(0, 255, 128, 0.5); /* BRIGHT GREEN BORDER */
                border-radius: 8px;
                font-family: Inter, sans-serif;
                font-size: 12px;
                color: #fff;
                z-index: 2147483647; /* MAX INT JS Z-INDEX */
                min-width: 220px;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.8);
            }
            
            .ml-header {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px 12px;
                cursor: pointer;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            
            .ml-header:hover {
                background: rgba(255,255,255,0.05);
            }
            
            .ml-icon {
                font-size: 16px;
            }
            
            .ml-title {
                font-weight: 600;
                flex: 1;
            }
            
            .ml-badge {
                background: linear-gradient(135deg, #00e676, #00bcd4);
                color: #000;
                padding: 2px 8px;
                border-radius: 10px;
                font-weight: 700;
                font-size: 11px;
            }
            
            .ml-badge.error {
                background: linear-gradient(135deg, #ff5252, #ff1744);
            }
            
            .ml-badge.warning {
                background: linear-gradient(135deg, #ffd700, #ff9800);
            }
            
            .ml-expand {
                color: #8899aa;
                font-size: 10px;
            }
            
            .ml-content {
                padding: 12px;
            }
            
            .ml-stats {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .ml-stat-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .ml-stat-label {
                color: #8899aa;
            }
            
            .ml-stat-value {
                font-weight: 600;
            }
            
            .ml-stat-value.good {
                color: #00e676;
            }
            
            .ml-stat-value.bad {
                color: #ff5252;
            }
            
            .ml-stat-value.neutral {
                color: #ffd700;
            }

            .ml-stat-section {
                margin: 8px 0;
                padding: 8px;
                background: rgba(255,255,255,0.03);
                border-radius: 4px;
            }

            .ml-stat-header {
                font-size: 11px;
                font-weight: bold;
                margin-bottom: 6px;
                padding-bottom: 4px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }

            .ml-history {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            
            .ml-history-title {
                font-size: 11px;
                color: #8899aa;
                margin-bottom: 6px;
            }
            
            .ml-history-item {
                display: flex;
                justify-content: space-between;
                font-size: 10px;
                color: #8899aa;
                padding: 3px 0;
            }
            
            .ml-btn {
                width: 100%;
                padding: 8px;
                margin-top: 10px;
                background: linear-gradient(135deg, #1a2332, #2a3a4a);
                border: 1px solid rgba(0, 230, 118, 0.3);
                border-radius: 4px;
                color: #00e676;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .ml-btn:hover {
                background: linear-gradient(135deg, #00e676, #00bcd4);
                color: #000;
            }
            
            .ml-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            /* FAILSAFE HIDDEN CLASS */
            .hidden { display: none !important; }

            /* Signal List Styles */
            .ml-signals-section {
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid rgba(255,255,255,0.1);
            }

            .ml-signals-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }

            .ml-signals-title {
                font-size: 12px;
                font-weight: 600;
            }

            .ml-signals-tabs {
                display: flex;
                gap: 4px;
            }

            .ml-signals-tab {
                padding: 3px 8px;
                font-size: 10px;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 4px;
                background: transparent;
                color: #8899aa;
                cursor: pointer;
                transition: all 0.2s;
            }

            .ml-signals-tab:hover {
                border-color: rgba(255,255,255,0.4);
                color: #fff;
            }

            .ml-signals-tab.active {
                background: rgba(0, 230, 118, 0.2);
                border-color: #00e676;
                color: #00e676;
            }

            .ml-signals-tab.active.short {
                background: rgba(255, 82, 82, 0.2);
                border-color: #ff5252;
                color: #ff5252;
            }

            .ml-signals-list {
                max-height: 300px;
                overflow-y: auto;
            }

            .ml-signal-item {
                padding: 8px;
                margin-bottom: 6px;
                background: rgba(255,255,255,0.03);
                border-radius: 4px;
                border-left: 3px solid #00e676;
                cursor: pointer;
                transition: all 0.2s;
            }

            .ml-signal-item:hover {
                background: rgba(255,255,255,0.06);
            }

            .ml-signal-item.short {
                border-left-color: #ff5252;
            }

            .ml-signal-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 11px;
            }

            .ml-signal-direction {
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
            }

            .ml-signal-direction.long {
                background: rgba(0, 230, 118, 0.2);
                color: #00e676;
            }

            .ml-signal-direction.short {
                background: rgba(255, 82, 82, 0.2);
                color: #ff5252;
            }

            .ml-signal-time {
                color: #8899aa;
                font-size: 10px;
            }

            .ml-signal-prices {
                display: flex;
                gap: 12px;
                margin-top: 6px;
                font-size: 10px;
            }

            .ml-signal-price {
                display: flex;
                flex-direction: column;
            }

            .ml-signal-price-label {
                color: #8899aa;
                font-size: 9px;
            }

            .ml-signal-price-value {
                font-weight: 600;
            }

            .ml-signal-details {
                display: none;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid rgba(255,255,255,0.1);
            }

            .ml-signal-item.expanded .ml-signal-details {
                display: block;
            }

            .ml-signal-metrics {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 6px;
                font-size: 10px;
            }

            .ml-signal-metric {
                display: flex;
                justify-content: space-between;
            }

            .ml-signal-metric-label {
                color: #8899aa;
            }

            .ml-signal-metric-value {
                font-weight: 600;
            }

            .ml-signal-status {
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 9px;
                font-weight: 600;
            }

            .ml-signal-status.active {
                background: rgba(255, 193, 7, 0.2);
                color: #ffc107;
            }

            .ml-signal-status.win {
                background: rgba(0, 230, 118, 0.2);
                color: #00e676;
            }

            .ml-signal-status.loss {
                background: rgba(255, 82, 82, 0.2);
                color: #ff5252;
            }

            .ml-no-signals {
                text-align: center;
                color: #8899aa;
                font-size: 11px;
                padding: 20px;
            }
        `;
        document.head.appendChild(style);
    }

    async _fetchStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/status`);
            if (!response.ok) throw new Error('API error');
            this.status = await response.json();
            this._updateUI();
        } catch (error) {
            console.warn('ML API not available:', error.message);
            this.status = { error: true };
            this._updateUI();
        }
    }

    _startPolling() {
        this._fetchStatus().finally(() => {
            // Adaptive polling: 60s if error/offline, 30s if active
            const delay = this.status && this.status.error ? 60000 : 30000;
            this.pollTimeout = setTimeout(() => this._startPolling(), delay);
        });
    }

    _updateUI() {
        const badge = document.getElementById('mlBadge');
        const stats = document.getElementById('mlStats');

        if (!this.status || this.status.error) {
            badge.textContent = 'Offline';
            badge.className = 'ml-badge error';
            stats.innerHTML = `
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Status</span>
                    <span class="ml-stat-value bad">Service nicht erreichbar</span>
                </div>
                <div style="color: #8899aa; font-size: 11px; margin-top: 8px;">
                    The ML service runs on port 5001.<br>
                    Start: <code>python ml_service.py</code>
                </div>
            `;
            return;
        }

        const accuracy = this.status.accuracy ? (this.status.accuracy * 100).toFixed(1) : '--';
        const accuracyClass = this.status.accuracy >= 0.55 ? 'good' : (this.status.accuracy >= 0.45 ? 'neutral' : 'bad');

        badge.textContent = `${accuracy}%`;
        badge.className = `ml-badge ${accuracyClass === 'bad' ? 'error' : (accuracyClass === 'neutral' ? 'warning' : '')}`;

        const lastTraining = this.status.lastTraining
            ? new Date(this.status.lastTraining).toLocaleString('ko-KR', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            })
            : '--';

        let historyHtml = '';
        if (this.status.history && this.status.history.length > 0) {
            historyHtml = `
                <div class="ml-history">
                    <div class="ml-history-title">Training History</div>
                    ${this.status.history.slice(0, 5).map(h => `
                        <div class="ml-history-item">
                            <span>${new Date(h.timestamp).toLocaleString('ko-KR', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            })}</span>
                            <span>${(h.accuracy * 100).toFixed(1)}% (${h.samples} samples)</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // LONG/SHORT stats
        const longSamples = this.status.longSamples || 0;
        const shortSamples = this.status.shortSamples || 0;
        const longAcc = (this.status.longAccuracy || 0).toFixed(1);
        const shortAcc = (this.status.shortAccuracy || 0).toFixed(1);
        const longWR = (this.status.longWinrate || 0).toFixed(1);
        const shortWR = (this.status.shortWinrate || 0).toFixed(1);

        stats.innerHTML = `
            <div class="ml-stat-row">
                <span class="ml-stat-label">Total Accuracy</span>
                <span class="ml-stat-value ${accuracyClass}">${accuracy}%</span>
            </div>
            <div class="ml-stat-section">
                <div class="ml-stat-header">📈 LONG</div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Accuracy</span>
                    <span class="ml-stat-value" style="color: #00e676;">${longAcc}%</span>
                </div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Win Rate</span>
                    <span class="ml-stat-value">${longWR}%</span>
                </div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Samples</span>
                    <span class="ml-stat-value">${longSamples}</span>
                </div>
            </div>
            <div class="ml-stat-section">
                <div class="ml-stat-header">📉 SHORT</div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Accuracy</span>
                    <span class="ml-stat-value" style="color: #ff5252;">${shortAcc}%</span>
                </div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Win Rate</span>
                    <span class="ml-stat-value">${shortWR}%</span>
                </div>
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Samples</span>
                    <span class="ml-stat-value">${shortSamples}</span>
                </div>
            </div>
            <div class="ml-stat-row">
                <span class="ml-stat-label">Last Training</span>
                <span class="ml-stat-value">${lastTraining}</span>
            </div>
            <div class="ml-stat-row">
                <span class="ml-stat-label">Model Status</span>
                <span class="ml-stat-value ${this.status.modelLoaded ? 'good' : 'bad'}">
                    ${this.status.modelLoaded ? '✓ Active' : '✗ Not loaded'}
                </span>
            </div>
            ${historyHtml}
            <button class="ml-btn" id="mlTrainBtn">🔄 Manual Training</button>
            ${this._renderSignalsSection()}
        `;

        // Setup signal tab handlers
        this._setupSignalTabHandlers();
        this.fetchSignals();

        // Add train button handler logic
        const trainBtn = stats.querySelector('#mlTrainBtn');
        if (trainBtn) {
            trainBtn.addEventListener('click', async (e) => {
                console.log('Train button clicked');
                const btn = e.target;
                btn.disabled = true;
                btn.textContent = '⏳ Training in progress...';

                try {
                    console.log(`Calling ${this.apiUrl}/train`);
                    const response = await fetch(`${this.apiUrl}/train`, {
                        headers: {
                            'x-api-key': 'CryptoFlowMasterKey2025!'
                        }
                    });
                    const data = await response.json();
                    console.log('Training response:', data);
                    await this._fetchStatus();
                } catch (error) {
                    console.error('Training failed:', error);
                    alert('Training failed: ' + error.message);
                }

                btn.disabled = false;
                btn.textContent = '🔄 Manual Training';
            });
        }
    }

    updatePredictionUI(result) {
        // Update dashboard prediction display if exists
        let predEl = document.getElementById('ml-live-prediction');
        if (!predEl) {
            // INSERT AFTER HEADER (Always Visible)
            const header = document.getElementById('mlHeader');
            if (header) {
                const div = document.createElement('div');
                div.id = 'ml-live-prediction';
                div.style.padding = '10px';
                div.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
                div.style.textAlign = 'center';
                div.style.fontWeight = 'bold';
                div.style.fontSize = '12px';
                div.style.background = 'rgba(255,255,255,0.02)';

                // Insert after header
                header.insertAdjacentElement('afterend', div);
                predEl = div;
            }
        }

        if (predEl) {
            if (result.signal) {
                // Active Signal
                const confidence = (result.confidence * 100).toFixed(0);
                const rr = result.rr ? result.rr.toFixed(1) : '?';
                const direction = result.direction || 'LONG';
                const isShort = direction === 'SHORT';

                // Green for LONG, Red for SHORT
                const color = isShort ? '#ff5252' : '#00e676';
                const bgColor = isShort ? 'rgba(255, 82, 82, 0.15)' : 'rgba(0, 230, 118, 0.15)';
                const icon = isShort ? '📉' : '📈';

                predEl.style.background = bgColor;
                predEl.style.borderColor = color;

                predEl.innerHTML = `
                <div style="color: ${color}; font-size: 14px; margin-bottom: 4px;">${icon} ${direction} SIGNAL</div>
                <div style="font-size: 11px; color: #ccc;">
                    ${result.setupType || 'Zone Touch'} | Conf: ${confidence}% | RR: 1:${rr}
                </div>
            `;
            } else {
                // Scanning / No Signal
                predEl.style.background = 'rgba(255,255,255,0.05)';
                predEl.style.borderColor = 'rgba(255,255,255,0.1)';

                // Show reason if available (e.g. "Values too low")
                const msg = result.message || 'Scanning...';
                predEl.innerHTML = `
                <div style="color: #8899aa;">
                    <span class="pulse-dot"></span> ${msg}
                </div>
                `;
            }
        }
    }

    async predictFromCandles(candles) {
        try {
            const response = await fetch(`${this.apiUrl}/predict_raw`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ candles })
            });
            if (!response.ok) throw new Error(response.statusText);
            return await response.json();
        } catch (error) {
            return null;
        }
    }

    /**
     * Get prediction confidence for a signal
     * @param {Object} features - Signal features
     * @returns {Promise<{prediction: string, confidence: number}>}
     */
    async getPrediction(features) {
        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features })
            });
            return await response.json();
        } catch (error) {
            console.warn('Prediction error:', error);
            return { prediction: null, confidence: 0 };
        }
    }

    async fetchSignals() {
        try {
            const direction = this.signalFilter === 'ALL' ? '' : `?direction=${this.signalFilter}`;
            const response = await fetch(`${this.apiUrl}/signals${direction}`);
            if (!response.ok) throw new Error('API error');
            const data = await response.json();
            this.signals = data.signals || [];
            this.signalStats = data.stats || {};
            this._renderSignals();
        } catch (error) {
            console.warn('Failed to fetch signals:', error.message);
            this.signals = [];
        }
    }

    _renderSignals() {
        const container = document.getElementById('mlSignalsList');
        if (!container) return;

        if (this.signals.length === 0) {
            container.innerHTML = `
                <div class="ml-no-signals">
                    No signals yet.<br>
                    <span style="font-size: 10px;">Signals appear when confidence > 60%</span>
                </div>
            `;
            return;
        }

        container.innerHTML = this.signals.map(signal => {
            const isShort = signal.direction === 'SHORT';
            const dirClass = isShort ? 'short' : 'long';
            const statusClass = (signal.status || 'ACTIVE').toLowerCase();
            const time = new Date(signal.timestamp).toLocaleString('ko-KR', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            // Format prices
            const formatPrice = (p) => p ? p.toLocaleString('en-US', { maximumFractionDigits: 2 }) : '--';

            return `
                <div class="ml-signal-item ${dirClass}" data-id="${signal.id}">
                    <div class="ml-signal-row">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <span class="ml-signal-direction ${dirClass}">${signal.direction}</span>
                            <span>${signal.symbol}</span>
                            <span class="ml-signal-status ${statusClass}">${signal.status || 'ACTIVE'}</span>
                        </div>
                        <span class="ml-signal-time">${time}</span>
                    </div>
                    <div class="ml-signal-prices">
                        <div class="ml-signal-price">
                            <span class="ml-signal-price-label">Entry</span>
                            <span class="ml-signal-price-value">${formatPrice(signal.entry)}</span>
                        </div>
                        <div class="ml-signal-price">
                            <span class="ml-signal-price-label">SL</span>
                            <span class="ml-signal-price-value" style="color: #ff5252;">${formatPrice(signal.sl)}</span>
                        </div>
                        <div class="ml-signal-price">
                            <span class="ml-signal-price-label">TP</span>
                            <span class="ml-signal-price-value" style="color: #00e676;">${formatPrice(signal.tp)}</span>
                        </div>
                        <div class="ml-signal-price">
                            <span class="ml-signal-price-label">RR</span>
                            <span class="ml-signal-price-value">1:${signal.rr?.toFixed(1) || '--'}</span>
                        </div>
                        <div class="ml-signal-price">
                            <span class="ml-signal-price-label">Conf</span>
                            <span class="ml-signal-price-value">${(signal.confidence * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                    <div class="ml-signal-details">
                        <div class="ml-signal-metrics">
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Setup</span>
                                <span class="ml-signal-metric-value">${signal.setup_type || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Exchange</span>
                                <span class="ml-signal-metric-value">${signal.exchange || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Zone Strength</span>
                                <span class="ml-signal-metric-value">${signal.zone_strength?.toFixed(1) || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Zone Age</span>
                                <span class="ml-signal-metric-value">${signal.zone_age || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Delta</span>
                                <span class="ml-signal-metric-value">${signal.delta?.toFixed(0) || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Delta %</span>
                                <span class="ml-signal-metric-value">${signal.delta_pct?.toFixed(2) || '--'}%</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Volume Ratio</span>
                                <span class="ml-signal-metric-value">${signal.volume_ratio?.toFixed(2) || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Whale Intensity</span>
                                <span class="ml-signal-metric-value">${signal.whale_intensity?.toFixed(2) || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">Imbalance Ratio</span>
                                <span class="ml-signal-metric-value">${signal.imbalance_ratio?.toFixed(2) || '--'}</span>
                            </div>
                            <div class="ml-signal-metric">
                                <span class="ml-signal-metric-label">CVD Slope</span>
                                <span class="ml-signal-metric-value">${signal.cvd_slope?.toFixed(2) || '--'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers for expanding
        container.querySelectorAll('.ml-signal-item').forEach(item => {
            item.addEventListener('click', () => {
                item.classList.toggle('expanded');
            });
        });
    }

    _renderSignalsSection() {
        return `
            <div class="ml-signals-section">
                <div class="ml-signals-header">
                    <span class="ml-signals-title">📋 Signal History</span>
                    <div class="ml-signals-tabs">
                        <button class="ml-signals-tab ${this.signalFilter === 'ALL' ? 'active' : ''}" data-filter="ALL">All</button>
                        <button class="ml-signals-tab ${this.signalFilter === 'LONG' ? 'active' : ''}" data-filter="LONG">Long</button>
                        <button class="ml-signals-tab ${this.signalFilter === 'SHORT' ? 'active short' : ''}" data-filter="SHORT">Short</button>
                    </div>
                </div>
                <div class="ml-signals-list" id="mlSignalsList">
                    <div class="ml-no-signals">Loading...</div>
                </div>
            </div>
        `;
    }

    _setupSignalTabHandlers() {
        const tabs = this.container.querySelectorAll('.ml-signals-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.stopPropagation();
                this.signalFilter = tab.dataset.filter;
                // Update active state
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                if (this.signalFilter === 'SHORT') tab.classList.add('short');
                // Fetch with new filter
                this.fetchSignals();
            });
        });
    }
}
