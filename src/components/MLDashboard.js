/**
 * ML Dashboard Component
 * Displays ML training status, accuracy, and prediction confidence
 */

export class MLDashboard {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.apiUrl = '/api/ml';

        this.status = null;
        this.expanded = false;

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
            <div class="ml-dashboard" style="border-color: #555;">
                <div class="ml-header" id="mlHeader">
                    <span class="ml-icon">🔒</span>
                    <span class="ml-title">AI Disabled</span>
                    <span class="ml-badge" id="mlBadge" style="background: #333;">OFF</span>
                    <span class="ml-expand">▼</span>
                </div>
                <div class="ml-content" id="mlContent" style="display: none;">
                    <div style="padding: 10px; color: #aaa; text-align: center;">
                        AI Features are currently disabled.
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

        console.log('MLDashboard initialized (AI DISABLED)');
        this.setVisible(true);
    }

    _startPolling() {
        // AI Disabled - No polling
        return;
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

            .ml-btn-ai {
                background: linear-gradient(135deg, #6200ea, #b388ff);
                border: 1px solid rgba(179, 136, 255, 0.5);
                color: #fff;
                margin-top: 8px;
            }
            .ml-btn-ai:hover {
                background: linear-gradient(135deg, #7c4dff, #d1c4e9);
                box-shadow: 0 0 10px rgba(98, 0, 234, 0.5);
                color: #fff;
            }
            
            .ai-result {
                margin-top: 10px;
                padding: 8px;
                background: rgba(98, 0, 234, 0.1);
                border-left: 3px solid #b388ff;
                font-size: 11px;
                color: #e1bee7;
                white-space: pre-wrap;
            }
        `;
        document.head.appendChild(style);
    }

    setChart(chart) {
        this.chart = chart;
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

    // NOTE: Second _startPolling was removed - AI is disabled

    _updateUI() {
        const badge = document.getElementById('mlBadge');
        const stats = document.getElementById('mlModelStatus');

        if (!this.status || this.status.error) {
            badge.textContent = 'Offline';
            badge.className = 'ml-badge error';
            stats.innerHTML = `
                <div class="ml-stat-row">
                    <span class="ml-stat-label">Status</span>
                    <span class="ml-stat-value bad">Service nicht erreichbar</span>
                </div>
                <div style="color: #8899aa; font-size: 11px; margin-top: 8px;">
                    Der ML-Service läuft auf Port 5001.<br>
                    Starte: <code>python ml_service.py</code>
                </div>
            `;
            return;
        }

        const accuracy = this.status.accuracy ? (this.status.accuracy * 100).toFixed(1) : '--';
        const accuracyClass = this.status.accuracy >= 0.55 ? 'good' : (this.status.accuracy >= 0.45 ? 'neutral' : 'bad');

        badge.textContent = `${accuracy}%`;
        badge.className = `ml-badge ${accuracyClass === 'bad' ? 'error' : (accuracyClass === 'neutral' ? 'warning' : '')}`;

        const lastTraining = this.status.lastTraining
            ? new Date(this.status.lastTraining).toLocaleString('de-DE', {
                day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'
            })
            : '--';

        let historyHtml = '';
        if (this.status.history && this.status.history.length > 0) {
            historyHtml = `
                <div class="ml-history">
                    <div class="ml-history-title">Training History</div>
                    ${this.status.history.slice(0, 5).map(h => `
                        <div class="ml-history-item">
                            <span>${new Date(h.timestamp).toLocaleString('de-DE', {
                day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit'
            })}</span>
                            <span>${(h.accuracy * 100).toFixed(1)}% (${h.samples} samples)</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        stats.innerHTML = `
            <div class="ml-stat-row">
                <span class="ml-stat-label">Model Accuracy</span>
                <span class="ml-stat-value ${accuracyClass}">${accuracy}%</span>
            </div>
            <div class="ml-stat-row">
                <span class="ml-stat-label">Training Samples</span>
                <span class="ml-stat-value">${this.status.sampleCount || '--'}</span>
            </div>
            <div class="ml-stat-row">
                <span class="ml-stat-label">Letztes Training</span>
                <span class="ml-stat-value">${lastTraining}</span>
            </div>
            <div class="ml-stat-row">
                <span class="ml-stat-label">Model Status</span>
                <span class="ml-stat-value ${this.status.modelLoaded ? 'good' : 'bad'}">
                    ${this.status.modelLoaded ? '✓ Aktiv' : '✗ Nicht geladen'}
                </span>
            </div>
            ${historyHtml}
            <button class="ml-btn" id="mlTrainBtn">🔄 Manuelles Training</button>
            <button class="ml-btn ml-btn-ai" id="aiAnalystBtn">✨ AI Analyst</button>
            ${this.lastAIResult ? `
                <div class="ai-result">
                    <strong>🤖 AI Analyst:</strong><br/>
                    ${this.lastAIResult.explanation}
                    ${this.lastAIResult.trade ? `<br/><br/><strong>SETUP:</strong> ${this.lastAIResult.action} @ ${this.lastAIResult.trade.entry}` : ''}
                    <div style="margin-top:4px; font-weight:bold; color: ${this.lastAIResult.action === 'BUY' ? '#00e676' : (this.lastAIResult.action === 'SELL' ? '#ff5252' : '#ffd700')}">
                        Action: ${this.lastAIResult.action}
                    </div>
                </div>
            ` : ''}
        `;

        this._attachListeners(stats);
    }

    _attachListeners(stats) {
        // Train Button
        const trainBtn = stats.querySelector('#mlTrainBtn');
        if (trainBtn) {
            trainBtn.addEventListener('click', async (e) => {
                const btn = e.target;
                btn.disabled = true;
                btn.textContent = '⏳ Training läuft...';
                try {
                    await fetch(`${this.apiUrl}/train`, { headers: { 'x-api-key': 'CryptoFlowMasterKey2025!' } });
                    await this._fetchStatus();
                } catch (error) {
                    alert('Training failed: ' + error.message);
                }
                btn.disabled = false;
                btn.textContent = '🔄 Manuelles Training';
            });
        }

        // AI Button
        const aiBtn = stats.querySelector('#aiAnalystBtn');
        if (aiBtn) {
            aiBtn.addEventListener('click', async (e) => {
                if (!this.chart) { alert("Chart not linked!"); return; }
                const btn = e.target;
                const originalText = btn.textContent;
                btn.disabled = true;
                btn.textContent = '🤖 Thinking...';

                try {
                    // Capture Chart
                    const imgData = this.chart.canvas.toDataURL('image/png');
                    const context = {
                        symbol: this.chart.parentApp ? this.chart.parentApp.currentSymbol : 'BTCUSDT',
                        price: this.chart.currentPrice,
                        timeframe: this.chart.parentApp ? this.chart.parentApp.currentTimeframe : 1
                    };

                    const response = await fetch('/api/ai/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: imgData, context })
                    });

                    if (!response.ok) {
                        if (response.status === 404) {
                            throw new Error("Server Update Required! Please restart 'ml_service.py' on VPS.");
                        }
                        const errText = await response.text();
                        let errMsg = response.statusText;
                        try {
                            const errJson = JSON.parse(errText);
                            if (errJson.error) errMsg = errJson.error;
                        } catch (e) { errMsg = errText.substring(0, 100); }

                        throw new Error(`AI Error (${response.status}): ${errMsg}`);
                    }

                    const result = await response.json();

                    this.lastAIResult = result;

                    if (this.chart.setAISignal) { // Pass user intention signal
                        // If AI returns a trade, we visualize it
                        if (result.trade) {
                            // Normalize trade object
                            const aiSig = {
                                type: result.action, // BUY/SELL
                                entry: result.trade.entry,
                                sl: result.trade.sl,
                                tp: result.trade.tp,
                                timestamp: Date.now(),
                                source: 'AI'
                            };
                            this.chart.setAISignal(aiSig);
                        } else {
                            this.chart.setAISignal(null);
                        }
                    }

                    this._updateUI(); // Re-render to show result

                } catch (error) {
                    console.error(error);
                    alert("AI Error: " + error.message);
                }

                btn.disabled = false;
                btn.textContent = originalText;
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

                predEl.style.background = 'rgba(0, 230, 118, 0.15)';
                predEl.style.borderColor = '#00e676';

                predEl.innerHTML = `
                <div style="color: #00e676; font-size: 14px; margin-bottom: 4px;">🚀 TRADE FOUND</div>
                <div style="font-size: 11px; color: #ccc;">
                    ${result.direction || 'LONG'} | Acc: ${confidence}% | RR: 1:${rr}
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

    async backtestHistory(candles) {
        try {
            console.log("Requesting backtest for", candles.length, "candles");
            const response = await fetch(`${this.apiUrl}/backtest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ candles })
            });

            if (!response.ok) throw new Error('Backtest failed');

            const data = await response.json();

            // Handle new format { signals, stats } or old format [signals]
            let signals = [];
            let stats = null;

            if (data.signals) {
                signals = data.signals;
                stats = data.stats;
            } else if (Array.isArray(data)) {
                signals = data;
            }

            console.log("Backtest received:", signals.length, "signals");

            // Update Statistics UI if available
            if (stats) {
                this._renderStats(stats);
            }

            return signals;
        } catch (error) {
            console.error('Backtest error:', error);
            return [];
        }
    }

    _renderStats(stats) {
        const container = this.container.querySelector('#mlBacktestStats');
        if (!container) return;

        // Calculate WIN Rate colors
        const getWinRateColor = (rate) => {
            if (rate >= 60) return '#00ff80'; // Green
            if (rate >= 50) return '#ffd700'; // Gold
            return '#ff4444'; // Red
        };

        const botColor = getWinRateColor(stats.botWinRate);
        const mlColor = getWinRateColor(stats.mlWinRate);

        container.innerHTML = `
            <div style="display: flex; gap: 15px; padding: 10px; font-size: 13px;">
                <div style="flex: 1; text-align: center; border-right: 1px solid rgba(255,255,255,0.1);">
                    <div style="color: #888; margin-bottom: 4px;">🛡️ Bot Signals</div>
                    <div style="font-size: 16px; font-weight: bold; color: ${botColor}">
                        ${stats.botWinRate.toFixed(1)}%
                    </div>
                    <div style="font-size: 10px; opacity: 0.7;">
                        ${stats.botWins}/${stats.botTotal} Wins
                    </div>
                </div>
                <div style="flex: 1; text-align: center;">
                    <div style="color: #888; margin-bottom: 4px;">🤖 ML Signals</div>
                    <div style="font-size: 16px; font-weight: bold; color: ${mlColor}">
                        ${stats.mlWinRate.toFixed(1)}%
                    </div>
                     <div style="font-size: 10px; opacity: 0.7;">
                        ${stats.mlWins}/${stats.mlTotal} Wins
                    </div>
                </div>
            </div>
            <div style="padding: 8px; border-top: 1px solid rgba(255,255,255,0.1); text-align: center; font-size: 11px; color: #666;">
                Total Signals: ${stats.total} | Overall WR: ${stats.winRate.toFixed(1)}%
            </div>
        `;

        // Ensure content is visible
        const content = this.container.querySelector('#mlContent');
        if (content) content.style.display = 'block';
        this.expanded = true;
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
}
