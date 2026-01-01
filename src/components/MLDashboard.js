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

    toggle() {
        this.container.classList.toggle('hidden');
        return !this.container.classList.contains('hidden');
    }

    _init() {
        if (!this.container) {
            console.warn('MLDashboard container not found!');
            return;
        }
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

        // Add styles
        this._addStyles();

        // Toggle expand
        document.getElementById('mlHeader').addEventListener('click', () => {
            this.expanded = !this.expanded;
            document.getElementById('mlContent').style.display = this.expanded ? 'block' : 'none';
            document.querySelector('.ml-expand').textContent = this.expanded ? '▲' : '▼';
        });
    }

    _addStyles() {
        if (document.getElementById('ml-dashboard-styles')) return;

        const style = document.createElement('style');
        style.id = 'ml-dashboard-styles';
        style.textContent = `
            .ml-dashboard {
                position: absolute;
                top: 50px;
                left: 10px;
                background: rgba(15, 20, 25, 0.95);
                border: 1px solid rgba(0, 230, 118, 0.3);
                border-radius: 8px;
                font-family: Inter, sans-serif;
                font-size: 12px;
                color: #fff;
                z-index: 1000;
                min-width: 200px;
                backdrop-filter: blur(10px);
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
        `;

        // Add train button handler
        document.getElementById('mlTrainBtn')?.addEventListener('click', async (e) => {
            const btn = e.target;
            btn.disabled = true;
            btn.textContent = '⏳ Training läuft...';

            try {
                await fetch(`${this.apiUrl}/train`);
                await this._fetchStatus();
            } catch (error) {
                console.error('Training failed:', error);
            }

            btn.disabled = false;
            btn.textContent = '🔄 Manuelles Training';
        });
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
