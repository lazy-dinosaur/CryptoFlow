/**
 * VPS API Client
 * Connects to VPS backend for historical data and live updates
 */

export class VpsAPI {
    constructor(options = {}) {
        // Detect environment
        const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

        // Production: Use relative URL (same domain)
        // Development: Point to lag0.io (HTTPS/WSS)
        const domain = 'lag0.io';
        const defaultBase = isLocalHost ? `https://${domain}` : '';
        const defaultWs = isLocalHost ? `wss://${domain}` : `wss://${window.location.hostname}`;

        this.baseUrl = options.baseUrl || defaultBase;
        this.wsUrl = options.wsUrl || defaultWs;

        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000;

        // Callbacks
        this.callbacks = {
            trade: [],
            candle: [],
            history: [],
            connect: [],
            disconnect: [],
            error: []
        };

        // Current subscription
        this.currentSymbol = null;
    }

    /**
     * Set VPS server URL
     */
    setServerUrl(baseUrl, wsUrl) {
        this.baseUrl = baseUrl;
        this.wsUrl = wsUrl || baseUrl.replace('http', 'ws');
    }

    /**
     * Check if VPS is available
     */
    async isAvailable() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`, {
                method: 'GET',
                timeout: 5000
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * Get historical candles
     */
    async getCandles(symbol, timeframe = 1, limit = 100) {
        try {
            const response = await fetch(
                `${this.baseUrl}/api/candles?symbol=${symbol}&tf=${timeframe}&limit=${limit}`
            );
            const data = await response.json();
            return data.candles || [];
        } catch (err) {
            console.error('Failed to fetch candles from VPS:', err.message);
            return [];
        }
    }

    /**
     * Get historical trades
     */
    async getTrades(symbol, limit = 10000) {
        try {
            const response = await fetch(
                `${this.baseUrl}/api/trades?symbol=${symbol}&limit=${limit}`
            );
            const data = await response.json();
            return data.trades || [];
        } catch (err) {
            console.error('Failed to fetch trades from VPS:', err.message);
            return [];
        }
    }

    /**
     * Get session data (PVOC, etc)
     */
    async getSessionData(symbol) {
        try {
            const response = await fetch(
                `${this.baseUrl}/api/session?symbol=${symbol}`
            );
            const data = await response.json();
            return data.previous || null;
        } catch (err) {
            console.error('Failed to fetch session data from VPS:', err.message);
            return null;
        }
    }

    /**
     * Connect to WebSocket for live updates
     */
    connect(symbol) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Unsubscribe from old symbol
            if (this.currentSymbol && this.currentSymbol !== symbol) {
                this.ws.send(JSON.stringify({ type: 'unsubscribe', symbol: this.currentSymbol }));
            }
            // Subscribe to new symbol
            this.ws.send(JSON.stringify({ type: 'subscribe', symbol }));
            this.currentSymbol = symbol;
            return;
        }

        this.currentSymbol = symbol;

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                this.reconnectAttempts = 0;

                // Subscribe to symbol
                this.ws.send(JSON.stringify({ type: 'subscribe', symbol }));

                this._emit('connect', { symbol });
            };

            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);

                    if (msg.type === 'trade') {
                        this._emit('trade', msg.trade);
                    } else if (msg.type === 'candle') {
                        this._emit('candle', msg.candle);
                    } else if (msg.type === 'history') {
                        this._emit('history', { symbol: msg.symbol, candles: msg.candles });
                    }
                } catch (err) {
                    console.error('Error parsing VPS message:', err);
                }
            };

            this.ws.onerror = (err) => {
                console.error('VPS WebSocket error:', err);
                this._emit('error', { error: err });
            };

            this.ws.onclose = () => {
                this._emit('disconnect', {});

                // Attempt reconnect
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    setTimeout(() => this.connect(this.currentSymbol), this.reconnectDelay);
                }
            };
        } catch (err) {
            console.error('Failed to connect to VPS:', err);
            this._emit('error', { error: err });
        }
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Register event callback
     */
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        }
        return this;
    }

    /**
     * Emit event
     */
    _emit(event, data) {
        if (this.callbacks[event]) {
            for (const cb of this.callbacks[event]) {
                cb(data);
            }
        }
    }
}

// Singleton instance
export const vpsAPI = new VpsAPI();
