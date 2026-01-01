/**
 * Binance WebSocket Service
 * Handles real-time connections to Binance Futures API
 */

// Combined stream endpoint
// https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
const BINANCE_WS_BASE = 'wss://fstream.binance.com/stream?streams=';

export class BinanceWebSocket {
    constructor() {
        this.connections = new Map();
        this.callbacks = {
            trade: [],
            depth: [],
            ticker: [],
            connect: [],
            disconnect: [],
            error: []
        };
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.currentSymbol = 'btcusdt';
    }

    /**
     * Subscribe to a symbol's data streams
     * @param {string} symbol - Trading pair symbol (e.g., 'btcusdt')
     */
    subscribe(symbol) {
        symbol = symbol.toLowerCase();

        // Close previous subscriptions (this app uses one active symbol at a time)
        this.unsubscribeAll();
        this.currentSymbol = symbol;

        // Create combined stream URL
        const streams = [
            `${symbol}@aggTrade`,       // Aggregated trades
            `${symbol}@depth@100ms`,    // Order book updates
            `${symbol}@ticker`          // 24h ticker
        ];

        const wsUrl = `${BINANCE_WS_BASE}${streams.join('/')}`;

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            this.reconnectAttempts = 0;
            this._emit('connect', { symbol });
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                // Combined stream messages come as { stream, data }
                const data = msg && msg.data ? msg.data : msg;
                this._handleMessage(data);
            } catch (err) {
                console.error('[BinanceWS] Parse error:', err);
            }
        };

        ws.onerror = (error) => {
            console.error('[BinanceWS] Error:', error);
            this._emit('error', { symbol, error });
        };

        ws.onclose = () => {
            this._emit('disconnect', { symbol });
            this._attemptReconnect(symbol);
        };

        this.connections.set(symbol, ws);
    }

    /**
     * Handle incoming WebSocket messages
     * @param {Object} data - Parsed message data
     */
    _handleMessage(data) {
        const eventType = data.e;

        switch (eventType) {
            case 'aggTrade':
                // Aggregated trade data
                const trade = {
                    symbol: data.s,
                    price: parseFloat(data.p),
                    quantity: parseFloat(data.q),
                    time: data.T,
                    isBuyerMaker: data.m, // true = sell, false = buy
                    tradeId: data.a
                };
                this._emit('trade', trade);
                break;

            case 'depthUpdate':
                // Order book depth update
                const depth = {
                    symbol: data.s,
                    bids: data.b.map(([price, qty]) => ({
                        price: parseFloat(price),
                        quantity: parseFloat(qty)
                    })),
                    asks: data.a.map(([price, qty]) => ({
                        price: parseFloat(price),
                        quantity: parseFloat(qty)
                    })),
                    updateId: data.u,
                    time: data.E
                };
                this._emit('depth', depth);
                break;

            case '24hrTicker':
                // 24h ticker statistics
                const ticker = {
                    symbol: data.s,
                    priceChange: parseFloat(data.p),
                    priceChangePercent: parseFloat(data.P),
                    lastPrice: parseFloat(data.c),
                    highPrice: parseFloat(data.h),
                    lowPrice: parseFloat(data.l),
                    volume: parseFloat(data.v),
                    quoteVolume: parseFloat(data.q)
                };
                this._emit('ticker', ticker);
                break;
        }
    }

    /**
     * Attempt to reconnect after disconnect
     * @param {string} symbol - Symbol to reconnect
     */
    _attemptReconnect(symbol) {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('[BinanceWS] Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);


        setTimeout(() => {
            if (this.currentSymbol === symbol) {
                this.subscribe(symbol);
            }
        }, delay);
    }

    /**
     * Unsubscribe from a symbol
     * @param {string} symbol - Symbol to unsubscribe from
     */
    unsubscribe(symbol) {
        symbol = symbol.toLowerCase();
        const ws = this.connections.get(symbol);
        if (ws) {
            ws.close();
            this.connections.delete(symbol);
        }
    }

    /**
     * Unsubscribe from all symbols
     */
    unsubscribeAll() {
        for (const [symbol, ws] of this.connections) {
            ws.close();
        }
        this.connections.clear();
    }

    /**
     * Register an event callback
     * @param {string} event - Event type
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        }
        return this;
    }

    /**
     * Remove an event callback
     * @param {string} event - Event type
     * @param {Function} callback - Callback to remove
     */
    off(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event] = this.callbacks[event].filter(cb => cb !== callback);
        }
        return this;
    }

    /**
     * Emit an event to all registered callbacks
     * @param {string} event - Event type
     * @param {*} data - Event data
     */
    _emit(event, data) {
        if (this.callbacks[event]) {
            for (const callback of this.callbacks[event]) {
                try {
                    callback(data);
                } catch (err) {
                    console.error(`[BinanceWS] Callback error for ${event}:`, err);
                }
            }
        }
    }

    /**
     * Check if connected
     * @returns {boolean}
     */
    isConnected() {
        for (const ws of this.connections.values()) {
            if (ws.readyState === WebSocket.OPEN) {
                return true;
            }
        }
        return false;
    }
}

// Singleton instance
export const binanceWS = new BinanceWebSocket();
