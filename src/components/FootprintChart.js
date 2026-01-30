import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts';

export class FootprintChart {
    constructor(options = {}) {
        const containerId = typeof options === 'string' ? options : options.containerId;
        this.container = document.getElementById(containerId);

        if (!this.container) {
            console.error('Chart container not found:', containerId);
            this.isReady = false;
            return;
        }

        this.isReady = true;
        this.candles = [];
        this.tradeSignals = [];
        this.channel = null;
        this.currentPrice = null;
        this.selectedSignal = null;
        this.selectedSignalKey = null;
        this.signalLines = [];
        this.onSignalSelect = null;

        this.onNeedMoreHistory = null;
        this._isLoadingHistory = false;
        this._loadHistoryThreshold = 50;
        this._pendingRangeShift = 0;

        this.showML = true;
        this.showChannel = true;

        this._initChart();
        this._initResizeObserver();
    }

    _initChart() {
        const rect = this.container.getBoundingClientRect();
        this.chart = createChart(this.container, {
            width: rect.width || 300,
            height: rect.height || 300,
            layout: {
                background: { color: '#0e1012' },
                textColor: '#a0a0a0',
                fontFamily: 'Inter, sans-serif'
            },
            grid: {
                vertLines: { color: '#1f2226' },
                horzLines: { color: '#1f2226' }
            },
            crosshair: {
                mode: CrosshairMode.Normal
            },
            timeScale: {
                borderColor: '#1f2226',
                rightOffset: 8,
                barSpacing: 8
            },
            rightPriceScale: {
                borderColor: '#1f2226'
            }
        });

        this.candleSeries = this.chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
            borderVisible: false
        });

        this.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            this._handleRangeChange(range);
        });

        this.chart.subscribeClick((param) => {
            this._handleChartClick(param);
        });
    }

    _initResizeObserver() {
        this.resizeObserver = new ResizeObserver(() => {
            if (!this.chart) return;
            const rect = this.container.getBoundingClientRect();
            this.chart.applyOptions({ width: rect.width, height: rect.height });
        });
        this.resizeObserver.observe(this.container);
    }

    _handleRangeChange(range) {
        if (!range || this._isLoadingHistory || !this.onNeedMoreHistory) return;
        if (!this.candles.length) return;

        if (range.from < this._loadHistoryThreshold) {
            const oldest = this.candles[0]?.time;
            if (!oldest) return;
            this._isLoadingHistory = true;
            this.onNeedMoreHistory(oldest);
        }
    }

    updateCandles(candles) {
        this.candles = candles || [];
        if (!this.candleSeries) return;
        const shouldShift = this._pendingRangeShift !== 0;
        const range = shouldShift ? this.chart.timeScale().getVisibleLogicalRange() : null;

        const data = this.candles
            .map((candle) => {
                const timeValue = Number(candle.time);
                const time = Math.floor(timeValue / 1000);
                return {
                    time,
                    open: Number(candle.open),
                    high: Number(candle.high),
                    low: Number(candle.low),
                    close: Number(candle.close)
                };
            })
            .filter((item) => Number.isFinite(item.time)
                && Number.isFinite(item.open)
                && Number.isFinite(item.high)
                && Number.isFinite(item.low)
                && Number.isFinite(item.close)
                && item.time > 0)
            .sort((a, b) => a.time - b.time);

        const deduped = [];
        let lastTime = null;
        for (const item of data) {
            if (lastTime === item.time) {
                deduped[deduped.length - 1] = item;
            } else {
                deduped.push(item);
                lastTime = item.time;
            }
        }

        this.candleSeries.setData(deduped);
        if (deduped.length > 0) {
            this._seriesTimeRange = { from: deduped[0].time, to: deduped[deduped.length - 1].time };
        } else {
            this._seriesTimeRange = null;
        }

        this._applyMarkers();
        this._applySelectedSignalLines();
        this._refreshPriceLine();

        if (!this._hasInitialData && deduped.length > 0) {
            this.chart.timeScale().fitContent();
            this._hasInitialData = true;
        }

        if (range && shouldShift) {
            const shifted = { from: range.from + this._pendingRangeShift, to: range.to + this._pendingRangeShift };
            this.chart.timeScale().setVisibleLogicalRange(shifted);
        }

        this._pendingRangeShift = 0;
    }

    updatePrice(price) {
        this.currentPrice = price;
        this._refreshPriceLine(true);
    }

    _refreshPriceLine(throttle = false) {
        if (!this.candleSeries || !this.currentPrice) return;
        const now = Date.now();
        if (throttle) {
            if (this._lastPriceUpdate && now - this._lastPriceUpdate < 1000) return;
            this._lastPriceUpdate = now;
        }

        if (this.priceLine) {
            this.candleSeries.removePriceLine(this.priceLine);
        }

        this.priceLine = this.candleSeries.createPriceLine({
            price: this.currentPrice,
            color: '#f59e0b',
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'Last'
        });
    }

    setChannel(channel) {
        this.channel = channel;
        this._applyChannelLines();
    }

    setTradeSignals(signals) {
        this.tradeSignals = signals || [];
        this._applyMarkers();
    }

    setSelectedSignal(signal, notify = false) {
        if (!signal) {
            this.selectedSignal = null;
            this.selectedSignalKey = null;
        } else {
            this.selectedSignal = signal;
            this.selectedSignalKey = this._getSignalKey(signal);
        }
        this._applySelectedSignalLines();
        this._applyMarkers();
        if (notify && this.onSignalSelect) {
            this.onSignalSelect(this.selectedSignal);
        }
    }

    _applyMarkers() {
        if (!this.candleSeries) return;

        const range = this._seriesTimeRange;
        const selectedKey = this.selectedSignalKey;
        const markers = (this.tradeSignals || [])
            .map((signal) => {
                const timestamp = Number(signal.timestamp || 0);
                if (!timestamp) return null;
                const time = Math.floor(timestamp / 1000);
                if (range && (time < range.from || time > range.to)) return null;
                const isLong = signal.direction === 'LONG';
                const key = this._getSignalKey(signal);
                const isSelected = selectedKey && key === selectedKey;
                const label = signal.setup_type ? `${signal.setup_type}` : '';
                return {
                    time,
                    position: isLong ? 'belowBar' : 'aboveBar',
                    color: isSelected ? '#f59e0b' : (isLong ? '#10b981' : '#ef4444'),
                    shape: isLong ? 'arrowUp' : 'arrowDown',
                    text: isSelected ? `${label} •` : label
                };
            })
            .filter(Boolean)
            .sort((a, b) => a.time - b.time);

        this.candleSeries.setMarkers(markers);
    }

    _applyChannelLines() {
        if (!this.candleSeries) return;

        if (this.selectedSignal) {
            this._clearChannelLines();
            return;
        }

        this._clearChannelLines();

        if (!this.showChannel || !this.channel) return;

        this.supportLine = this.candleSeries.createPriceLine({
            price: Number(this.channel.support),
            color: '#00bcd4',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `S ${this.channel.support_touches ?? ''}`.trim()
        });

        this.resistanceLine = this.candleSeries.createPriceLine({
            price: Number(this.channel.resistance),
            color: '#f97316',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `R ${this.channel.resistance_touches ?? ''}`.trim()
        });
    }

    _applySelectedSignalLines() {
        if (!this.candleSeries) return;

        this._clearSignalLines();
        if (!this.selectedSignal) {
            this._applyChannelLines();
            return;
        }

        this._clearChannelLines();

        const signal = this.selectedSignal;
        const lines = [
            { price: signal.channel_support, color: '#00bcd4', style: LineStyle.Dashed, title: 'S' },
            { price: signal.channel_resistance, color: '#f97316', style: LineStyle.Dashed, title: 'R' },
            { price: signal.entry_price, color: '#e2e8f0', style: LineStyle.Solid, title: 'Entry' },
            { price: signal.sl_price, color: '#ef4444', style: LineStyle.Solid, title: 'SL' },
            { price: signal.tp1_price, color: '#22c55e', style: LineStyle.Solid, title: 'TP1' },
            { price: signal.tp2_price, color: '#16a34a', style: LineStyle.Dotted, title: 'TP2' }
        ];

        for (const line of lines) {
            const price = Number(line.price);
            if (!Number.isFinite(price)) continue;
            const priceLine = this.candleSeries.createPriceLine({
                price,
                color: line.color,
                lineWidth: 2,
                lineStyle: line.style,
                axisLabelVisible: true,
                title: line.title
            });
            this.signalLines.push(priceLine);
        }
    }

    _handleChartClick(param) {
        if (!param || typeof param.time !== 'number') return;
        const signal = this._findSignalByTime(param.time);
        if (!signal) return;
        const key = this._getSignalKey(signal);
        const isSelected = this.selectedSignalKey && key === this.selectedSignalKey;
        this.setSelectedSignal(isSelected ? null : signal, true);
    }

    _clearSignalLines() {
        if (!this.candleSeries) return;
        if (this.signalLines && this.signalLines.length > 0) {
            this.signalLines.forEach((line) => this.candleSeries.removePriceLine(line));
        }
        this.signalLines = [];
    }

    _clearChannelLines() {
        if (this.supportLine) this.candleSeries.removePriceLine(this.supportLine);
        if (this.resistanceLine) this.candleSeries.removePriceLine(this.resistanceLine);
        this.supportLine = null;
        this.resistanceLine = null;
    }

    shiftVisibleRange(count) {
        this._pendingRangeShift += count;
    }

    resetView() {
        if (!this.chart) return;
        this.chart.timeScale().fitContent();
    }

    historyLoadComplete() {
        this._isLoadingHistory = false;
    }

    requestDraw() {
        this._applyChannelLines();
        this._applySelectedSignalLines();
        this._applyMarkers();
    }

    updateSessionMarkers() {}
    setTickSize() {}
    setHeatmapIntensityThreshold() {}
    setHeatmapHistoryPercent() {}
    setZoom() {}
    setBigTradeThreshold() {}
    updateDepthHeatmap() {}
    setLiquidityLevels() {}
    trackTrade() {}
    addBigTrade() { return false; }
    setMLPrediction() {}

    _calculatePriceRange() {
        if (!this.candles.length) return { min: null, max: null };
        let min = Infinity;
        let max = -Infinity;
        for (const candle of this.candles) {
            if (candle.low < min) min = candle.low;
            if (candle.high > max) max = candle.high;
        }
        return { min, max };
    }

    _getSignalKey(signal) {
        const ts = Number(signal.timestamp || 0);
        const dir = signal.direction || '';
        const entry = Number(signal.entry_price || 0);
        return `${ts}:${dir}:${entry}`;
    }

    _findSignalByTime(time) {
        const matches = (this.tradeSignals || []).filter((signal) => {
            const timestamp = Number(signal.timestamp || 0);
            if (!timestamp) return false;
            return Math.floor(timestamp / 1000) === time;
        });
        if (matches.length === 0) return null;
        return matches[0];
    }
}
