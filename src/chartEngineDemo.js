/**
 * Demo for the unused layer-based chart engine.
 *
 * Open in Vite dev server:
 *   /chart-engine-demo.html
 */

import { ChartState } from './components/chart/core/ChartState.js';
import { CoordinateSystem } from './components/chart/core/CoordinateSystem.js';
import { RenderEngine } from './components/chart/core/RenderEngine.js';
import { InputHandler } from './components/chart/core/InputHandler.js';

import { HeatmapLayer } from './components/chart/layers/HeatmapLayer.js';
import { GridLayer } from './components/chart/layers/GridLayer.js';
import { CandleLayer } from './components/chart/layers/CandleLayer.js';
import { AnalysisLayer } from './components/chart/layers/AnalysisLayer.js';
import { CrosshairLayer } from './components/chart/layers/CrosshairLayer.js';

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function makeDemoCandles({ count, startPrice, tickSize }) {
  const r = mulberry32(42);
  const candles = [];

  let price = startPrice;
  const startTime = Date.now() - count * 60_000;

  for (let i = 0; i < count; i++) {
    const drift = (r() - 0.5) * 25;
    const open = price;
    const close = Math.max(1, open + drift);

    const wick = 10 + r() * 20;
    const high = Math.max(open, close) + wick;
    const low = Math.max(1, Math.min(open, close) - wick);

    // Simple footprint clusters around close
    const clusters = {};
    const levels = 18;
    const base = Math.round(close / tickSize) * tickSize;

    for (let j = 0; j < levels; j++) {
      const p = base + (levels / 2 - j) * tickSize;
      const bid = Math.max(0, (r() * 120) ** 1.2);
      const ask = Math.max(0, (r() * 120) ** 1.2);
      if (bid + ask < 8) continue;
      clusters[p.toFixed(10)] = {
        bid,
        ask,
        delta: bid - ask,
        imbalance: null,
      };
    }

    candles.push({
      time: startTime + i * 60_000,
      open,
      high,
      low,
      close,
      delta: (r() - 0.5) * 200,
      clusters,
    });

    price = close;
  }

  return candles;
}

function makeDemoHeatmap({ candles, tickSize }) {
  const r = mulberry32(7);

  return candles.map((c) => {
    const bids = [];
    const asks = [];

    const center = Math.round(c.close / tickSize) * tickSize;
    for (let i = 0; i < 60; i++) {
      const bidP = center - (i + 1) * tickSize;
      const askP = center + (i + 1) * tickSize;

      const wall = i % 17 === 0 ? 8 : 1;
      bids.push({ p: bidP, q: (r() * 6 + 0.1) * wall });
      asks.push({ p: askP, q: (r() * 6 + 0.1) * wall });
    }

    const maxVolume = Math.max(
      ...bids.map((b) => b.q),
      ...asks.map((a) => a.q),
      0
    );

    return { time: c.time, bids, asks, maxVolume };
  });
}

function setup() {
  const canvas = document.getElementById('chartCanvas');
  if (!canvas) return;

  const state = new ChartState();
  // CandleLayer expects `state.colors` for some fields.
  state.colors = {
    candleBody: 'rgba(30, 34, 45, 0.65)',
  };

  // Make it look “alive”
  state.tickSize = 1;
  state.zoomX = 10;
  state.zoomY = 2;
  state.heatmapOpacity = 0.55;
  state.heatmapIntensityThreshold = 0.05;

  const coords = new CoordinateSystem(state);
  const engine = new RenderEngine(canvas, state, coords);

  // Layers (order matters)
  const heatmapLayer = new HeatmapLayer(state, coords);
  engine.addLayer(heatmapLayer);
  engine.addLayer(new GridLayer());
  engine.addLayer(new CandleLayer());
  engine.addLayer(new AnalysisLayer());
  engine.addLayer(new CrosshairLayer());

  const requestDraw = () => engine.requestDraw();

  // Input wiring
  new InputHandler(canvas, state, coords, requestDraw);

  const candles = makeDemoCandles({
    count: 260,
    startPrice: 10_000,
    tickSize: state.tickSize,
  });

  const heatmapData = makeDemoHeatmap({ candles, tickSize: state.tickSize });

  // Calculate heatmap max
  state.maxVolumeInHistory = Math.max(...heatmapData.map((s) => s.maxVolume || 0), 10);

  state.setData(candles);
  state.setHeatmapData(heatmapData);

  function resize() {
    const rect = canvas.getBoundingClientRect();
    state.pixelRatio = window.devicePixelRatio || 1;
    state.updateDimensions(rect.width, rect.height);

    // Auto-pan so latest candles are near the right edge.
    const totalWidth = candles.length * state.zoomX;
    state.offsetX = Math.min(0, rect.width - totalWidth - 40);

    // Auto-center Y around last close.
    const last = candles[candles.length - 1];
    if (last?.close) {
      // coords.getY(price) = height - ((price/tick*zoomY) + offsetY)
      // Put price at ~55% height.
      const targetY = rect.height * 0.55;
      const value = (rect.height - targetY);
      state.offsetY = value - (last.close / state.tickSize) * state.zoomY;
      state.currentPrice = last.close;
    }

    requestDraw();
  }

  // Redraw when state changes
  state.subscribe(() => requestDraw());

  window.addEventListener('resize', resize);
  resize();

  // Animate current price slightly so the price line moves
  setInterval(() => {
    const last = candles[candles.length - 1];
    if (!last) return;
    state.currentPrice = last.close + (Math.random() - 0.5) * 3;
    requestDraw();
  }, 250);
}

document.addEventListener('DOMContentLoaded', setup);
