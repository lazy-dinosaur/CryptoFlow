# Chart Engine Knowledge Base

## OVERVIEW

Layered Canvas 2D rendering engine for high-performance orderflow visualization. No framework dependencies.

## STRUCTURE

```
chart/
├── core/
│   ├── ChartState.js       # Viewport, zoom, pan state
│   ├── CoordinateSystem.js # Price↔pixel, time↔x mapping
│   ├── RenderEngine.js     # Layer compositor, render loop
│   └── InputHandler.js     # Mouse/keyboard/touch events
└── layers/
    ├── GridLayer.js        # Price/time grid lines
    ├── CandleLayer.js      # OHLC candle rendering
    ├── HeatmapLayer.js     # Bookmap-style depth heatmap
    ├── AnalysisLayer.js    # Overlays, drawings, indicators
    └── CrosshairLayer.js   # Cursor tracking, tooltips
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Zoom/pan behavior | `core/ChartState.js` | Viewport bounds |
| Price scaling | `core/CoordinateSystem.js` | `priceToY()`, `yToPrice()` |
| Add new layer | `layers/` + `RenderEngine.js` | Register in render order |
| Heatmap colors | `layers/HeatmapLayer.js` | Gamma-corrected gradient |
| Candle styles | `layers/CandleLayer.js` | Fill, wick, colors |

## RENDER PIPELINE

```
InputHandler (events)
    ↓
ChartState (viewport update)
    ↓
RenderEngine.render()
    ↓
[GridLayer → CandleLayer → HeatmapLayer → AnalysisLayer → CrosshairLayer]
    ↓
Canvas 2D context
```

## CONVENTIONS

- Layers render bottom-to-top (Grid first, Crosshair last)
- All coordinates through `CoordinateSystem` - never raw pixel math
- State changes trigger full re-render via `requestAnimationFrame`

## ANTI-PATTERNS

### Coordinate System Alignment
```javascript
// NOTE: CoordinateSystem may not match FootprintChart.js implementation
// Test BOTH when modifying coordinate transforms
```

### Heatmap Buffer Sizing
```javascript
// IMPORTANT: Buffer = VIEWPORT size, not entire history
// Caching 24h of heatmap data is impossible
```

### Time-to-Index Assumption
```javascript
// TODO: Weak assumption - assumes 1:1 candle-to-snapshot mapping
// Will break if data not pre-aligned
```

## NOTES

- Performance target: 60fps with 2000+ candles visible
- Heatmap uses gamma correction (0.4) for heat gradient
- Touch events normalized to mouse events in InputHandler
