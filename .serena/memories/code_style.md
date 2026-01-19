# CryptoFlow - Code Style & Conventions

## JavaScript
- **Module System**: ES modules (`import`/`export`)
- **Indentation**: 4 spaces
- **Semicolons**: Required
- **Quotes**: Single quotes preferred

### File Naming
| Type | Convention | Example |
|------|------------|---------|
| Components (src/components/) | PascalCase | `FootprintChart.js`, `MLDashboard.js` |
| Services (src/services/) | camelCase | `dataAggregator.js`, `binanceWS.js` |

### Patterns
- **Event-Driven Architecture**: Services emit events (`candleUpdate`, `trade`, `statsUpdate`) consumed by UI components
- **Canvas Rendering**: All visual logic uses vanilla JS Canvas 2D (no frameworks)

## Python
- **File Naming**: snake_case (`ml_service.py`, `run_backtest.py`)
- **Function/Variable Naming**: snake_case
- **No specific framework** - uses standard pandas/numpy patterns

## Git Commits
Use **Conventional Commits** format:
```
feat: add new feature
fix: bug fix
docs: documentation changes
refactor: code refactoring
test: add tests
chore: maintenance tasks
```

## Key Visual Features (DO NOT BREAK)
1. **Bookmap Heatmap**: Blue→Cyan→Green→Yellow→Red→White gradient, time-aligned
2. **3D Bubbles**: Big trades as 3D spheres with radial gradients (Green/Red)
3. **Smart Axis**: TradingView-style dual labels (Time + Date)
4. **Zoom Logic**: Deep squeeze support (`minZoomX: 0.5`)
