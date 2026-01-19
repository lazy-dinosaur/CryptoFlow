# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the frontend Vite app (Canvas chart engine, UI components, services, and utilities).
- `server/` hosts the Node/Express API, collectors, and ML service scripts; SQLite data lives under `server/data/`.
- `backtest/` contains Python backtesting/analysis scripts, datasets, and outputs (`backtest/data/`, `backtest/results/`).
- `docs/`, `research/`, and `pinescripts/` hold supplemental documentation and experiments.
- `dist/` is the build output; treat it as generated.

## Build, Test, and Development Commands
- `npm install` installs frontend dependencies.
- `npm run dev` starts the Vite dev server at `http://localhost:5173`.
- `npm run build` builds the frontend into `dist/`.
- `npm run preview` serves the production build locally.
- `npm --prefix server install` installs server dependencies.
- `npm --prefix server run start` or `node server/api.js` starts the API server (default port 5001).
- `npm --prefix server run start:collector` runs the standalone market data collector.
- `python3 server/ml_service.py` runs the ML service when needed.

## Coding Style & Naming Conventions
- JavaScript uses ES modules, 4-space indentation, and semicolons; keep imports organized by feature area.
- Component files in `src/components/` are PascalCase (e.g., `FootprintChart.js`); service modules in `src/services/` use camelCase filenames.
- Python scripts use snake_case filenames and functions.

## Testing Guidelines
- There is no formal unit test runner in this repo.
- Backtesting and validation live in `backtest/` with scripts named `test_*.py` or `ml_test_*.py`.
- Run targeted scripts directly, e.g., `python backtest/run_backtest.py`, and review outputs in `backtest/results/`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits as seen in history: `feat: ...`, `fix: ...`, `docs: ...`.
- PRs should include a clear description, reproducible steps, and screenshots for UI changes.
- Call out data/schema changes (e.g., `server/data/cryptoflow.db`) and any migration or re-seed steps.

## Security & Configuration Tips
- Avoid committing credentials or API keys; use local environment configuration.
- The SQLite DB is stored at `server/data/cryptoflow.db`; back it up before destructive changes.
- `ecosystem.config.cjs` and `setup-*` scripts are used for VPS/Windows setup; update them when deployment steps change.
