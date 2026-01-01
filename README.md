# CryptoFlow - Institutional Footprint & Heatmap Chart

**CryptoFlow** is a high-performance, professional-grade trading visualization tool designed to replicate the depth and precision of institutional platforms like **Bookmap** and **ATAS**.

It features a custom-built Canvas rendering engine capable of displaying tick-level data, 3D wide-imbalance bubbles, and a time-aligned liquidity heatmap.

![CryptoFlow Demo](https://via.placeholder.com/800x400?text=CryptoFlow+Chart+Demo)

## 🚀 Key Features

*   **⚡ High-Performance Canvas Engine**: Zero-lag rendering of thousands of candles and ticks.
*   **🔥 Bookmap-Style Heatmap**:
    *   **Time-Aligned**: Heatmap snapshots are strictly anchored to candle time.
    *   **Gamma Corrected**: Uses a specialized "Heat Gradient" (Blue -> White) with gamma correction (0.4) to visualize true liquidity intensity.
    *   **Drift-Free**: Does not desync from price action.
    *   **Interactive Slider**: Filter out low-liquidity noise in real-time.
*   **🫧 3D Imbalance Bubbles**: Large trades are visualized as 3D spheres with radial gradients for instant visual recognition.
*   **🔍 Advanced Tools**:
    *   **Footprint / DOM**: Detailed volume breakdown inside candles.
    *   **Magnifier Lens (L-Key)**: Hover over candles to see granular tick data.
    *   **Delta Summary**: Integrated Delta and Imbalance bars below the chart.
*   **📅 Smart Axis**: TradingView-style dual labels (Time + Date) for easy navigation.

## 🛠️ Tech Stack

*   **Frontend**: Vanilla JavaScript (No Frameworks), HTML5 Canvas, Vite.
*   **Backend**: Node.js, Express, `better-sqlite3`.
*   **Data**: Binance WebSocket API.
*   **Deployment**: Windows VPS / PM2.

## 📦 Installation (Local Dev)

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/dkay95/ATAS2.git
    cd ATAS2
    ```

2.  **Install Dependencies**
    ```bash
    npm install
    ```

3.  **Start Development Server**
    ```bash
    npm run dev
    ```
    Open `http://localhost:5173`.

## 🚢 Deployment

To deploy to a standard VPS:

1.  **Build**
    ```bash
    npm run build
    ```
2.  **Upload**
    Copy the `dist/` folder and `server/` folder to your VPS.
3.  **Run**
    ```bash
    node server/api.js
    ```

## 📜 License
Private Project. All rights reserved.
