import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import io
import base64
import json
from ml_trainer import load_candles, DB_PATH
import os

class ChartRenderer:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        # Style: Dark theme with distinct colors
        self.style = mpf.make_mpf_style(base_mpf_style='nightclouds', 
                                      rc={'font.size': 7, 'axes.labelsize': 7},
                                      marketcolors=mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True))
        
    def extract_poc(self, clusters_json):
        """Find Price of Control (max vol level) from clusters JSON"""
        try:
            if not clusters_json: return np.nan
            # Support both dict formats: {'price': vol} or {'price': {'b': 1, 's': 1}}
            max_vol = -1
            poc_price = np.nan
            
            for price_str, data in clusters_json.items():
                vol = 0
                if isinstance(data, dict):
                    vol = data.get('b', 0) + data.get('s', 0) + data.get('v', 0) # Handle b/s or v key
                else:
                    vol = float(data)
                
                if vol > max_vol:
                    max_vol = vol
                    poc_price = float(price_str)
            
            return poc_price
        except:
            return np.nan

    def get_data(self, symbol, timeframe, limit=100):
        try:
            df = load_candles(self.db_path, symbol, timeframe)
            if df.empty: return None
            
            # Filter and Sort
            df['date'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('date', inplace=True)
            df = df.sort_index().tail(limit)
            
            # Calculate POC
            df['poc'] = df['clusters'].apply(self.extract_poc)
            
            # Ensure required columns for mpf
            return df[['open', 'high', 'low', 'close', 'volume', 'poc']]
        except Exception as e:
            print(f"Error loading data for {symbol} {timeframe}m: {e}")
            return None

    def render_multi_timeframe(self, symbol):
        """
        Generates a 3-panel chart (15m, 5m, 1m) with POCs.
        """
        # Fetch Data
        df_15m = self.get_data(symbol, 15, limit=50)
        df_5m = self.get_data(symbol, 5, limit=70)
        df_1m = self.get_data(symbol, 1, limit=100)
        
        if df_1m is None: return None

        # setup figure
        fig = mpf.figure(style=self.style, figsize=(14, 10))
        
        # Add subplots
        ax1 = fig.add_subplot(2, 2, 1) # 15m
        ax2 = fig.add_subplot(2, 2, 2) # 5m
        ax3 = fig.add_subplot(2, 1, 2) # 1m
        
        # Helper to plot with POC
        def plot_tf(df, ax, title, vol=False):
            if df is None or df.empty: return
            addplots = []
            # Add POC markers (Blue dots)
            if not df['poc'].isna().all():
                addplots.append(mpf.make_addplot(df['poc'], ax=ax, type='scatter', markersize=15, marker='.', color='#00bfff', alpha=0.8))
            
            mpf.plot(df, type='candle', ax=ax, volume=vol, addplot=addplots, show_nontrading=False, axtitle=title)

        plot_tf(df_15m, ax1, f'{symbol} 15m (Structure + POC)', vol=False)
        plot_tf(df_5m, ax2, f'{symbol} 5m (Trend + POC)', vol=False)
        plot_tf(df_1m, ax3, f'{symbol} 1m (Entry + POC)', vol=True)
        
        # Save
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
