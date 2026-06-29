"""
plot_indicators.py
------------------
Visualize main TA-Lib indicators on 1-minute OHLCV data pulled from the DB.

Layout:
  Panel 1 (price):   Candlestick + BBANDS + SMA(20) + EMA(20) + SAR
  Panel 2 (RSI):     RSI(14) with overbought/oversold lines
  Panel 3 (MACD):    MACD line, signal line, histogram
  Panel 4 (volume):  Volume bars

Usage:
    python plot_indicators.py
"""

from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crypto_pipeline.utils.db_utils import get_db_connection, get_candles_from_db
from crypto_pipeline.indicators.talib_indicators import (
    overlap_bbands,
    overlap_sma,
    overlap_ema,
    overlap_sar,
    momentum_rsi,
    momentum_macd,
)

# ---------------------------------------------------------------------------
# Config — change these to match what you have in the DB
# ---------------------------------------------------------------------------

EXCHANGE = "binance"
SYMBOL   = "btc"
START = datetime(2026, 6, 29, 0, 0, 0)
END   = datetime(2026, 6, 29, 5, 41, 0)

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

conn = get_db_connection()
df   = get_candles_from_db(conn, EXCHANGE, SYMBOL, START, END)
conn.close()

if df.empty:
    raise ValueError(f"No data found for {EXCHANGE} | {SYMBOL} between {START} and {END}. "
                     "Check your DB or adjust the date range.")

print(f"Loaded {len(df)} candles from DB.")
print(df.head())

# ---------------------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------------------

bbands  = overlap_bbands(df, period=20)
sma20   = overlap_sma(df, period=20)
ema20   = overlap_ema(df, period=20)
sar     = overlap_sar(df)
rsi     = momentum_rsi(df, period=14)
macd    = momentum_macd(df)

# ---------------------------------------------------------------------------
# Build plot
# ---------------------------------------------------------------------------

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.17, 0.17, 0.16],
    subplot_titles=("BTCUSDT 1m — Price + Overlays", "RSI (14)", "MACD (12/26/9)", "Volume"),
)

# --- Panel 1: Candlestick ---
fig.add_trace(go.Candlestick(
    x=df["datetime"],
    open=df["open"], high=df["high"],
    low=df["low"],   close=df["close"],
    name="OHLC",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
), row=1, col=1)

# BBANDS
fig.add_trace(go.Scatter(
    x=df["datetime"], y=bbands["upper"],
    name="BB Upper", line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df["datetime"], y=bbands["middle"],
    name="BB Middle", line=dict(color="rgba(100,149,237,0.9)", width=1),
    fill=None,
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df["datetime"], y=bbands["lower"],
    name="BB Lower", line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(100,149,237,0.05)",
), row=1, col=1)

# SMA / EMA
fig.add_trace(go.Scatter(
    x=df["datetime"], y=sma20,
    name="SMA 20", line=dict(color="#f59e0b", width=1.2),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df["datetime"], y=ema20,
    name="EMA 20", line=dict(color="#a78bfa", width=1.2),
), row=1, col=1)

# SAR (dots, not a line)
fig.add_trace(go.Scatter(
    x=df["datetime"], y=sar,
    name="SAR", mode="markers",
    marker=dict(size=2, color="#f97316"),
), row=1, col=1)

# --- Panel 2: RSI ---
fig.add_trace(go.Scatter(
    x=df["datetime"], y=rsi,
    name="RSI 14", line=dict(color="#818cf8", width=1.2),
), row=2, col=1)

# Overbought / oversold reference lines
for level, color in [(70, "rgba(239,83,80,0.5)"), (30, "rgba(38,166,154,0.5)")]:
    fig.add_hline(y=level, line=dict(color=color, width=1, dash="dash"), row=2, col=1)

fig.add_hline(y=50, line=dict(color="rgba(150,150,150,0.3)", width=1), row=2, col=1)

# --- Panel 3: MACD ---
# Histogram colored by positive/negative
hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd["hist"].fillna(0)]

fig.add_trace(go.Bar(
    x=df["datetime"], y=macd["hist"],
    name="MACD Hist", marker_color=hist_colors, opacity=0.6,
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df["datetime"], y=macd["macd"],
    name="MACD", line=dict(color="#60a5fa", width=1.2),
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df["datetime"], y=macd["signal"],
    name="Signal", line=dict(color="#f97316", width=1.2),
), row=3, col=1)

# --- Panel 4: Volume ---
vol_colors = [
    "#26a69a" if df["close"].iloc[i] >= df["open"].iloc[i] else "#ef5350"
    for i in range(len(df))
]

fig.add_trace(go.Bar(
    x=df["datetime"], y=df["volume"],
    name="Volume", marker_color=vol_colors, opacity=0.7,
), row=4, col=1)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

fig.update_layout(
    title=f"BTCUSDT 1m — {START.date()} | Indicators Sanity Check",
    height=900,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=50, r=50, t=80, b=40),
)

# Y-axis labels
fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
fig.update_yaxes(title_text="RSI",          row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD",         row=3, col=1)
fig.update_yaxes(title_text="Volume",       row=4, col=1)

fig.show()