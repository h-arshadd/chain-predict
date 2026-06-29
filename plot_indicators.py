"""
plot_indicators.py
------------------
Sanity-check each indicator one at a time — every indicator gets its own
standalone interactive HTML chart so you can inspect them individually.

Layout per chart:
  - Price panel always on top (candlestick + any price-overlay indicators)
  - Indicator panel below (for oscillators / sub-panel indicators)

10 indicators tested:
  1.  RSI (14)
  2.  MACD (12/26/9)
  3.  Bollinger Bands (20)
  4.  EMA (20)
  5.  ATR (14)
  6.  Stochastic (5/3/3)
  7.  ADX (14)
  8.  CCI (14)
  9.  OBV
  10. Parabolic SAR

Output:
  One HTML file per indicator saved to ./indicator_charts/
  Open any file in a browser — fully interactive (zoom, hover, etc.)

Usage:
    python plot_indicators.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crypto_pipeline.utils.db_utils import get_db_connection, get_candles_from_db
from crypto_pipeline.indicators.talib_indicators import (
    # overlap
    overlap_bbands,
    overlap_ema,
    overlap_sar,
    # momentum
    momentum_rsi,
    momentum_macd,
    momentum_adx,
    momentum_cci,
    momentum_stoch,
    # volatility
    volatility_atr,
    # volume
    volume_obv,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXCHANGE   = "binance"
SYMBOL     = "btc"
START      = datetime(2026, 6, 29, 0, 0, 0)
END        = datetime(2026, 6, 29, 5, 41, 0)
OUTPUT_DIR = "indicator_charts"

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = get_db_connection()
df   = get_candles_from_db(conn, EXCHANGE, SYMBOL, START, END)
conn.close()

if df.empty:
    raise ValueError(
        f"No data found for {EXCHANGE} | {SYMBOL} between {START} and {END}. "
        "Check your DB or adjust the date range."
    )

print(f"Loaded {len(df)} candles  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

# ---------------------------------------------------------------------------
# Compute all indicators up front
# ---------------------------------------------------------------------------

bbands = overlap_bbands(df, period=20)
ema20  = overlap_ema(df, period=20)
sar    = overlap_sar(df)
rsi    = momentum_rsi(df, period=14)
macd   = momentum_macd(df)
adx    = momentum_adx(df, period=14)
cci    = momentum_cci(df, period=14)
stoch  = momentum_stoch(df)
atr    = volatility_atr(df, period=14)
obv    = volume_obv(df)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANDLE_UP = "#26a69a"
CANDLE_DN = "#ef5350"

COMMON_LAYOUT = dict(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=60, r=40, t=80, b=40),
    height=700,
)


def candle_trace(df):
    return go.Candlestick(
        x=df["datetime"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="OHLC",
        increasing_line_color=CANDLE_UP,
        decreasing_line_color=CANDLE_DN,
    )


def vol_colors(df):
    return [CANDLE_UP if df["close"].iloc[i] >= df["open"].iloc[i]
            else CANDLE_DN for i in range(len(df))]


def make_fig_2panel(title, price_row_height=0.65, sub_title=""):
    """Price panel on top, indicator sub-panel below. Returns (fig, price_row, ind_row)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[price_row_height, 1 - price_row_height],
        subplot_titles=(f"Price — {title}", sub_title),
    )
    fig.update_layout(title=title, **COMMON_LAYOUT)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    return fig


def make_fig_1panel(title):
    """Single panel — indicator overlaid on price."""
    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(title=title, **COMMON_LAYOUT)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    return fig


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    print(f"  ✓  {path}")


def sanity_print(name, series_or_dict):
    """Print basic stats to console for a quick eyes-on check."""
    items = series_or_dict.items() if isinstance(series_or_dict, dict) else [(name, series_or_dict)]
    for label, s in items:
        valid = s.notna().sum()
        total = len(s)
        print(f"     {label:30s}  valid={valid}/{total}  "
              f"min={s.min():.4f}  max={s.max():.4f}")
        if valid == 0:
            print(f"     ⚠  ALL NaN — data too short for period, or wrong columns.")
        elif valid < total * 0.5:
            print(f"     ⚠  More than half NaN — check period vs data length.")


# ---------------------------------------------------------------------------
# 1. RSI
# ---------------------------------------------------------------------------

print("\n[01] RSI (14)")
sanity_print("rsi", rsi)

fig = make_fig_2panel("RSI (14)  —  range [0,100], >70 overbought, <30 oversold",
                      sub_title="RSI (14)")
fig.add_trace(candle_trace(df), row=1, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=rsi,
                         name="RSI 14", line=dict(color="#818cf8", width=1.4)),
              row=2, col=1)
fig.add_hline(y=70, line=dict(color="rgba(239,83,80,0.6)",   width=1, dash="dash"), row=2, col=1)
fig.add_hline(y=30, line=dict(color="rgba(38,166,154,0.6)",  width=1, dash="dash"), row=2, col=1)
fig.add_hline(y=50, line=dict(color="rgba(150,150,150,0.3)", width=1),              row=2, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
save(fig, "01_rsi.html")

# ---------------------------------------------------------------------------
# 2. MACD
# ---------------------------------------------------------------------------

print("\n[02] MACD (12/26/9)")
sanity_print("macd", macd)

hist_colors = [CANDLE_UP if v >= 0 else CANDLE_DN for v in macd["hist"].fillna(0)]

fig = make_fig_2panel("MACD (12/26/9)  —  macd, signal, histogram", sub_title="MACD (12/26/9)")
fig.add_trace(candle_trace(df), row=1, col=1)
fig.add_trace(go.Bar(x=df["datetime"], y=macd["hist"],
                     name="MACD Hist", marker_color=hist_colors, opacity=0.6), row=2, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=macd["macd"],
                         name="MACD", line=dict(color="#60a5fa", width=1.3)), row=2, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=macd["signal"],
                         name="Signal", line=dict(color="#f97316", width=1.3)), row=2, col=1)
fig.update_yaxes(title_text="MACD", row=2, col=1)
save(fig, "02_macd.html")

# ---------------------------------------------------------------------------
# 3. Bollinger Bands  (overlaid on price — single panel)
# ---------------------------------------------------------------------------

print("\n[03] Bollinger Bands (20, 2σ)")
sanity_print("bbands", bbands)

fig = make_fig_1panel("Bollinger Bands (20, 2σ)  —  upper / middle / lower overlaid on price")
fig.add_trace(candle_trace(df), row=1, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=bbands["upper"],
                         name="BB Upper", line=dict(color="rgba(100,149,237,0.7)", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=df["datetime"], y=bbands["middle"],
                         name="BB Middle", line=dict(color="rgba(100,149,237,1.0)", width=1)))
fig.add_trace(go.Scatter(x=df["datetime"], y=bbands["lower"],
                         name="BB Lower", line=dict(color="rgba(100,149,237,0.7)", width=1, dash="dot"),
                         fill="tonexty", fillcolor="rgba(100,149,237,0.06)"))
save(fig, "03_bbands.html")

# ---------------------------------------------------------------------------
# 4. EMA (20)  (overlaid on price — single panel)
# ---------------------------------------------------------------------------

print("\n[04] EMA (20)")
sanity_print("ema20", ema20)

fig = make_fig_1panel("EMA (20)  —  overlaid on price")
fig.add_trace(candle_trace(df), row=1, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=ema20,
                         name="EMA 20", line=dict(color="#a78bfa", width=1.5)))
save(fig, "04_ema.html")

# ---------------------------------------------------------------------------
# 5. ATR (14)
# ---------------------------------------------------------------------------

print("\n[05] ATR (14)")
sanity_print("atr", atr)

fig = make_fig_2panel("ATR (14)  —  average true range (volatility in price units)",
                      sub_title="ATR (14)")
fig.add_trace(candle_trace(df), row=1, col=1)
fig.add_trace(go.Scatter(x=df["datetime"], y=atr,
                         name="ATR 14", line=dict(color="#34d399", width=1.3)),
              row=2, col=1)
fig.update_yaxes(title_text="ATR", row=2, col=1)
save(fig, "05_atr.html")


print(f"\nDone. All charts saved to ./{OUTPUT_DIR}/")
print("Open any .html file in your browser — fully interactive.\n")