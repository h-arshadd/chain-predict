"""
plot_indicators.py
------------------
Sanity-check each indicator one at a time — every indicator gets its own
standalone interactive HTML chart so you can inspect them individually.

Now runs for BOTH binance and bybit, each into its own output subfolder:
  ./indicator_charts/binance/
  ./indicator_charts/bybit/

Layout per chart:
  - Price panel always on top (candlestick + any price-overlay indicators)
  - Indicator panel below (for oscillators / sub-panel indicators)

5 indicators tested:
  1.  RSI (14)
  2.  MACD (12/26/9)
  3.  Bollinger Bands (20)
  4.  EMA (20)
  5.  ATR (14)

Data comes from data_downloader.get_data(), not a direct DB read — get_data
already merges whatever's stored with any live gap (fetching from the
exchange itself if the DB hasn't caught up) and resamples it into the
configured timeframe before handing it back.

Output (per exchange):
  - One HTML file per indicator saved to ./indicator_charts/<exchange>/
    Open any file in a browser — fully interactive (zoom, hover, etc.)
  - One CSV per indicator saved alongside it (e.g. 01_rsi_values.csv) with
    open/high/low/close + that indicator's exact plotted values, so you can
    cross-check the numbers by hand.

Usage:
    python plot_indicators.py
"""

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.indicators.talib_indicators import (
    # overlap
    overlap_bbands,
    overlap_ema,
    # momentum
    momentum_rsi,
    momentum_macd,
    # volatility
    volatility_atr,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL          = "btc"
START           = datetime(2026, 6, 29, 0, 0, 0)
END             = datetime(2026, 6, 29, 5, 41, 0)
BASE_OUTPUT_DIR = "indicator_charts"

# Exchange names to run (must match a key in data_downloader's
# _EXCHANGE_FETCHERS, i.e. "binance"/"bybit"). Add/remove here.
EXCHANGES = ["binance", "bybit"]

# ---------------------------------------------------------------------------
# Helpers (exchange-agnostic)
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


def make_fig_2panel(title, price_row_height=0.65, sub_title=""):
    """Price panel on top, indicator sub-panel below. Returns fig."""
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


def save(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
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


def save_values_csv(df, output_dir, filename, columns: dict):
    """
    Dump datetime + whatever columns you pass in to a CSV next to the HTML
    chart, so every number that went into the plot is on disk and you can
    cross-check it by hand.

    columns: dict of {column_name: pd.Series}, all already aligned to `df`.
    """
    out = pd.DataFrame({"datetime": df["datetime"]})
    for col_name, series in columns.items():
        out[col_name] = series.values
    path = os.path.join(output_dir, filename)
    out.to_csv(path, index=False)
    print(f"  ✓  {path}")
    return out


# ---------------------------------------------------------------------------
# Per-exchange pipeline
# ---------------------------------------------------------------------------

def run_for_exchange(exchange_name):
    print(f"\n{'=' * 70}")
    print(f"  {exchange_name.upper()}")
    print(f"{'=' * 70}")

    output_dir = os.path.join(BASE_OUTPUT_DIR, exchange_name)
    os.makedirs(output_dir, exist_ok=True)

    df = get_data(exchange=exchange_name, symbol=SYMBOL, start_date=START, end_date=END)["resampled"]

    if df.empty:
        print(
            f"No data found for {exchange_name} | {SYMBOL} between {START} and {END}. "
            "Check your DB or adjust the date range. Skipping this exchange."
        )
        return

    print(f"Loaded {len(df)} candles  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

    # -----------------------------------------------------------------
    # Compute all indicators up front
    # -----------------------------------------------------------------

    bbands = overlap_bbands(df, period=20)
    ema20  = overlap_ema(df, period=20)
    rsi    = momentum_rsi(df, period=14)
    macd   = momentum_macd(df)
    atr    = volatility_atr(df, period=14)

    # -----------------------------------------------------------------
    # Drop leading NaN rows so every chart starts where all indicators are valid
    # -----------------------------------------------------------------

    # Find the first index where every indicator has a value.
    # MACD (26 slow + 9 signal) is the slowest so it drives this.
    valid_from = max(
        rsi.first_valid_index(),
        macd["macd"].first_valid_index(),
        bbands["upper"].first_valid_index(),
        ema20.first_valid_index(),
        atr.first_valid_index(),
    )

    df     = df.iloc[valid_from:].reset_index(drop=True)
    bbands = {k: v.iloc[valid_from:].reset_index(drop=True) for k, v in bbands.items()}
    ema20  = ema20.iloc[valid_from:].reset_index(drop=True)
    rsi    = rsi.iloc[valid_from:].reset_index(drop=True)
    macd   = {k: v.iloc[valid_from:].reset_index(drop=True) for k, v in macd.items()}
    atr    = atr.iloc[valid_from:].reset_index(drop=True)

    print(f"After NaN trim: {len(df)} candles  ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

    # -----------------------------------------------------------------
    # 1. RSI
    # -----------------------------------------------------------------

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
    save(fig, output_dir, "01_rsi.html")
    save_values_csv(df, output_dir, "01_rsi_values.csv", {
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "rsi_14": rsi,
    })

    # -----------------------------------------------------------------
    # 2. MACD
    # -----------------------------------------------------------------

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
    save(fig, output_dir, "02_macd.html")
    save_values_csv(df, output_dir, "02_macd_values.csv", {
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "macd": macd["macd"], "signal": macd["signal"], "hist": macd["hist"],
    })

    # -----------------------------------------------------------------
    # 3. Bollinger Bands  (overlaid on price — single panel)
    # -----------------------------------------------------------------

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
    save(fig, output_dir, "03_bbands.html")
    save_values_csv(df, output_dir, "03_bbands_values.csv", {
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "bb_upper": bbands["upper"], "bb_middle": bbands["middle"], "bb_lower": bbands["lower"],
    })

    # -----------------------------------------------------------------
    # 4. EMA (20)  (overlaid on price — single panel)
    # -----------------------------------------------------------------

    print("\n[04] EMA (20)")
    sanity_print("ema20", ema20)

    fig = make_fig_1panel("EMA (20)  —  overlaid on price")
    fig.add_trace(candle_trace(df), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=ema20,
                             name="EMA 20", line=dict(color="#a78bfa", width=1.5)))
    save(fig, output_dir, "04_ema.html")
    save_values_csv(df, output_dir, "04_ema_values.csv", {
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "ema_20": ema20,
    })

    # -----------------------------------------------------------------
    # 5. ATR (14)
    # -----------------------------------------------------------------

    print("\n[05] ATR (14)")
    sanity_print("atr", atr)

    fig = make_fig_2panel("ATR (14)  —  average true range (volatility in price units)",
                          sub_title="ATR (14)")
    fig.add_trace(candle_trace(df), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=atr,
                             name="ATR 14", line=dict(color="#34d399", width=1.3)),
                  row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=2, col=1)
    save(fig, output_dir, "05_atr.html")
    save_values_csv(df, output_dir, "05_atr_values.csv", {
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "atr_14": atr,
    })

    print(f"\nDone with {exchange_name}. All charts + value CSVs saved to ./{output_dir}/")


# ---------------------------------------------------------------------------
# Run for every exchange
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for exchange_name in EXCHANGES:
        run_for_exchange(exchange_name)

    print("\nAll exchanges complete.")
    print("Open any .html file in your browser — fully interactive.")
    print("Open any .csv file to cross-check exact numbers.\n")