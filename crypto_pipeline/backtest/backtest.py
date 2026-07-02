"""
backtest.py
-----------
Vectorized backtesting engine.

Given 1-minute OHLCV data, a signal series (which may come from any
timeframe), and a backtest config, simulates trade execution and returns
the trade ledger plus summary results.

Notes on "vectorized":
  - Finding stop-loss/take-profit exits uses NumPy boolean arrays and
    np.argmax to locate the first hit, not a per-bar Python loop. This is
    the part that scales with data size (thousands of 1-minute bars).
  - We do loop once per *signal* (not once per 1-minute bar). This is
    unavoidable and cheap: max_open_positions=1 means trades are opened
    strictly one after another, and position sizing compounds off the
    running balance, so trade N's size depends on trade N-1's outcome.
    Signals are far fewer than 1-minute bars, so this loop is negligible.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path=None) -> dict:
    """Load the backtest configuration from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _position_size(balance, config, entry_price):
    """Quantity to trade, given the current balance and config.position_size."""
    sizing = config["position_size"]
    if sizing["type"] != "fixed_percentage":
        raise ValueError(f"Unsupported position_size type: {sizing['type']}")
    position_value = balance * (sizing["value"] / 100)
    return position_value / entry_price


def _tp_sl_prices(entry_price, direction, config):
    """Take-profit / stop-loss price levels for a new position."""
    if config["take_profit"]["type"] != "percentage" or config["stop_loss"]["type"] != "percentage":
        raise NotImplementedError("Version 1 only supports take_profit/stop_loss type = percentage")
    tp_pct = config["take_profit"]["value"] / 100
    sl_pct = config["stop_loss"]["value"] / 100
    if direction == 1:  # long
        take_profit = entry_price * (1 + tp_pct)
        stop_loss = entry_price * (1 - sl_pct)
    else:  # short
        take_profit = entry_price * (1 - tp_pct)
        stop_loss = entry_price * (1 + sl_pct)
    return take_profit, stop_loss


def _find_exit(high, low, open_, entry_idx, direction, take_profit, stop_loss):
    """
    Vectorized search for the first bar after entry_idx where price touches
    take_profit or stop_loss. Returns (exit_idx, exit_price).

    If neither level is ever touched before the data runs out, the position
    is closed on the last available bar's open.
    """
    future_high = high[entry_idx + 1:]
    future_low = low[entry_idx + 1:]

    if direction == 1:  # long: TP hit when high >= tp, SL hit when low <= sl
        tp_hit = future_high >= take_profit
        sl_hit = future_low <= stop_loss
    else:  # short: TP hit when low <= tp, SL hit when high >= sl
        tp_hit = future_low <= take_profit
        sl_hit = future_high >= stop_loss

    hit = tp_hit | sl_hit
    if not hit.any():
        last_idx = len(high) - 1
        return last_idx, open_[last_idx]

    first_hit_offset = np.argmax(hit)
    exit_idx = entry_idx + 1 + first_hit_offset

    # If TP and SL are both touched within the same bar, we can't tell from
    # OHLC alone which came first. Assume the worse outcome (stop-loss).
    if sl_hit[first_hit_offset]:
        exit_price = stop_loss
    else:
        exit_price = take_profit

    return exit_idx, exit_price


def run_backtest(ohlcv: pd.DataFrame, signals: pd.DataFrame, config: dict = None) -> dict:
    """
    Run the vectorized backtest.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        1-minute OHLCV data. Must contain: datetime, open, high, low, close.
    signals : pd.DataFrame
        Signal data with columns: datetime, signal (1=Buy, -1=Sell, 0=No signal).
        May be on any timeframe -- entries always execute on the 1-minute grid.
    config : dict, optional
        Backtest configuration. Loaded from config.yaml if not given.

    Returns
    -------
    dict with keys:
        trade_ledger, equity_curve, balance_history, drawdown_series,
        final_balance, total_net_profit, total_trades, win_loss
    """
    config = config or load_config()

    if config.get("entry_price") != "next_open" or config.get("exit_price") != "next_open":
        raise NotImplementedError("Version 1 only supports entry_price/exit_price = next_open")
    if config.get("max_open_positions", 1) != 1:
        raise NotImplementedError("Version 1 only supports max_open_positions = 1")

    ohlcv = ohlcv.sort_values("datetime").reset_index(drop=True)
    signals = signals.sort_values("datetime").reset_index(drop=True)

    # Map every non-zero signal onto a row of the 1-minute series. merge_asof
    # (backward) finds, for each signal timestamp, the 1-minute bar at or
    # just before it -- this is what lets signals generated on a coarser
    # timeframe (e.g. 1h) line up with the 1-minute execution grid.
    signal_events = signals[signals["signal"] != 0][["datetime", "signal"]]
    bar_index = ohlcv[["datetime"]].reset_index().rename(columns={"index": "bar_idx"})
    aligned = pd.merge_asof(signal_events, bar_index, on="datetime", direction="backward")
    aligned = aligned.dropna(subset=["bar_idx"])
    aligned["bar_idx"] = aligned["bar_idx"].astype(int)

    open_ = ohlcv["open"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    datetimes = ohlcv["datetime"].to_numpy()
    n_bars = len(ohlcv)

    allow_long = config.get("allow_long", True)
    allow_short = config.get("allow_short", True)
    commission_rate = config["commission"]
    slippage_rate = config["slippage"]
    initial_balance = config["initial_balance"]

    balance = initial_balance
    trades = []
    next_free_idx = 0  # earliest bar index a new trade may enter on

    # Loop over the (small) list of signal events, not over the OHLCV
    # DataFrame -- plain NumPy arrays, not .iterrows(), to stay out of
    # pandas row-by-row territory even here.
    signal_bar_idxs = aligned["bar_idx"].to_numpy()
    signal_directions = aligned["signal"].to_numpy()

    for signal_bar_idx, direction in zip(signal_bar_idxs, signal_directions):
        direction = int(direction)  # 1 = long, -1 = short

        if direction == 1 and not allow_long:
            continue
        if direction == -1 and not allow_short:
            continue

        entry_idx = signal_bar_idx + 1  # Step 2: entry price = next candle open
        if entry_idx <= next_free_idx or entry_idx >= n_bars:
            # A position is already open through this bar, or there's no
            # next candle left to enter on -- skip this signal.
            continue

        # Step 2: Entry Price = Next Candle Open
        entry_price = open_[entry_idx]

        quantity = _position_size(balance, config, entry_price)
        take_profit, stop_loss = _tp_sl_prices(entry_price, direction, config)

        # Step 4: exit at whichever of take_profit/stop_loss is touched first
        # (or the last available open if neither is ever touched).
        exit_idx, exit_price = _find_exit(high, low, open_, entry_idx, direction, take_profit, stop_loss)

        # Step 5: commission and slippage are each charged on entry and exit,
        # then deducted from gross PnL (Step 6) -- kept as separate ledger
        # columns rather than baked into entry_price/exit_price, so
        # Net PnL = Gross PnL - Commission - Slippage reconciles directly
        # from the ledger's own columns.
        entry_commission = entry_price * quantity * commission_rate
        exit_commission = exit_price * quantity * commission_rate
        total_commission = entry_commission + exit_commission

        entry_slippage = entry_price * quantity * slippage_rate
        exit_slippage = exit_price * quantity * slippage_rate
        total_slippage = entry_slippage + exit_slippage

        # Step 6: Gross / Net PnL
        if direction == 1:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        net_pnl = gross_pnl - total_commission - total_slippage

        # Step 7: New Balance = Previous Balance + Net PnL
        balance += net_pnl

        trades.append({
            "entry_time": datetimes[entry_idx],
            "exit_time": datetimes[exit_idx],
            "direction": "long" if direction == 1 else "short",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_pnl": gross_pnl,
            "commission": total_commission,
            "slippage": total_slippage,
            "net_pnl": net_pnl,
            "balance_after_trade": balance,
        })

        next_free_idx = exit_idx

    trade_ledger = pd.DataFrame(trades)

    # Equity curve on the full 1-minute timeline: balance is flat between
    # trades and steps at each trade's exit. Built with merge_asof (a step
    # function via "as of" lookup), not a per-bar loop.
    equity_events = pd.DataFrame({
        "datetime": [ohlcv["datetime"].iloc[0]] + [t["exit_time"] for t in trades],
        "balance": [initial_balance] + [t["balance_after_trade"] for t in trades],
    })
    equity_curve = pd.merge_asof(ohlcv[["datetime"]], equity_events, on="datetime")
    equity_curve = pd.Series(equity_curve["balance"].to_numpy(), index=ohlcv["datetime"])

    running_max = equity_curve.cummax()
    drawdown_series = (equity_curve - running_max) / running_max

    balance_history = pd.Series([initial_balance] + [t["balance_after_trade"] for t in trades])

    total_trades = len(trade_ledger)
    if total_trades > 0:
        wins = int((trade_ledger["net_pnl"] > 0).sum())
        losses = int((trade_ledger["net_pnl"] <= 0).sum())
        win_rate = wins / total_trades
    else:
        wins = losses = 0
        win_rate = 0.0

    return {
        "trade_ledger": trade_ledger,
        "equity_curve": equity_curve,
        "balance_history": balance_history,
        "drawdown_series": drawdown_series,
        "final_balance": balance,
        "total_net_profit": balance - initial_balance,
        "total_trades": total_trades,
        "win_loss": {"wins": wins, "losses": losses, "win_rate": win_rate},
    }