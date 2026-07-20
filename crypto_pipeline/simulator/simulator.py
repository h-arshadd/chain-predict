"""
simulator.py
------------
Simulation engine. Processes 1-minute OHLCV candles one at a time, exactly
as if they were arriving live, and executes trades against a strategy's
signals.

Unlike backtest.py (vectorized, whole-history-at-once), this walks forward
candle by candle so it can be called repeatedly (once per scheduler run)
and resume exactly where it left off, using the Position Table it was
given as the current state.

Position state (open position, if any) and the Trade Ledger both come in
and go back out as plain dicts/DataFrames -- main.py owns loading them from
and saving them to the DB. This file only knows how to advance the
simulation forward, not how to persist anything.
"""

from pathlib import Path

import yaml


def load_config(config_path=None) -> dict:
    """Load the simulator configuration from config.yaml."""
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
    tp_pct = config["take_profit"]["value"] / 100
    sl_pct = config["stop_loss"]["value"] / 100
    if direction == 1:  # long
        take_profit = entry_price * (1 + tp_pct)
        stop_loss = entry_price * (1 - sl_pct)
    else:  # short
        take_profit = entry_price * (1 - tp_pct)
        stop_loss = entry_price * (1 + sl_pct)
    return take_profit, stop_loss


def _open_position(candle, direction, balance, config):
    """Step 3: Open Position. Returns a new position dict."""
    entry_price = float(candle["open"])
    quantity = float(_position_size(balance, config, entry_price))
    take_profit, stop_loss = _tp_sl_prices(entry_price, direction, config)

    # float(...) on every numeric field here, not just entry_price: numpy
    # scalar arithmetic is "sticky" -- entry_price * anything stays
    # numpy.float64 even when the other operand is a plain Python float, so
    # take_profit/stop_loss/quantity could still end up numpy even if only
    # entry_price started out that way. Cast everything at this single
    # return point so a position dict is guaranteed to hold plain Python
    # types, however it was constructed above.
    return {
        "direction": "long" if direction == 1 else "short",
        "entry_time": candle["datetime"],
        "entry_price": entry_price,
        "quantity": quantity,
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "current_price": entry_price,
        "unrealized_pnl": 0.0,
        "status": "open",
    }


def _check_exit(position, candle):
    """
    Step 5: Check Exit Conditions against the current candle's high/low.
    Returns (exit_price, exit_reason) or (None, None) if no exit triggered
    by price this candle. Opposite-signal and reversal exits are handled
    separately in step_candle(), since those depend on the signal, not price.
    """
    direction = 1 if position["direction"] == "long" else -1
    high, low = candle["high"], candle["low"]

    if direction == 1:
        tp_hit = high >= position["take_profit"]
        sl_hit = low <= position["stop_loss"]
    else:
        tp_hit = low <= position["take_profit"]
        sl_hit = high >= position["stop_loss"]

    # Both touched in the same candle: can't tell which came first from
    # OHLC alone, assume the worse outcome (stop-loss) -- same convention
    # as backtest.py's _find_exit.
    if sl_hit:
        return position["stop_loss"], "stop_loss"
    if tp_hit:
        return position["take_profit"], "take_profit"
    return None, None


def _close_position(position, exit_price, exit_time, exit_reason, balance, config):
    """Step 6: Close Position. Returns (trade record dict, new balance)."""
    direction = 1 if position["direction"] == "long" else -1
    quantity = position["quantity"]
    entry_price = position["entry_price"]

    commission_rate = config["commission"] / 100
    slippage_rate = config["slippage"] / 100

    entry_commission = entry_price * quantity * commission_rate
    exit_commission = exit_price * quantity * commission_rate
    total_commission = entry_commission + exit_commission

    entry_slippage = entry_price * quantity * slippage_rate
    exit_slippage = exit_price * quantity * slippage_rate
    total_slippage = entry_slippage + exit_slippage

    if direction == 1:
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    net_pnl = gross_pnl - total_commission - total_slippage
    new_balance = float(balance + net_pnl)

    # Same reasoning as _open_position: cast every numeric field to plain
    # Python float at this single return point, since numpy-ness in
    # entry_price/quantity (however it got there) would otherwise ride
    # along through every one of these computed values and eventually
    # break save_simulator_state()'s raw INSERT, which -- unlike the COPY
    # path insert_signals/insert_trades/append_simulator_trades use --
    # cannot adapt numpy scalar types.
    trade = {
        "direction": position["direction"],
        "entry_time": position["entry_time"],
        "exit_time": exit_time,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "quantity": float(quantity),
        "gross_pnl": float(gross_pnl),
        "commission": float(total_commission),
        "slippage": float(total_slippage),
        "net_pnl": float(net_pnl),
        "exit_reason": exit_reason,
        "balance_after_trade": new_balance,
    }
    return trade, new_balance


def step_candle(candle, signal, position, balance, config):
    """
    Advance the simulation by exactly one candle (Steps 1-6 of the spec).

    Parameters
    ----------
    candle : dict-like with datetime/open/high/low/close
    signal : int -- 1 (buy), -1 (sell), 0 (no signal) for this candle.
        Pass 0 for every candle where no new resampled-timeframe signal
        has closed yet (see main.py's time-horizon gating).
    position : dict or None -- current open position (Position Table),
        or None if flat.
    balance : float -- current account balance.
    config : dict -- simulator config. Execution settings (initial_balance,
        position_size, commission, slippage) plus strategy rules
        (allow_long, allow_short, take_profit, stop_loss,
        max_open_positions), same as backtest/config.yaml.

    Direction-change rule: if a position is open and a new signal arrives
    pointing the OPPOSITE way, the old position is closed immediately
    (exit_reason "reversal") EVEN IF neither TP nor SL has been hit yet,
    and a new position is opened in the new direction on this same candle.
    If the new signal points the SAME way as the open position, nothing
    changes -- the existing trade is simply left running (no re-entry, no
    resizing, no double-counting).

    Returns
    -------
    (position, balance, closed_trade)
        position    : updated position dict, or None if now flat
        balance     : updated balance
        closed_trade: trade dict if a position closed this candle, else None
    """
    allow_long = config.get("allow_long", True)
    allow_short = config.get("allow_short", True)
    closed_trade = None

    # Step 4: Monitor Open Position (update mark price / unrealized PnL)
    if position is not None:
        direction = 1 if position["direction"] == "long" else -1
        close_price = float(candle["close"])
        position["current_price"] = close_price
        if direction == 1:
            position["unrealized_pnl"] = (close_price - position["entry_price"]) * position["quantity"]
        else:
            position["unrealized_pnl"] = (position["entry_price"] - close_price) * position["quantity"]

        # Step 5: TP / SL
        exit_price, exit_reason = _check_exit(position, candle)

        # Direction-change rule: if TP/SL wasn't hit but a new signal points
        # the opposite way from the open position, close here at this
        # candle's close and record it as "reversal" -- the flip itself is
        # what closed it, whether or not the position was about to hit its
        # levels anyway (TP/SL not being hit doesn't matter; the direction
        # change still forces the close). A same-direction signal is a
        # no-op: the existing trade just keeps running untouched.
        if exit_price is None and signal != 0:
            current_direction = 1 if position["direction"] == "long" else -1
            if signal == -current_direction:
                exit_price = candle["close"]
                exit_reason = "reversal"
            else:
                signal = 0

        if exit_price is not None:
            # Step 6: Close Position
            closed_trade, balance = _close_position(
                position, exit_price, candle["datetime"], exit_reason, balance, config
            )
            position = None
            # signal keeps its value here (still the opposite-direction
            # signal that caused the reversal) so Step 3 below can
            # immediately open the new position on this same candle.

    # Step 3: Open Position (only if flat -- Version 1: max_open_positions = 1)
    # Runs both for a normal flat-market entry AND right after a same-candle
    # reversal close above.
    if position is None and signal != 0:
        if signal == 1 and not allow_long:
            signal = 0
        if signal == -1 and not allow_short:
            signal = 0

        if signal != 0:
            position = _open_position(candle, signal, balance, config)

    return position, balance, closed_trade