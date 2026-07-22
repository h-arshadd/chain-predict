"""
main.py
-------

Entry point of the Execution Module.

Shares its DECISION math with the Simulator Module (imports the pure
sizing/TP-SL-price/leaning helpers from simulator.py -- that logic is
just arithmetic, nothing simulator-specific about it) but NOT its fill
math, and NOT its candle-based TP/SL exit check. simulator.py's
step_candle() assumes a paper fill at the candle's open/close price --
that assumption is wrong here. Execution instead:

  1. Uses the same signal + direction-change RULES as the simulator to
     decide WHEN to open/close a position. TP/SL is NOT decided by
     walking candle high/low here -- it's registered natively with
     Bybit at order time (see _open_live_position/place_market_order)
     and enforced by Bybit's own engine. Each run starts by checking
     whether Bybit already auto-closed a position that way since the
     last run (see _reconcile_bybit_close) and, if so, logs the real
     exit before doing anything else.
  2. When a decision is made, places a REAL market order on Bybit
     (bybit_client.place_market_order) and reads back what it actually
     filled at (avgPrice/cumExecQty/fee from Bybit itself).
  3. Builds the position/ledger entirely from that real fill data -- NOT
     from the candle price the decision was triggered on. This is the
     one place execution genuinely differs from simulator: the ledger
     reflects what really happened on the exchange, not a paper fill.

Nothing pair/strategy-specific is hardcoded or read from a local config
file -- the universe (which (exchange, symbol) pairs to trade, and which
single strategy to run per pair) is entirely DB-driven, same as
simulator: every row in execution.config (see
db_utils.get_execution_universe/get_execution_config/
save_execution_config). The ONLY thing local to this machine is the
Bybit API key/secret, read from .env (BYBIT_API_KEY/BYBIT_API_SECRET/
BYBIT_DEMO) via bybit_client.get_client_from_env() -- secrets never
go in the DB or a checked-in config file.

Meant to be run repeatedly (Task Scheduler -> run_execution.bat), same as
simulator. Each run, per registered (exchange, symbol) pair:
  1. Loads saved state (last processed candle, balance, open position)
     from execution.positions -- or starts fresh on first run.
  2. Pulls recent live 1-minute candles from Bybit directly (only the new
     ones since last_processed).
  3. Resamples to the strategy's time_horizon and generates signals with
     the exact same signal pipeline every other module uses.
  4. Walks forward candle by candle, deciding open/close/hold exactly
     like simulator does, but on a decision to open or close, places the
     real order first and only then records what happened, from the
     order's real fill.
  5. Saves state back to execution.positions, appends any newly-closed
     trades to execution.{exchange}_{symbol}_{strategy}_trades.

Execution settings (initial_balance, position_size, commission, slippage,
allow_long, allow_short, max_open_positions) come from execution.config
(see db_utils.get_execution_config), same pattern as simulator.config.
commission/slippage in execution.config are effectively unused for real
trades now (the real fee comes back from Bybit itself on every order) --
kept only because get_execution_config still returns them. Which
strategy to run for a pair is NOT stored in execution.config -- it's
derived from metadata.strategy (whichever row for that exchange/symbol
has execution_enabled=True). take_profit/stop_loss and the strategy's
indicators/conditions/time_horizon also come from metadata.strategy,
same as simulator.
"""

from datetime import datetime

import pandas as pd

from crypto_pipeline.simulator.simulator import _position_size, _tp_sl_prices, _leaning
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.execution.bybit_client import (
    get_client_from_env,
    get_live_ohlcv,
    place_market_order,
    get_open_position,
    get_last_closed_pnl,
)
from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_state,
    save_execution_state,
    open_execution_trade,
    close_execution_trade,
    append_execution_trades,
    get_execution_summary,
    get_execution_config,
    get_execution_universe,
    build_execution_equity_curve_from_ledger,
    save_execution_stats,
)
from crypto_pipeline.utils.metadata_utils import (
    get_db_connection as get_metadata_connection,
    get_strategies,
)
from crypto_pipeline.stats.calculator import compute_stats

# stats/config.yaml -- same config compute_stats() takes everywhere else,
# loaded the same way simulator/main.py loads it (see that module's
# _load_stats_config() for the reasoning on not reusing stats_runner's
# leading-underscore internal default).
def _load_stats_config():
    import yaml
    from pathlib import Path
    stats_config_path = Path(__file__).parent.parent / "stats" / "config.yaml"
    with open(stats_config_path, "r") as f:
        return yaml.safe_load(f)


STATS_CONFIG = _load_stats_config()


def build_strategy_config_dict(strategy_row: dict) -> dict:
    """
    Reassemble a metadata.strategy row back into the full config dict
    shape generate_signals() expects. Identical to simulator/main.py's
    version of this function.
    """
    config = dict(strategy_row["strategy_config"])
    config["strategy_name"] = strategy_row["strategy_name"]
    config["time_horizon"] = strategy_row["time_horizon"]
    config["take_profit"] = {
        "type": strategy_row["take_profit_type"],
        "value": float(strategy_row["take_profit_value"]),
    }
    config["stop_loss"] = {
        "type": strategy_row["stop_loss_type"],
        "value": float(strategy_row["stop_loss_value"]),
    }
    return config


def parse_start_date(start_date_str: str):
    """Same date formats as simulator/main.py's parse_simulator_start_date."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(start_date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format for start_date: {start_date_str!r}")


def build_resampled_signals(resampled_df, strategy_config_dict):
    """Identical to simulator/main.py's version -- runs the signal pipeline for one strategy."""
    indicator_df, condition_df, signal_series = generate_signals(resampled_df, config_dict=strategy_config_dict)
    combined = pd.concat([indicator_df, condition_df], axis=1)
    combined["signal"] = signal_series
    combined = combined.dropna().reset_index(drop=True)
    return combined[["datetime", "signal"]]


def _open_live_position(client, symbol, direction, candle, balance, config,
                         take_profit_pct, stop_loss_pct):
    """
    Place a real opening order on Bybit and build the position dict from
    its actual fill -- the live equivalent of simulator.py's
    _open_position(), but priced/sized off the real exchange fill instead
    of the candle's open price.

    quantity is still SIZED off the candle open (balance * position_size%
    / current price) since that's the best estimate available before the
    order exists -- but the position actually opened (entry_price,
    quantity, fee) reflects Bybit's real fill once placed.

    take_profit/stop_loss are computed off estimate_price (candle open)
    and sent to Bybit ON the opening order itself, so Bybit registers
    them as native exchange-side TP/SL (visible on the Positions tab,
    same as clicking "+ Add" under TP/SL) -- Bybit's own engine then
    watches price and auto-closes on hit, independent of whether this
    script is running. This is a deliberate change from computing
    TP/SL off entry_price after the fact: it needs to be known BEFORE
    placing the order so it can be attached to that same order.
    Direction-change exits are unaffected -- Bybit has no concept of a
    "signal", so those are still detected and closed by this script
    itself, same as before (see run_execution()).
    """
    estimate_price = float(candle["open"])
    target_quantity = _position_size(balance, config, estimate_price)
    take_profit, stop_loss = _tp_sl_prices(estimate_price, direction, take_profit_pct, stop_loss_pct)

    order = place_market_order(
        client, symbol, "long" if direction == 1 else "short", target_quantity,
        take_profit=round(float(take_profit), 4), stop_loss=round(float(stop_loss), 4),
    )

    entry_price = order["avg_price"]
    quantity = order["filled_qty"]

    return {
        "direction": "long" if direction == 1 else "short",
        "entry_time": candle["datetime"],
        "entry_price": round(entry_price, 4),
        "quantity": round(quantity, 4),
        "take_profit": round(float(take_profit), 4),
        "stop_loss": round(float(stop_loss), 4),
        "entry_fee": round(order["fee"], 4),
        "current_price": round(entry_price, 4),
        "unrealized_pnl": 0.0,
        "leaning": _leaning(entry_price, take_profit, stop_loss),
        "status": "open",
    }


def _close_live_position(client, symbol, position, exit_time, exit_reason, balance, config):
    """
    Place a real closing order on Bybit (opposite side of the open
    position) and build the closed-trade record from its actual fill --
    the live equivalent of simulator.py's _close_position(), but priced
    off the real exchange fill instead of a candle price.

    net_pnl = gross PnL from real entry/exit fill prices, minus the real
    entry + exit fees Bybit charged (no slippage estimate needed -- the
    fill price already IS whatever slippage happened).
    """
    direction = 1 if position["direction"] == "long" else -1
    quantity = position["quantity"]
    entry_price = position["entry_price"]

    # Closing order is the opposite side of however the position was opened.
    close_direction = "short" if position["direction"] == "long" else "long"
    order = place_market_order(client, symbol, close_direction, quantity)

    exit_price = order["avg_price"]
    exit_fee = order["fee"]

    if direction == 1:
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity

    # entry_fee is only present if this position was opened AND closed
    # within the same run. If execution.main.py was restarted while this
    # position was open, execution.positions has no entry_fee column to
    # resume it from -- estimate it from config["commission"] instead of
    # silently treating it as zero (better an estimate than understating
    # real fees paid).
    entry_fee = position.get("entry_fee")
    if entry_fee is None:
        entry_fee = entry_price * quantity * (config.get("commission", 0.0) / 100)

    total_fee = entry_fee + exit_fee
    net_pnl = gross_pnl - total_fee
    new_balance = float(balance + net_pnl)

    trade = {
        "direction": position["direction"],
        "entry_date_time": position["entry_time"],
        "exit_date_time": exit_time,
        "entry_price": round(float(entry_price), 4),
        "exit_price": round(float(exit_price), 4),
        "quantity": round(float(quantity), 4),
        "gross_pnl": round(float(gross_pnl), 4),
        "commission": round(float(total_fee), 4),
        "slippage": 0.0,
        "net_pnl": round(float(net_pnl), 4),
        "exit_reason": exit_reason,
        "balance": round(new_balance, 4),
    }
    return trade, new_balance


def _reconcile_bybit_close(client, symbol, position, balance, config):
    """
    Check whether Bybit already closed `position` on its own since our
    last run -- i.e. its native TP/SL (see _open_live_position) fired
    and the exchange auto-closed the position without this script
    placing the closing order itself. Our own candle walk in
    run_execution() has no way of knowing about that on its own, since
    it only ever finds out about a close it triggered itself.

    Returns (position, balance, closed_trade):
      - If our DB says flat (position is None), nothing to check --
        returns (None, balance, None) immediately.
      - If Bybit still shows the position open, nothing changed --
        returns (position, balance, None) unchanged.
      - If Bybit shows it closed, pulls the REAL exit fill from
        get_last_closed_pnl() and builds a closed-trade record in the
        exact same shape _close_live_position() returns (so
        run_execution() can append it to closed_trades/the ledger the
        same way either way) -- returns (None, new_balance, closed_trade).

    Only price-based (TP/SL) exits can happen this way -- direction-
    change exits are still detected and closed by run_execution() itself
    further down, unaffected by this check, since Bybit has no concept
    of a strategy signal.
    """
    if position is None:
        return None, balance, None

    live_position = get_open_position(client, symbol)
    if live_position is not None:
        # Still open on Bybit -- nothing to reconcile.
        return position, balance, None

    # DB says open, Bybit says flat -- Bybit auto-closed it (native
    # TP/SL hit). Pull the real fill instead of guessing exit_price
    # from a candle.
    fill = get_last_closed_pnl(client, symbol)

    entry_price = position["entry_price"]
    quantity = position["quantity"]
    exit_price = fill["exit_price"]

    # closed_pnl from Bybit is net of Bybit's own fees already -- use it
    # directly as net_pnl rather than recomputing gross - fees ourselves,
    # since we don't have the real exit fee split out separately here.
    net_pnl = fill["closed_pnl"]
    gross_pnl = (exit_price - entry_price) * quantity if position["direction"] == "long" \
        else (entry_price - exit_price) * quantity
    commission = round(gross_pnl - net_pnl, 4)
    new_balance = float(balance + net_pnl)

    # exit_reason: Bybit's own execType doesn't cleanly say "TakeProfit"
    # vs "StopLoss" on closed-pnl -- but since this path only fires when
    # our own TP/SL fields were the only exit condition registered with
    # Bybit at open time, tag it by whichever level exit_price actually
    # landed closer to, same convention _leaning() already uses elsewhere.
    exit_reason = _leaning(exit_price, position["take_profit"], position["stop_loss"])

    closed_trade = {
        "direction": position["direction"],
        "entry_date_time": position["entry_time"],
        "exit_date_time": fill["exit_time"],
        "entry_price": round(float(entry_price), 4),
        "exit_price": round(float(exit_price), 4),
        "quantity": round(float(quantity), 4),
        "gross_pnl": round(float(gross_pnl), 4),
        "commission": commission,
        "slippage": 0.0,
        "net_pnl": round(float(net_pnl), 4),
        "exit_reason": exit_reason,
        "balance": round(new_balance, 4),
    }
    return None, new_balance, closed_trade


def run_execution(client, exchange, symbol, config, strategy_name, time_horizon,
                   strategy_config_dict, take_profit_pct, stop_loss_pct):
    """
    Advance the live execution by however many new 1-minute candles are
    available from Bybit. Same signal/direction-change RULES as
    simulator/main.py's run_simulator(), but every open/close is a real
    Bybit order, and the ledger is built from that order's real fill
    (see _open_live_position/_close_live_position above) instead of the
    candle price the decision was triggered on.

    TP/SL is no longer checked candle-by-candle here -- it's registered
    natively with Bybit at order time (see _open_live_position) and
    Bybit's own engine enforces it. This function's job re: TP/SL is
    only to notice, at the top of each run, if that already happened
    (see _reconcile_bybit_close) and log it properly. Direction-change
    exits are unaffected and still fully handled by this script, since
    Bybit has no concept of a strategy signal.
    """
    allow_long = config.get("allow_long", True)
    allow_short = config.get("allow_short", True)

    conn = get_db_connection()
    try:
        state = get_execution_state(conn, exchange, symbol, strategy_name)
    finally:
        conn.close()

    is_first_run = state is None

    if state is None:
        balance = config["initial_balance"]
        position = None
        last_processed = None
    else:
        balance = state["balance"]
        position = state["position"]
        last_processed = state["last_processed"]

    reconciled_trades = []
    position, balance, reconciled_trade = _reconcile_bybit_close(client, symbol, position, balance, config)
    if reconciled_trade is not None:
        reconciled_trade["cumulative_pnl"] = round(balance - config["initial_balance"], 4)
        reconciled_trades.append(reconciled_trade)
        print(
            f"{exchange} {symbol} ({strategy_name}): position was closed by Bybit "
            f"(native {reconciled_trade['exit_reason']}) since last run -- "
            f"net PnL {reconciled_trade['net_pnl']:.4f}."
        )

    # Pull recent live candles straight from Bybit -- enough bars to cover
    # both the 1-minute walk-forward and the resample window the strategy's
    # time_horizon needs (e.g. 200 1-minute candles comfortably covers a
    # "3min" or "1h" resample warm-up).
    ohlcv_1m = get_live_ohlcv(client, symbol, limit=500)

    if is_first_run:
        # First time this (exchange, symbol, strategy) has ever run --
        # there's no last_processed to resume from, but that does NOT
        # mean "walk everything Bybit happens to return." Bybit's kline
        # endpoint always returns its most recent candles regardless of
        # when execution last ran -- with no state at all yet, that's
        # up to 500 minutes (8+ hours) of ALREADY-PASSED price action.
        # Walking that as if it were live would generate signals off
        # stale data and could immediately fire a REAL order on this
        # run, priced at whatever the market is doing right now, off a
        # decision made from hours-old candles.
        #
        # Correct first-run behavior: start clean from THIS moment
        # forward. Record the latest pulled candle as last_processed and
        # do nothing else this run -- no signals generated, no orders
        # placed. The very next run will only see genuinely new candles
        # from here on, same as every subsequent run already works.
        last_seen = ohlcv_1m["datetime"].iloc[-1].to_pydatetime() if not ohlcv_1m.empty else None
        conn = get_db_connection()
        try:
            cumulative_pnl = round(balance - config["initial_balance"], 4)
            save_execution_state(conn, exchange, symbol, strategy_name, time_horizon, last_seen, round(balance, 4), position, cumulative_pnl)
            if reconciled_trades:
                append_execution_trades(conn, exchange, symbol, strategy_name, pd.DataFrame(reconciled_trades))
        finally:
            conn.close()
        print(
            f"{exchange} {symbol} ({strategy_name}): first run -- starting fresh from "
            f"{last_seen}, no historical backlog processed. Next run will pick up new candles from here."
        )
        return 0

    if last_processed is not None:
        ohlcv_1m = ohlcv_1m[ohlcv_1m["datetime"] > last_processed].reset_index(drop=True)

    if ohlcv_1m.empty:
        if reconciled_trades:
            # Still need to persist the reconciled close even though
            # there's no candle work to do this run -- otherwise it's
            # silently lost until candles happen to show up again.
            conn = get_db_connection()
            try:
                cumulative_pnl = round(balance - config["initial_balance"], 4)
                save_execution_state(conn, exchange, symbol, strategy_name, time_horizon, last_processed, round(balance, 4), position, cumulative_pnl)
                append_execution_trades(conn, exchange, symbol, strategy_name, pd.DataFrame(reconciled_trades))
            finally:
                conn.close()
        print(f"{exchange} {symbol} ({strategy_name}): no new candles.")
        return 0

    ohlcv_resampled = (
        ohlcv_1m.set_index("datetime")
        .resample(time_horizon)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )

    if ohlcv_resampled.empty:
        signals = pd.DataFrame(columns=["datetime", "signal"])
    else:
        signals = build_resampled_signals(ohlcv_resampled, strategy_config_dict)

    # Same time-horizon gate as simulator/main.py: a signal only takes
    # effect on the 1-minute candle where a new resampled candle just closed.
    if not signals.empty:
        aligned = pd.merge_asof(
            ohlcv_1m[["datetime"]], signals, on="datetime", direction="backward"
        )
        aligned["signal"] = aligned["signal"].fillna(0)
        is_new_signal_bar = aligned["datetime"].isin(signals["datetime"])
        aligned.loc[~is_new_signal_bar, "signal"] = 0
    else:
        aligned = ohlcv_1m[["datetime"]].copy()
        aligned["signal"] = 0

    closed_trades = []
    last_candle_time = last_processed

    for i in range(len(ohlcv_1m)):
        candle = {
            "datetime": ohlcv_1m["datetime"].iloc[i].to_pydatetime(),
            "open": float(ohlcv_1m["open"].iloc[i]),
            "high": float(ohlcv_1m["high"].iloc[i]),
            "low": float(ohlcv_1m["low"].iloc[i]),
            "close": float(ohlcv_1m["close"].iloc[i]),
        }
        signal = int(aligned["signal"].iloc[i])
        closed_trade = None

        # Step: monitor open position (mark price / unrealized PnL / leaning) --
        # same math as simulator.step_candle(), no order involved.
        if position is not None:
            direction = 1 if position["direction"] == "long" else -1
            close_price = candle["close"]
            position["current_price"] = round(close_price, 4)
            if direction == 1:
                position["unrealized_pnl"] = round((close_price - position["entry_price"]) * position["quantity"], 4)
            else:
                position["unrealized_pnl"] = round((position["entry_price"] - close_price) * position["quantity"], 4)
            position["leaning"] = _leaning(close_price, position["take_profit"], position["stop_loss"])

            # TP/SL is no longer checked here -- it's registered natively
            # with Bybit at order time (see _open_live_position) and
            # enforced by Bybit's own engine, not by walking candle
            # high/low ourselves. A Bybit-side auto-close is picked up
            # at the top of the NEXT run by _reconcile_bybit_close(),
            # not mid-loop here.
            exit_reason = None

            # Direction-change rule: opposite-direction signal closes the
            # open position even if TP/SL wasn't hit. Same-direction signal
            # is a no-op -- existing trade keeps running untouched.
            if exit_reason is None and signal != 0:
                current_direction = 1 if position["direction"] == "long" else -1
                if signal == -current_direction:
                    exit_reason = "direction_change"
                else:
                    signal = 0

            if exit_reason is not None:
                # Real close order placed here -- ledger built from its
                # actual fill, not from exit_price/candle close.
                #
                # Note on TP/SL here: this closing order uses the same
                # qty as the full open position, so it fully closes the
                # position on Bybit. Bybit's native TP/SL (set at open,
                # see _open_live_position) is a position-attached
                # parameter, not a separate resting order -- once the
                # position it's attached to is fully closed, Bybit drops
                # it automatically. No manual cancel needed here, and no
                # stale TP/SL is left behind for the next position opened
                # on this symbol below.
                closed_trade, balance = _close_live_position(
                    client, symbol, position, candle["datetime"], exit_reason, balance, config
                )
                position = None
                # signal keeps its value so a direction-change can reopen
                # in the new direction on this same candle, below.

        # Step: open position (only if flat).
        if position is None and signal != 0:
            if signal == 1 and not allow_long:
                signal = 0
            if signal == -1 and not allow_short:
                signal = 0

            if signal != 0:
                # Real open order placed here -- position built from its
                # actual fill, not from candle open.
                position = _open_live_position(
                    client, symbol, signal, candle, balance, config, take_profit_pct, stop_loss_pct
                )
                # Trade row is written to the DB right now, as soon as the
                # position opens -- not held until it closes. Only the
                # entry-side columns are filled; exit columns stay NULL
                # until close_execution_trade() fills them in later.
                conn = get_db_connection()
                try:
                    open_execution_trade(conn, exchange, symbol, strategy_name, position)
                finally:
                    conn.close()

        if closed_trade is not None:
            closed_trade["cumulative_pnl"] = round(closed_trade["balance"] - config["initial_balance"], 4)
            closed_trades.append(closed_trade)

        last_candle_time = candle["datetime"]

    closed_trades = reconciled_trades + closed_trades

    conn = get_db_connection()
    try:
        cumulative_pnl = round(balance - config["initial_balance"], 4)
        save_execution_state(conn, exchange, symbol, strategy_name, time_horizon, last_candle_time, round(balance, 4), position, cumulative_pnl)
        trade_ledger = pd.DataFrame(closed_trades)
        append_execution_trades(conn, exchange, symbol, strategy_name, trade_ledger)
    finally:
        conn.close()

    print(
        f"{exchange} {symbol} ({strategy_name}, {time_horizon}): processed {len(ohlcv_1m)} candle(s), "
        f"{len(closed_trades)} trade(s) closed, balance {balance:.2f}, "
        f"position {'open (' + position['direction'] + ')' if position else 'flat'}"
    )

    conn = get_db_connection()
    try:
        summary = get_execution_summary(conn, exchange, symbol, strategy_name)
    finally:
        conn.close()

    if summary is not None:
        wl = summary["win_loss"]
        print(
            f"    summary: {summary['total_trades']} total trade(s), "
            f"net PnL {summary['total_net_profit']:.2f}, "
            f"wins {wl['wins']} / losses {wl['losses']} "
            f"(win rate {wl['win_rate']:.1%})"
        )

    # Stats: one shared execution.stats table, one row per
    # exchange+symbol+strategy -- same pattern as simulator/main.py's
    # equivalent block, just computed from execution's own (real-money)
    # ledger via build_execution_equity_curve_from_ledger() instead of
    # simulator's paper one. Skipped if there are no closed trades yet:
    # quantstats' metrics need at least one return to be meaningful.
    if summary is not None and summary["total_trades"] > 0:
        conn = get_db_connection()
        try:
            equity_curve = build_execution_equity_curve_from_ledger(
                conn, exchange, symbol, strategy_name, config["initial_balance"]
            )
        finally:
            conn.close()

        if equity_curve is not None and len(equity_curve) > 1:
            stats_dict = compute_stats(
                {"equity_curve": equity_curve, "total_trades": summary["total_trades"]},
                STATS_CONFIG,
            )
            stats_row = dict(stats_dict["metrics"])
            stats_row["total_trades"] = summary["total_trades"]

            conn = get_db_connection()
            try:
                save_execution_stats(conn, exchange, symbol, strategy_name, time_horizon, stats_row)
            finally:
                conn.close()

    return len(ohlcv_1m)


if __name__ == "__main__":

    # Bybit credentials only -- BYBIT_API_KEY/BYBIT_API_SECRET/BYBIT_DEMO
    # in .env. Everything else (which pair, which strategy, execution
    # settings) is DB-driven, read below.
    client = get_client_from_env()

    # Universe: every (exchange, symbol) pair currently registered in
    # execution.config -- same DB-driven pattern as simulator/main.py's
    # get_simulator_universe(). Add a pair by calling
    # save_execution_config(); remove one by deleting its row.
    conn = get_db_connection()
    try:
        universe = get_execution_universe(conn)
    finally:
        conn.close()

    if not universe:
        raise RuntimeError(
            "No (exchange, symbol) pairs found in execution.config. "
            "Call save_execution_config() for at least one pair first."
        )

    print(f"Active execution universe: {universe}")

    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            config = get_execution_config(conn, exchange, symbol)
        finally:
            conn.close()

        if config is None:
            print(f"{exchange} {symbol}: no execution.config row -- skipping.")
            continue

        # Which strategy to run for this pair is no longer pinned in
        # execution.config -- it's derived here from metadata.strategy
        # instead: whichever row for this (exchange, symbol) has
        # execution_enabled=True. get_strategies() can return multiple
        # rows for this exchange/symbol (strategy history/iterations),
        # so this still guards against more than one being marked
        # execution_enabled at the same time (a misconfiguration) --
        # that pair is skipped rather than guessing which one to run.
        metadata_conn = get_metadata_connection()
        try:
            strategy_rows = get_strategies(metadata_conn, exchange=exchange, coin=symbol)
        finally:
            metadata_conn.close()

        enabled_rows = [s for s in strategy_rows if s.get("execution_enabled", True)]

        if len(enabled_rows) == 0:
            print(f"{exchange} {symbol}: no execution_enabled strategy found in metadata.strategy -- skipping.")
            continue

        if len(enabled_rows) > 1:
            enabled_strategy_names = [s["strategy_name"] for s in enabled_rows]
            print(
                f"{exchange} {symbol}: {len(enabled_rows)} strategies are "
                f"execution_enabled ({enabled_strategy_names}) -- only one strategy is "
                f"allowed per coin. Disable all but one in metadata.strategy before this "
                f"pair will run. Skipping."
            )
            continue

        strategy_row = enabled_rows[0]
        strategy_name = strategy_row["strategy_name"]

        time_horizon = strategy_row["time_horizon"]
        strategy_config_dict = build_strategy_config_dict(strategy_row)
        take_profit_pct = float(strategy_row["take_profit_value"])
        stop_loss_pct = float(strategy_row["stop_loss_value"])

        run_execution(
            client, exchange, symbol, config, strategy_name, time_horizon,
            strategy_config_dict, take_profit_pct, stop_loss_pct
        )