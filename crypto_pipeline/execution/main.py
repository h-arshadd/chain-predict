"""
main.py
-------

Entry point of the Execution Module.

Same engine as the Simulator Module (imports step_candle() from
simulator.py directly -- that logic is pure position/PnL math, nothing
simulator-specific about it, so it's reused as-is rather than copied).
Two differences from simulator/main.py:

  1. Data comes live from Bybit's REST API (bybit_client.get_live_ohlcv),
     not from the DB via get_data().
  2. When step_candle() opens or closes a position, a REAL market order
     is placed on Bybit (bybit_client.place_market_order) -- this moves
     real money, unlike simulator which only ever writes to the DB.

Nothing pair/strategy-specific is hardcoded or read from a local config
file -- the universe (which (exchange, symbol) pairs to trade, and which
single strategy to run per pair) is entirely DB-driven, same as
simulator: every row in execution.config where enabled is TRUE (see
db_utils.get_execution_universe/get_execution_config/
save_execution_config). The ONLY thing local to this machine is the
Bybit API key/secret, read from .env (BYBIT_API_KEY/BYBIT_API_SECRET/
BYBIT_TESTNET) via bybit_client.get_client_from_env() -- secrets never
go in the DB or a checked-in config file.

Meant to be run repeatedly (Task Scheduler -> run_execution.bat), same as
simulator. Each run, per active (exchange, symbol) pair:
  1. Loads saved state (last processed candle, balance, open position)
     from execution.positions -- or starts fresh on first run.
  2. Pulls recent live 1-minute candles from Bybit directly (only the new
     ones since last_processed).
  3. Resamples to the strategy's time_horizon and generates signals with
     the exact same signal pipeline every other module uses.
  4. Walks forward candle by candle, calling simulator.step_candle() for
     each one -- exactly like simulator/main.py.
  5. Whenever step_candle() opens or closes a position, places the
     matching real market order on Bybit before saving state.
  6. Saves state back to execution.positions, appends any newly-closed
     trades to execution.{exchange}_{symbol}_{strategy}_trades.

Execution settings (initial_balance, position_size, commission, slippage,
allow_long, allow_short, max_open_positions, enabled, strategy_name) come
from execution.config (see db_utils.get_execution_config), same pattern
as simulator.config plus strategy_name. take_profit/stop_loss and the
strategy's indicators/conditions/time_horizon come from metadata.strategy,
same as simulator.
"""

from datetime import datetime

import pandas as pd

from crypto_pipeline.simulator.simulator import step_candle
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.execution.bybit_client import get_client_from_env, get_live_ohlcv, place_market_order
from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_state,
    save_execution_state,
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


def run_execution(client, exchange, symbol, config, strategy_name, time_horizon,
                   strategy_config_dict, take_profit_pct, stop_loss_pct):
    """
    Advance the live execution by however many new 1-minute candles are
    available from Bybit. Same walk-forward logic as simulator/main.py's
    run_simulator(), with one addition: a real market order is placed on
    Bybit every time step_candle() opens or closes a position.
    """
    conn = get_db_connection()
    try:
        state = get_execution_state(conn, exchange, symbol, strategy_name)
    finally:
        conn.close()

    if state is None:
        balance = config["initial_balance"]
        position = None
        last_processed = None
    else:
        balance = state["balance"]
        position = state["position"]
        last_processed = state["last_processed"]

    # Pull recent live candles straight from Bybit -- enough bars to cover
    # both the 1-minute walk-forward and the resample window the strategy's
    # time_horizon needs (e.g. 200 1-minute candles comfortably covers a
    # "3min" or "1h" resample warm-up).
    ohlcv_1m = get_live_ohlcv(client, symbol, limit=500)

    if last_processed is not None:
        ohlcv_1m = ohlcv_1m[ohlcv_1m["datetime"] > last_processed].reset_index(drop=True)

    if ohlcv_1m.empty:
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

        position_before = position

        position, balance, closed_trade = step_candle(
            candle, signal, position, balance, config, take_profit_pct, stop_loss_pct
        )

        # Place the real order to match whatever step_candle() just did.
        # Closed (position_before -> None or position_before -> new position
        # via direction_change): close the OLD position's direction.
        # Opened (None -> position, or a direction_change's re-open):
        # open the NEW position's direction.
        if closed_trade is not None:
            place_market_order(client, symbol, position_before["direction"], position_before["quantity"])
            closed_trade["cumulative_pnl"] = round(closed_trade["balance"] - config["initial_balance"], 4)
            closed_trades.append(closed_trade)

        if position is not None and (position_before is None or position is not position_before):
            place_market_order(client, symbol, position["direction"], position["quantity"])

        last_candle_time = candle["datetime"]

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

    # Bybit credentials only -- BYBIT_API_KEY/BYBIT_API_SECRET/BYBIT_TESTNET
    # in .env. Everything else (which pair, which strategy, execution
    # settings) is DB-driven, read below.
    client = get_client_from_env()

    # Universe: every (exchange, symbol) pair currently active in
    # execution.config -- same DB-driven pattern as simulator/main.py's
    # get_simulator_universe(). Add a pair by calling
    # save_execution_config(), stop trading it by flipping enabled to False.
    conn = get_db_connection()
    try:
        universe = get_execution_universe(conn)
    finally:
        conn.close()

    if not universe:
        raise RuntimeError(
            "No active (exchange, symbol) pairs found in execution.config. "
            "Call save_execution_config() for at least one pair first."
        )

    print(f"Active execution universe: {universe}")

    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            config = get_execution_config(conn, exchange, symbol)
        finally:
            conn.close()

        if config is None or not config["enabled"]:
            print(f"{exchange} {symbol}: no active execution.config row -- skipping.")
            continue

        strategy_name = config["strategy_name"]

        metadata_conn = get_metadata_connection()
        try:
            strategy_rows = get_strategies(metadata_conn, exchange=exchange, coin=symbol)
        finally:
            metadata_conn.close()

        strategy_row = next((s for s in strategy_rows if s["strategy_name"] == strategy_name), None)

        if strategy_row is None:
            print(f"{exchange} {symbol}: strategy {strategy_name!r} not found in metadata.strategy -- skipping.")
            continue

        time_horizon = strategy_row["time_horizon"]
        strategy_config_dict = build_strategy_config_dict(strategy_row)
        take_profit_pct = float(strategy_row["take_profit_value"])
        stop_loss_pct = float(strategy_row["stop_loss_value"])

        run_execution(
            client, exchange, symbol, config, strategy_name, time_horizon,
            strategy_config_dict, take_profit_pct, stop_loss_pct
        )