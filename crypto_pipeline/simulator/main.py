"""
main.py
-------

Entry point of the Simulator Module.

Meant to be run repeatedly (Task Scheduler -> run_simulator.bat), the same
way run_pipeline.bat drives the data pipelines. Each run:

  1. Loads every strategy for a pair from metadata.strategy (10-20
     strategies, each a row: indicators, long/short rules, strategy_name,
     and this strategy's own time_horizon/take_profit/stop_loss).
  2. For every (exchange, symbol) pair registered (and active) in
     simulator.config, and every strategy for that pair:
       a. Pulls live OHLCV via get_data() (same call every other module
          uses -- reads whatever's in the DB and fetches any live gap from
          the exchange -- but with drop_last_1m=False so the still-forming
          1m candle is kept), resampled to THIS strategy's time_horizon.
       b. Loads saved state (last processed candle, balance, open position)
          from simulator.positions -- one shared table, row matched on
          (exchange, symbol, strategy) -- or starts fresh if this is the
          first run for that combo.
       c. Walks forward candle by candle over whatever 1-minute candles are
          new since last_processed, calling simulator.step_candle() for
          each one.
       d. Signals only change once a full resampled-timeframe candle closes
          (this strategy's time horizon), so a new signal is only fed in on
          the 1-minute candle that lines up with a newly-closed resampled
          candle -- every other 1-minute candle just monitors the open
          position (signal=0). TP/SL is still checked every 1-minute candle
          regardless of time horizon -- see simulator.py's step_candle().
       e. Saves state back to simulator.positions (its own row, one per
          exchange+symbol+strategy), appends any newly-closed trades to
          that combo's own Trade Ledger table. The DB is the only source
          of truth -- no CSV is written; query
          simulator.{exchange}_{symbol}_{strategy}_trades directly to
          inspect a strategy's ledger.

Execution settings (initial_balance, position_size, commission, slippage,
allow_long, allow_short, max_open_positions, is_active) come from
simulator.config (see db_utils.get_simulator_config), one row per
(exchange, symbol) pair, and are shared across every strategy run against
that pair. take_profit/stop_loss are PER-STRATEGY instead -- every
strategy's own metadata.strategy row has its own take_profit_value/
stop_loss_value columns. Each strategy's own indicators/conditions/rule
and time_horizon also live only in its own metadata.strategy row
(strategy_config JSONB + time_horizon column).

The universe (which (exchange, symbol) pairs to run every strategy on) is
simply every row in simulator.config where is_active is TRUE -- see
db_utils.get_simulator_universe(). Add a pair by inserting a row (or
flipping is_active back to TRUE), remove one by setting is_active to
FALSE -- no code changes needed either way. simulator/config.yaml is no
longer read.
"""

from datetime import datetime

import pandas as pd

from crypto_pipeline.simulator.simulator import step_candle
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_simulator_state,
    save_simulator_state,
    append_simulator_trades,
    get_simulator_summary,
    build_equity_curve_from_ledger,
    save_simulator_stats,
    get_simulator_universe,
    get_simulator_config,
)
from crypto_pipeline.utils.metadata_utils import (
    get_db_connection as get_metadata_connection,
    get_strategies,
)
from crypto_pipeline.stats.calculator import compute_stats

# stats/config.yaml -- same config compute_stats() takes everywhere else
# (risk_free_rate, periods_per_year, resample_freq, exclude_metrics,
# generate_plots). Loaded once here rather than importing stats_runner's
# _default_stats_config() (leading underscore = that module's own
# internal default, not meant to be reused elsewhere).
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
    shape generate_signals()/signals/strategies/*.yaml used to have:
    strategy_name, time_horizon, take_profit, stop_loss, plus the
    indicator/strategy blocks from strategy_config JSONB, all in one dict.

    metadata.strategy stores strategy_name/time_horizon/take_profit_*/
    stop_loss_* as their own columns (not inside strategy_config), so this
    just merges them back together -- the inverse of the split done in
    metadata_utils.load_strategies_from_yaml().
    """
    config = dict(strategy_row["strategy_config"])
    config["strategy_name"] = strategy_row["strategy_name"]
    config["time_horizon"] = strategy_row["time_horizon"]
    # take_profit_value/stop_loss_value are Postgres NUMERIC -> psycopg2
    # returns them as decimal.Decimal; cast to float so this dict matches
    # the plain-float shape a parsed yaml file always had.
    config["take_profit"] = {
        "type": strategy_row["take_profit_type"],
        "value": float(strategy_row["take_profit_value"]),
    }
    config["stop_loss"] = {
        "type": strategy_row["stop_loss_type"],
        "value": float(strategy_row["stop_loss_value"]),
    }
    return config


def build_resampled_signals(resampled_df, strategy_config_dict):
    """
    Run the signal pipeline on resampled OHLCV for ONE strategy (a dict
    loaded from metadata.strategy -- strategy_config merged with its
    time_horizon/take_profit/stop_loss columns, same shape a parsed
    signals/strategies/*.yaml file used to have), same pattern as
    backtest/main.py's build_signals().
    Returns datetime/signal only, warm-up rows dropped.
    """
    indicator_df, condition_df, signal_series = generate_signals(resampled_df, config_dict=strategy_config_dict)

    combined = pd.concat([indicator_df, condition_df], axis=1)
    combined["signal"] = signal_series
    combined = combined.dropna().reset_index(drop=True)

    return combined[["datetime", "signal"]]


def run_simulator(exchange, symbol, config, strategy_name, time_horizon, strategy_config_dict,
                   take_profit_pct, stop_loss_pct):
    """
    Advance one exchange+symbol+strategy simulation by however many new
    1-minute candles are available. Returns the number of candles processed.

    time_horizon is THIS strategy's own resampled timeframe (e.g. "2h"),
    read from its metadata.strategy row -- controls both how signals are
    generated (resample target) and the entry gate below (a new signal
    only takes effect on the 1-minute candle where a new time_horizon
    candle closes). It does NOT change how often TP/SL is checked --
    that's every 1-minute candle inside step_candle(), independent of
    time horizon.

    strategy_config_dict : dict -- this strategy's full config as loaded
    from metadata.strategy (strategy_config JSONB merged with its
    time_horizon/take_profit/stop_loss columns), same shape a parsed
    signals/strategies/*.yaml file used to have. Passed straight through
    to build_resampled_signals() / generate_signals().

    take_profit_pct, stop_loss_pct : float -- THIS strategy's own TP/SL
    percentages, read directly from its metadata.strategy row. Passed
    straight through to step_candle().

    A pair's very first run no longer falls back to a configured
    start_date (that column/field is gone) -- see the is_first_run
    branch below, which mirrors execution/main.py's first-run behavior
    instead: start clean from "now", record only last_processed, no
    historical backlog processed.
    """
    conn = get_db_connection()
    try:
        state = get_simulator_state(conn, exchange, symbol, strategy_name)
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

    if is_first_run:
        # First time this (exchange, symbol, strategy) has ever run --
        # there's no last_processed to resume from. Previously this fell
        # back to simulator.config's start_date and walked everything
        # from there, which could be days/weeks of historical backlog
        # processed as if it were live. Same fix as execution/main.py:
        # start clean from THIS moment forward instead. Pull just enough
        # recent data to know "now", record the latest 1-minute candle as
        # last_processed, and do nothing else this run -- no signals
        # generated, no trades opened. The very next run will only see
        # genuinely new candles from here on, same as every subsequent
        # run already works.
        result = get_data(
            exchange=exchange,
            symbol=symbol,
            start_date="now",
            end_date="now",
            timeframe=time_horizon,
            df_1m=True,
            drop_last_1m=False,
        )
        ohlcv_1m = result["one_min"]
        last_seen = ohlcv_1m["datetime"].iloc[-1].to_pydatetime() if not ohlcv_1m.empty else None

        conn = get_db_connection()
        try:
            cumulative_pnl = round(balance - config["initial_balance"], 4)
            save_simulator_state(conn, exchange, symbol, strategy_name, time_horizon, last_seen, round(balance, 4), position, cumulative_pnl)
        finally:
            conn.close()

        print(
            f"{exchange} {symbol} ({strategy_name}): first run -- starting fresh from "
            f"{last_seen}, no historical backlog processed. Next run will pick up new candles from here."
        )
        return 0

    # Pull live 1m data (+ resampled, for signals) starting right after
    # whatever we've already processed.
    result = get_data(
        exchange=exchange,
        symbol=symbol,
        start_date=last_processed,
        end_date="now",
        timeframe=time_horizon,
        df_1m=True,
        drop_last_1m=False,
    )
    ohlcv_1m = result["one_min"]
    ohlcv_resampled = result["resampled"]

    if last_processed is not None:
        ohlcv_1m = ohlcv_1m[ohlcv_1m["datetime"] > last_processed].reset_index(drop=True)

    if ohlcv_1m.empty:
        print(f"{exchange} {symbol} ({strategy_name}): no new candles.")
        return 0

    if ohlcv_resampled.empty:
        signals = pd.DataFrame(columns=["datetime", "signal"])
    else:
        signals = build_resampled_signals(ohlcv_resampled, strategy_config_dict)

    # Time-horizon gate: a signal only takes effect on the first 1-minute
    # candle at or after its resampled-timeframe candle has closed. Look
    # this up per 1-minute candle via merge_asof (backward), same alignment
    # approach backtest.py uses, then only keep the signal on the exact
    # 1-minute row where a *new* resampled candle just closed -- every
    # other row gets signal=0 so step_candle() only monitors, doesn't
    # re-enter on a signal that already fired earlier.
    #
    # NOTE: this gate only controls NEW/CHANGED signal entries. TP/SL is
    # still evaluated every 1-minute candle inside step_candle() itself,
    # regardless of time_horizon.
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
        # .iloc[i] on a DataFrame returns numpy scalar types (numpy.float64,
        # pandas.Timestamp, etc.) inside the Series -- psycopg2 can adapt
        # native Python types but not numpy ones, and that numpy-ness would
        # otherwise silently ride along through step_candle() into
        # balance/position and break save_simulator_state()'s INSERT
        # ("schema np does not exist" -- psycopg2 rendering np.float64(...)
        # literally instead of binding it as a parameter). Cast to plain
        # Python types once, right here, so nothing downstream ever sees a
        # numpy scalar.
        candle = {
            "datetime": ohlcv_1m["datetime"].iloc[i].to_pydatetime(),
            "open": float(ohlcv_1m["open"].iloc[i]),
            "high": float(ohlcv_1m["high"].iloc[i]),
            "low": float(ohlcv_1m["low"].iloc[i]),
            "close": float(ohlcv_1m["close"].iloc[i]),
        }
        signal = int(aligned["signal"].iloc[i])

        position, balance, closed_trade = step_candle(
            candle, signal, position, balance, config, take_profit_pct, stop_loss_pct
        )

        if closed_trade is not None:
            # exchange/symbol are NOT stored as columns -- the ledger
            # table itself is already named per exchange+symbol+strategy
            # (simulator.{exchange}_{symbol}_{strategy_name}_trades), so
            # repeating them in every row would be redundant. trade_id
            # (simple incrementing 1, 2, 3...) is added in
            # append_simulator_trades instead of here, since it needs to
            # continue counting across every past run's trades already in
            # the table, not just this run's batch.
            #
            # Same column backtest.py's ledger has: running P&L since this
            # strategy's very first trade (not just this run's batch), so
            # it's correct across resumed runs too. balance
            # already reflects every trade ever closed for this
            # (exchange, symbol, strategy), so this is just that minus
            # where the account started -- no extra running state needed.
            closed_trade["cumulative_pnl"] = round(closed_trade["balance"] - config["initial_balance"], 4)
            closed_trades.append(closed_trade)

        last_candle_time = candle["datetime"]

    conn = get_db_connection()
    try:
        cumulative_pnl = round(balance - config["initial_balance"], 4)
        save_simulator_state(conn, exchange, symbol, strategy_name, time_horizon, last_candle_time, round(balance, 4), position, cumulative_pnl)
        trade_ledger = pd.DataFrame(closed_trades)
        append_simulator_trades(conn, exchange, symbol, strategy_name, trade_ledger)
    finally:
        conn.close()

    print(
        f"{exchange} {symbol} ({strategy_name}, {time_horizon}): processed {len(ohlcv_1m)} candle(s), "
        f"{len(closed_trades)} trade(s) closed, balance {balance:.2f}, "
        f"position {'open (' + position['direction'] + ')' if position else 'flat'}"
    )

    # Spec output: Trade Ledger (already in DB), Final Account Balance
    # (already in DB via state), Total Profit/Loss, Total Number of Trades,
    # Win/Loss Summary -- rolled up here from the DB so it reflects the
    # strategy's full history, not just this run's candles.
    conn = get_db_connection()
    try:
        summary = get_simulator_summary(conn, exchange, symbol, strategy_name)
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

    # Stats: one shared simulator.stats table, one row per
    # exchange+symbol+strategy (see save_simulator_stats in db_utils.py),
    # holding the same headline metrics stats_runner.py's comparison CSV
    # already treats as "most important" (sharpe, sortino, calmar,
    # max_drawdown, cagr, profit_factor, win_rate, recovery_factor,
    # risk_of_ruin) plus total_trades. Computed from the strategy's full
    # ledger-to-date (not just this run's candles), same equity-curve
    # shape compute_stats() expects from run_backtest() -- built here via
    # build_equity_curve_from_ledger() since the simulator itself only
    # persists balance + a trade ledger, never an in-memory equity curve.
    # Skipped if there are no closed trades yet: quantstats' metrics need
    # at least one return to be meaningful.
    if summary is not None and summary["total_trades"] > 0:
        conn = get_db_connection()
        try:
            equity_curve = build_equity_curve_from_ledger(
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
                save_simulator_stats(conn, exchange, symbol, strategy_name, time_horizon, stats_row)
            finally:
                conn.close()

    return len(ohlcv_1m)


if __name__ == "__main__":

    # Universe: every (exchange, symbol) pair currently active in
    # simulator.config -- replaces simulator/config.yaml's exchanges/symbols
    # lists. Flip a pair's is_active to False (e.g. from a frontend) to
    # stop it from running without deleting its history.
    conn = get_db_connection()
    try:
        universe = get_simulator_universe(conn)
    finally:
        conn.close()

    if not universe:
        raise RuntimeError(
            "No active (exchange, symbol) pairs found in simulator.config. "
            "Call save_simulator_config() for at least one pair first."
        )

    print(f"Active universe: {universe}")

    for exchange, symbol in universe:
        # Execution settings for THIS pair (initial_balance, position_size,
        # commission, slippage, allow_long, allow_short, max_open_positions)
        # -- shared across every strategy run against this pair, same as
        # simulator/config.yaml used to be shared across all strategies.
        conn = get_db_connection()
        try:
            config = get_simulator_config(conn, exchange, symbol)
        finally:
            conn.close()

        if config is None:
            print(f"{exchange} {symbol}: no simulator.config row -- skipping.")
            continue

        # Every strategy registered for THIS (exchange, symbol) pair in
        # metadata.strategy, filtered down to only the ones with
        # simulator_enabled = True -- False means this specific strategy
        # is turned off for simulator (independent of execution_enabled,
        # which execution/main.py checks instead).
        metadata_conn = get_metadata_connection()
        try:
            strategy_rows = get_strategies(metadata_conn, exchange=exchange, coin=symbol)
        finally:
            metadata_conn.close()

        strategy_rows = [s for s in strategy_rows if s.get("simulator_enabled", True)]

        if not strategy_rows:
            print(f"{exchange} {symbol}: no simulator-enabled strategies found in metadata.strategy -- skipping.")
            continue

        print(f"{exchange} {symbol}: {len(strategy_rows)} strategies -- "
              f"{[s['strategy_name'] for s in strategy_rows]}")

        # One shared simulator.positions table (one row per exchange+symbol+strategy)
        # plus one Trade Ledger table per (exchange, symbol, strategy) -- see
        # get_simulator_state/save_simulator_state/append_simulator_trades in
        # db_utils.py.
        for strategy_row in strategy_rows:
            strategy_name = strategy_row["strategy_name"]
            time_horizon = strategy_row["time_horizon"]
            strategy_config_dict = build_strategy_config_dict(strategy_row)

            # Every strategy row has its own take_profit_value/stop_loss_value
            # -- no shared default across strategies.
            # Postgres NUMERIC columns come back from psycopg2 as
            # decimal.Decimal, not float -- cast here so downstream math in
            # simulator.py (entry_price * (1 + tp_pct), plain floats) never
            # hits a "float * Decimal" TypeError.
            take_profit_pct = float(strategy_row["take_profit_value"])
            stop_loss_pct = float(strategy_row["stop_loss_value"])

            run_simulator(
                exchange, symbol, config, strategy_name, time_horizon, strategy_config_dict,
                take_profit_pct, stop_loss_pct
            )