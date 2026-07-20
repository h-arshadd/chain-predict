"""
main.py
-------

Entry point of the Simulator Module.

Meant to be run repeatedly (Task Scheduler -> run_simulator.bat), the same
way run_pipeline.bat drives the data pipelines. Each run:

  1. Loads every strategy config in signals/strategies/*.yaml (10-20
     strategies, each a self-contained file: indicators, long/short rules,
     strategy_name, and this strategy's own time_horizon).
  2. For every (exchange, symbol, strategy) combination:
       a. Pulls live OHLCV via get_data() (same call every other module
          uses -- reads whatever's in the DB and fetches any live gap from
          the exchange), resampled to THIS strategy's time_horizon.
       b. Loads saved state (last processed candle, balance, open position)
          from simulator.{exchange}_{symbol}_{strategy}_state -- or starts
          fresh if this is the first run for that strategy.
       c. Walks forward candle by candle over whatever 1-minute candles are
          new since last_processed, calling simulator.step_candle() for
          each one.
       d. Signals only change once a full resampled-timeframe candle closes
          (this strategy's time horizon), so a new signal is only fed in on
          the 1-minute candle that lines up with a newly-closed resampled
          candle -- every other 1-minute candle just monitors the open
          position (signal=0). TP/SL is still checked every 1-minute candle
          regardless of time horizon -- see simulator.py's step_candle().
       e. Saves state back to the DB, appends any newly-closed trades to
          the running Trade Ledger table (its own Position Table + Trade
          Ledger per strategy, per exchange, per symbol). The DB is the
          only source of truth -- no CSV is written; query
          simulator.{exchange}_{symbol}_{strategy}_trades directly to
          inspect a strategy's ledger.

Execution settings (initial_balance, position_size, commission, slippage,
allow_long, allow_short, take_profit, stop_loss, max_open_positions) come
from simulator/config.yaml and are shared across every strategy -- that
file has no strategy-specific fields (see Simulator Module spec: execution
settings must stay separate from strategy rules). Each strategy's own
indicators/conditions/rule and time_horizon live only in its own file
under signals/strategies/.

The universe (which exchanges and coins/symbols to run every strategy on)
also lives in simulator/config.yaml, under "exchanges" and "symbols" --
add or remove a coin there and it's automatically picked up for every
strategy, no code changes needed.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from crypto_pipeline.simulator.simulator import load_config, step_candle
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_simulator_state,
    save_simulator_state,
    append_simulator_trades,
    get_simulator_summary,
)

# Path is crypto_pipeline/simulator/main.py -> parent.parent gets to
# crypto_pipeline/ -- then straight into signals/strategies/ (flat layout:
# crypto_pipeline/signals/ and crypto_pipeline/simulator/ are siblings).
STRATEGIES_DIR = Path(__file__).parent.parent / "signals" / "strategies"


def load_strategies(strategies_dir=None):
    """
    Load every *.yaml file under signals/strategies/ as one strategy config
    each. Returns a list of dicts, each the full parsed YAML (so it has
    strategy_name, time_horizon, indicator blocks, and the strategy rules
    all in one place) plus "_config_path" (str) so generate_signals() can
    be pointed at that exact file.

    Add or remove a strategy by adding/removing a file here -- nothing else
    needs to change to pick it up.
    """
    if strategies_dir is None:
        strategies_dir = STRATEGIES_DIR

    import yaml

    strategies = []
    for path in sorted(Path(strategies_dir).glob("*.yaml")):
        with open(path, "r") as f:
            strategy_config = yaml.safe_load(f)

        if "strategy_name" not in strategy_config:
            raise ValueError(f"{path} is missing required key 'strategy_name'.")
        if "time_horizon" not in strategy_config:
            raise ValueError(f"{path} is missing required key 'time_horizon'.")

        strategy_config["_config_path"] = str(path)
        strategies.append(strategy_config)

    return strategies


def parse_simulator_start_date(config: dict):
    """
    Parse config["start_date"] (simulator/config.yaml) into a datetime.
    Only used as the fallback data-pull start for a (exchange, symbol,
    strategy) combo that has never run before -- once state exists,
    last_processed from the DB always wins instead (see run_simulator()).

    Same date formats as backtest's parse_backtest_dates, minus the "now"
    special case since start_date is never "now" here (this is the start,
    not the end -- end_date is always "now" for a live/paper simulator,
    handled directly in run_simulator()).
    """
    value = config["start_date"]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format for start_date: {value!r}")


def build_resampled_signals(resampled_df, config_path):
    """
    Run the signal pipeline on resampled OHLCV for ONE strategy (identified
    by config_path), same pattern as backtest/main.py's build_signals().
    Returns datetime/signal only, warm-up rows dropped.
    """
    indicator_df, condition_df, signal_series = generate_signals(resampled_df, config_path=config_path)

    combined = pd.concat([indicator_df, condition_df], axis=1)
    combined["signal"] = signal_series
    combined = combined.dropna().reset_index(drop=True)

    return combined[["datetime", "signal"]]


def run_simulator(exchange, symbol, config, strategy_name, time_horizon, strategy_config_path, default_start_date):
    """
    Advance one exchange+symbol+strategy simulation by however many new
    1-minute candles are available. Returns the number of candles processed.

    time_horizon is THIS strategy's own resampled timeframe (e.g. "2h"),
    read from its config file -- controls both how signals are generated
    (resample target) and the entry gate below (a new signal only takes
    effect on the 1-minute candle where a new time_horizon candle closes).
    It does NOT change how often TP/SL is checked -- that's every 1-minute
    candle inside step_candle(), independent of time horizon.

    default_start_date : datetime -- config["start_date"] from
    simulator/config.yaml, parsed. Only used if this is the very first run
    for this (exchange, symbol, strategy) combo (no saved state yet).
    """
    conn = get_db_connection()
    try:
        state = get_simulator_state(conn, exchange, symbol, strategy_name)
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

    # Pull live 1m data (+ resampled, for signals) starting right after
    # whatever we've already processed. First run ever for this
    # (exchange, symbol, strategy): no last_processed to resume from, so
    # fall back to config["start_date"] (simulator/config.yaml) instead --
    # set that to wherever your DB's data actually begins.
    start_date = last_processed if last_processed is not None else default_start_date

    result = get_data(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date="now",
        timeframe=time_horizon,
        df_1m=True,
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
        signals = build_resampled_signals(ohlcv_resampled, strategy_config_path)

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

        position, balance, closed_trade = step_candle(candle, signal, position, balance, config)

        if closed_trade is not None:
            closed_trade["exchange"] = exchange
            closed_trade["symbol"] = symbol
            # Same column backtest.py's ledger has: running P&L since this
            # strategy's very first trade (not just this run's batch), so
            # it's correct across resumed runs too. balance_after_trade
            # already reflects every trade ever closed for this
            # (exchange, symbol, strategy), so this is just that minus
            # where the account started -- no extra running state needed.
            closed_trade["cumulative_pnl"] = closed_trade["balance_after_trade"] - config["initial_balance"]
            closed_trades.append(closed_trade)

        last_candle_time = candle["datetime"]

    conn = get_db_connection()
    try:
        save_simulator_state(conn, exchange, symbol, strategy_name, last_candle_time, balance, position)
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

    return len(ohlcv_1m)


if __name__ == "__main__":

    config = load_config()  # simulator/config.yaml -- execution settings + universe (exchanges/symbols), shared by all strategies
    default_start_date = parse_simulator_start_date(config)

    strategies = load_strategies()
    if not strategies:
        raise RuntimeError(f"No strategy files found under {STRATEGIES_DIR}. Add at least one *.yaml there.")

    print(f"Loaded {len(strategies)} strategies: {[s['strategy_name'] for s in strategies]}")

    # Universe (which exchanges/coins to run every strategy on) comes from
    # simulator/config.yaml -- edit the "exchanges"/"symbols" lists there to
    # add or remove a coin, nothing here needs to change.
    exchanges = config["exchanges"]
    symbols = config["symbols"]

    # One live Position Table + Trade Ledger per (exchange, symbol, strategy)
    # -- see get_simulator_state/save_simulator_state/append_simulator_trades
    # in db_utils.py, keyed on all three.
    for strategy_config in strategies:
        strategy_name = strategy_config["strategy_name"]
        time_horizon = strategy_config["time_horizon"]
        strategy_config_path = strategy_config["_config_path"]

        for exchange in exchanges:
            for symbol in symbols:
                run_simulator(exchange, symbol, config, strategy_name, time_horizon, strategy_config_path, default_start_date)