"""
main.py
-------

Entry point of the Backtest Module.

Pulls OHLCV data via get_data() the same way the Signal Module does, runs
the signal pipeline live (generate_signals), then feeds those signals plus
1-minute OHLCV into the vectorized backtest engine and stores the resulting
trade ledger per symbol in Postgres (backtest.{exchange}_{symbol} -- see
insert_trades() in utils/db_utils.py).

The date range (start_date/end_date) lives in backtest/config.yaml and is
the single source of truth used for both the signal-generation data pull
and the execution data pull, so both sides of the backtest always cover
the exact same window.
"""

from datetime import datetime

import pandas as pd

from crypto_pipeline.backtest.backtest import load_config, run_backtest
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.utils.db_utils import get_db_connection, get_candles_from_db, insert_trades


def parse_backtest_dates(config: dict) -> dict:
    """
    Parse config["start_date"]/config["end_date"] into datetime objects.

    Unlike utils.pipeline_utils.parse_config_dates (date-only, "%Y-%m-%d"),
    this also accepts a full timestamp ("%Y-%m-%d %H:%M:%S") since backtest
    windows often need to be pinned to an exact hour/minute of DB coverage.
    "now" is left as-is -- get_data() resolves it at call time.
    """
    for key in ("start_date", "end_date"):
        value = config[key]
        if value == "now":
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                config[key] = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized date format for {key}: {value!r}")
    return config


def get_1m_data(exchange: str, symbol: str, start_date, end_date) -> pd.DataFrame:
    """
    Pure DB read of stored 1-minute candles -- no resampling, no exchange
    API fallback for a live gap. Lives here (not in data_downloader.py)
    because it's specific to how the backtest pulls execution data: we
    already know the DB covers [start_date, end_date] (end_date is pinned
    to a known-good timestamp in config.yaml), so there's nothing to
    resample and no gap to fetch live -- get_data()'s extra machinery
    would just be unused overhead here.

    Opens/closes its own connection, same pattern as get_data().
    """
    conn = get_db_connection()
    try:
        return get_candles_from_db(conn, exchange, symbol, start_date, end_date)
    finally:
        conn.close()


def build_signals(resampled_df):
    """
    Run the signal pipeline on resampled (1h) OHLCV and return a plain
    datetime/signal DataFrame, ready for the backtest engine.

    Drops warm-up rows where indicators/conditions aren't fully formed yet
    (e.g. an SMA_20 needs 20 bars before it produces a value) -- same idea
    as the dropna() step in signals/main.py, just done generically instead
    of hardcoding indicator column names.
    """
    indicator_df, condition_df, signal_series = generate_signals(resampled_df)

    combined = pd.concat([indicator_df, condition_df], axis=1)
    combined["signal"] = signal_series
    combined = combined.dropna().reset_index(drop=True)

    return combined[["datetime", "signal"]]


if __name__ == "__main__":

    # start_date/end_date come from backtest/config.yaml -- the one place
    # that defines the window for this backtest run, used for both data
    # pulls below.
    backtest_config = parse_backtest_dates(load_config())

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    conn = get_db_connection()

    try:
        for exchange in exchanges:
            for symbol in symbols:

                # 1h data, just for signal generation -- same call pattern
                # signals/main.py itself uses (df_1m left False, so only
                # "resampled" comes back).
                hourly_result = get_data(
                    exchange=exchange,
                    symbol=symbol,
                    start_date=backtest_config["start_date"],
                    end_date=backtest_config["end_date"],
                )
                ohlcv_1h = hourly_result["resampled"]

                if ohlcv_1h.empty:
                    print(f"Skipping {exchange} {symbol}: no hourly data returned.")
                    continue

                signals = build_signals(ohlcv_1h)

                # 1-minute data, for backtest execution. Same start/end date as
                # above so both pulls cover the same window. This reads straight
                # from the DB (no resample, no exchange fallback) since we
                # already know the DB covers this exact window.
                ohlcv_1m = get_1m_data(
                    exchange=exchange,
                    symbol=symbol,
                    start_date=backtest_config["start_date"],
                    end_date=backtest_config["end_date"],
                )

                if ohlcv_1m.empty:
                    print(f"Skipping {exchange} {symbol}: no 1-minute data returned.")
                    continue

                backtest_result = run_backtest(ohlcv_1m, signals, backtest_config)

                print(
                    f"{exchange} {symbol}: "
                    f"{backtest_result['total_trades']} trades, "
                    f"final balance {backtest_result['final_balance']:.2f}, "
                    f"net profit {backtest_result['total_net_profit']:.2f}"
                )

                # Store in DB: backtest.{exchange}_{symbol}, full rebuild each run
                insert_trades(conn, exchange, symbol, backtest_result["trade_ledger"])

    finally:
        conn.close()