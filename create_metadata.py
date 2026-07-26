"""
run_backtest_direct.py
-----------------------
Runs one real backtest end-to-end and writes it straight into the DB --
no API server, no frontend needed. Same pipeline api/repos/backtests_repo.py
uses (get_data -> generate_signals -> run_backtest), just called directly
from a script instead of a FastAPI BackgroundTask.

What it does:
    1. Picks a strategy from metadata.strategy (by name, or the first
       available if EXCHANGE/COIN/STRATEGY_NAME below don't match one).
    2. Inserts a 'pending' row into metadata.backtest.
    3. Pulls OHLCV data, generates signals, runs the vectorized backtest.
    4. Writes the trade ledger to backtest.run_{backtest_id} and the
       equity curve to backtest.run_{backtest_id}_equity.
    5. Marks the metadata.backtest row 'completed' (or 'failed' with the
       error message) -- exactly what shows up in the frontend's
       Backtest Requests table.

Run from the repo root:
    python run_backtest_direct.py
"""

from datetime import datetime

import pandas as pd
import yaml

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_candles_from_db,
    insert_backtest_trades,
    save_backtest_equity_curve,
)
from crypto_pipeline.utils.metadata_utils import (
    get_strategies,
    create_backtest_table,
    insert_backtest,
    start_backtest,
    complete_backtest,
    fail_backtest,
)
from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.signals.main import generate_signals
from crypto_pipeline.backtest.backtest import run_backtest, load_config as load_backtest_config

# ------------------------------------------------------------------
# Edit these to pick which strategy/pair to backtest. If STRATEGY_NAME
# is None, the first strategy row found for EXCHANGE/COIN is used.
# ------------------------------------------------------------------
EXCHANGE = "bybit"
COIN = "btc"
STRATEGY_NAME = None  # e.g. "RSI_14_reversal", or None for "just pick one"


def parse_date(value):
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value!r}")


def main():
    conn = get_db_connection()
    try:
        # Make sure metadata.backtest exists (safe/no-op if it already does).
        create_backtest_table(conn)

        # ---- 1. Pick a strategy row ----
        strategies = get_strategies(conn, exchange=EXCHANGE, coin=COIN)
        if not strategies:
            print(f"No strategy rows found for {EXCHANGE}/{COIN}.")
            print("Load strategies first, e.g.:")
            print("  python -c \"from crypto_pipeline.utils.metadata_utils import get_db_connection, load_strategies_from_yaml; conn = get_db_connection(); load_strategies_from_yaml(conn, 'crypto_pipeline/signals/strategies'); conn.close()\"")
            return

        if STRATEGY_NAME:
            matches = [s for s in strategies if s["strategy_name"] == STRATEGY_NAME]
            if not matches:
                print(f"Strategy {STRATEGY_NAME!r} not found for {EXCHANGE}/{COIN}. Available:")
                for s in strategies:
                    print(f"  - {s['strategy_name']}")
                return
            strategy_row = matches[0]
        else:
            strategy_row = strategies[0]

        print(f"Using strategy: {strategy_row['strategy_name']!r} (strategy_id={strategy_row['strategy_id']}, {EXCHANGE}/{COIN})")

        # ---- 2. Build backtest config (defaults from backtest/config.yaml) ----
        config = load_backtest_config()
        config["start_date"] = parse_date(config["start_date"])
        config["end_date"] = parse_date(config["end_date"])

        # ---- 3. Insert a pending row into metadata.backtest ----
        # backtest_config stored back as strings (JSON-friendly), same as
        # backtests_repo.create_backtest_request does with the raw config.
        config_for_storage = dict(config)
        config_for_storage["start_date"] = config["start_date"].strftime("%Y-%m-%d %H:%M:%S")
        config_for_storage["end_date"] = config["end_date"].strftime("%Y-%m-%d %H:%M:%S")

        backtest_id = insert_backtest(
            conn,
            strategy_name=strategy_row["strategy_name"],
            backtest_config=config_for_storage,
            strategy_id=strategy_row["strategy_id"],
            status="pending",
        )
        print(f"Inserted metadata.backtest row: backtest_id={backtest_id}")

        start_backtest(conn, backtest_id)

        # ---- 4. Pull data, generate signals, run the backtest ----
        timeframe = strategy_row.get("time_horizon") or "1h"

        hourly_result = get_data(
            exchange=EXCHANGE,
            symbol=COIN,
            start_date=config["start_date"],
            end_date=config["end_date"],
            timeframe=timeframe,
        )
        ohlcv_resampled = hourly_result["resampled"]
        if ohlcv_resampled.empty:
            fail_backtest(conn, backtest_id, f"No {timeframe} data available for {EXCHANGE}/{COIN} in that date range.")
            print("FAILED: no resampled data in that date range. Have you run the data pipeline for this pair?")
            return

        indicator_df, condition_df, signal_series = generate_signals(
            ohlcv_resampled, config_dict=strategy_row["strategy_config"]
        )
        combined = pd.concat([indicator_df, condition_df], axis=1)
        combined["signal"] = signal_series
        combined = combined.dropna().reset_index(drop=True)
        signals = combined[["datetime", "signal"]]

        ohlcv_1m = get_candles_from_db(conn, EXCHANGE, COIN, config["start_date"], config["end_date"])
        if ohlcv_1m.empty:
            fail_backtest(conn, backtest_id, f"No 1-minute data available for {EXCHANGE}/{COIN} in that date range.")
            print("FAILED: no 1-minute data in that date range.")
            return

        result = run_backtest(ohlcv_1m, signals, config)

        # ---- 5. Persist trades/equity, mark completed ----
        insert_backtest_trades(conn, backtest_id, result["trade_ledger"])
        save_backtest_equity_curve(conn, backtest_id, result["equity_curve"])

        complete_backtest(conn, backtest_id, {
            "final_balance": result["final_balance"],
            "total_net_profit": result["total_net_profit"],
            "total_trades": result["total_trades"],
            "win_loss": result["win_loss"],
        })

        print(
            f"COMPLETED backtest_id={backtest_id}: "
            f"{result['total_trades']} trades, "
            f"final balance {result['final_balance']:.2f}, "
            f"net profit {result['total_net_profit']:.2f}"
        )

    except Exception as exc:
        # If we got as far as creating backtest_id, mark it failed so the
        # frontend's Failed tab reflects it instead of leaving it stuck.
        if "backtest_id" in dir():
            fail_backtest(conn, backtest_id, str(exc))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()