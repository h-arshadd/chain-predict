"""
run_metadata_tables.py
-----------------------
Creates the `metadata` schema + tables in the real DB, and demos
inserting/reading the (now global, not per-pair) strategy.

Usage:
    python run_metadata_tables.py

Needs the same .env as the rest of the pipeline (DB_HOST, DB_PORT,
DB_NAME, DB_USER, DB_PASSWORD) -- metadata_utils.get_db_connection()
reads it via python-dotenv, same as db_utils.py.
"""

from crypto_pipeline.signals.main import load_config as load_signals_config, split_config
from crypto_pipeline.backtest.backtest import load_config as load_backtest_config
from crypto_pipeline.utils.metadata_utils import (
    get_db_connection,
    create_all_metadata_tables,
    seed_data_pairs,
    get_data_rows,
    seed_sentiment_pairs,
    get_sentiment_rows,
    insert_strategy,
    get_current_strategy,
    get_strategies,
    insert_backtest,
    get_backtests,
)


def main():
    conn = get_db_connection()

    try:
        # 1. Create metadata schema + all three tables (data, strategy,
        #    sentiment). Safe to run repeatedly -- CREATE ... IF NOT EXISTS.
        create_all_metadata_tables(conn)

        # 1b. Seed metadata.data with the 16 (exchange, symbol) pairs from
        #     data/binance/config_binance.yml + data/bybit/config_bybit.yml
        #     (8 symbols x 2 exchanges). Only seed if the table is empty,
        #     so re-running this script doesn't duplicate rows.
        existing_data_rows = get_data_rows(conn)
        if not existing_data_rows:
            seeded_ids = seed_data_pairs(conn)
            print(f"Seeded {len(seeded_ids)} rows into metadata.data")
        else:
            print(f"metadata.data already has {len(existing_data_rows)} rows -- skipping seed")

        # 1c. Seed metadata.sentiment with the 6 (coin, subreddit) pairs from
        #     sentiment_pipeline/config.yaml (2 coins x 3 subreddits each).
        #     Same empty-check guard as metadata.data above.
        existing_sentiment_rows = get_sentiment_rows(conn)
        if not existing_sentiment_rows:
            seeded_sentiment_ids = seed_sentiment_pairs(conn)
            print(f"Seeded {len(seeded_sentiment_ids)} rows into metadata.sentiment")
        else:
            print(f"metadata.sentiment already has {len(existing_sentiment_rows)} rows -- skipping seed")

        # 2. Register the ACTIVE strategy straight from signals/config.yaml,
        #    including its strategy_name. This is the ACTIVE strategy and
        #    runs against EVERY tracked pair now -- no data_id, no
        #    exchange/symbol passed in, since it's global.
        raw_config = load_signals_config()
        strategy_name = raw_config.get("strategy_name", "unnamed_strategy")
        indicator_config, strategy_config = split_config(raw_config)

        # strategy_config here is the indicator blocks (RSI, EMA, ...) plus
        # the "strategy" long/short conditions -- same shape insert_strategy
        # expects, so it's reassembled the same way signals/config.yaml
        # itself is structured.
        full_strategy_config = {**indicator_config, "strategy": strategy_config}

        strategy_id = insert_strategy(conn, strategy_name, full_strategy_config, timeframe="1h")
        print(f"Inserted strategy_id={strategy_id} ({strategy_name!r})")

        # 3. Read back "the current strategy" -- what signals/main.py would
        #    call once per run, then loop it over every tracked pair.
        current = get_current_strategy(conn)
        print("Current strategy:", current)

        # 4. Full history, newest first.
        print(f"Total strategies on record: {len(get_strategies(conn))}")

        # 5. Register a backtest run against that same strategy_name,
        #    straight from backtest/config.yaml (date range, position
        #    sizing, commission/slippage, TP/SL, execution, portfolio
        #    limits) -- stored whole as JSON.
        backtest_config = load_backtest_config()
        backtest_id = insert_backtest(conn, strategy_name, backtest_config)
        print(f"Inserted backtest_id={backtest_id} for strategy {strategy_name!r}")

        # 6. Full backtest history for this strategy, newest first.
        print(f"Total backtests on record for {strategy_name!r}: "
              f"{len(get_backtests(conn, strategy_name=strategy_name))}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()