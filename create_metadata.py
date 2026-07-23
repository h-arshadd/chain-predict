"""
switch_to_rsi.py
-----------------
One-off script: for every (exchange, coin) pair currently running
ema_50_200_long_only_trend in execution, flip execution_enabled OFF for
that strategy and ON for rsi_14_reversal on the same pair.

Looks rows up by name via get_strategies() instead of hardcoding
strategy_id -- safer than guessing IDs from a screenshot. Prints a
before/after summary per pair; does NOT touch pairs where either strategy
row doesn't exist (prints a warning and skips instead of guessing).

Run once: python switch_to_rsi.py
"""

from crypto_pipeline.utils.metadata_utils import (
    get_db_connection,
    get_strategies,
    set_strategy_enabled,
)

OLD_STRATEGY = "ema_50_200_long_only_trend"
NEW_STRATEGY = "RSI_14_reversal"

# Same 8-coin universe as execution.config.
COINS = ["btc", "eth", "sol", "doge", "ada", "ltc", "mina", "sui"]
EXCHANGE = "bybit"


def find_row(rows, strategy_name):
    matches = [r for r in rows if r["strategy_name"] == strategy_name]
    if not matches:
        return None
    # get_strategies() orders by created_at DESC, strategy_id DESC --
    # first match is the most recent row for this name.
    return matches[0]


def main():
    conn = get_db_connection()
    try:
        for coin in COINS:
            rows = get_strategies(conn, exchange=EXCHANGE, coin=coin)

            old_row = find_row(rows, OLD_STRATEGY)
            new_row = find_row(rows, NEW_STRATEGY)

            if old_row is None and new_row is None:
                print(f"{EXCHANGE} {coin}: neither {OLD_STRATEGY!r} nor {NEW_STRATEGY!r} found -- skipping.")
                continue

            if new_row is None:
                print(f"{EXCHANGE} {coin}: {NEW_STRATEGY!r} not found -- skipping (nothing to enable).")
                continue

            if old_row is not None and old_row.get("execution_enabled", True):
                set_strategy_enabled(conn, old_row["strategy_id"], execution_enabled=False)
                print(f"{EXCHANGE} {coin}: disabled {OLD_STRATEGY!r} (strategy_id={old_row['strategy_id']})")

            if not new_row.get("execution_enabled", True):
                set_strategy_enabled(conn, new_row["strategy_id"], execution_enabled=True)
            else:
                # Already True in the DB -- still call it to be explicit/idempotent.
                set_strategy_enabled(conn, new_row["strategy_id"], execution_enabled=True)
            print(f"{EXCHANGE} {coin}: enabled {NEW_STRATEGY!r} (strategy_id={new_row['strategy_id']})")

    finally:
        conn.close()


if __name__ == "__main__":
    main()