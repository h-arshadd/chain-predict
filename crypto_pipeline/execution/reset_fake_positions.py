"""
reset_fake_positions.py
------------------------
One-off cleanup, not part of the app. The 8 pairs in execution.config
all show an OPEN position in execution.positions from a single seeded/
test run (2026-07-24 05:57:00) that never actually placed an order on
Bybit -- confirmed by Bybit's own Positions tab showing 0 open positions.

Left in place, the next real main.py run would hit
_reconcile_bybit_close(): DB says open, Bybit says flat -> it assumes
Bybit auto-closed the position via native TP/SL and tries to pull a
"real" closing fill via get_last_closed_pnl() for a trade that was never
opened -- producing a bogus closed trade (or an error/crash) instead of
correctly treating the pair as flat.

This script, for every pair in execution.config:
  1. Prints the current (fake) execution.positions row.
  2. Resets execution.positions to flat: position fields NULL,
     balance back to initial_balance, cumulative_pnl 0, last_processed
     NULL (so the next run behaves exactly like a genuine first run --
     same is_first_run=True path, no backlog replay, per the earlier
     backlog-replay fix).
  3. Deletes any OPEN row (exit_date_time IS NULL) from that pair's
     trade ledger table -- the phantom trade must not show up in trade
     history either. CLOSED rows, if any, are left untouched --
     wiping trade history isn't the goal, only the never-real open row.

Nothing here touches Bybit -- there's nothing on Bybit to touch, that's
the whole point. Pure DB cleanup.

Usage: python -m crypto_pipeline.execution.reset_fake_positions
       python -m crypto_pipeline.execution.reset_fake_positions --dry-run
"""

import sys

from psycopg2 import sql

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_universe,
    get_execution_config,
    get_execution_state,
    _execution_trades_table,
)
from crypto_pipeline.utils.metadata_utils import get_strategies


def _strategy_name_for(conn, exchange, symbol):
    strategies = get_strategies(conn, exchange, symbol)
    for s in strategies:
        if s.get("execution_enabled"):
            return s["strategy_name"]
    return strategies[0]["strategy_name"] if strategies else None


def _reset_position_row(conn, exchange, symbol, strategy_name, initial_balance, dry_run):
    cursor = conn.cursor()
    if dry_run:
        cursor.close()
        return
    cursor.execute(sql.SQL("""
        UPDATE {schema}.positions
        SET last_processed = NULL,
            balance = %s,
            cumulative_pnl = 0,
            direction = NULL,
            entry_time = NULL,
            entry_price = NULL,
            quantity = NULL,
            take_profit = NULL,
            stop_loss = NULL,
            leaning = NULL,
            status = NULL
        WHERE exchange = %s AND symbol = %s AND strategy_name = %s
    """).format(schema=sql.Identifier("execution")), (initial_balance, exchange, symbol, strategy_name))
    conn.commit()
    cursor.close()


def _delete_open_ledger_row(conn, table_name, dry_run):
    cursor = conn.cursor()
    qualified_name = sql.SQL(".").join(
        [sql.Identifier("execution"), sql.Identifier(table_name)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    if cursor.fetchone()[0] is None:
        cursor.close()
        return 0

    if dry_run:
        cursor.execute(sql.SQL(
            "SELECT count(*) FROM {schema}.{table} WHERE exit_date_time IS NULL"
        ).format(schema=sql.Identifier("execution"), table=sql.Identifier(table_name)))
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    cursor.execute(sql.SQL(
        "DELETE FROM {schema}.{table} WHERE exit_date_time IS NULL"
    ).format(schema=sql.Identifier("execution"), table=sql.Identifier(table_name)))
    count = cursor.rowcount
    conn.commit()
    cursor.close()
    return count


def main():
    dry_run = "--dry-run" in sys.argv

    conn = get_db_connection()
    try:
        universe = get_execution_universe(conn)
    finally:
        conn.close()

    print(f"{'mode':<10}: {'DRY RUN (no changes)' if dry_run else 'LIVE (will modify DB)'}\n")

    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            config = get_execution_config(conn, exchange, symbol)
            strategy_name = _strategy_name_for(conn, exchange, symbol)
            if strategy_name is None:
                print(f"{exchange}/{symbol}: no strategy found -- skipping.")
                continue

            state = get_execution_state(conn, exchange, symbol, strategy_name)
            if state is None or state["position"] is None:
                print(f"{exchange}/{symbol} ({strategy_name}): already flat -- nothing to reset.")
                continue

            position = state["position"]
            print(
                f"{exchange}/{symbol} ({strategy_name}): fake {position['direction']} "
                f"qty={position['quantity']} entry={position['entry_price']} "
                f"last_processed={state['last_processed']}"
            )

            initial_balance = config["initial_balance"] if config else state["balance"]
            _reset_position_row(conn, exchange, symbol, strategy_name, initial_balance, dry_run)

            table_name = _execution_trades_table(exchange, symbol, strategy_name)
            deleted = _delete_open_ledger_row(conn, table_name, dry_run)
            action = "would delete" if dry_run else "deleted"
            print(f"  -> reset execution.positions to flat, {action} {deleted} open ledger row(s).")
        finally:
            conn.close()

    print("\nDone." if not dry_run else "\nDry run complete -- re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()