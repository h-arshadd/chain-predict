"""
show_recent_trades.py
-----------------------
One-off diagnostic, not part of the app. Lists every trade row (open or
closed) from each pair's ledger table, most recent first -- so you can
see the real entry_date_time for every position opened by the last
main.py run, instead of just "it happened at the same time I ran it."

If entry_date_time values are spread across many distinct timestamps
(e.g. every few minutes going back hours) rather than all landing on the
exact same wall-clock moment, that confirms the run walked historical
1-minute candles one at a time and fired a real order at each signal it
found along the way -- not a single look-ahead order priced off live
data at the instant the script launched.

Usage: python -m crypto_pipeline.execution.show_recent_trades
"""

from psycopg2 import sql

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_universe,
    _execution_trades_table,
)
from crypto_pipeline.utils.metadata_utils import get_strategies


def _strategy_name_for(conn, exchange, symbol):
    strategies = get_strategies(conn, exchange, symbol)
    for s in strategies:
        if s.get("execution_enabled"):
            return s["strategy_name"]
    return strategies[0]["strategy_name"] if strategies else None


def main():
    conn = get_db_connection()
    try:
        universe = get_execution_universe(conn)
    finally:
        conn.close()

    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            strategy_name = _strategy_name_for(conn, exchange, symbol)
            if strategy_name is None:
                continue
            table_name = _execution_trades_table(exchange, symbol, strategy_name)

            cursor = conn.cursor()
            qualified_name = sql.SQL(".").join(
                [sql.Identifier("execution"), sql.Identifier(table_name)]
            ).as_string(conn)
            cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
            if cursor.fetchone()[0] is None:
                cursor.close()
                continue

            cursor.execute(sql.SQL("""
                SELECT entry_date_time, direction, entry_price, quantity,
                       exit_date_time, exit_price, net_pnl, exit_reason, status
                FROM {schema}.{table}
                ORDER BY entry_date_time DESC
                LIMIT 10
            """).format(schema=sql.Identifier("execution"), table=sql.Identifier(table_name)))
            rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()

        if not rows:
            continue

        print(f"\n{exchange}/{symbol} ({strategy_name}):")
        print(f"  {'entry_time':<22}{'dir':<8}{'entry_px':<12}{'qty':<10}{'exit_time':<22}{'status'}")
        for r in rows:
            entry_dt, direction, entry_px, qty, exit_dt, exit_px, net_pnl, exit_reason, status = r
            print(f"  {str(entry_dt):<22}{direction:<8}{entry_px:<12}{qty:<10}{str(exit_dt):<22}{status}")


if __name__ == "__main__":
    main()