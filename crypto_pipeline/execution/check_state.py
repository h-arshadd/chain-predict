"""
check_state.py
---------------
One-off diagnostic, not part of the app. Prints, for every pair in
execution.config: whether a wallet is assigned, and how stale
execution.positions is (last_processed vs now) -- so you can see
exactly which rows are live vs frozen before running main.py again.

Usage: python -m crypto_pipeline.execution.check_state
"""

from datetime import datetime, timezone

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_universe,
    get_execution_config,
    get_execution_state,
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

    print(f"{'pair':<12}{'wallet':<20}{'strategy':<20}{'status':<10}{'last_processed':<22}{'age'}")
    print("-" * 100)

    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            config = get_execution_config(conn, exchange, symbol)
            account_name = config.get("account_name") if config else None

            strategy_name = _strategy_name_for(conn, exchange, symbol)
            if strategy_name is None:
                print(f"{exchange}/{symbol:<8}{str(account_name):<20}{'—':<20}{'no strategy':<10}")
                continue

            state = get_execution_state(conn, exchange, symbol, strategy_name)
        finally:
            conn.close()

        if state is None:
            print(f"{exchange}/{symbol:<8}{str(account_name):<20}{strategy_name:<20}{'never_run':<10}")
            continue

        last_processed = state["last_processed"]
        position = state["position"]
        status = "OPEN" if position else "flat"

        age_str = "—"
        if last_processed is not None:
            lp = last_processed if last_processed.tzinfo else last_processed.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - lp
            hours = age.total_seconds() / 3600
            age_str = f"{hours:.1f}h ago" if hours < 48 else f"{hours/24:.1f}d ago"

        print(f"{exchange}/{symbol:<8}{str(account_name):<20}{strategy_name:<20}{status:<10}{str(last_processed):<22}{age_str}")


if __name__ == "__main__":
    main()