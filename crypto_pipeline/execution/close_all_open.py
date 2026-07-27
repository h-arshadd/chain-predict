"""
close_all_open.py
-------------------
One-off cleanup, not part of the app. Closes every currently open
position in execution.positions with a REAL market order on Bybit (via
the same _close_live_position() main.py itself uses), then records the
close in both execution.positions and the trade ledger.

This exists because a bad reset (last_processed set to NULL instead of
deleting the row) caused execution/main.py's is_first_run guard to miss,
so its last run replayed ~500 historical 1-minute candles as if they
were live and fired REAL opening orders off stale, already-passed
signals -- not off genuinely live data. This script unwinds those real
positions with real closing orders (there is no "undo" on an exchange,
only a new opposing trade), so the account goes back to flat and the
DB matches reality again.

This places REAL orders on the wallet(s) assigned in execution.config.
Confirms before doing anything.

Usage: python -m crypto_pipeline.execution.close_all_open
       python -m crypto_pipeline.execution.close_all_open --yes   (skip confirmation)
"""

import sys
from datetime import datetime

from crypto_pipeline.utils.db_utils import (
    get_db_connection,
    get_execution_universe,
    get_execution_config,
    get_execution_state,
    save_execution_state,
    close_execution_trade,
)
from crypto_pipeline.utils.metadata_utils import get_strategies
from crypto_pipeline.accounts.accounts_utils import get_account_api_key
from crypto_pipeline.execution.bybit_client import get_client
from crypto_pipeline.execution.main import _close_live_position


def _strategy_name_for(conn, exchange, symbol):
    strategies = get_strategies(conn, exchange, symbol)
    for s in strategies:
        if s.get("execution_enabled"):
            return s["strategy_name"]
    return strategies[0]["strategy_name"] if strategies else None


def main():
    skip_confirm = "--yes" in sys.argv

    conn = get_db_connection()
    try:
        universe = get_execution_universe(conn)
    finally:
        conn.close()

    to_close = []
    for exchange, symbol in universe:
        conn = get_db_connection()
        try:
            config = get_execution_config(conn, exchange, symbol)
            strategy_name = _strategy_name_for(conn, exchange, symbol)
            if strategy_name is None or config is None:
                continue
            state = get_execution_state(conn, exchange, symbol, strategy_name)
            if state is None or state["position"] is None:
                continue
            to_close.append((exchange, symbol, strategy_name, config, state))
        finally:
            conn.close()

    if not to_close:
        print("Nothing open -- all pairs already flat.")
        return

    print(f"{len(to_close)} open position(s) will be CLOSED WITH REAL MARKET ORDERS on Bybit:\n")
    for exchange, symbol, strategy_name, config, state in to_close:
        p = state["position"]
        print(f"  {exchange}/{symbol} ({strategy_name}): {p['direction']} qty={p['quantity']} entry={p['entry_price']}")

    if not skip_confirm:
        answer = input("\nType 'close them' to proceed, anything else to abort: ")
        if answer.strip().lower() != "close them":
            print("Aborted -- nothing was touched.")
            return

    for exchange, symbol, strategy_name, config, state in to_close:
        account_name = config.get("account_name")

        conn = get_db_connection()
        try:
            wallet = get_account_api_key(conn, account_name) if account_name else None
        finally:
            conn.close()

        if wallet is None:
            print(f"{exchange}/{symbol}: no wallet found for '{account_name}' -- SKIPPED, close manually on Bybit.")
            continue

        client = get_client(api_key=wallet["api_key"], api_secret=wallet["api_secret"], demo=wallet["demo"])
        position = state["position"]
        balance = state["balance"]

        try:
            closed_trade, new_balance = _close_live_position(
                client, symbol, position, datetime.utcnow(), "manual_close", balance, config
            )
        except Exception as e:
            print(f"{exchange}/{symbol}: FAILED to close on Bybit -- {e}. Left open, check manually.")
            continue

        conn = get_db_connection()
        try:
            close_execution_trade(conn, exchange, symbol, strategy_name, closed_trade)
            cumulative_pnl = round(new_balance - config["initial_balance"], 4)
            save_execution_state(
                conn, exchange, symbol, strategy_name, state.get("time_horizon"),
                state["last_processed"], round(new_balance, 4), None, cumulative_pnl,
            )
        finally:
            conn.close()

        print(
            f"{exchange}/{symbol}: closed at {closed_trade['exit_price']}, "
            f"net PnL {closed_trade['net_pnl']:.4f}, new balance {new_balance:.2f}"
        )

    print("\nDone. Run check_state.py to confirm everything is flat.")


if __name__ == "__main__":
    main()