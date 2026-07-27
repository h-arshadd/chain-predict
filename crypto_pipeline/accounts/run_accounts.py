"""
run_accounts.py
----------------
Single script for everything accounts-related. Run it directly:

    python -m crypto_pipeline.accounts.run_accounts

Each run does two things:

  1. Registers/updates the account in accounts.api_keys (only inserts
     the key the FIRST time -- if the account already exists, this does
     NOT touch its stored key, so re-running this script never
     overwrites a key with whatever is currently in your .env). Set the
     env vars described in the SECURITY note below before your first
     run; they're only read/needed for that first registration.

  2. Refreshes accounts.history and accounts.stats from EVERY
     (exchange, symbol, strategy) combo currently in execution.config --
     rebuilds accounts.history by pulling this account's fill history
     LIVE from Bybit (get_executions) for each symbol, then recomputes
     accounts.stats -- ONE ROW FOR THE WHOLE ACCOUNT (the 85-stat
     ledger_stats block, computed over all of that account's history
     pooled together, not split per symbol/strategy). Safe to run
     repeatedly (a full rebuild each time, not an append), and safe to
     run even if Bybit has no fills yet for any symbol (writes a
     zero-trade row in that case).

Run this after execution/main.py (or on its own schedule) so
accounts.history/accounts.stats stay current -- it does not place any
orders. It reads execution.config (to know which symbols this account
trades) but pulls the actual trade/fill data live from Bybit itself,
not from execution's own stored ledger.

SECURITY: credentials are read from environment variables (via a local
.env file, same convention as db_utils.py/bybit_client.py -- see
python-dotenv's load_dotenv() below), never hardcoded in this file.
Add these to your own .env (never commit .env -- it's already
gitignored):

    ACCOUNTS_ACCOUNT_NAME=bybit_demo_1
    ACCOUNTS_BYBIT_API_KEY=your_key_here
    ACCOUNTS_BYBIT_API_SECRET=your_secret_here
    ACCOUNTS_BYBIT_DEMO=true

If a key is ever exposed (e.g. committed to git before this fix), revoke
it on Bybit and generate a new one before using it further -- editing
this file to remove it afterward does not undo an already-public commit.
"""

import os
from dotenv import load_dotenv

from crypto_pipeline.utils.db_utils import get_db_connection, get_execution_config, get_execution_universe
from crypto_pipeline.utils.metadata_utils import (
    get_db_connection as get_metadata_connection,
    get_strategies,
)
from crypto_pipeline.accounts.accounts_utils import (
    get_account_api_key,
    save_account_api_key,
    refresh_account_history,
    refresh_account_stats,
)

load_dotenv()

# ---- Set these in your .env, not here (see SECURITY note above) ----
ACCOUNT_NAME = os.getenv("ACCOUNTS_ACCOUNT_NAME", "bybit_demo_1")
EXCHANGE = "bybit"
API_KEY = os.getenv("ACCOUNTS_BYBIT_API_KEY")
API_SECRET = os.getenv("ACCOUNTS_BYBIT_API_SECRET")
DEMO = os.getenv("ACCOUNTS_BYBIT_DEMO", "true").strip().lower() == "true"
# ----------------------------------------------------------------------


def _get_strategy_combos(conn):
    """
    Every (exchange, symbol, strategy_name) combo currently configured
    for execution -- one combo per execution.config row.

    strategy_name is NOT stored in execution.config (see
    db_utils.get_execution_config's docstring), so it's looked up the
    same way execution/main.py does: from metadata.strategy, taking
    whichever row for that (exchange, symbol) has execution_enabled=True.
    A pair is skipped if it has no execution_enabled strategy, or more
    than one (ambiguous) -- same guard execution/main.py uses.

    Returns:
        combos               : list of (exchange, symbol, strategy_name)
                                tuples -- what refresh_account_history()
                                needs to find each combo's trades table.
        combo_configs         : list of dicts, one per combo -- each
                                pair's full execution.config (exchange,
                                symbol, strategy_name, initial_balance,
                                position_size, commission, slippage, ...)
                                merged with exchange/symbol/strategy_name
                                -- what refresh_account_stats() writes as
                                one row per combo in accounts.stats.
    """
    combos = []
    combo_configs = []

    metadata_conn = get_metadata_connection()
    try:
        for exchange, symbol in get_execution_universe(conn):
            config = get_execution_config(conn, exchange, symbol)
            if config is None:
                continue

            strategy_rows = get_strategies(metadata_conn, exchange=exchange, coin=symbol)
            enabled_rows = [s for s in strategy_rows if s.get("execution_enabled", True)]

            if len(enabled_rows) == 0:
                print(f"{exchange} {symbol}: no execution_enabled strategy found in metadata.strategy -- skipping.")
                continue

            if len(enabled_rows) > 1:
                enabled_strategy_names = [s["strategy_name"] for s in enabled_rows]
                print(
                    f"{exchange} {symbol}: {len(enabled_rows)} strategies are "
                    f"execution_enabled ({enabled_strategy_names}) -- only one strategy is "
                    f"allowed per coin. Skipping."
                )
                continue

            strategy_name = enabled_rows[0]["strategy_name"]

            combos.append((exchange, symbol, strategy_name))
            combo_configs.append({"exchange": exchange, "symbol": symbol, "strategy_name": strategy_name, **config})
    finally:
        metadata_conn.close()

    return combos, combo_configs


def main():
    # Only actually needed on this account's very first run (see Step 1
    # below) -- but checked upfront so a missing .env fails clearly and
    # immediately, instead of registering an account with a NULL/empty
    # key that would only surface as a confusing Bybit auth error later.
    existing_check_conn = get_db_connection()
    try:
        already_registered = get_account_api_key(existing_check_conn, ACCOUNT_NAME) is not None
    finally:
        existing_check_conn.close()

    if not already_registered and (not API_KEY or not API_SECRET):
        raise RuntimeError(
            f"Account {ACCOUNT_NAME!r} isn't registered yet and ACCOUNTS_BYBIT_API_KEY/"
            f"ACCOUNTS_BYBIT_API_SECRET aren't set in your .env -- see the SECURITY note "
            f"at the top of this file for the required .env keys."
        )

    conn = get_db_connection()
    try:
        # Step 1: register the account, but only if it doesn't already
        # exist -- re-running this script should never clobber a
        # previously-saved key with whatever placeholder happens to be
        # sitting in the constants above.
        existing = get_account_api_key(conn, ACCOUNT_NAME)
        if existing is None:
            save_account_api_key(conn, ACCOUNT_NAME, EXCHANGE, API_KEY, API_SECRET, DEMO)
            print(f"Registered new account {ACCOUNT_NAME!r}.")
        else:
            print(f"Account {ACCOUNT_NAME!r} already registered (updated_at={existing['updated_at']}) -- not overwriting its stored key.")

        # Step 2: every (exchange, symbol, strategy) combo execution
        # currently tracks.
        combos, combo_configs = _get_strategy_combos(conn)

        if not combos:
            print("No (exchange, symbol) pairs found in execution.config -- nothing to refresh.")
            return

        print(f"Refreshing history/stats for {len(combos)} combo(s): {combos}")

        # Step 3: rebuild accounts.history from every combo's
        # execution.*_trades table.
        refresh_account_history(conn, ACCOUNT_NAME, combos)

        # Step 4: recompute accounts.stats -- one overall row for the account.
        refresh_account_stats(conn, ACCOUNT_NAME)

        print(f"accounts.history and accounts.stats refreshed for {ACCOUNT_NAME!r}.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()