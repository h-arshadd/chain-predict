"""
run_accounts.py
----------------
Single script for everything accounts-related. Run it directly:

    python -m crypto_pipeline.accounts.run_accounts

Each run does two things:

  1. Registers/updates the account in accounts.api_keys (only inserts
     the key the FIRST time -- if the account already exists, this does
     NOT touch its stored key, so re-running this script never
     overwrites a key with the placeholder below). Edit ACCOUNT_NAME /
     API_KEY / API_SECRET / DEMO below before your first run, then you
     can leave them as-is afterward.

  2. Refreshes accounts.history and accounts.stats from EVERY
     (exchange, symbol) pair currently in execution.config -- rebuilds
     accounts.history by pulling this account's fill history LIVE from
     Bybit (get_executions) for each symbol, then recomputes
     accounts.stats (trade facts + a live wallet snapshot + quantstats)
     from that combined history. Safe to run repeatedly (a full rebuild
     each time, not an append), and safe to run whether or not Bybit has
     any fills yet for a given symbol (it just does nothing for combos
     with no fills).

Run this after execution/main.py (or on its own schedule) so
accounts.history/accounts.stats stay current -- it does not place any
orders. It reads execution.config (to know which symbols this account
trades) but pulls the actual trade/fill data live from Bybit itself,
not from execution's own stored ledger.

SECURITY: put your REAL Bybit API key/secret below only on your own
machine, in your own copy of this file -- never commit a real key, and
never paste one into a chat/ticket/PR. If a key is ever exposed that
way, revoke it on Bybit and generate a new one before using it further.
"""

from crypto_pipeline.utils.db_utils import get_db_connection, get_execution_config, get_execution_universe
from crypto_pipeline.utils.metadata_utils import (
    get_db_connection as get_metadata_connection,
    get_strategies,
)
from crypto_pipeline.utils.accounts_utils import (
    get_account_api_key,
    save_account_api_key,
    save_account_combos,
    refresh_account_history,
    refresh_account_stats,
)

# ---- EDIT THESE before your first run ----
ACCOUNT_NAME = "bybit_demo_1"     # your own label for this account -- pick anything
EXCHANGE = "bybit"
API_KEY = "dqImKT6rBvUGeKSbBl"
API_SECRET = "b28lVBRZhHl7QZjHIbzghgj8qz9yyavxwFLM"
DEMO = True                        # True = Bybit Demo Trading, False = production
# -------------------------------------------


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
                                -- what save_account_combos() writes as
                                plain rows to accounts.combos so the
                                account's makeup is visible before any
                                trade closes.
        total_initial_balance : combined initial_balance across every
                                combo, used as the account's overall
                                starting point for total_net_profit.
    """
    combos = []
    combo_configs = []
    total_initial_balance = 0.0

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
            total_initial_balance += float(config["initial_balance"])
    finally:
        metadata_conn.close()

    return combos, combo_configs, total_initial_balance


def _load_stats_config():
    import yaml
    import crypto_pipeline
    from pathlib import Path
    stats_config_path = Path(crypto_pipeline.__file__).parent / "stats" / "config.yaml"
    with open(stats_config_path, "r") as f:
        return yaml.safe_load(f)


def main():
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
        # currently tracks, plus their combined initial_balance.
        combos, combo_configs, total_initial_balance = _get_strategy_combos(conn)

        if not combos:
            print("No (exchange, symbol) pairs found in execution.config -- nothing to refresh.")
            return

        print(f"Refreshing history/stats for {len(combos)} combo(s): {combos}")

        # Step 3: rewrite accounts.combos -- plain rows, one per
        # (exchange, symbol, strategy) pair this account trades.
        save_account_combos(conn, ACCOUNT_NAME, combo_configs)

        # Step 4: rebuild accounts.history from every combo's
        # execution.*_trades table.
        refresh_account_history(conn, ACCOUNT_NAME, combos)

        # Step 5: recompute accounts.stats from that refreshed history.
        stats_config = _load_stats_config()
        refresh_account_stats(conn, ACCOUNT_NAME, total_initial_balance, stats_config)

        print(f"accounts.history and accounts.stats refreshed for {ACCOUNT_NAME!r}.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()