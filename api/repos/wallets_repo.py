"""
repos/wallets_repo.py
----------------------
DB access for the Wallets module. Builds on top of accounts.api_keys
(defined in crypto_pipeline.accounts.accounts_utils) rather than
duplicating it -- save_account_api_key / get_account_api_key are reused
as-is for create/read.

Two things accounts_utils.py doesn't have yet, added here:
  - an `enabled` column on accounts.api_keys (blocks new executions from
    being placed against a disabled wallet -- the execution module will
    check this when we build it)
  - delete_account() -- removing a wallet wasn't previously supported

Also: list_assignable_coins()/assign_strategy()/unassign_strategy() --
the "one strategy per coin, on this wallet" picker for the Wallets page,
built on top of execution.config.account_name (db_utils.set_execution_account)
and metadata.strategy.execution_enabled (strategies_repo.set_execution_enabled),
both of which already existed for the Execution/Deployment page -- this
just exposes them together as a single per-wallet action.

_ensure_enabled_column() runs an idempotent ALTER TABLE the same way the
rest of this codebase self-heals schema (see accounts_utils.py's own
CREATE TABLE IF NOT EXISTS pattern) -- safe to call on every request,
no-ops after the first time.
"""

from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from crypto_pipeline.accounts.accounts_utils import (
    save_account_api_key,
    get_account_api_key,
)
from crypto_pipeline.utils.metadata_utils import get_strategies
from crypto_pipeline.utils.db_utils import (
    get_execution_universe,
    get_execution_config,
    set_execution_account,
)
from api.repos.strategies_repo import set_execution_enabled


def _ensure_schema(conn):
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.api_keys (
            id            SERIAL PRIMARY KEY,
            account_name  TEXT NOT NULL UNIQUE,
            exchange      TEXT NOT NULL,
            api_key       TEXT NOT NULL,
            api_secret    TEXT NOT NULL,
            demo          BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at    TIMESTAMP NOT NULL DEFAULT now()
        )
    """))
    cursor.execute(sql.SQL("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'accounts' AND table_name = 'api_keys' AND column_name = 'enabled'
            ) THEN
                ALTER TABLE accounts.api_keys ADD COLUMN enabled BOOLEAN NOT NULL DEFAULT TRUE;
            END IF;
        END $$;
    """))
    conn.commit()
    cursor.close()


def list_wallets(conn) -> list[dict]:
    """All wallets with masked keys, no secrets. enabled included."""
    _ensure_schema(conn)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT account_name, exchange, demo, enabled, api_key, updated_at
        FROM accounts.api_keys
        ORDER BY account_name
    """))
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    return rows


def get_wallet(conn, account_name: str) -> dict | None:
    _ensure_schema(conn)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT account_name, exchange, demo, enabled, api_key, api_secret, updated_at
        FROM accounts.api_keys WHERE account_name = %s
    """), (account_name,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def create_wallet(conn, account_name: str, exchange: str, api_key: str, api_secret: str, demo: bool) -> dict:
    _ensure_schema(conn)
    # Reuse the pipeline's own upsert rather than reimplementing the INSERT.
    save_account_api_key(conn, account_name, exchange, api_key, api_secret, demo)
    return get_wallet(conn, account_name)


def update_wallet(conn, account_name: str, exchange: str | None, api_key: str | None,
                   api_secret: str | None, demo: bool | None) -> dict | None:
    """Partial update. None fields keep their current DB value."""
    _ensure_schema(conn)
    current = get_wallet(conn, account_name)
    if current is None:
        return None

    save_account_api_key(
        conn,
        account_name,
        exchange if exchange is not None else current["exchange"],
        api_key if api_key else current["api_key"],
        api_secret if api_secret else current["api_secret"],
        demo if demo is not None else current["demo"],
    )
    return get_wallet(conn, account_name)


def delete_wallet(conn, account_name: str) -> bool:
    _ensure_schema(conn)
    cursor = conn.cursor()
    cursor.execute(sql.SQL("DELETE FROM accounts.api_keys WHERE account_name = %s"), (account_name,))
    deleted = cursor.rowcount > 0
    conn.commit()
    cursor.close()
    return deleted


def set_enabled(conn, account_name: str, enabled: bool) -> dict | None:
    _ensure_schema(conn)
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        UPDATE accounts.api_keys SET enabled = %s, updated_at = now()
        WHERE account_name = %s
    """), (enabled, account_name))
    updated = cursor.rowcount > 0
    conn.commit()
    cursor.close()
    if not updated:
        return None
    return get_wallet(conn, account_name)


# ----------------------------------------------------------------------
# Per-wallet strategy assignment -- "one strategy per coin" picker shown
# when a wallet row on the Wallets page is expanded/managed.
#
# execution.config already has ONE account_name per (exchange, symbol)
# pair, and metadata.strategy already allows at most one execution_enabled
# strategy per pair (enforced by strategies_repo.set_execution_enabled).
# This just exposes both together as a single per-wallet action: picking
# a strategy for a coin (a) makes that strategy the execution_enabled one
# for the pair (disabling any sibling, same exclusivity rule the
# Execution page already enforces) and (b) points that pair's
# execution.config row at this wallet -- so execution/main.py's next run
# picks up both changes together.
# ----------------------------------------------------------------------

def list_assignable_coins(conn, account_name: str) -> list[dict]:
    """
    Every (exchange, symbol) pair that has an execution.config row (the
    same universe the Execution/Deployment page already trades from --
    see get_execution_universe), each with the strategies available for
    that pair and which one (if any) is currently live for it.

    Returns a list of:
        {
            "exchange", "symbol",
            "assigned_account": current execution.config.account_name for
                this pair (None if unassigned) -- lets the frontend show
                "currently assigned to <other wallet>" instead of
                silently stealing it,
            "strategies": [
                {"strategy_id", "strategy_name", "time_horizon", "execution_enabled"}
            ],
        }
    A pair with zero strategy rows in metadata.strategy still shows (with
    an empty strategies list) rather than being dropped, so the user can
    see it exists and needs strategies loaded first.
    """
    results = []
    for exchange, symbol in get_execution_universe(conn):
        config = get_execution_config(conn, exchange, symbol)
        strategy_rows = get_strategies(conn, exchange=exchange, coin=symbol)
        results.append({
            "exchange": exchange,
            "symbol": symbol,
            "assigned_account": (config or {}).get("account_name"),
            "strategies": [
                {
                    "strategy_id": s["strategy_id"],
                    "strategy_name": s["strategy_name"],
                    "time_horizon": s.get("time_horizon"),
                    "execution_enabled": s.get("execution_enabled", True),
                }
                for s in strategy_rows
            ],
        })
    return results


def assign_strategy(conn, account_name: str, exchange: str, symbol: str, strategy_id: int) -> dict:
    """
    Make `strategy_id` the live execution strategy for (exchange, symbol)
    and point that pair's execution.config at this wallet -- the two
    halves of "enable this strategy for this coin, on this wallet".

    Reuses strategies_repo.set_execution_enabled() for the exclusivity
    rule (turns off any other strategy already live for this pair) rather
    than re-implementing it here.

    Raises ValueError if the pair has no execution.config row yet (see
    set_execution_account) or the strategy doesn't exist.
    """
    strategy_row = set_execution_enabled(conn, strategy_id, True)
    if strategy_row is None:
        raise ValueError(f"Strategy {strategy_id} not found")
    if strategy_row["exchange"] != exchange or strategy_row["coin"] != symbol:
        raise ValueError(
            f"Strategy {strategy_id} belongs to {strategy_row['exchange']}/{strategy_row['coin']}, "
            f"not {exchange}/{symbol}"
        )

    assigned = set_execution_account(conn, exchange, symbol, account_name)
    if not assigned:
        raise ValueError(f"No execution.config row for {exchange}/{symbol} -- set it up before assigning a wallet")

    return {"exchange": exchange, "symbol": symbol, "strategy_id": strategy_id, "account_name": account_name}


def unassign_strategy(conn, exchange: str, symbol: str) -> None:
    """
    Turn execution off for this pair entirely -- disable whichever
    strategy is currently execution_enabled for (exchange, symbol) and
    clear execution.config's account_name, so execution/main.py skips
    this pair on its next run (mirrors its own "no wallet assigned --
    skipping" / "no execution_enabled strategy -- skipping" checks).
    """
    for row in get_strategies(conn, exchange=exchange, coin=symbol):
        if row.get("execution_enabled", True):
            set_execution_enabled(conn, row["strategy_id"], False)
    set_execution_account(conn, exchange, symbol, None)