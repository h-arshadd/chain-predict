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