"""
accounts_utils.py
------------------
Schema: accounts -- overall, per-trading-account view that sits ABOVE
execution's per-(exchange, symbol, strategy) tables. Execution tracks
things per strategy (execution.positions, execution.{...}_trades,
execution.stats); accounts rolls that up to "how is this whole Bybit
account doing, across every strategy/symbol it's ever run".

CHANGED FROM BEFORE: accounts.history and accounts.stats are no longer
rebuilt by re-reading execution.*_trades. They're rebuilt by calling
Bybit's live API directly, per account, for whatever (exchange, symbol)
pairs that account trades -- so accounts.* reflects Bybit's own record
of what happened, with as many raw Bybit fields kept as possible,
independent of whatever execution's own ledger happened to store.

Three tables:
    accounts.api_keys -- one row per account: exchange, label, demo flag,
                         and the CURRENT live api_key/api_secret. Updatable
                         in place when a key is rotated.
    accounts.history  -- every fill Bybit has on record for this account,
                         across every (exchange, symbol) pair it trades,
                         pulled LIVE from Bybit's execution list
                         (get_executions) each refresh. One row per fill,
                         tagged with account_name/exchange/symbol, with
                         Bybit's own fields kept close to as-is (order id,
                         exec id, price, qty, fee, side, exec type, etc).
    accounts.stats    -- ONE ROW PER ACCOUNT (true overall, not per
                         symbol/strategy): the 85-stat block from
                         ledger_stats.get_ledger_stats(), computed over
                         this account's ENTIRE accounts.history pooled
                         together across every (exchange, symbol) pair it
                         trades (FIFO-derived realized PnL, win rate,
                         profit factor, streaks, drawdown, fees, volume,
                         holding time, time-of-day/day-of-week
                         breakdowns, and a per_symbol sub-dict for
                         coin-level detail within that one row).

    NOTE: accounts.history is still tagged with exchange/symbol per fill
    (Bybit has no concept of "which strategy" placed an order, so
    strategy_name never appears in accounts.history at all). accounts.stats
    pools everything for the account together into one row -- if you need
    a coin-level view, use that row's per_symbol sub-dict rather than
    looking for separate rows per symbol/strategy.

Why get_executions (fill-level) instead of get_closed_pnl (closed-trade
level): get_closed_pnl is documented by Bybit as NOT supported on demo
trading accounts (only production/pre-upgrade) -- and this pipeline runs
on demo by default (see setup_accounts_example.py's DEMO flag). get_executions
works on both demo and production, so it's what actually runs regardless
of which mode the account is in. The tradeoff: it's fill-level, not
closed-trade level, so a position closed in three partial fills is three
rows here, not one -- net_pnl per row isn't meaningful the way execution's
own ledger's net_pnl is (that still comes from execution.*_trades, which
this file no longer touches). accounts.history is Bybit's raw record of
what happened, not a re-derivation of execution's own trade-level P&L.

SECURITY NOTE: api_key/api_secret are real trading credentials. Nothing
in this file hardcodes any key -- save_account_api_key() takes them as
plain parameters. Call it yourself with the real value; never commit a
real key into a source file.
"""

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from crypto_pipeline.utils.db_utils import _pg_type_for, _copy_dataframe
from crypto_pipeline.execution.bybit_client import get_client, to_bybit_symbol, utcnow_from_ms


# ==========================================================
# accounts.api_keys
# ==========================================================

def save_account_api_key(conn, account_name, exchange, api_key, api_secret, demo=True):
    """
    Create or update one account's row. Matched on account_name -- calling
    this again with the same account_name (e.g. after rotating a key on
    Bybit) UPDATES the existing row in place, so accounts.history/stats
    (which reference account_name, not the key itself) stay attached to
    the same account across a key rotation.

    account_name : your own label for this account, e.g. "bybit_demo_1" --
        used everywhere else in this module to refer to this account.
    exchange     : e.g. "bybit".
    api_key, api_secret : the real credentials. Pass them in directly when
        you call this -- never hardcode a real key in a source file.
    demo         : True for Bybit Demo Trading, False for production --
        same meaning as BYBIT_DEMO in .env / bybit_client.get_client().
    """
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
    conn.commit()

    cursor.execute(sql.SQL("""
        INSERT INTO accounts.api_keys (account_name, exchange, api_key, api_secret, demo, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (account_name) DO UPDATE SET
            exchange   = EXCLUDED.exchange,
            api_key    = EXCLUDED.api_key,
            api_secret = EXCLUDED.api_secret,
            demo       = EXCLUDED.demo,
            updated_at = now()
    """), (account_name, exchange, api_key, api_secret, demo))
    conn.commit()
    cursor.close()


def get_account_api_key(conn, account_name):
    """
    Return one account's current credentials, or None if account_name
    doesn't exist. Returns dict: account_name, exchange, api_key,
    api_secret, demo, enabled, updated_at.

    `enabled` is also self-healed here (not just in the API layer's
    wallets_repo._ensure_schema()) since execution/main.py reads it
    through this same function to decide whether a wallet is allowed to
    open new positions -- one source of truth for the column, safe to
    call from either place.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

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
    conn.commit()

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

    cursor.execute(sql.SQL("""
        SELECT account_name, exchange, api_key, api_secret, demo, enabled, updated_at
        FROM accounts.api_keys WHERE account_name = %s
    """), (account_name,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def list_accounts(conn):
    """Return every account_name currently registered (no secrets)."""
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

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
    conn.commit()

    cursor.execute(sql.SQL("SELECT account_name, exchange, demo FROM accounts.api_keys"))
    rows = cursor.fetchall()
    cursor.close()
    return [{"account_name": r[0], "exchange": r[1], "demo": r[2]} for r in rows]


def _get_bybit_client(conn, account_name):
    """Build a pybit HTTP session from this account's own stored credentials."""
    account = get_account_api_key(conn, account_name)
    if account is None:
        raise ValueError(f"Account {account_name!r} is not registered in accounts.api_keys.")
    return get_client(api_key=account["api_key"], api_secret=account["api_secret"], demo=account["demo"])


# ==========================================================
# accounts.history -- live from Bybit's execution list
# ==========================================================

def _fetch_symbol_executions(client, bybit_symbol, limit=200, max_pages=50):
    """
    Pull every fill Bybit has on record for one symbol via get_executions,
    paging back through nextPageCursor until exhausted (or max_pages hit,
    as a safety cap against an unbounded loop if Bybit ever returns a
    cursor that doesn't terminate).

    Returns a list of raw Bybit execution dicts (unmodified, still with
    Bybit's own field names/types as strings) -- caller normalizes them.
    """
    rows = []
    cursor = None
    for _ in range(max_pages):
        params = dict(category="linear", symbol=bybit_symbol, limit=limit)
        if cursor:
            params["cursor"] = cursor
        response = client.get_executions(**params)
        page = response["result"]["list"]
        rows.extend(page)
        cursor = response["result"].get("nextPageCursor")
        if not cursor or not page:
            break
    return rows


def _normalize_execution_row(account_name, exchange, symbol, row):
    """
    Bybit's raw execution dict -> one accounts.history row. Keeps as many
    of Bybit's own fields as are useful, renamed to plain snake_case, cast
    to numeric/datetime where it's a number/timestamp. Anything Bybit adds
    that isn't listed here just isn't kept -- add a line below to pick up
    a new field, no schema migration needed since the table columns are
    derived from whatever keys are actually present (see refresh_account_history).
    """

    def _f(key):
        val = row.get(key)
        return float(val) if val not in (None, "") else None

    return {
        "account_name": account_name,
        "exchange": exchange,
        "symbol": symbol,
        "order_id": row.get("orderId"),
        "exec_id": row.get("execId"),
        "side": row.get("side"),
        "order_type": row.get("orderType"),
        "exec_type": row.get("execType"),
        "exec_price": _f("execPrice"),
        "exec_qty": _f("execQty"),
        "exec_value": _f("execValue"),
        "exec_fee": _f("execFee"),
        "fee_rate": _f("feeRate"),
        "closed_size": _f("closedSize"),
        "leverage": _f("leverage"),
        "is_maker": row.get("isMaker"),
        "mark_price": _f("markPrice"),
        "index_price": _f("indexPrice"),
        "mark_iv": _f("markIv"),
        "trade_time": utcnow_from_ms(int(row["execTime"])) if row.get("execTime") else None,
    }


def refresh_account_history(conn, account_name, strategy_combos):
    """
    Rebuild accounts.history for one account from scratch, by pulling
    every fill Bybit has on record LIVE, for every (exchange, symbol)
    pair in strategy_combos (strategy_name is accepted for signature
    compatibility with existing callers but isn't needed here -- Bybit's
    executions aren't tagged by strategy, only by symbol).

    Full rebuild (not append) each call -- always reflects exactly what
    Bybit currently reports for this account, nothing stale left behind
    from a pair that was later removed.

    strategy_combos : list of (exchange, symbol, strategy_name) tuples,
        e.g. from get_execution_universe() + each pair's strategy_name
        (strategy_name is ignored here, kept only for call-site parity
        with refresh_account_stats/setup_accounts_example.py).
    """
    client = _get_bybit_client(conn, account_name)

    seen_symbols = set()
    frames = []
    for exchange, symbol, _strategy_name in strategy_combos:
        if (exchange, symbol) in seen_symbols:
            continue
        seen_symbols.add((exchange, symbol))

        bybit_symbol = to_bybit_symbol(symbol)
        raw_rows = _fetch_symbol_executions(client, bybit_symbol)
        if not raw_rows:
            continue

        normalized = [_normalize_execution_row(account_name, exchange, symbol, r) for r in raw_rows]
        frames.append(pd.DataFrame(normalized))

    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.history (
            account_name TEXT,
            exchange     TEXT,
            symbol       TEXT,
            order_id     TEXT,
            exec_id      TEXT,
            side         TEXT,
            order_type   TEXT,
            exec_type    TEXT,
            exec_price   DOUBLE PRECISION,
            exec_qty     DOUBLE PRECISION,
            exec_value   DOUBLE PRECISION,
            exec_fee     DOUBLE PRECISION,
            fee_rate     DOUBLE PRECISION,
            closed_size  DOUBLE PRECISION,
            leverage     DOUBLE PRECISION,
            is_maker     BOOLEAN,
            mark_price   DOUBLE PRECISION,
            index_price  DOUBLE PRECISION,
            mark_iv      DOUBLE PRECISION,
            trade_time   TIMESTAMP
        )
    """))
    conn.commit()

    cursor.execute(sql.SQL("DELETE FROM accounts.history WHERE account_name = %s"), (account_name,))
    conn.commit()
    cursor.close()

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)

    fixed_columns = [
        "account_name", "exchange", "symbol", "order_id", "exec_id", "side",
        "order_type", "exec_type", "exec_price", "exec_qty", "exec_value",
        "exec_fee", "fee_rate", "closed_size", "leverage", "is_maker",
        "mark_price", "index_price", "mark_iv", "trade_time",
    ]
    for col in fixed_columns:
        if col not in combined.columns:
            combined[col] = None
    combined = combined[fixed_columns]

    _copy_dataframe(conn, combined, "accounts", "history")


def get_account_history(conn, account_name):
    """Return this account's full fill history as a DataFrame, oldest first."""
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()
    cursor.close()

    return pd.read_sql(
        sql.SQL(
            "SELECT * FROM accounts.history WHERE account_name = %s ORDER BY trade_time ASC"
        ).as_string(conn),
        conn,
        params=(account_name,)
    )


# ==========================================================
# accounts.stats -- ledger_stats (85-stat block) only
# ==========================================================


def refresh_account_stats(conn, account_name):
    """
    Recompute accounts.stats for one account: ONE ROW PER ACCOUNT (true
    overall, not per combo/symbol) -- the 85-stat block from
    ledger_stats.get_ledger_stats(), computed over this account's ENTIRE
    accounts.history pooled together across every (exchange, symbol) pair
    it trades. accounts.history must be refreshed first via
    refresh_account_history() in the same run.

    Per-coin detail isn't lost: ledger_stats' own per_symbol sub-dict
    (one of the 85 stats) already breaks trade_count/total_qty/
    total_value/total_fees/avg_price/realized_pnl/win_rate_pct down by
    symbol, computed from this same pooled history -- so a per-coin view
    is still available inside this one row, without a separate row per
    combo.

    Full rewrite each call (old row for this account deleted first). If
    accounts.history has no rows for this account, a single row of
    all-zero/empty stats is still written (so accounts.stats always has
    exactly one row per registered account).
    """
    from crypto_pipeline.accounts.ledger_stats import get_ledger_stats

    history = get_account_history(conn, account_name)

    if history.empty:
        stats_row = {"row_count": 0}
    else:
        stats_row = get_ledger_stats(history.sort_values("trade_time"))

    # ledger_stats returns a handful of dict/list-valued fields
    # (per_symbol, symbols_traded, trades_by_hour_of_day,
    # trades_by_day_of_week, trades_by_date) that _pg_type_for can't
    # infer a scalar column type for -- store those as JSON text so the
    # rest of the row still writes as plain scalar columns.
    import json as _json
    for _col in ("per_symbol", "symbols_traded", "trades_by_hour_of_day",
                 "trades_by_day_of_week", "trades_by_date"):
        if _col in stats_row and not isinstance(stats_row[_col], (str, type(None))):
            stats_row[_col] = _json.dumps(stats_row[_col], default=str)

    def _pg_type_for_stats_col(col, val):
        if col in ("first_trade_time", "last_trade_time"):
            return "TIMESTAMP"
        return _pg_type_for(pd.Series([val]))

    all_columns = list(stats_row.keys())
    sample_values = {col: stats_row[col] for col in all_columns if stats_row[col] is not None}

    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for_stats_col(col, sample_values.get(col)))
        )
        for col in all_columns
    )
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.stats (
            account_name TEXT NOT NULL,
            updated_at   TIMESTAMP NOT NULL DEFAULT now(),
            {column_defs},
            PRIMARY KEY (account_name)
        )
    """).format(column_defs=column_defs))
    conn.commit()

    # Self-heal: add any metric columns that didn't exist yet (e.g. a new
    # ledger_stats field added later).
    for col in all_columns:
        cursor.execute(sql.SQL("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'accounts' AND table_name = 'stats' AND column_name = %s
                ) THEN
                    ALTER TABLE accounts.stats ADD COLUMN {col} {pg_type};
                END IF;
            END $$;
        """).format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for_stats_col(col, sample_values.get(col)))
        ), (col,))
    conn.commit()

    columns = ["account_name"] + all_columns
    update_clause = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
        for col in all_columns
    )
    values = [account_name] + [stats_row.get(col) for col in all_columns]

    cursor.execute(sql.SQL("""
        INSERT INTO accounts.stats ({columns}, updated_at)
        VALUES ({placeholders}, now())
        ON CONFLICT (account_name) DO UPDATE SET
            {update_clause},
            updated_at = now()
    """).format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        update_clause=update_clause
    ), values)

    conn.commit()
    cursor.close()


def get_account_stats(conn, account_name):
    """Return this account's single overall stats row as a dict, or None if not yet computed."""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()
    cursor.execute(sql.SQL(
        "SELECT to_regclass('accounts.stats')"
    ))
    # RealDictCursor returns dicts, not tuples -- fetchone()[0] would
    # KeyError here since dict keys are column names, not ints. Index by
    # the actual column name (Postgres names it after the function call).
    if cursor.fetchone()["to_regclass"] is None:
        cursor.close()
        return None

    cursor.execute(sql.SQL("SELECT * FROM accounts.stats WHERE account_name = %s"), (account_name,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None