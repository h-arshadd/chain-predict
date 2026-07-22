"""
accounts_utils.py
------------------
Schema: accounts -- overall, per-trading-account view that sits ABOVE
execution's per-(exchange, symbol, strategy) tables. Execution tracks
things per strategy (execution.positions, execution.{...}_trades,
execution.stats); accounts rolls that up to "how is this whole Bybit
account doing, across every strategy/symbol it's ever run" -- and
persists that identity across an API key ever being rotated.

Three tables:
    accounts.api_keys -- one row per account: exchange, label, demo flag,
                         and the CURRENT live api_key/api_secret. Updatable
                         in place when a key is rotated -- the account's
                         history/stats stay attached to the account, not
                         to whichever key happened to be active at the
                         time.
    accounts.history  -- every closed trade from every
                         execution.{exchange}_{symbol}_{strategy}_trades
                         table, pulled into one combined table, each row
                         tagged with account_name/exchange/symbol/
                         strategy_name so you can see (or filter) the
                         full trade history in one place.
    accounts.stats    -- one row per account: initial_balance, the
                         combos this account trades (JSONB list of each
                         (exchange, symbol) pair's full execution.config
                         -- strategy_name, initial_balance, position_size,
                         commission, slippage), plus aggregate performance
                         metrics (quantstats, same shape as
                         execution.stats) computed over the account's
                         ENTIRE combined history. A row is written as
                         soon as the account is registered/refreshed --
                         initial_balance/combos are there immediately,
                         trade-derived metrics start at 0 and only
                         reflect real numbers once accounts.history has
                         closed trades. Refreshed at the end of an
                         execution run alongside accounts.history -- not
                         recomputed from scratch each time, just
                         re-derived from whatever's currently in
                         accounts.history.

Nothing here duplicates execution's own per-strategy tables -- those
stay the source of truth. accounts.history/stats are a rollup built FROM
them, refreshed by re-reading them, never written to independently.

SECURITY NOTE: api_key/api_secret are real trading credentials. Nothing
in this file hardcodes any key -- save_account_api_key() takes them as
plain parameters, the same way bybit_client.get_client_from_env() reads
them from .env today. Call it yourself with the real value; never commit
a real key into a source file.
"""

import re

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

from crypto_pipeline.utils.db_utils import _pg_type_for, _copy_dataframe


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
    api_secret, demo, updated_at.
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
        SELECT account_name, exchange, api_key, api_secret, demo, updated_at
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


# ==========================================================
# accounts.history
# ==========================================================

def refresh_account_history(conn, account_name, strategy_combos):
    """
    Rebuild accounts.history for one account from scratch, by re-reading
    every execution.{exchange}_{symbol}_{strategy_name}_trades table
    listed in strategy_combos. Full rebuild (not append) -- cheap, since
    it's just re-reading tables that already exist, and guarantees
    history never drifts out of sync with the underlying per-strategy
    ledgers (e.g. if a strategy is ever removed from execution.config,
    its trades still show correctly here without a manual cleanup step).

    strategy_combos : list of (exchange, symbol, strategy_name) tuples --
        pass every combo this account trades, e.g. from
        execution.get_execution_universe() + each pair's strategy_name.

    Each row = one closed trade, same columns as an execution trade
    ledger row (direction, entry_date_time, exit_date_time, entry_price,
    exit_price, quantity, gross_pnl, commission, slippage, net_pnl,
    exit_reason, balance, cumulative_pnl, trade_id), plus account_name/
    exchange/symbol/strategy_name so trades from different strategies
    can sit in one table without colliding.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

    frames = []
    for exchange, symbol, strategy_name in strategy_combos:
        safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
        trades_table = f"{exchange}_{symbol}_{safe_strategy_name}_trades"

        qualified_name = sql.SQL(".").join(
            [sql.Identifier("execution"), sql.Identifier(trades_table)]
        ).as_string(conn)
        cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
        if cursor.fetchone()[0] is None:
            continue  # this strategy hasn't closed any trades yet

        df = pd.read_sql(
            sql.SQL("SELECT * FROM {schema}.{table}").format(
                schema=sql.Identifier("execution"),
                table=sql.Identifier(trades_table)
            ).as_string(conn),
            conn
        )
        if df.empty:
            continue

        df.insert(0, "account_name", account_name)
        df.insert(1, "exchange", exchange)
        df.insert(2, "symbol", symbol)
        df.insert(3, "strategy_name", strategy_name)
        frames.append(df)

    # Clear out this account's old rows before reinserting -- full
    # rebuild, not append, so a trade that somehow changed or a strategy
    # that got removed never leaves a stale row behind.
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.history (
            account_name    TEXT,
            exchange        TEXT,
            symbol          TEXT,
            strategy_name   TEXT,
            trade_id        INTEGER,
            direction       TEXT,
            entry_date_time TIMESTAMP,
            exit_date_time  TIMESTAMP,
            entry_price     DOUBLE PRECISION,
            exit_price      DOUBLE PRECISION,
            quantity        DOUBLE PRECISION,
            gross_pnl       DOUBLE PRECISION,
            commission      DOUBLE PRECISION,
            slippage        DOUBLE PRECISION,
            net_pnl         DOUBLE PRECISION,
            exit_reason     TEXT,
            balance         DOUBLE PRECISION,
            cumulative_pnl  DOUBLE PRECISION
        )
    """))
    conn.commit()

    cursor.execute(sql.SQL("DELETE FROM accounts.history WHERE account_name = %s"), (account_name,))
    conn.commit()
    cursor.close()

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # accounts.history's own fixed column list (declared above) is the
    # contract -- keep only those columns, in that order, so differing
    # per-strategy ledger columns (or column order) never break the
    # table shape. Any column not present in a given trade row (rare,
    # only if an old ledger predates a schema change) comes through as
    # NaN/None.
    fixed_columns = [
        "account_name", "exchange", "symbol", "strategy_name", "trade_id",
        "direction", "entry_date_time", "exit_date_time", "entry_price",
        "exit_price", "quantity", "gross_pnl", "commission", "slippage",
        "net_pnl", "exit_reason", "balance", "cumulative_pnl",
    ]
    for col in fixed_columns:
        if col not in combined.columns:
            combined[col] = None
    combined = combined[fixed_columns]

    _copy_dataframe(conn, combined, "accounts", "history")


def get_account_history(conn, account_name):
    """Return this account's full trade history as a DataFrame, oldest first."""
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()
    cursor.close()

    return pd.read_sql(
        sql.SQL(
            "SELECT * FROM accounts.history WHERE account_name = %s ORDER BY exit_date_time ASC"
        ).as_string(conn),
        conn,
        params=(account_name,)
    )


# ==========================================================
# accounts.stats
# ==========================================================

def refresh_account_stats(conn, account_name, initial_balance, stats_config, combos=None):
    """
    Recompute accounts.stats for one account from accounts.history (must
    be refreshed first via refresh_account_history() in the same run).
    Same win/loss + quantstats metrics shape as execution.stats, just
    aggregated across every strategy/symbol this account has ever traded
    instead of one strategy at a time.

    initial_balance : this account's starting balance (sum of whatever
        each contributing (exchange, symbol) pair's execution.config
        initial_balance was, or your own account-level starting number --
        caller decides, this function just needs one total to measure
        profit against).
    stats_config    : same dict compute_stats() takes everywhere else
        (loaded from stats/config.yaml).
    combos          : optional list of dicts, one per (exchange, symbol)
        pair this account trades -- each dict is that pair's full
        execution.config, e.g. {"exchange", "symbol", "strategy_name",
        "initial_balance", "position_size", "commission", "slippage",
        ...}. Stored as-is (JSONB) in the "combos" column so the
        account's makeup -- which pairs, at what size/cost -- is visible
        immediately after registration, before any trade has closed.
        Pass None/omit to leave the column untouched on refresh.

    Always writes a row, even with zero closed trades yet -- so the
    account's initial_balance/combos are visible right after
    registration instead of waiting for a first closed trade. With no
    trades, every trade-derived metric (win_rate, quantstats, etc.) is
    written as 0/zeroed rather than computed, and final_balance just
    equals initial_balance.
    """
    from crypto_pipeline.stats.calculator import compute_stats

    history = get_account_history(conn, account_name)

    if history.empty:
        # No closed trades yet -- write the pre-trade shell row (balance/
        # combo info only) instead of skipping, so the account shows up
        # in accounts.stats immediately after registration.
        stats_row = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_balance": float(initial_balance),
            "total_net_profit": 0.0,
        }
    else:
        total_trades = len(history)
        wins = int((history["net_pnl"] > 0).sum())
        losses = int((history["net_pnl"] <= 0).sum())
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        history = history.sort_values("exit_date_time")
        final_balance = float(history["balance"].iloc[-1])
        total_net_profit = final_balance - float(initial_balance)

        # Equity curve across the WHOLE account: every strategy's closed
        # trades interleaved by exit time, starting from initial_balance.
        # Note this sums each strategy's own "balance" column (that
        # strategy's own running total, not a shared pot) at the point it
        # closed -- fine for total_net_profit/win-loss (those only look at
        # net_pnl per trade, which is strategy-agnostic), but the equity
        # curve itself is only a true reflection of combined capital if each
        # strategy was funded from the same starting pot rather than its own
        # separate initial_balance. Good enough for an overall performance
        # trend; treat cumulative dollar levels on the curve as approximate
        # if strategies run with different initial_balances.
        equity_index = pd.to_datetime(history["exit_date_time"])
        equity_values = float(initial_balance) + history["net_pnl"].cumsum()
        equity = pd.Series(equity_values.values, index=equity_index)
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()

        stats_dict = compute_stats(
            {"equity_curve": equity, "total_trades": total_trades},
            stats_config,
        )
        stats_row = dict(stats_dict["metrics"])
        stats_row["total_trades"] = total_trades
        stats_row["wins"] = wins
        stats_row["losses"] = losses
        stats_row["win_rate"] = win_rate
        stats_row["final_balance"] = final_balance
        stats_row["total_net_profit"] = total_net_profit

    stats_row["initial_balance"] = float(initial_balance)
    if combos is not None:
        stats_row["combos"] = Json(combos)

    def _pg_type_for_stats_col(col, val):
        # combos is a Json-wrapped list of dicts -- _pg_type_for() only
        # understands plain pandas dtypes, so route this one column to
        # JSONB directly instead of letting it fall through to TEXT.
        if col == "combos":
            return "JSONB"
        return _pg_type_for(pd.Series([val]))

    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for_stats_col(col, val))
        )
        for col, val in stats_row.items()
    )
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.stats (
            account_name TEXT PRIMARY KEY,
            updated_at   TIMESTAMP NOT NULL DEFAULT now(),
            {column_defs}
        )
    """).format(column_defs=column_defs))
    conn.commit()

    # Self-heal: add any metric columns that didn't exist yet (e.g. a new
    # quantstats metric that appeared after this table was first
    # created), same defensive pattern used elsewhere for state tables.
    for col, val in stats_row.items():
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
            pg_type=sql.SQL(_pg_type_for_stats_col(col, val))
        ), (col,))
    conn.commit()

    columns = ["account_name"] + list(stats_row.keys())
    values = [account_name] + list(stats_row.values())
    update_clause = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
        for col in stats_row.keys()
    )

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
    """Return this account's stats row as a dict, or None if never computed."""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()
    cursor.execute(sql.SQL(
        "SELECT to_regclass('accounts.stats')"
    ))
    if cursor.fetchone()[0] is None:
        cursor.close()
        return None

    cursor.execute(sql.SQL("SELECT * FROM accounts.stats WHERE account_name = %s"), (account_name,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None