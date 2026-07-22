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
    accounts.combos   -- one row per (exchange, symbol, strategy_name)
                         pair this account trades, plain columns (exchange,
                         symbol, strategy_name, initial_balance,
                         position_size_type, position_size_value,
                         commission, slippage, allow_long, allow_short,
                         max_open_positions) -- no JSON. Rewritten in full
                         each refresh (old rows for this account deleted
                         first, so a pair dropped from execution.config
                         also disappears here).
    accounts.stats    -- one row per account: initial_balance, a block of
                         plain trade facts (total trades, wins, losses,
                         win_rate, total fees, total volume, per-coin
                         breakdown, best/worst trade, ...), a live wallet
                         snapshot (equity/available balance/unrealized
                         pnl, pulled from Bybit right now), and every
                         quantstats metric computed over the account's
                         combined fill history.

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

import re

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor

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


def _get_bybit_client(conn, account_name):
    """Build a pybit HTTP session from this account's own stored credentials."""
    account = get_account_api_key(conn, account_name)
    if account is None:
        raise ValueError(f"Account {account_name!r} is not registered in accounts.api_keys.")
    return get_client(api_key=account["api_key"], api_secret=account["api_secret"], demo=account["demo"])


# ==========================================================
# accounts.combos -- plain columns, one row per (exchange, symbol, strategy)
# ==========================================================

def save_account_combos(conn, account_name, combo_configs):
    """
    Rewrite accounts.combos for one account: one row per (exchange,
    symbol, strategy_name) pair it trades, as plain columns (no JSON).

    combo_configs : list of dicts, one per combo, each shaped like
        {"exchange": ..., "symbol": ..., "strategy_name": ...,
         "initial_balance": ..., "position_size_type": ...,
         "position_size_value": ..., "commission": ..., "slippage": ...,
         "allow_long": ..., "allow_short": ..., "max_open_positions": ...}
        (exactly what _get_strategy_combos() in setup_accounts_example.py
        builds -- execution.config's fields plus the looked-up
        strategy_name).

    Full rewrite each call (old rows for this account deleted first, new
    ones inserted) -- always reflects exactly what execution.config +
    metadata.strategy currently say this account trades, nothing stale
    left behind from a pair that was later removed.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS accounts"))
    conn.commit()

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS accounts.combos (
            id                   SERIAL PRIMARY KEY,
            account_name         TEXT NOT NULL,
            exchange             TEXT NOT NULL,
            symbol               TEXT NOT NULL,
            strategy_name        TEXT NOT NULL,
            initial_balance      DOUBLE PRECISION,
            position_size_type   TEXT,
            position_size_value  DOUBLE PRECISION,
            commission           DOUBLE PRECISION,
            slippage             DOUBLE PRECISION,
            allow_long           BOOLEAN,
            allow_short          BOOLEAN,
            max_open_positions   INTEGER,
            updated_at           TIMESTAMP NOT NULL DEFAULT now()
        )
    """))
    conn.commit()

    cursor.execute(sql.SQL("DELETE FROM accounts.combos WHERE account_name = %s"), (account_name,))

    for combo in combo_configs:
        position_size = combo.get("position_size") or {}
        cursor.execute(sql.SQL("""
            INSERT INTO accounts.combos (
                account_name, exchange, symbol, strategy_name, initial_balance,
                position_size_type, position_size_value, commission, slippage,
                allow_long, allow_short, max_open_positions, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        """), (
            account_name,
            combo.get("exchange"),
            combo.get("symbol"),
            combo.get("strategy_name"),
            combo.get("initial_balance"),
            position_size.get("type") if position_size else combo.get("position_size_type"),
            position_size.get("value") if position_size else combo.get("position_size_value"),
            combo.get("commission"),
            combo.get("slippage"),
            combo.get("allow_long"),
            combo.get("allow_short"),
            combo.get("max_open_positions"),
        ))

    conn.commit()
    cursor.close()


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
# accounts.stats -- live wallet snapshot + trade facts + quantstats
# ==========================================================

def _fetch_wallet_snapshot(client):
    """
    Live wallet balance right now, straight from Bybit (get_wallet_balance,
    UNIFIED account). Returns a flat dict of floats, prefixed wallet_ so
    they're obviously a live snapshot rather than something derived from
    history. Returns all-None if Bybit has no UNIFIED wallet row (e.g.
    brand new demo account before demo funds are applied) rather than
    raising, since a missing wallet snapshot shouldn't block the rest of
    accounts.stats from being written.
    """
    keys = [
        "totalEquity", "totalWalletBalance", "totalAvailableBalance",
        "totalMarginBalance", "totalPerpUPL", "totalInitialMargin",
        "totalMaintenanceMargin", "accountIMRate", "accountMMRate",
    ]
    try:
        response = client.get_wallet_balance(accountType="UNIFIED")
        rows = response["result"]["list"]
        row = rows[0] if rows else {}
    except Exception:
        row = {}

    snapshot = {}
    for key in keys:
        val = row.get(key)
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        snapshot[f"wallet_{snake}"] = float(val) if val not in (None, "") else None
    return snapshot


def refresh_account_stats(conn, account_name, initial_balance, stats_config):
    """
    Recompute accounts.stats for one account from accounts.history (must
    be refreshed first via refresh_account_history() in the same run),
    plus a live wallet snapshot pulled from Bybit right now.

    Three blocks end up in the one row written:
      1. Plain trade facts, derived from accounts.history -- total fills,
         wins/losses (by exec-level realized value where determinable),
         total fees paid, total volume traded, per-coin breakdown, best/
         worst fill by value, average fill size, first/last trade time.
      2. Live wallet snapshot -- current equity/available balance/
         unrealized PnL etc, fetched from Bybit at refresh time (see
         _fetch_wallet_snapshot). Reflects right now, not history.
      3. Every quantstats metric (sharpe, sortino, max_drawdown, ...)
         computed over an approximate equity curve built by walking
         accounts.history's net cash flow (exec_value net of fees,
         signed by side) forward from initial_balance. This is an
         approximation of the account's trade-level P&L curve since
         fill-level data doesn't carry a "closed trade" P&L the way
         execution.*_trades does -- treat the quantstats block as
         directionally useful, not as precise as execution.stats'
         own per-strategy version (which is still trade-level and
         untouched by this file).

    initial_balance : this account's starting balance (caller decides
        what this means -- sum of contributing execution.config
        initial_balances, or your own account-level number).
    stats_config    : same dict compute_stats() takes everywhere else
        (loaded from stats/config.yaml).

    Which (exchange, symbol, strategy) combos this account trades no
    longer lives on this row -- see save_account_combos() / accounts.combos.

    Always writes a row, even with zero fills yet, so the account shows
    up in accounts.stats immediately after registration.
    """
    from crypto_pipeline.stats.calculator import compute_stats

    history = get_account_history(conn, account_name)
    client = _get_bybit_client(conn, account_name)
    wallet_snapshot = _fetch_wallet_snapshot(client)

    if history.empty:
        stats_row = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_fees_paid": 0.0,
            "total_volume": 0.0,
            "avg_trade_size": 0.0,
            "best_trade_value": None,
            "worst_trade_value": None,
            "coins_traded": Json([]),
            "first_trade_time": None,
            "last_trade_time": None,
            "final_balance": float(initial_balance),
            "total_net_profit": 0.0,
        }
    else:
        history = history.sort_values("trade_time")

        total_trades = len(history)
        total_fees_paid = float(history["exec_fee"].fillna(0).sum())
        total_volume = float(history["exec_value"].fillna(0).sum())
        avg_trade_size = float(history["exec_value"].fillna(0).mean())

        # Net cash flow per fill: negative for Buy (cash out), positive
        # for Sell (cash in), fee always a cost -- a rough per-fill P&L
        # proxy since fill-level data has no direct "this trade won/lost"
        # flag the way a closed-trade ledger row does.
        signed_value = history.apply(
            lambda r: (r["exec_value"] if r["side"] == "Sell" else -r["exec_value"]) - r["exec_fee"]
            if pd.notna(r["exec_value"]) else 0.0,
            axis=1,
        )
        wins = int((signed_value > 0).sum())
        losses = int((signed_value <= 0).sum())
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        best_trade_value = float(signed_value.max()) if total_trades > 0 else None
        worst_trade_value = float(signed_value.min()) if total_trades > 0 else None

        coins_traded = sorted(history["symbol"].dropna().unique().tolist())

        equity_values = float(initial_balance) + signed_value.cumsum()
        final_balance = float(equity_values.iloc[-1])
        total_net_profit = final_balance - float(initial_balance)

        equity_index = pd.to_datetime(history["trade_time"])
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
        stats_row["total_fees_paid"] = total_fees_paid
        stats_row["total_volume"] = total_volume
        stats_row["avg_trade_size"] = avg_trade_size
        stats_row["best_trade_value"] = best_trade_value
        stats_row["worst_trade_value"] = worst_trade_value
        stats_row["coins_traded"] = Json(coins_traded)
        stats_row["first_trade_time"] = history["trade_time"].iloc[0]
        stats_row["last_trade_time"] = history["trade_time"].iloc[-1]
        stats_row["final_balance"] = final_balance
        stats_row["total_net_profit"] = total_net_profit

    stats_row["initial_balance"] = float(initial_balance)
    stats_row.update(wallet_snapshot)

    def _pg_type_for_stats_col(col, val):
        # coins_traded is Json-wrapped -- _pg_type_for() only understands
        # plain pandas dtypes, so route it to JSONB directly instead of
        # letting it fall through to TEXT.
        if col == "coins_traded":
            return "JSONB"
        if col in ("first_trade_time", "last_trade_time"):
            return "TIMESTAMP"
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
    # quantstats metric, or a new wallet/facts field added later).
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