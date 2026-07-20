"""
db_utils.py
-----------
Shared database utility functions for the chain-predict pipeline.
Handles:
    - PostgreSQL connection
    - Schema and table creation for Binance and Bybit
    - Inserting OHLCV candle data
    - Checking last stored timestamp for incremental loading
    - Storing signal-pipeline output (insert_signals) and backtest trade
      ledgers (insert_trades), both as full rebuild-on-every-run tables
      (see their docstrings below for why)

All candles are 1-minute timeframe, so tables are named like:
    binance.doge_1m, bybit.sol_1m

Signals/backtest tables are named like:
    signals.binance_doge, backtest.bybit_sol
"""

import os
import io
import re
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TIMEFRAME = "1m"

# Maps pandas dtypes to Postgres column types for the dynamic signals/trades
# tables (see insert_signals/insert_trades). Order matters: bool must be
# checked before int, since pandas bool dtype would otherwise also satisfy
# an int check.
_PG_TYPE_MAP = [
    (pd.api.types.is_bool_dtype, "BOOLEAN"),
    (pd.api.types.is_integer_dtype, "INTEGER"),
    (pd.api.types.is_float_dtype, "DOUBLE PRECISION"),
    (pd.api.types.is_datetime64_any_dtype, "TIMESTAMP"),
]


def _pg_type_for(series):
    """Best-effort pandas dtype -> Postgres column type mapping."""
    for check, pg_type in _PG_TYPE_MAP:
        if check(series.dtype):
            return pg_type
    return "TEXT"  # fallback for anything unexpected (e.g. object dtype)


def _round4(value):
    """
    Round a single numeric value to 4 decimal places before it goes into
    Postgres. Leaves None, strings, bools, datetimes, and anything else
    non-numeric untouched. Used at the simulator's write points (state/
    positions, trade ledger, stats) so every float landing in the DB is
    consistently rounded, not just whichever ones happened to get a
    round() call at the source.
    """
    if value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
                return value  # NaN/inf -- leave as-is, not a rounding concern
            return round(float(value), 4)
        except (TypeError, ValueError):
            return value
    return value


def _round4_df(df, exclude_cols=()):
    """
    Return a copy of df with every numeric (float) column rounded to 4
    decimal places. Integer/bool/datetime/object columns are left alone.
    exclude_cols: column names to skip (e.g. trade_id, which is already
    a plain int and never needs rounding).
    """
    import numpy as np
    df = df.copy()
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].round(4)
    return df


def _copy_dataframe(conn, df, schema, table):
    """
    Shared COPY helper used by insert_signals/insert_trades: bulk-load every
    row of df into schema.table via COPY (same fast-path insert_candles()
    already uses). Caller is responsible for creating/rebuilding the table
    first with matching columns.
    """
    cursor = conn.cursor()

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    columns = sql.SQL(", ").join(sql.Identifier(col) for col in df.columns)
    copy_query = sql.SQL(
        "COPY {schema}.{table} ({columns}) FROM STDIN WITH (FORMAT csv)"
    ).format(
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        columns=columns,
    )

    cursor.copy_expert(copy_query, buffer)

    conn.commit()
    cursor.close()
    logger.info(f"Copied {len(df)} rows into {schema}.{table}")


def get_db_connection():
    """
    Create and return a PostgreSQL connection using credentials from .env file.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        logger.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise


def create_tables(conn, exchange, symbols):
    """
    Create the exchange schema (if missing) and one table per symbol.
    Table name: {symbol}_1m, e.g. doge_1m
    """
    cursor = conn.cursor()

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(exchange)))

    for symbol in symbols:
        table_name = f"{symbol}_{TIMEFRAME}"

        cursor.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                datetime TIMESTAMP PRIMARY KEY,
                open      DOUBLE PRECISION NOT NULL,
                high      DOUBLE PRECISION NOT NULL,
                low       DOUBLE PRECISION NOT NULL,
                close     DOUBLE PRECISION NOT NULL,
                volume    DOUBLE PRECISION NOT NULL
            )
        """).format(
            schema=sql.Identifier(exchange),
            table=sql.Identifier(table_name)
        ))

        logger.info(f"Table ensured: {exchange}.{table_name}")

    conn.commit()
    cursor.close()


def get_last_timestamp(conn, exchange, symbol):
    """
    Return the most recent datetime stored for this symbol, or None if empty.
    Used for incremental loading.
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

    cursor.execute(sql.SQL("SELECT MAX(datetime) FROM {schema}.{table}").format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    ))
    result = cursor.fetchone()[0]
    cursor.close()

    if result:
        logger.info(f"Last stored timestamp for {exchange}.{table_name}: {result}")
    else:
        logger.info(f"No existing data found for {exchange}.{table_name}. Will fetch from start_date.")

    return result


def insert_candles(conn, exchange, symbol, df):
    """
    Bulk-insert OHLCV candle rows into the symbol's table using COPY.
    df must have columns: datetime, open, high, low, close, volume
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

    buffer = io.StringIO()
    df[["datetime", "open", "high", "low", "close", "volume"]].to_csv(
        buffer, index=False, header=False
    )
    buffer.seek(0)

    copy_query = sql.SQL(
        "COPY {schema}.{table} (datetime, open, high, low, close, volume) FROM STDIN WITH (FORMAT csv)"
    ).format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    )

    cursor.copy_expert(copy_query, buffer)

    conn.commit()
    cursor.close()
    logger.info(f"Copied {len(df)} rows into {exchange}.{table_name}")


def get_candles_from_db(conn, exchange, symbol, start_date, end_date):
    """
    Return stored 1m candles for a symbol between start_date and end_date (inclusive), as a DataFrame.
    Columns: datetime, open, high, low, close, volume

    Uses the caller's existing connection — like every other function in
    this file, it does NOT open or close its own. The caller (e.g.
    DataDownloader, which already has self.conn) owns the connection's
    lifecycle.
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

    cursor.execute(sql.SQL(
        "SELECT datetime, open, high, low, close, volume FROM {schema}.{table} "
        "WHERE datetime BETWEEN %s AND %s ORDER BY datetime"
    ).format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    ), (start_date, end_date))
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def insert_signals(conn, exchange, symbol, strategy_name, df):
    """
    Store the signals/main.py output DataFrame for one exchange+symbol+
    strategy run, replacing whatever was there before.

    Schema: signals.{exchange}_{symbol}_{strategy_name}
    e.g. signals.binance_doge_RSI_14_reversal

    strategy_name is baked into the TABLE NAME (not stored as a column in
    df) -- so which strategy produced a table is identifiable from its
    name alone, and different strategies for the same pair land in
    different tables instead of overwriting each other. strategy_name is
    sanitized (non-alphanumeric -> "_") since it flows into a raw
    identifier.

    NOTE: if you insert a new strategy row in metadata.strategy but keep
    the same strategy_name (e.g. just tweaking indicator params), this
    table gets overwritten on the next signals run -- the table is keyed
    by name only, not strategy_id, so re-using a name means re-using the
    table.

    df's column set changes whenever signals/config.yaml's active strategy
    changes (different indicators -> different ind_* columns, different
    condition counts -> different long_cond_N/short_cond_N columns). Rather
    than trying to migrate an existing table's columns to match, this always
    drops and recreates the table from df's current columns/dtypes, then
    COPYs the full DataFrame in. So every run of the signal pipeline for the
    same (exchange, symbol, strategy_name) fully replaces that table -- there
    is no append/incremental mode within a single strategy's table.

    df must include a "datetime" column; every other column is taken as-is
    (indicators, condition booleans, the final "signal" column, etc.).
    """
    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    table_name = f"{exchange}_{symbol}_{safe_strategy_name}"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS signals"))

    # Full rebuild: drop first so a strategy change (different/renamed
    # columns) can never collide with the previous run's table shape.
    cursor.execute(sql.SQL("DROP TABLE IF EXISTS {schema}.{table}").format(
        schema=sql.Identifier("signals"),
        table=sql.Identifier(table_name)
    ))

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL("TIMESTAMP" if col == "datetime" else _pg_type_for(df[col]))
        )
        for col in df.columns
    )

    cursor.execute(sql.SQL("CREATE TABLE {schema}.{table} ({column_defs})").format(
        schema=sql.Identifier("signals"),
        table=sql.Identifier(table_name),
        column_defs=column_defs
    ))

    conn.commit()
    cursor.close()
    logger.info(f"Table rebuilt: signals.{table_name} ({len(df.columns)} columns)")

    _copy_dataframe(conn, df, "signals", table_name)


def insert_trades(conn, exchange, symbol, trade_ledger):
    """
    Store the backtest trade ledger for one exchange+symbol, replacing
    whatever was there before.

    Schema: backtest.{exchange}_{symbol}, e.g. backtest.bybit_sol

    trade_ledger's columns are fixed by backtest.py's trades.append(...) --
    entry_time, exit_time, direction, entry_price, exit_price, quantity,
    gross_pnl, commission, slippage, net_pnl, balance_after_trade, plus
    cumulative_pnl if run_backtest added it. Still rebuilt from the
    DataFrame's own columns/dtypes rather than hardcoded here, so this keeps
    working even if backtest.py's ledger columns change later -- same
    rebuild-on-every-run behavior as insert_signals, for the same reason (a
    strategy/backtest config change shouldn't have to migrate an old table).

    If trade_ledger is empty (no trades), the table is still created (with
    the right columns, in case something downstream queries it) but no rows
    are copied in.
    """
    cursor = conn.cursor()
    table_name = f"{exchange}_{symbol}"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS backtest"))

    cursor.execute(sql.SQL("DROP TABLE IF EXISTS {schema}.{table}").format(
        schema=sql.Identifier("backtest"),
        table=sql.Identifier(table_name)
    ))

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for(trade_ledger[col]))
        )
        for col in trade_ledger.columns
    )

    cursor.execute(sql.SQL("CREATE TABLE {schema}.{table} ({column_defs})").format(
        schema=sql.Identifier("backtest"),
        table=sql.Identifier(table_name),
        column_defs=column_defs
    ))

    conn.commit()
    cursor.close()
    logger.info(f"Table rebuilt: backtest.{table_name} ({len(trade_ledger.columns)} columns)")

    if trade_ledger.empty:
        logger.info(f"No trades for {exchange} | {symbol}. Table created empty.")
        return

    _copy_dataframe(conn, trade_ledger, "backtest", table_name)

# ============================================================
# The functions below are new additions for the Simulator Module.
# Append them to the end of the existing crypto_pipeline/utils/db_utils.py
# (they use the same helpers -- sql, _pg_type_for, _copy_dataframe -- 
# already defined earlier in that file).
# ============================================================


def get_simulator_state(conn, exchange, symbol, strategy_name):
    """
    Return the simulator's saved state for this exchange+symbol+strategy,
    or None if it has never run before.

    Schema: simulator.positions -- ONE shared table for every strategy/
    exchange/symbol combo (not one table per combo). Each row = one
    combo's current execution state: exchange, symbol, strategy_name,
    last processed candle timestamp, running balance, cumulative_pnl,
    and the open position's fields (all NULL if flat). This is the
    Simulator Module spec's "Position Table" -- at most one open
    position per combo (per Version 1), separate from the Trade
    Ledger's permanent history.

    Returns a dict with keys: last_processed, balance, cumulative_pnl,
    time_horizon, position (dict or None). position dict has: direction,
    entry_time, entry_price, quantity, take_profit, stop_loss, status.

    No separate trade_id column: entry_time already uniquely identifies
    this trade (a strategy can only have one open position at a time), and
    is the same value used as the PRIMARY KEY on simulator.*_trades once
    this position closes and lands in the Trade Ledger -- see
    append_simulator_trades. Keeping one field instead of two avoids ever
    having them drift apart.

    "id" here is a normal auto-incrementing PRIMARY KEY, unrelated to
    trade identity -- it exists purely so this table has something to
    upsert against in save_simulator_state (matched there via exchange +
    symbol + strategy_name, not via id).

    NOTE: the CREATE TABLE below must stay byte-for-byte in sync with the
    one in save_simulator_state (including the UNIQUE constraint), since
    "IF NOT EXISTS" means only whichever of these two functions runs
    first actually creates the table. If they ever drift, ON CONFLICT in
    save_simulator_state will fail with "no unique or exclusion
    constraint matching the ON CONFLICT specification".
    """
    cursor = conn.cursor()

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.positions (
            id              SERIAL PRIMARY KEY,
            exchange        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            time_horizon    TEXT,
            last_processed  TIMESTAMP,
            balance         DOUBLE PRECISION NOT NULL,
            cumulative_pnl  DOUBLE PRECISION,
            direction       TEXT,
            entry_time      TIMESTAMP,
            entry_price     DOUBLE PRECISION,
            quantity        DOUBLE PRECISION,
            take_profit     DOUBLE PRECISION,
            stop_loss       DOUBLE PRECISION,
            leaning         TEXT,
            status          TEXT,
            UNIQUE (exchange, symbol, strategy_name)
        )
    """).format(
        schema=sql.Identifier("simulator")
    ))
    conn.commit()

    cursor.execute(sql.SQL(
        "SELECT last_processed, balance, cumulative_pnl, direction, entry_time, "
        "entry_price, quantity, take_profit, stop_loss, leaning, status, time_horizon FROM {schema}.positions "
        "WHERE exchange = %s AND symbol = %s AND strategy_name = %s LIMIT 1"
    ).format(
        schema=sql.Identifier("simulator")
    ), (exchange, symbol, strategy_name))
    row = cursor.fetchone()
    cursor.close()

    if row is None:
        return None

    columns = ["last_processed", "balance", "cumulative_pnl", "direction", "entry_time",
               "entry_price", "quantity", "take_profit", "stop_loss", "leaning", "status", "time_horizon"]
    values = dict(zip(columns, row))

    position = None
    if values["direction"] is not None:
        position = {
            "direction": values["direction"],
            "entry_time": values["entry_time"],
            "entry_price": values["entry_price"],
            "quantity": values["quantity"],
            "take_profit": values["take_profit"],
            "stop_loss": values["stop_loss"],
            "leaning": values["leaning"],
            "status": values["status"] or "open",
        }

    return {
        "last_processed": values["last_processed"],
        "balance": values["balance"],
        "cumulative_pnl": values["cumulative_pnl"],
        "time_horizon": values["time_horizon"],
        "position": position,
    }


def save_simulator_state(conn, exchange, symbol, strategy_name, time_horizon, last_processed, balance, position, cumulative_pnl):
    """
    Overwrite the simulator's saved state for this exchange+symbol+strategy.
    Called at the end of every run so the next scheduled run resumes from
    here. position=None is stored with status="closed" and every other
    position field (direction, entry_time, entry_price, quantity,
    take_profit, stop_loss) left NULL -- there is no trade to describe, so
    only status gets an explicit value.

    Schema: simulator.positions -- ONE shared table for every strategy/
    exchange/symbol combo.

    This is a genuine UPSERT (INSERT ... ON CONFLICT DO UPDATE), not
    DELETE-then-INSERT. Matched on (exchange, symbol, strategy_name) --
    each combo gets exactly one row, updated in place every run.

    NOTE: the CREATE TABLE below must stay byte-for-byte in sync with the
    one in get_simulator_state -- see the note there.
    """
    cursor = conn.cursor()

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.positions (
            id              SERIAL PRIMARY KEY,
            exchange        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            time_horizon    TEXT,
            last_processed  TIMESTAMP,
            balance         DOUBLE PRECISION NOT NULL,
            cumulative_pnl  DOUBLE PRECISION,
            direction       TEXT,
            entry_time      TIMESTAMP,
            entry_price     DOUBLE PRECISION,
            quantity        DOUBLE PRECISION,
            take_profit     DOUBLE PRECISION,
            stop_loss       DOUBLE PRECISION,
            leaning         TEXT,
            status          TEXT,
            UNIQUE (exchange, symbol, strategy_name)
        )
    """).format(
        schema=sql.Identifier("simulator")
    ))
    conn.commit()

    # Defensive self-heal: if simulator.positions already exists from a
    # prior run without the UNIQUE constraint (e.g. it was created by an
    # older version of get_simulator_state that lacked it), add the
    # constraint now instead of failing on ON CONFLICT below. No-op if
    # the constraint is already present.
    cursor.execute(sql.SQL("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'positions_exchange_symbol_strategy_name_key'
                  AND conrelid = 'simulator.positions'::regclass
            ) THEN
                ALTER TABLE simulator.positions
                    ADD CONSTRAINT positions_exchange_symbol_strategy_name_key
                    UNIQUE (exchange, symbol, strategy_name);
            END IF;
        END $$;
    """))
    conn.commit()

    position = position or {}
    values = (
        exchange,
        symbol,
        strategy_name,
        time_horizon,
        last_processed,
        balance,
        cumulative_pnl,
        position.get("direction"),
        position.get("entry_time"),
        position.get("entry_price"),
        position.get("quantity"),
        position.get("take_profit"),
        position.get("stop_loss"),
        position.get("leaning"),
        position.get("status") if position else "closed",
    )

    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.positions
            (exchange, symbol, strategy_name, time_horizon, last_processed, balance, cumulative_pnl,
             direction, entry_time, entry_price, quantity, take_profit, stop_loss, leaning, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (exchange, symbol, strategy_name) DO UPDATE SET
            time_horizon   = EXCLUDED.time_horizon,
            last_processed = EXCLUDED.last_processed,
            balance        = EXCLUDED.balance,
            cumulative_pnl = EXCLUDED.cumulative_pnl,
            direction      = EXCLUDED.direction,
            entry_time     = EXCLUDED.entry_time,
            entry_price    = EXCLUDED.entry_price,
            quantity       = EXCLUDED.quantity,
            take_profit    = EXCLUDED.take_profit,
            stop_loss      = EXCLUDED.stop_loss,
            leaning        = EXCLUDED.leaning,
            status         = EXCLUDED.status
    """).format(
        schema=sql.Identifier("simulator")
    ), values)

    conn.commit()
    cursor.close()
    logger.info(f"Saved simulator state: simulator.positions ({exchange}/{symbol}/{strategy_name})")


def append_simulator_trades(conn, exchange, symbol, strategy_name, trade_ledger):
    """
    Append newly-closed trades to the simulator's running Trade Ledger.
    Unlike insert_trades() (backtest -- full rebuild every run), this is a
    live, ever-growing ledger across scheduler runs, so rows are appended,
    never dropped or replaced. The table is only created (not recreated) if
    missing, so it survives across runs.

    Schema: simulator.{exchange}_{symbol}_{strategy_name}_trades

    trade_ledger : DataFrame of newly-closed trades from this run only
    (same columns as one closed_trade dict from simulator.py: direction,
    entry_date_time, exit_date_time, entry_price, exit_price, quantity,
    gross_pnl, commission, slippage, net_pnl, exit_reason, balance).
    Does nothing if trade_ledger is empty.

    No exchange/symbol columns -- the table itself is already named per
    exchange+symbol+strategy, so repeating them in every row would be
    redundant.

    trade_id : plain incrementing 1, 2, 3... column, added here (not by
    the caller) so it can continue counting across every past run's
    trades already in the table, not just this run's batch. It's a
    convenience label only -- NOT the primary key, and NOT unique-
    constrained, so nothing downstream should rely on it for identity.

    entry_date_time is still this table's real PRIMARY KEY -- it's the
    PDF's "Trade ID" column in the sense that matters (guaranteed
    unique): a strategy can only ever have one open position at a time
    (max_open_positions=1, enforced in simulator.py's step_candle), so
    entry_date_time is already a unique identifier for a trade within
    this strategy's own table, and it's assigned the moment the position
    opens (Step 3 of the spec) rather than only once it closes -- so the
    same value identifies the trade on both the Position Table (while
    open, see get_simulator_state, where the column is still named
    entry_time) and the Trade Ledger (once closed, here, named
    entry_date_time). If entry_date_time is ever duplicated (would mean
    two trades opened on the exact same candle for the same strategy --
    shouldn't happen given max_open_positions=1), the COPY below fails
    loudly on the PK violation rather than silently overwriting a row.
    """
    if trade_ledger.empty:
        return

    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    table_name = f"{exchange}_{symbol}_{safe_strategy_name}_trades"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    qualified_name = sql.SQL(".").join(
        [sql.Identifier("simulator"), sql.Identifier(table_name)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None

    # Continue trade_id from wherever the existing table left off, so
    # numbering stays 1, 2, 3... across scheduler runs instead of
    # restarting at 1 every time. 0 if the table doesn't exist yet (first
    # run for this combo).
    next_trade_id = 1
    if table_exists:
        cursor.execute(sql.SQL(
            "SELECT COALESCE(MAX(trade_id), 0) FROM {schema}.{table}"
        ).format(
            schema=sql.Identifier("simulator"),
            table=sql.Identifier(table_name)
        ))
        next_trade_id = cursor.fetchone()[0] + 1

    trade_ledger = trade_ledger.copy()
    trade_ledger.insert(0, "trade_id", range(next_trade_id, next_trade_id + len(trade_ledger)))

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}{pk}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for(trade_ledger[col])),
            pk=sql.SQL(" PRIMARY KEY" if col == "entry_date_time" else "")
        )
        for col in trade_ledger.columns
    )

    cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {schema}.{table} ({column_defs})").format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(table_name),
        column_defs=column_defs
    ))
    conn.commit()
    cursor.close()

    _copy_dataframe(conn, trade_ledger, "simulator", table_name)
    logger.info(f"Appended {len(trade_ledger)} trade(s) to simulator.{table_name}")

def get_simulator_summary(conn, exchange, symbol, strategy_name):
    """
    Roll up the simulator's Trade Ledger for one exchange+symbol+strategy
    into the summary fields the Simulator Module spec requires as output:
    Final Account Balance, Total Profit/Loss, Total Number of Trades, and
    a Win/Loss Summary -- same shape as run_backtest()'s return dict
    (final_balance, total_net_profit, total_trades, win_loss), just read
    back from the DB instead of computed in-memory from a fresh run.

    Schema read: simulator.{exchange}_{symbol}_{strategy_name}_trades
    (see append_simulator_trades) for the ledger, and simulator.positions
    (see get_simulator_state) for the current balance/position.

    Returns a dict:
        final_balance     : float -- current balance from the positions
                             table (starting balance if the strategy has
                             never traded), or None if this strategy has
                             never run at all (no positions table yet).
        total_net_profit  : float -- final_balance - starting balance.
                             starting balance is read from the ledger's
                             own first trade's (balance - net_pnl) if any
                             trades exist, otherwise falls back to
                             final_balance itself (0 profit, no trades
                             yet).
        total_trades      : int -- row count in the Trade Ledger table.
        win_loss          : dict -- {"wins", "losses", "win_rate"}, same
                             convention as backtest (win = net_pnl > 0).
        open_position     : dict or None -- current open position, same
                             shape as get_simulator_state()'s "position".

    Returns None entirely if this (exchange, symbol, strategy) has never
    been run (no state table exists yet).
    """
    state = get_simulator_state(conn, exchange, symbol, strategy_name)
    if state is None:
        return None

    final_balance = state["balance"]
    open_position = state["position"]

    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    trades_table = f"{exchange}_{symbol}_{safe_strategy_name}_trades"

    # to_regclass() takes a plain string that Postgres parses like any
    # other identifier reference: unquoted, it folds to lowercase before
    # the catalog lookup. But the table was created via sql.Identifier(),
    # which always emits a double-quoted, case-preserved identifier (e.g.
    # CREATE TABLE simulator."binance_btc_RSI_14_reversal_trades"). Any
    # strategy_name with uppercase letters (RSI_14_reversal,
    # SMA_20_price_cross, ...) then has a table to_regclass can never
    # find -- table_exists comes back False even though the table exists
    # and is full of rows, so the summary silently falls into the
    # "never run" branch and reports 0 trades / 0 PnL forever.
    #
    # Fix: render the qualified name through sql.Identifier + as_string()
    # the same way CREATE TABLE did, so the quoting matches and
    # to_regclass looks up the exact case-preserved name.
    qualified_name = sql.SQL(".").join(
        [sql.Identifier("simulator"), sql.Identifier(trades_table)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None

    if not table_exists:
        cursor.close()
        return {
            "final_balance": final_balance,
            "total_net_profit": 0.0,
            "total_trades": 0,
            "win_loss": {"wins": 0, "losses": 0, "win_rate": 0.0},
            "open_position": open_position,
        }

    cursor.execute(sql.SQL(
        "SELECT COUNT(*), "
        "COUNT(*) FILTER (WHERE net_pnl > 0), "
        "COUNT(*) FILTER (WHERE net_pnl <= 0), "
        "MIN(balance - net_pnl) "
        "FROM {schema}.{table}"
    ).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(trades_table)
    ))
    total_trades, wins, losses, starting_balance = cursor.fetchone()
    cursor.close()

    total_trades = int(total_trades or 0)
    wins = int(wins or 0)
    losses = int(losses or 0)
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0

    # starting_balance comes from the earliest trade's pre-trade balance
    # (balance - net_pnl on that row). If there are no trades
    # yet, there's nothing to compute profit against -- net profit is 0.
    if starting_balance is not None:
        total_net_profit = float(final_balance) - float(starting_balance)
    else:
        total_net_profit = 0.0

    return {
        "final_balance": float(final_balance),
        "total_net_profit": total_net_profit,
        "total_trades": total_trades,
        "win_loss": {"wins": wins, "losses": losses, "win_rate": win_rate},
        "open_position": open_position,
    }


# ============================================================
# Simulator Stats (simulator.stats)
# ============================================================


def build_equity_curve_from_ledger(conn, exchange, symbol, strategy_name, initial_balance):
    """
    Reconstruct a datetime-indexed equity curve from the simulator's own
    Trade Ledger table (simulator.{exchange}_{symbol}_{strategy_name}_trades),
    the same shape run_backtest() already returns as "equity_curve" (flat
    between trades, steps at each exit) -- this is what
    crypto_pipeline.stats.calculator.compute_stats() requires as input.

    The simulator itself never builds this in memory (unlike backtest,
    which walks one contiguous candle range in a single process): it only
    persists final balance (simulator.positions) and a trade ledger, across
    possibly many separate scheduled runs. So stats has to read it back
    from the ledger table instead of receiving it directly.

    Returns a pandas Series indexed by exit_date_time, values = balance
    after each trade, with one synthetic point at the very start
    (index = first trade's entry_date_time, value = initial_balance) so
    the curve doesn't start mid-air. Returns None if the ledger table
    doesn't exist yet or has no rows (nothing to build a curve from).
    """
    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    trades_table = f"{exchange}_{symbol}_{safe_strategy_name}_trades"

    qualified_name = sql.SQL(".").join(
        [sql.Identifier("simulator"), sql.Identifier(trades_table)]
    ).as_string(conn)
    cursor.execute(sql.SQL("SELECT to_regclass(%s)"), (qualified_name,))
    table_exists = cursor.fetchone()[0] is not None

    if not table_exists:
        cursor.close()
        return None

    cursor.execute(sql.SQL(
        "SELECT entry_date_time, exit_date_time, balance FROM {schema}.{table} "
        "ORDER BY exit_date_time ASC"
    ).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(trades_table)
    ))
    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        return None

    first_entry_date_time = rows[0][0]
    index = [first_entry_date_time] + [r[1] for r in rows]
    values = [float(initial_balance)] + [float(r[2]) for r in rows]

    equity = pd.Series(values, index=pd.to_datetime(index))
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    return equity


def save_simulator_stats(conn, exchange, symbol, strategy_name, time_horizon, stats_dict):
    """
    Save one row of ALL performance stats for this exchange+symbol+
    strategy combo into simulator.stats -- ONE shared table for every
    combo (same pattern as simulator.positions), one row per
    (exchange, symbol, strategy_name), upserted in place each run.

    stats_dict: the "metrics" dict from
    crypto_pipeline.stats.calculator.compute_stats(), which itself
    dynamically discovers and computes every quantstats.stats metric
    (55+ as of this quantstats version -- sharpe, sortino, calmar,
    max_drawdown, cagr, profit_factor, win_rate, kelly_criterion, var,
    cvar, ulcer_index, ... everything metrics.py's discover_metrics()
    finds), plus "total_trades" added by the caller. Every key in
    stats_dict becomes its own column -- nothing is filtered down to a
    "headline" subset anymore, so a metric excluded via stats/config.yaml's
    exclude_metrics (or dropped in a future quantstats version) just
    means one less column next time the table's shape is checked, not a
    missing value in an otherwise-fixed schema.

    Columns are derived from stats_dict's own keys (quantstats' own
    function names -- a fixed, trusted vocabulary, not arbitrary user
    input) via ADD COLUMN IF NOT EXISTS, so the table grows to fit
    whatever metrics.py actually discovers rather than needing a
    hardcoded column list kept in sync by hand.

    This is a genuine UPSERT (INSERT ... ON CONFLICT DO UPDATE), matched
    on (exchange, symbol, strategy_name) -- each combo gets exactly one
    row, replaced in place every run (stats are a snapshot of the
    strategy's full history-to-date, not something to append to).
    """
    cursor = conn.cursor()

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    # Base table: identity columns + total_trades + bookkeeping only.
    # Every metric column is added on demand below, so this function
    # works whether stats_dict has 9 keys or 55+.
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.stats (
            id              SERIAL PRIMARY KEY,
            exchange        TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            strategy_name   TEXT NOT NULL,
            time_horizon    TEXT,
            total_trades    INTEGER,
            updated_at      TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (exchange, symbol, strategy_name)
        )
    """).format(
        schema=sql.Identifier("simulator")
    ))
    conn.commit()

    # Defensive self-heal, same reasoning as save_simulator_state: if
    # simulator.stats already exists from an earlier version of this
    # function without the UNIQUE constraint, add it now instead of
    # failing on ON CONFLICT below.
    cursor.execute(sql.SQL("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'stats_exchange_symbol_strategy_name_key'
                  AND conrelid = 'simulator.stats'::regclass
            ) THEN
                ALTER TABLE simulator.stats
                    ADD CONSTRAINT stats_exchange_symbol_strategy_name_key
                    UNIQUE (exchange, symbol, strategy_name);
            END IF;
        END $$;
    """))
    conn.commit()

    # Reserved/base column names -- never treated as a metric column even
    # if a metric somehow shared the name (defensive, shouldn't happen
    # with quantstats' actual function names).
    _RESERVED = {"id", "exchange", "symbol", "strategy_name", "time_horizon", "total_trades", "updated_at"}
    metric_cols = [k for k in stats_dict.keys() if k not in _RESERVED]

    # Every metric value is a plain float (or None) from quantstats --
    # add any column that isn't there yet. Safe to run every call;
    # IF NOT EXISTS makes it a no-op once the column already exists.
    for col in metric_cols:
        cursor.execute(sql.SQL(
            "ALTER TABLE {schema}.stats ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"
        ).format(
            schema=sql.Identifier("simulator"),
            col=sql.Identifier(col)
        ))
    conn.commit()

    all_cols = ["exchange", "symbol", "strategy_name", "time_horizon", "total_trades"] + metric_cols
    values = (
        exchange,
        symbol,
        strategy_name,
        time_horizon,
        stats_dict.get("total_trades"),
        *[_round4(stats_dict.get(m)) for m in metric_cols],
    )

    insert_cols = sql.SQL(", ").join(sql.Identifier(c) for c in all_cols) + sql.SQL(", updated_at")
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in all_cols) + sql.SQL(", NOW()")
    update_set = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
        for c in (["time_horizon", "total_trades"] + metric_cols)
    ) + sql.SQL(", updated_at = NOW()")

    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.stats ({insert_cols})
        VALUES ({placeholders})
        ON CONFLICT (exchange, symbol, strategy_name) DO UPDATE SET
            {update_set}
    """).format(
        schema=sql.Identifier("simulator"),
        insert_cols=insert_cols,
        placeholders=placeholders,
        update_set=update_set,
    ), values)

    conn.commit()
    cursor.close()
    logger.info(
        f"Saved simulator stats ({len(metric_cols)} metric column(s)): "
        f"simulator.stats ({exchange}/{symbol}/{strategy_name})"
    )