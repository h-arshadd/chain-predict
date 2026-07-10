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