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

    Schema: simulator.{exchange}_{symbol}_{strategy_name}_state
    Single-row table: last processed candle timestamp, running balance,
    and the open position's fields (all NULL if flat). This is the
    Simulator Module spec's "Position Table" -- it represents the current
    execution state (at most one open position, per Version 1), separate
    from the Trade Ledger's permanent history.

    Returns a dict with keys: last_processed, balance, position (dict or
    None). position dict has: direction, entry_time, entry_price,
    quantity, take_profit, stop_loss, status (always "open" when position
    is not None -- the PDF's Position Table lists Status as a column;
    since this table only ever holds zero or one row, "open" is the only
    status a present row can have -- there's nothing to store for "flat",
    that's just no position at all, i.e. position=None).

    No separate trade_id column: entry_time already uniquely identifies
    this trade (a strategy can only have one open position at a time), and
    is the same value used as the PRIMARY KEY on simulator.*_trades once
    this position closes and lands in the Trade Ledger -- see
    append_simulator_trades. Keeping one field instead of two avoids ever
    having them drift apart.

    "id" here is a normal auto-incrementing PRIMARY KEY, unrelated to
    trade identity -- it exists purely so this single-row table (which
    must still have a row even when flat, i.e. entry_time IS NULL) has
    something to upsert against in save_simulator_state. entry_time can't
    serve as this table's PK the way it does on the trades table, because
    it's NULL whenever there's no open position.
    """
    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    table_name = f"{exchange}_{symbol}_{safe_strategy_name}_state"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            id              SERIAL PRIMARY KEY,
            last_processed  TIMESTAMP,
            balance         DOUBLE PRECISION NOT NULL,
            direction       TEXT,
            entry_time      TIMESTAMP,
            entry_price     DOUBLE PRECISION,
            quantity        DOUBLE PRECISION,
            take_profit     DOUBLE PRECISION,
            stop_loss       DOUBLE PRECISION,
            status          TEXT
        )
    """).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(table_name)
    ))
    conn.commit()

    cursor.execute(sql.SQL(
        "SELECT last_processed, balance, direction, entry_time, "
        "entry_price, quantity, take_profit, stop_loss, status FROM {schema}.{table} LIMIT 1"
    ).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(table_name)
    ))
    row = cursor.fetchone()
    cursor.close()

    if row is None:
        return None

    columns = ["last_processed", "balance", "direction", "entry_time",
               "entry_price", "quantity", "take_profit", "stop_loss", "status"]
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
            "status": values["status"] or "open",
        }

    return {
        "last_processed": values["last_processed"],
        "balance": values["balance"],
        "position": position,
    }


def save_simulator_state(conn, exchange, symbol, strategy_name, last_processed, balance, position):
    """
    Overwrite the simulator's saved state for this exchange+symbol+strategy.
    Called at the end of every run so the next scheduled run resumes from
    here. position=None is stored as all-NULL (flat) -- including status.

    Schema: simulator.{exchange}_{symbol}_{strategy_name}_state

    This is a genuine UPSERT (INSERT ... ON CONFLICT (id) DO UPDATE), not
    DELETE-then-INSERT. The table only ever holds exactly one row, so an
    UPDATE-in-place is strictly correct here and avoids doing two
    statements (plus MVCC bloat from a dead row) every single scheduler
    tick, for every strategy/pair, even on ticks where nothing changed.
    "id" is a normal auto-incrementing PRIMARY KEY (SERIAL) -- the table
    is only ever written to via this function, which always targets id=1,
    so it stays a true singleton in practice even though nothing stops a
    second row from existing structurally.
    """
    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    table_name = f"{exchange}_{symbol}_{safe_strategy_name}_state"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.{table} (
            id              SERIAL PRIMARY KEY,
            last_processed  TIMESTAMP,
            balance         DOUBLE PRECISION NOT NULL,
            direction       TEXT,
            entry_time      TIMESTAMP,
            entry_price     DOUBLE PRECISION,
            quantity        DOUBLE PRECISION,
            take_profit     DOUBLE PRECISION,
            stop_loss       DOUBLE PRECISION,
            status          TEXT
        )
    """).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(table_name)
    ))
    conn.commit()

    position = position or {}
    values = (
        last_processed,
        balance,
        position.get("direction"),
        position.get("entry_time"),
        position.get("entry_price"),
        position.get("quantity"),
        position.get("take_profit"),
        position.get("stop_loss"),
        position.get("status") if position else None,
    )

    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.{table}
            (id, last_processed, balance, direction, entry_time,
             entry_price, quantity, take_profit, stop_loss, status)
        VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            last_processed = EXCLUDED.last_processed,
            balance        = EXCLUDED.balance,
            direction      = EXCLUDED.direction,
            entry_time     = EXCLUDED.entry_time,
            entry_price    = EXCLUDED.entry_price,
            quantity       = EXCLUDED.quantity,
            take_profit    = EXCLUDED.take_profit,
            stop_loss      = EXCLUDED.stop_loss,
            status         = EXCLUDED.status
    """).format(
        schema=sql.Identifier("simulator"),
        table=sql.Identifier(table_name)
    ), values)

    conn.commit()
    cursor.close()
    logger.info(f"Saved simulator state: simulator.{table_name}")


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
    entry_time, exit_time, entry_price, exit_price, quantity, gross_pnl,
    commission, slippage, net_pnl, exit_reason, balance_after_trade).
    Does nothing if trade_ledger is empty.

    entry_time is this table's PRIMARY KEY -- it's the PDF's "Trade ID"
    column: a strategy can only ever have one open position at a time
    (max_open_positions=1, enforced in simulator.py's step_candle), so
    entry_time is already a unique identifier for a trade within this
    strategy's own table, and it's assigned the moment the position opens
    (Step 3 of the spec) rather than only once it closes -- so the same
    value identifies the trade on both the Position Table (while open,
    see get_simulator_state) and the Trade Ledger (once closed, here).
    If entry_time is ever duplicated (would mean two trades opened on the
    exact same candle for the same strategy -- shouldn't happen given
    max_open_positions=1), the COPY below fails loudly on the PK
    violation rather than silently overwriting a row.
    """
    if trade_ledger.empty:
        return

    cursor = conn.cursor()
    safe_strategy_name = re.sub(r"[^0-9a-zA-Z_]", "_", strategy_name)
    table_name = f"{exchange}_{symbol}_{safe_strategy_name}_trades"

    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS simulator"))

    column_defs = sql.SQL(", ").join(
        sql.SQL("{col} {pg_type}{pk}").format(
            col=sql.Identifier(col),
            pg_type=sql.SQL(_pg_type_for(trade_ledger[col])),
            pk=sql.SQL(" PRIMARY KEY" if col == "entry_time" else "")
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
    (see append_simulator_trades) for the ledger, and
    simulator.{exchange}_{symbol}_{strategy_name}_state (see
    get_simulator_state) for the current balance/position.

    Returns a dict:
        final_balance     : float -- current balance from the state table
                             (starting balance if the strategy has never
                             traded), or None if this strategy has never
                             run at all (no state table yet).
        total_net_profit  : float -- final_balance - starting balance.
                             starting balance is read from the ledger's
                             own first trade's (balance_after_trade -
                             net_pnl) if any trades exist, otherwise falls
                             back to final_balance itself (0 profit, no
                             trades yet).
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
        "MIN(balance_after_trade - net_pnl) "
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
    # (balance_after_trade - net_pnl on that row). If there are no trades
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