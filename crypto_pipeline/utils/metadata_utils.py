"""
metadata_utils.py
------------------
Database utilities for the `metadata` schema.

Why this exists
----------------
Right now every knob (which exchange/symbol to pull, which indicators and
conditions make up a strategy, which coin/subreddits to track sentiment
for) lives in hardcoded yaml files (data/binance/config_binance.yml,
data/bybit/config_bybit.yml, signals/config.yaml,
sentiment_pipeline/config.yaml). That's fine for one person running
scripts locally, but it means every option is fixed at deploy time.

The `metadata` schema replaces those yaml files as the source of truth.
The plan is a frontend where a user picks an exchange/symbol/date range,
picks indicators + conditions for a strategy, picks a coin to track
sentiment for, etc. Those choices get INSERTed here instead of edited
into a yaml file, and the pipeline reads its config from these tables
instead.

Tables (schema: metadata)
--------------------------
metadata.data
    One row per (exchange, symbol) pair that should be tracked. Mirrors
    what data/binance/config_binance.yml and data/bybit/config_bybit.yml
    currently hardcode per exchange. Points at the real 1m candle table:
    {exchange}.{symbol}_1m (see db_utils.create_tables / insert_candles).

metadata.strategy
    One row per strategy DEFINITION -- the indicator + long/short condition
    rules from signals/config.yaml, as JSON, plus which (exchange, symbol)
    it's meant to run against (via a data_id foreign key, so exchange/symbol
    aren't duplicated here) and the timeframe it runs on (e.g. "1h" --
    the resample target, same as resample_timeframe in
    data_downloader.get_data()). Points at the real output table
    signals.{exchange}_{symbol} (see db_utils.insert_signals) once run.

    One strategy + its backtest for now (see backtest/config.yaml) --
    this table does not yet have a separate backtest_config column; that's
    an intentional simplification, not an oversight, until there's a
    reason to run more than one backtest per strategy.

metadata.sentiment
    One row per coin tracked by the sentiment pipeline (subreddits +
    search query, from sentiment_pipeline/config.yaml). Points at the real
    sentiment_clean.{coin}_posts table (see sentiment_pipeline/database.py).

Relationships
-------------
    metadata.data <--- metadata.strategy   (strategy.data_id -> data.data_id)

    metadata.sentiment  (standalone -- no FK to the others; sentiment is
                          tracked per-coin, not per-exchange/symbol/strategy)

This file only defines create_*_table()/get_*()/insert_*() functions. It
does not seed data from the yaml configs on its own -- see
discover_data_pairs() below for backfilling metadata.data from tables that
already exist in the DB; metadata.strategy/metadata.sentiment are meant to
be inserted directly (by you, or later by the frontend) since there's no
reliable way to reverse-engineer a strategy's config or a coin's
subreddits from tables that already exist.
"""

import os
import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA = "metadata"


def get_db_connection():
    """
    Create and return a PostgreSQL connection using credentials from .env.
    Same credentials/env vars as utils/db_utils.py -- this is the same
    database, just a different schema.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        logger.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise


def create_metadata_schema(conn):
    """Create the `metadata` schema if it doesn't already exist."""
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Schema ensured: {SCHEMA}")


# ==========================================================
# metadata.data
# ==========================================================

def create_data_table(conn):
    """
    One row per (exchange, symbol) pair to track, e.g. ("binance", "btc").
    Mirrors data/binance/config_binance.yml + data/bybit/config_bybit.yml.

    Points at the real candle table {exchange}.{symbol}_1m -- created
    separately by db_utils.create_tables() once data collection actually
    starts for that pair.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.data (
            data_id                 SERIAL PRIMARY KEY,
            exchange                TEXT NOT NULL,
            symbol                  TEXT NOT NULL,
            start_date              TIMESTAMP NOT NULL,
            end_date                TIMESTAMP,
            filling_missing_method  TEXT NOT NULL DEFAULT 'interpolation',
            fill_zero_volume        TEXT NOT NULL DEFAULT 'ffill',
            created_at              TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE (exchange, symbol)
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.data")


def insert_data(conn, exchange, symbol, start_date, end_date=None,
                 filling_missing_method="interpolation", fill_zero_volume="ffill"):
    """
    Register an (exchange, symbol) pair to track.

    end_date=None means "now" (open-ended, same meaning as the yaml configs'
    end_date: "now"). If the pair already exists (same exchange+symbol),
    updates its settings instead of erroring.

    Returns the data_id (existing or newly inserted).
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.data
            (exchange, symbol, start_date, end_date, filling_missing_method, fill_zero_volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (exchange, symbol) DO UPDATE SET
            start_date              = EXCLUDED.start_date,
            end_date                = EXCLUDED.end_date,
            filling_missing_method  = EXCLUDED.filling_missing_method,
            fill_zero_volume        = EXCLUDED.fill_zero_volume
        RETURNING data_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        exchange, symbol, start_date, end_date, filling_missing_method, fill_zero_volume,
    ))
    data_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Upserted {SCHEMA}.data: {exchange} | {symbol} -> data_id={data_id}")
    return data_id


def get_data_rows(conn, exchange=None, symbol=None):
    """
    Fetch rows from metadata.data, optionally filtered by exchange and/or
    symbol. Returns a list of dicts. This is what a frontend dropdown
    (e.g. "which exchange/symbol pairs are available") would call.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = sql.SQL("SELECT * FROM {schema}.data").format(schema=sql.Identifier(SCHEMA))
    conditions = []
    params = []

    if exchange is not None:
        conditions.append(sql.SQL("exchange = %s"))
        params.append(exchange)
    if symbol is not None:
        conditions.append(sql.SQL("symbol = %s"))
        params.append(symbol)

    if conditions:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# metadata.strategy
# ==========================================================

def create_strategy_table(conn):
    """
    One row per strategy definition -- the indicator + long/short condition
    rules currently hardcoded in signals/config.yaml, stored as JSON so the
    frontend can build arbitrary indicator/condition combinations without
    new columns per indicator.

    data_id is a foreign key into metadata.data, so exchange/symbol are
    looked up through it instead of being duplicated as their own columns
    here. timeframe is the resample target the strategy runs on (e.g. "1h"),
    same as resample_timeframe in data_downloader.get_data().

    One strategy + one backtest for now -- backtest settings aren't stored
    separately yet (see backtest/config.yaml); this table can grow a
    backtest_config column later if/when that's needed.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.strategy (
            strategy_id      SERIAL PRIMARY KEY,
            data_id          INTEGER NOT NULL REFERENCES {schema}.data(data_id) ON DELETE CASCADE,
            timeframe        TEXT NOT NULL DEFAULT '1h',
            strategy_config  JSONB NOT NULL,
            created_at       TIMESTAMP NOT NULL DEFAULT now()
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.strategy")


def insert_strategy(conn, data_id, strategy_config, timeframe="1h"):
    """
    Register a new strategy definition against a given (exchange, symbol),
    identified by data_id (see metadata.data / insert_data()).

    strategy_config: dict matching the shape of signals/config.yaml minus
    the top-level "strategy" split -- i.e. indicator blocks (RSI, EMA, ...)
    plus a "strategy" key with "long"/"short" condition lists. Stored as-is
    in JSONB; signals/main.py's load_config()/split_config() can be swapped
    to read this dict directly instead of parsing a yaml file.

    Returns the new strategy_id. Each insert creates a new row (no upsert)
    since a strategy is a new definition, not something keyed by a natural
    unique constraint.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.strategy (data_id, timeframe, strategy_config)
        VALUES (%s, %s, %s)
        RETURNING strategy_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        data_id, timeframe, Json(strategy_config),
    ))
    strategy_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Inserted {SCHEMA}.strategy: data_id={data_id} -> strategy_id={strategy_id}")
    return strategy_id


def get_strategy(conn, strategy_id):
    """
    Fetch one strategy row by id, joined with its data row, so the caller
    gets exchange/symbol/dates alongside strategy_config/timeframe in one
    call. Returns None if it doesn't exist.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT
            st.strategy_id, st.data_id, st.timeframe, st.strategy_config, st.created_at,
            d.exchange, d.symbol, d.start_date, d.end_date
        FROM {schema}.strategy st
        JOIN {schema}.data d ON d.data_id = st.data_id
        WHERE st.strategy_id = %s
    """).format(schema=sql.Identifier(SCHEMA)), (strategy_id,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def get_strategies(conn, data_id=None):
    """
    Fetch strategy rows (joined with their data row for exchange/symbol),
    optionally filtered to one data_id. What a frontend "pick a strategy"
    list would call.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = sql.SQL("""
        SELECT
            st.strategy_id, st.data_id, st.timeframe, st.strategy_config, st.created_at,
            d.exchange, d.symbol, d.start_date, d.end_date
        FROM {schema}.strategy st
        JOIN {schema}.data d ON d.data_id = st.data_id
    """).format(schema=sql.Identifier(SCHEMA))
    params = []

    if data_id is not None:
        query = query + sql.SQL(" WHERE st.data_id = %s")
        params.append(data_id)

    query = query + sql.SQL(" ORDER BY st.created_at DESC")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# metadata.sentiment
# ==========================================================

def create_sentiment_table(conn):
    """
    One row per coin tracked by the sentiment pipeline -- subreddits +
    search query, currently hardcoded per-coin in sentiment_pipeline's
    config.yaml under `coins:`.

    Points at the real table sentiment_clean.{coin}_posts (see
    sentiment_pipeline/database.py's create_tables/insert_analysis).
    No FK to data/strategy -- sentiment is tracked per-coin, independent
    of any specific exchange, symbol, or strategy.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.sentiment (
            sentiment_id  SERIAL PRIMARY KEY,
            coin          TEXT NOT NULL UNIQUE,
            subreddits    JSONB NOT NULL,
            search_query  TEXT NOT NULL,
            post_limit    INTEGER NOT NULL DEFAULT 5,
            created_at    TIMESTAMP NOT NULL DEFAULT now()
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.sentiment")


def insert_sentiment(conn, coin, subreddits, search_query, post_limit=5):
    """
    Register (or update) which subreddits/query to track for a coin.

    subreddits: list of subreddit name strings, e.g. ["Bitcoin", "BitcoinMarkets"].
    If the coin already has a row, updates its settings instead of erroring.

    Returns the sentiment_id (existing or newly inserted).
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.sentiment (coin, subreddits, search_query, post_limit)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (coin) DO UPDATE SET
            subreddits    = EXCLUDED.subreddits,
            search_query  = EXCLUDED.search_query,
            post_limit    = EXCLUDED.post_limit
        RETURNING sentiment_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        coin, Json(subreddits), search_query, post_limit,
    ))
    sentiment_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Upserted {SCHEMA}.sentiment: {coin} -> sentiment_id={sentiment_id}")
    return sentiment_id


def get_sentiment_rows(conn, coin=None):
    """
    Fetch rows from metadata.sentiment, optionally filtered by coin.
    Returns a list of dicts. What a frontend "which coins are tracked"
    list would call.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = sql.SQL("SELECT * FROM {schema}.sentiment").format(schema=sql.Identifier(SCHEMA))
    params = []

    if coin is not None:
        query = query + sql.SQL(" WHERE coin = %s")
        params.append(coin)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# Convenience: create everything at once
# ==========================================================

def create_all_metadata_tables(conn):
    """
    Create the metadata schema and all three tables in the correct order
    (data before strategy, since strategy.data_id references it). Safe to
    call repeatedly -- everything uses CREATE ... IF NOT EXISTS.
    """
    create_metadata_schema(conn)
    create_data_table(conn)
    create_strategy_table(conn)   # FK -> data
    create_sentiment_table(conn)
    logger.info("All metadata tables ensured.")


# ==========================================================
# Discovery: backfill metadata.data from tables that already exist
# ==========================================================
#
# The pipeline has been running before metadata.data existed, so
# {exchange}.{symbol}_1m tables are already populated in the real DB. This
# scans information_schema for those existing tables and registers them
# in metadata.data with INSERT ... ON CONFLICT DO NOTHING (via
# insert_data()'s upsert), so it's safe to run once now to backfill, and
# safe to re-run later without duplicating rows.
#
# There's no equivalent discovery for metadata.strategy or
# metadata.sentiment -- there's no reliable way to reverse-engineer a
# strategy's indicator/condition config or a coin's subreddits from tables
# that already exist (signals.{exchange}_{symbol} doesn't record which
# strategy config produced it). Those are meant to be inserted directly.

def find_existing_candle_tables(conn, exchanges=("binance", "bybit")):
    """
    Query information_schema for every {symbol}_1m table under the given
    exchange schemas. Returns a list of (exchange, symbol) tuples.

    This is read-only introspection -- it does not touch metadata.data
    itself. See discover_data_pairs() to actually register what's found.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema = ANY(%s)
          AND table_name LIKE %s
    """), (list(exchanges), "%\\_1m"))
    rows = cursor.fetchall()
    cursor.close()

    pairs = []
    for schema_name, table_name in rows:
        # table_name is "{symbol}_1m" -- strip the trailing "_1m"
        symbol = table_name[: -len("_1m")]
        pairs.append((schema_name, symbol))
    return pairs


def discover_data_pairs(conn, exchanges=("binance", "bybit"),
                         filling_missing_method="interpolation", fill_zero_volume="ffill"):
    """
    Find every existing {exchange}.{symbol}_1m table and register it in
    metadata.data, using that table's own MIN(datetime) as start_date (the
    real date the data begins at, not whatever a yaml config claims) and
    leaving end_date NULL (open-ended -- these tables keep growing via
    incremental loads, see data_downloader.DataDownloader.download()).

    Safe to re-run: insert_data() upserts on (exchange, symbol), so running
    this again just refreshes start_date to whatever MIN(datetime) is now.

    Returns the list of data_id's that were inserted/updated.
    """
    pairs = find_existing_candle_tables(conn, exchanges=exchanges)
    data_ids = []

    for exchange, symbol in pairs:
        table_name = f"{symbol}_1m"
        cursor = conn.cursor()
        cursor.execute(sql.SQL("SELECT MIN(datetime) FROM {schema}.{table}").format(
            schema=sql.Identifier(exchange),
            table=sql.Identifier(table_name),
        ))
        min_datetime = cursor.fetchone()[0]
        cursor.close()

        if min_datetime is None:
            logger.warning(f"{exchange}.{table_name} exists but is empty -- skipping.")
            continue

        data_id = insert_data(
            conn, exchange, symbol,
            start_date=min_datetime,
            end_date=None,
            filling_missing_method=filling_missing_method,
            fill_zero_volume=fill_zero_volume,
        )
        data_ids.append(data_id)

    logger.info(f"Discovered and registered {len(data_ids)} (exchange, symbol) pairs into {SCHEMA}.data.")
    return data_ids