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
    rules from signals/config.yaml, as JSON, plus the timeframe it runs on
    (e.g. "1h" -- the resample target, same as resample_timeframe in
    data_downloader.get_data()). GLOBAL, not per-pair: there is no FK to
    metadata.data -- the most recently inserted strategy is "the active
    strategy" and runs against every pair in metadata.data (see
    get_current_strategy()). Points at the real output tables
    signals.{exchange}_{symbol} (see db_utils.insert_signals) once run,
    once per tracked pair.

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
    metadata.data        (standalone -- 16 rows, one per (exchange, symbol)
                           pair: 8 symbols x 2 exchanges from
                           config_binance.yml / config_bybit.yml)

    metadata.strategy     (standalone -- global, not linked to metadata.data;
                           the newest row is the active strategy and applies
                           to every row in metadata.data)

    metadata.sentiment    (standalone -- no FK to the others; sentiment is
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
            time_horizon            TEXT NOT NULL DEFAULT '1m',
            created_at              TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE (exchange, symbol)
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.data")


def insert_data(conn, exchange, symbol, start_date, end_date=None, time_horizon="1m"):
    """
    Register an (exchange, symbol) pair to track, or update it in place if
    that pair is already registered (upsert on the (exchange, symbol)
    UNIQUE constraint) -- so calling this twice for the same pair updates
    the existing row instead of creating a duplicate.

    end_date=None means "now" (open-ended, same meaning as the yaml configs'
    end_date: "now").

    Returns the data_id of the inserted or existing row.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.data
            (exchange, symbol, start_date, end_date, time_horizon)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (exchange, symbol) DO UPDATE SET
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            time_horizon = EXCLUDED.time_horizon
        RETURNING data_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        exchange, symbol, start_date, end_date, time_horizon,
    ))
    data_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Inserted {SCHEMA}.data: {exchange} | {symbol} -> data_id={data_id}")
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

    Strategy is GLOBAL now, not per-pair: the active strategy runs against
    EVERY tracked pair in metadata.data, so there's no data_id FK here --
    that's an intentional change from the earlier per-pair design, not an
    oversight. timeframe is the resample target the strategy runs on (e.g.
    "1h"), same as resample_timeframe in data_downloader.get_data().

    strategy_name is a short human-readable label (e.g. "RSI_14_reversal")
    -- purely descriptive, not used for lookups, so it isn't unique.

    One strategy + one backtest for now -- backtest settings aren't stored
    separately yet (see backtest/config.yaml); this table can grow a
    backtest_config column later if/when that's needed.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.strategy (
            strategy_id      SERIAL PRIMARY KEY,
            strategy_name    TEXT NOT NULL,
            timeframe        TEXT NOT NULL DEFAULT '1h',
            strategy_config  JSONB NOT NULL,
            created_at       TIMESTAMP NOT NULL DEFAULT now()
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.strategy")


def insert_strategy(conn, strategy_name, strategy_config, timeframe="1h"):
    """
    Register a new global strategy definition. It applies to every pair in
    metadata.data -- no data_id / exchange / symbol taken here.

    strategy_name: short human-readable label for this strategy (e.g.
    "RSI_14_reversal", "EMA_cross_trend") -- pick something that describes
    what the strategy does, since strategy_config alone is just raw JSON.

    strategy_config: dict matching the shape of signals/config.yaml minus
    the top-level "strategy" split -- i.e. indicator blocks (RSI, EMA, ...)
    plus a "strategy" key with "long"/"short" condition lists. Stored as-is
    in JSONB; signals/main.py's load_config()/split_config() can be swapped
    to read this dict directly instead of parsing a yaml file.

    Returns the new strategy_id. Each insert creates a new row (no upsert)
    since a strategy is a new definition, not something keyed by a natural
    unique constraint. The most recently inserted row is "the current
    strategy" -- see get_current_strategy().
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.strategy (strategy_name, timeframe, strategy_config)
        VALUES (%s, %s, %s)
        RETURNING strategy_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        strategy_name, timeframe, Json(strategy_config),
    ))
    strategy_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Inserted {SCHEMA}.strategy: {strategy_name!r} -> strategy_id={strategy_id}")
    return strategy_id


def get_strategy(conn, strategy_id):
    """
    Fetch one strategy row by id. Returns None if it doesn't exist.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT strategy_id, strategy_name, timeframe, strategy_config, created_at
        FROM {schema}.strategy
        WHERE strategy_id = %s
    """).format(schema=sql.Identifier(SCHEMA)), (strategy_id,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def get_current_strategy(conn):
    """
    Fetch the single active strategy -- the most recently inserted row in
    metadata.strategy. This is what signals/main.py would call once per
    run, then loop over every pair in metadata.data. Returns None if no
    strategy has been inserted yet.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT strategy_id, strategy_name, timeframe, strategy_config, created_at
        FROM {schema}.strategy
        ORDER BY created_at DESC, strategy_id DESC
        LIMIT 1
    """).format(schema=sql.Identifier(SCHEMA)))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def get_strategies(conn):
    """
    Fetch all strategy rows, newest first. What a frontend "strategy
    history" list would call.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT strategy_id, strategy_name, timeframe, strategy_config, created_at
        FROM {schema}.strategy
        ORDER BY created_at DESC, strategy_id DESC
    """).format(schema=sql.Identifier(SCHEMA)))
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# metadata.backtest
# ==========================================================

def create_backtest_table(conn):
    """
    One row per backtest run -- the full settings from backtest/config.yaml
    (date range, position sizing, commission/slippage, TP/SL, execution,
    portfolio limits), stored as-is in JSON so this table doesn't need a
    column per setting.

    strategy_name is a plain TEXT column (not a FK into metadata.strategy)
    -- it's here so a backtest row can be identified/filtered by which
    strategy it ran, without needing a join. It's a label, same as
    metadata.strategy.strategy_name, not a foreign key.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.backtest (
            backtest_id      SERIAL PRIMARY KEY,
            strategy_name    TEXT NOT NULL,
            backtest_config  JSONB NOT NULL,
            created_at       TIMESTAMP NOT NULL DEFAULT now()
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.backtest")


def insert_backtest(conn, strategy_name, backtest_config):
    """
    Register a new backtest run against a given strategy_name.

    backtest_config: dict matching the shape of backtest/config.yaml --
    start_date, end_date, initial_balance, position_size, commission,
    slippage, allow_long/allow_short, take_profit, stop_loss, entry_price,
    exit_price, max_open_positions. Stored as-is in JSONB.

    Returns the new backtest_id. Each insert creates a new row (no upsert)
    since every backtest run is its own record -- there's no natural
    unique constraint to key off of.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.backtest (strategy_name, backtest_config)
        VALUES (%s, %s)
        RETURNING backtest_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        strategy_name, Json(backtest_config),
    ))
    backtest_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Inserted {SCHEMA}.backtest: {strategy_name!r} -> backtest_id={backtest_id}")
    return backtest_id


def get_backtest(conn, backtest_id):
    """
    Fetch one backtest row by id. Returns None if it doesn't exist.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(sql.SQL("""
        SELECT backtest_id, strategy_name, backtest_config, created_at
        FROM {schema}.backtest
        WHERE backtest_id = %s
    """).format(schema=sql.Identifier(SCHEMA)), (backtest_id,))
    row = cursor.fetchone()
    cursor.close()
    return dict(row) if row else None


def get_backtests(conn, strategy_name=None):
    """
    Fetch backtest rows, newest first, optionally filtered to one
    strategy_name. What a frontend "backtest history" list would call.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = sql.SQL("""
        SELECT backtest_id, strategy_name, backtest_config, created_at
        FROM {schema}.backtest
    """).format(schema=sql.Identifier(SCHEMA))
    params = []

    if strategy_name is not None:
        query = query + sql.SQL(" WHERE strategy_name = %s")
        params.append(strategy_name)

    query = query + sql.SQL(" ORDER BY created_at DESC, backtest_id DESC")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# metadata.sentiment
# ==========================================================

def create_sentiment_table(conn):
    """
    One row per (coin, subreddit, search_query) combination tracked by the sentiment pipeline.
    Allows multiple search queries per coin+subreddit for different sentiment analyses.
    
    Points at the real table sentiment_clean.{coin}_posts (see
    sentiment_pipeline/database.py's create_tables/insert_analysis).
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        CREATE TABLE IF NOT EXISTS {schema}.sentiment (
            sentiment_id  SERIAL PRIMARY KEY,
            coin          TEXT NOT NULL,
            subreddit     TEXT NOT NULL,
            search_query  TEXT NOT NULL,
            post_limit    INTEGER NOT NULL DEFAULT 5,
            created_at    TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE (coin, subreddit)
        )
    """).format(schema=sql.Identifier(SCHEMA)))
    conn.commit()
    cursor.close()
    logger.info(f"Table ensured: {SCHEMA}.sentiment")


def insert_sentiment(conn, coin, subreddit, search_query, post_limit=5):
    """
    Register a (coin, subreddit) combination for sentiment tracking, or
    update it in place if that combination is already registered (upsert
    on the (coin, subreddit) UNIQUE constraint) -- so re-running this for
    the same coin+subreddit updates search_query/post_limit instead of
    creating a duplicate.

    Returns the sentiment_id of the inserted or existing row.
    """
    cursor = conn.cursor()
    cursor.execute(sql.SQL("""
        INSERT INTO {schema}.sentiment (coin, subreddit, search_query, post_limit)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (coin, subreddit) DO UPDATE SET
            search_query = EXCLUDED.search_query,
            post_limit = EXCLUDED.post_limit
        RETURNING sentiment_id
    """).format(schema=sql.Identifier(SCHEMA)), (
        coin, subreddit, search_query, post_limit,
    ))
    sentiment_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    logger.info(f"Registered sentiment: {coin} | {subreddit} | {search_query} -> sentiment_id={sentiment_id}")
    return sentiment_id


def get_sentiment_rows(conn, coin=None, subreddit=None):
    """
    Fetch rows from metadata.sentiment, optionally filtered by coin and/or subreddit.
    Returns a list of dicts.
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = sql.SQL("SELECT * FROM {schema}.sentiment").format(schema=sql.Identifier(SCHEMA))
    conditions = []
    params = []

    if coin is not None:
        conditions.append(sql.SQL("coin = %s"))
        params.append(coin)
    
    if subreddit is not None:
        conditions.append(sql.SQL("subreddit = %s"))
        params.append(subreddit)

    if conditions:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    cursor.close()
    return [dict(row) for row in rows]


# ==========================================================
# Convenience: create everything at once
# ==========================================================

def create_all_metadata_tables(conn):
    """
    Create the metadata schema and all four tables. Safe to call
    repeatedly -- everything uses CREATE ... IF NOT EXISTS. (strategy no
    longer has a data_id FK -- it's global now -- so table order doesn't
    matter, but data is still created first for readability.)
    """
    create_metadata_schema(conn)
    create_data_table(conn)
    create_strategy_table(conn)
    create_backtest_table(conn)
    create_sentiment_table(conn)
    logger.info("All metadata tables ensured.")


# Exchanges + symbols currently hardcoded in data/binance/config_binance.yml
# and data/bybit/config_bybit.yml. Both exchanges track the same 8 symbols,
# so this is 16 (exchange, symbol) pairs total. Used by seed_data_pairs()
# below to backfill metadata.data without needing live {exchange}.{symbol}_1m
# candle tables to already exist (unlike discover_data_pairs(), which reads
# information_schema and requires the tables to be there).
KNOWN_EXCHANGE_SYMBOLS = {
    "binance": ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"],
    "bybit":   ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"],
}


def seed_data_pairs(conn, start_date="2024-01-01", end_date=None, time_horizon="1m"):
    """
    Register all 16 (exchange, symbol) pairs from KNOWN_EXCHANGE_SYMBOLS
    into metadata.data, using the start_date/end_date/time_horizon from
    the yaml configs (data/binance/config_binance.yml,
    data/bybit/config_bybit.yml -- both currently say start_date:
    "2024-01-01", end_date: "now", time_horizon: "1m").

    Unlike discover_data_pairs(), this doesn't require the real
    {exchange}.{symbol}_1m candle tables to exist yet -- it seeds from the
    known config, not from introspecting the DB. Safe to call repeatedly:
    insert_data() upserts on the (exchange, symbol) UNIQUE constraint, so
    re-running this just refreshes start_date/end_date/time_horizon on the
    existing 16 rows instead of duplicating them.

    Returns the list of data_id's for all 16 pairs (existing or newly
    inserted), in binance-then-bybit / symbol-list order.
    """
    data_ids = []
    for exchange, symbols in KNOWN_EXCHANGE_SYMBOLS.items():
        for symbol in symbols:
            data_id = insert_data(
                conn, exchange, symbol,
                start_date=start_date,
                end_date=end_date,
                time_horizon=time_horizon,
            )
            data_ids.append(data_id)

    logger.info(f"Seeded {len(data_ids)} (exchange, symbol) pairs into {SCHEMA}.data.")
    return data_ids


# Coins + subreddits + search queries currently hardcoded in
# sentiment_pipeline/config.yaml. 2 coins x 3 subreddits each = 6
# (coin, subreddit) rows total. post_limit comes from that same config's
# top-level reddit.post_limit (shared across all coins/subreddits there).
KNOWN_SENTIMENT_CONFIG = {
    "BTC": {
        "subreddits": ["Bitcoin", "BitcoinMarkets", "CryptoCurrency"],
        "search_query": "BTC OR Bitcoin",
    },
    "ETH": {
        "subreddits": ["ethereum", "ethtrader", "CryptoCurrency"],
        "search_query": "ETH OR Ethereum",
    },
}
KNOWN_SENTIMENT_POST_LIMIT = 5


def seed_sentiment_pairs(conn, post_limit=None):
    """
    Register all 6 (coin, subreddit) combinations from
    KNOWN_SENTIMENT_CONFIG into metadata.sentiment, using each coin's
    search_query from that same config and post_limit from
    sentiment_pipeline/config.yaml's reddit.post_limit (default 5).

    Safe to call repeatedly: insert_sentiment() upserts on the
    (coin, subreddit) UNIQUE constraint, so re-running this refreshes the
    existing 6 rows instead of duplicating them.

    Returns the list of sentiment_id's for all 6 combinations (existing or
    newly inserted).
    """
    if post_limit is None:
        post_limit = KNOWN_SENTIMENT_POST_LIMIT

    sentiment_ids = []
    for coin, cfg in KNOWN_SENTIMENT_CONFIG.items():
        for subreddit in cfg["subreddits"]:
            sentiment_id = insert_sentiment(
                conn, coin, subreddit,
                search_query=cfg["search_query"],
                post_limit=post_limit,
            )
            sentiment_ids.append(sentiment_id)

    logger.info(f"Seeded {len(sentiment_ids)} (coin, subreddit) pairs into {SCHEMA}.sentiment.")
    return sentiment_ids


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


def discover_data_pairs(conn, exchanges=("binance", "bybit"), time_horizon="1m"):
    """
    Find every existing {exchange}.{symbol}_1m table and register it in
    metadata.data, using that table's own MIN(datetime) as start_date (the
    real date the data begins at, not whatever a yaml config claims) and
    leaving end_date NULL (open-ended -- these tables keep growing via
    incremental loads, see data_downloader.DataDownloader.download()).

    Inserts each discovered pair into metadata.data. Safe to re-run --
    insert_data() upserts on the (exchange, symbol) UNIQUE constraint, so
    re-discovery refreshes existing rows instead of duplicating them.

    Returns the list of data_id's that were inserted.
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
            time_horizon=time_horizon,
        )
        data_ids.append(data_id)

    logger.info(f"Discovered and registered {len(data_ids)} (exchange, symbol) pairs into {SCHEMA}.data.")
    return data_ids