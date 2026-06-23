"""
db_utils.py
-----------
Shared database utility functions for the chain-predict pipeline.
Handles:
    - PostgreSQL connection
    - Schema and table creation for Binance and Bybit
    - Inserting OHLCV candle data
    - Checking last stored timestamp for incremental loading
"""

import os
import logging
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger for this module
logger = logging.getLogger(__name__)


# ── Database connection ────────────────────────────────────────────────────────

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


# ── Table creation ─────────────────────────────────────────────────────────────

def create_tables(conn, exchange: str, symbols: list, timeframes: list):
    """
    Create OHLCV tables for all symbols and timeframes under the given exchange schema.

    Each table is named like: binance.doge_1m or bybit.sol_1m

    Columns:
        - date_time : timestamp of the candle (primary key, no duplicates)
        - open      : opening price
        - high      : highest price in the candle
        - low       : lowest price in the candle
        - close     : closing price
        - volume    : trading volume

    Args:
        conn       : active psycopg2 connection
        exchange   : schema name, either 'binance' or 'bybit'
        symbols    : list of coin names from config
        timeframes : list of timeframe strings from config
    """
    cursor = conn.cursor()

    # Ensure the schema exists before creating tables inside it
    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(exchange)))

    for symbol in symbols:
        for timeframe in timeframes:
            table_name = f"{symbol}_{timeframe}"

            create_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {schema}.{table} (
                    date_time TIMESTAMP PRIMARY KEY,
                    open      DOUBLE PRECISION NOT NULL,
                    high      DOUBLE PRECISION NOT NULL,
                    low       DOUBLE PRECISION NOT NULL,
                    close     DOUBLE PRECISION NOT NULL,
                    volume    DOUBLE PRECISION NOT NULL
                )
            """).format(
                schema=sql.Identifier(exchange),
                table=sql.Identifier(table_name)
            )

            cursor.execute(create_query)
            logger.info(f"Table ensured: {exchange}.{table_name}")

    conn.commit()
    cursor.close()


# ── Last timestamp check ───────────────────────────────────────────────────────

def get_last_timestamp(conn, exchange: str, symbol: str, timeframe: str):
    """
    Get the most recent date_time stored in a table.
    Used for incremental loading — so we don't re-fetch data we already have.

    Returns None if table is empty (fetch from start_date in config).
    Returns datetime if data exists (fetch from there onwards).

    Args:
        conn      : active psycopg2 connection
        exchange  : schema name, either 'binance' or 'bybit'
        symbol    : coin name e.g. 'btc'
        timeframe : e.g. '1m'

    Returns:
        datetime of last stored candle, or None if table is empty
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{timeframe}"

    query = sql.SQL("""
        SELECT MAX(date_time) FROM {schema}.{table}
    """).format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    )

    cursor.execute(query)
    result = cursor.fetchone()[0]
    cursor.close()

    if result:
        logger.info(f"Last stored timestamp for {exchange}.{table_name}: {result}")
    else:
        logger.info(f"No existing data found for {exchange}.{table_name}. Will fetch from start_date.")

    return result


# ── Data insertion ─────────────────────────────────────────────────────────────

def insert_candles(conn, exchange: str, symbol: str, timeframe: str, df):
    """
    Insert OHLCV candle rows into the correct table.
    Skips rows that already exist (ON CONFLICT DO NOTHING).

    Args:
        conn      : active psycopg2 connection
        exchange  : schema name, either 'binance' or 'bybit'
        symbol    : coin name e.g. 'btc'
        timeframe : e.g. '1m'
        df        : pandas DataFrame with columns [date_time, open, high, low, close, volume]
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{timeframe}"

    insert_query = sql.SQL("""
        INSERT INTO {schema}.{table} (date_time, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (date_time) DO NOTHING
    """).format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    )

    rows = list(df[["date_time", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None))

    inserted = 0
    for row in rows:
        cursor.execute(insert_query, row)
        inserted += cursor.rowcount

    conn.commit()
    logger.info(f"Inserted {inserted} new rows into {exchange}.{table_name} (skipped {len(rows) - inserted} duplicates)")
    cursor.close()