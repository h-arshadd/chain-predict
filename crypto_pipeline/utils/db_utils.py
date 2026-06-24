"""
db_utils.py
-----------
Shared database utility functions for the chain-predict pipeline.
Handles:
    - PostgreSQL connection
    - Schema and table creation for Binance and Bybit
    - Inserting OHLCV candle data
    - Checking last stored timestamp for incremental loading

All candles are 1-minute timeframe, so tables are named like:
    binance.doge_1m, bybit.sol_1m
"""

import os
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TIMEFRAME = "1m"


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
        ))

        logger.info(f"Table ensured: {exchange}.{table_name}")

    conn.commit()
    cursor.close()


def get_last_timestamp(conn, exchange, symbol):
    """
    Return the most recent date_time stored for this symbol, or None if empty.
    Used for incremental loading.
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

    cursor.execute(sql.SQL("SELECT MAX(date_time) FROM {schema}.{table}").format(
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
    Insert OHLCV candle rows into the symbol's table. Skips duplicates.
    df must have columns: date_time, open, high, low, close, volume
    """
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

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
    cursor.close()
    logger.info(f"Inserted {inserted} new rows into {exchange}.{table_name} (skipped {len(rows) - inserted} duplicates)")


def get_candles(conn, exchange, symbol):
    """
    Return all stored 1m candles for a symbol as a DataFrame.
    Columns: date_time, open, high, low, close, volume
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    table_name = f"{symbol}_{TIMEFRAME}"

    cursor.execute(sql.SQL(
        "SELECT date_time, open, high, low, close, volume FROM {schema}.{table} ORDER BY date_time"
    ).format(
        schema=sql.Identifier(exchange),
        table=sql.Identifier(table_name)
    ))
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=["date_time", "open", "high", "low", "close", "volume"])
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df