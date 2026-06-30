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
import io
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