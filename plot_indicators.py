"""
plot_indicators_sma_all_symbols.py
-----------------------------------
Fetch all symbols (1h resampled), store in DB, compute SMA, save to CSV.

Outputs per exchange in ./sma_output/:
  - ada_binance_1h_sma.csv
  - btc_binance_1h_sma.csv
  - ... (all symbols)
"""

import os
from datetime import datetime

import pandas as pd

from crypto_pipeline.data.data_downloader import get_data
from crypto_pipeline.indicators.talib_indicators import overlap_sma
from crypto_pipeline.utils.db_utils import get_db_connection


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["ada", "btc", "doge", "eth", "ltc", "mina", "sol", "sui"]
START = datetime(2025, 1, 1, 0, 0, 0)
END = datetime.now()
BASE_OUTPUT_DIR = "sma_output"
SMA_PERIOD = 20

# Exchange names to run
EXCHANGES = ["binance", "bybit"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_csv(df, output_dir, filename):
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"  ✓  {path}")


def create_1h_table(conn, exchange_name, symbol):
    """Create 1h table if it doesn't exist."""
    cursor = conn.cursor()
    table_name = f"{symbol}_1h"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {exchange_name}.{table_name} (
            datetime TIMESTAMP PRIMARY KEY,
            open      DOUBLE PRECISION NOT NULL,
            high      DOUBLE PRECISION NOT NULL,
            low       DOUBLE PRECISION NOT NULL,
            close     DOUBLE PRECISION NOT NULL,
            volume    DOUBLE PRECISION NOT NULL
        )
    """)
    conn.commit()
    cursor.close()


def insert_1h_data(conn, exchange_name, symbol, df):
    """Insert 1h data using COPY, skipping duplicates."""
    import io
    from psycopg2 import sql
    
    cursor = conn.cursor()
    table_name = f"{symbol}_1h"
    
    buffer = io.StringIO()
    df[["datetime", "open", "high", "low", "close", "volume"]].to_csv(
        buffer, index=False, header=False
    )
    buffer.seek(0)
    
    copy_query = sql.SQL(
        "COPY {schema}.{table} (datetime, open, high, low, close, volume) FROM STDIN WITH (FORMAT csv)"
    ).format(
        schema=sql.Identifier(exchange_name),
        table=sql.Identifier(table_name)
    )
    
    try:
        cursor.copy_expert(copy_query, buffer)
        conn.commit()
    except Exception as e:
        conn.rollback()
        # If duplicate, just skip
        if "duplicate key" in str(e).lower():
            pass
        else:
            raise
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Per-exchange, per-symbol pipeline
# ---------------------------------------------------------------------------

def run_for_symbol(exchange_name, symbol):
    """Fetch resampled data, store in DB, compute SMA, and save CSV."""
    
    print(f"\n  [{symbol.upper()}]")

    # get_data already resamples (to 1h via data_downloader.py)
    print(f"    Fetching {START} to {END}...")
    result = get_data(exchange=exchange_name, symbol=symbol, start_date=START, end_date=END)
    df = result["resampled"]

    if df.empty:
        print(f"    ⚠  No data found. Skipping.")
        return False

    print(f"    Loaded {len(df)} candles")

    # -----------------------------------------------------------------
    # Store 1h data in DB
    # -----------------------------------------------------------------
    
    conn = get_db_connection()
    try:
        create_1h_table(conn, exchange_name, symbol)
        
        # Check last timestamp in 1h table
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(datetime) FROM {exchange_name}.{symbol}_1h")
        last_1h_timestamp = cursor.fetchone()[0]
        cursor.close()
        
        # Only insert rows newer than last_1h_timestamp
        if last_1h_timestamp:
            df_to_insert = df[df["datetime"] > last_1h_timestamp]
            print(f"    Last 1h timestamp: {last_1h_timestamp}, inserting {len(df_to_insert)} new rows")
        else:
            df_to_insert = df
            print(f"    No existing 1h data, inserting all {len(df_to_insert)} rows")
        
        if not df_to_insert.empty:
            insert_1h_data(conn, exchange_name, symbol, df_to_insert)
            print(f"    ✓  Stored in DB: {exchange_name}.{symbol}_1h")
        else:
            print(f"    ✓  1h data already up to date")
    except Exception as e:
        print(f"    ⚠  DB insert error: {e}")
    finally:
        conn.close()

    # -----------------------------------------------------------------
    # Compute SMA
    # -----------------------------------------------------------------

    print(f"    Computing SMA({SMA_PERIOD})...")
    sma = overlap_sma(df, period=SMA_PERIOD)

    # -----------------------------------------------------------------
    # Trim leading NaN rows
    # -----------------------------------------------------------------

    valid_from = sma.first_valid_index()

    if valid_from is None:
        print(f"    ⚠  SMA all NaN. Skipping.")
        return False

    df_trimmed = df.iloc[valid_from:].reset_index(drop=True)
    sma_trimmed = sma.iloc[valid_from:].reset_index(drop=True)

    # -----------------------------------------------------------------
    # Build and save CSV
    # -----------------------------------------------------------------

    output_df = pd.DataFrame({
        "datetime": df_trimmed["datetime"],
        "open": df_trimmed["open"],
        "high": df_trimmed["high"],
        "low": df_trimmed["low"],
        "close": df_trimmed["close"],
        f"sma_{SMA_PERIOD}": sma_trimmed,
    })

    output_dir = os.path.join(BASE_OUTPUT_DIR, exchange_name)
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = f"{symbol}_{exchange_name}_1h_sma.csv"
    save_csv(output_df, output_dir, csv_filename)

    return True


def run_for_exchange(exchange_name):
    """Process all symbols for a given exchange."""
    print(f"\n{'=' * 70}")
    print(f"  {exchange_name.upper()}")
    print(f"{'=' * 70}")

    success_count = 0
    for symbol in SYMBOLS:
        try:
            if run_for_symbol(exchange_name, symbol):
                success_count += 1
        except Exception as e:
            print(f"    ✗ Error: {e}")

    print(f"\n  {success_count}/{len(SYMBOLS)} symbols processed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    print(f"\nProcessing {len(SYMBOLS)} symbols from {len(EXCHANGES)} exchanges")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframe: 1h | SMA Period: {SMA_PERIOD}")
    print(f"Date range: {START} to {END}\n")

    for exchange_name in EXCHANGES:
        run_for_exchange(exchange_name)

    print(f"\n{'=' * 70}")
    print(f"All done.")
    print(f"✓ 1h resampled data stored in DB: {SYMBOLS[0]}_1h, {SYMBOLS[1]}_1h, etc.")
    print(f"✓ SMA({SMA_PERIOD}) computed and saved to CSV in ./{BASE_OUTPUT_DIR}/<exchange>/")
    print(f"{'=' * 70}\n")