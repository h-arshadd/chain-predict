"""
exchange_binance.py
-------------------
Handles all Binance API interactions for fetching OHLCV candle data.
Uses the python-binance library.

Key details:
    - Binance timestamps are in MILLISECONDS — we convert to datetime
    - Uses futures klines (future_klines) as specified
    - Uses LINEAR perpetual contracts
    - Fetches 1000 candles per API call (Binance max)
    - Retries on failure with configurable delay
"""

import time
import logging
import pandas as pd
from datetime import datetime, timezone
from binance.client import Client

# Set up logger for this module
logger = logging.getLogger(__name__)

# Binance client — no API key needed for public market data
client = Client()


# ── Timeframe mapping ──────────────────────────────────────────────────────────

# Map our config timeframe strings to Binance interval constants
TIMEFRAME_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}


# ── Raw candle fetcher ─────────────────────────────────────────────────────────

def fetch_batch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """
    Fetch a single batch of up to 1000 candles from Binance Futures.
    Uses get_future_klines (linear perpetual contracts).

    Args:
        symbol    : trading pair e.g. "DOGEUSDT"
        interval  : Binance interval constant e.g. Client.KLINE_INTERVAL_1MINUTE
        start_ms  : start timestamp in milliseconds
        end_ms    : end timestamp in milliseconds

    Returns:
        List of raw candle data from Binance API
    """
    return client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=1000  # maximum allowed per call
    )


# ── Candle parser ──────────────────────────────────────────────────────────────

def parse_candles(raw_candles: list) -> pd.DataFrame:
    """
    Parse raw Binance API response into a clean OHLCV DataFrame.

    Binance returns each candle as a list:
        [open_time, open, high, low, close, volume, close_time, ...]

    Timestamps from Binance are in MILLISECONDS — divide by 1000 to get seconds,
    then convert to UTC datetime.

    Args:
        raw_candles : list of raw candle lists from Binance API

    Returns:
        Clean pandas DataFrame with columns [date_time, open, high, low, close, volume]
    """
    if not raw_candles:
        return pd.DataFrame()

    # Binance open_time is in milliseconds → convert to naive UTC datetime
    df = pd.DataFrame([{
        "date_time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
        "open":      float(c[1]),
        "high":      float(c[2]),
        "low":       float(c[3]),
        "close":     float(c[4]),
        "volume":    float(c[5]),
    } for c in raw_candles])

    return df


# ── Main fetch function ────────────────────────────────────────────────────────

def fetch_candles(symbol: str, timeframe: str, start_date, end_date: datetime, config: dict) -> pd.DataFrame:
    """
    Fetch all historical OHLCV candles for a symbol from Binance in batches.

    Since Binance only returns 1000 candles per call, we loop — moving the
    start time forward after each batch — until we reach the end date.

    Args:
        symbol     : coin name from config e.g. "doge"
        timeframe  : e.g. "1m"
        start_date : start date as either a "YYYY-MM-DD" string (first run,
                     straight from the yml config) or a datetime/pandas
                     Timestamp (incremental run, computed from the last
                     stored timestamp). Both forms are handled here.
        end_date   : datetime object for end of range
        config     : full config dictionary (for retries, delay, etc.)

    Returns:
        Combined DataFrame of all fetched candles
    """
    # Append USDT to make the full trading pair symbol
    full_symbol = f"{symbol.upper()}USDT"

    # Get Binance interval constant
    interval = TIMEFRAME_MAP.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # start_date arrives as either a "YYYY-MM-DD" string (first run) or a
    # datetime/Timestamp (incremental run / gap-fill). Normalize both into
    # a tz-aware UTC datetime so the millisecond math below is correct
    # either way, instead of assuming it's always a string.
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        # Already a datetime or pandas Timestamp. Treat any naive value as
        # UTC (the rest of the pipeline standardizes on naive-UTC), and
        # respect existing tzinfo if it's already tz-aware.
        start_dt = start_date
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

    # Likewise, end_date should be tz-aware UTC for the ms conversion below.
    # data_downloader.py passes this as naive UTC, so treat naive as UTC.
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_date.timestamp() * 1000)

    # Read retry settings from config
    retries     = config.get("retries", 5)
    retry_delay = config.get("retry_delay", 10)

    all_candles = []
    current_start_ms = start_ms

    logger.info(f"Fetching Binance candles for {full_symbol} | {timeframe} | from {start_date} to {end_date}")

    # ── Pagination loop ────────────────────────────────────────────────────────
    # Keep fetching 1000-candle batches until we pass the end date
    while current_start_ms < end_ms:
        attempt = 0
        batch = None

        # ── Retry loop ─────────────────────────────────────────────────────────
        while attempt < retries:
            try:
                batch = fetch_batch(full_symbol, interval, current_start_ms, end_ms)
                break  # success — exit retry loop

            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt}/{retries} failed for {full_symbol}: {e}")
                if attempt < retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {retries} attempts failed for {full_symbol}. Skipping batch.")
                    break

        # If batch is empty or None, we've reached the end
        if not batch:
            break

        all_candles.extend(batch)

        # Move start forward to after the last candle we just fetched
        # +1ms to avoid re-fetching the last candle
        current_start_ms = batch[-1][0] + 1

        logger.info(f"Fetched batch of {len(batch)} candles. Total so far: {len(all_candles)}")

        # Small delay to be respectful to the API rate limits
        time.sleep(0.1)

    if not all_candles:
        logger.warning(f"No candles fetched for {full_symbol}.")
        return pd.DataFrame()

    # Parse all raw candles into a clean DataFrame
    df = parse_candles(all_candles)
    logger.info(f"Total candles fetched for {full_symbol}: {len(df)}")

    return df