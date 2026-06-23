"""
exchange_bybit.py
-----------------
Handles all Bybit API interactions for fetching OHLCV candle data.
Uses the pybit library.

Key details:
    - Bybit timestamps are in MILLISECONDS in the response — we convert to datetime
    - Uses linear perpetual contracts (linear category)
    - Fetches 200 candles per API call (Bybit max)
    - Retries on failure with configurable delay
    - Bybit returns candles NEWEST FIRST — we must account for that during
      pagination, not just at the end when we build the DataFrame
"""

import time
import logging
import pandas as pd
from datetime import datetime, timezone
from pybit.unified_trading import HTTP

# Set up logger for this module
logger = logging.getLogger(__name__)

# Bybit client — no API key needed for public market data
client = HTTP()


# ── Timeframe mapping ──────────────────────────────────────────────────────────

# Map our config timeframe strings to Bybit interval strings
TIMEFRAME_MAP = {
    "1m":  "1",
    "5m":  "5",
    "15m": "15",
    "1h":  "60",
    "1d":  "D",
}


# ── Raw candle fetcher ─────────────────────────────────────────────────────────

def fetch_batch(symbol: str, interval: str, start_sec: int, end_sec: int) -> list:
    """
    Fetch a single batch of up to 200 candles from Bybit.
    Uses linear category (linear perpetual contracts).

    Args:
        symbol    : trading pair e.g. "DOGEUSDT"
        interval  : Bybit interval string e.g. "1"
        start_sec : start timestamp in SECONDS
        end_sec   : end timestamp in SECONDS

    Returns:
        List of raw candle data from Bybit API, NEWEST FIRST (as Bybit returns it)
    """
    response = client.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        start=start_sec * 1000,   # Bybit API actually expects ms here
        end=end_sec * 1000,
        limit=200                  # maximum allowed per call
    )

    # Bybit response structure: response["result"]["list"]
    if response["retCode"] != 0:
        raise Exception(f"Bybit API error: {response['retMsg']}")

    return response["result"]["list"]


# ── Candle parser ──────────────────────────────────────────────────────────────

def parse_candles(raw_candles: list) -> pd.DataFrame:
    """
    Parse raw Bybit API response into a clean OHLCV DataFrame.

    Bybit returns each candle as a list:
        [start_time, open, high, low, close, volume, turnover]

    Important: Bybit returns candles in DESCENDING order (newest first)
    so we sort them into chronological order here.

    Timestamps from Bybit are in MILLISECONDS in the response
    — divide by 1000 to get seconds, then convert to UTC datetime.

    Args:
        raw_candles : list of raw candle lists from Bybit API

    Returns:
        Clean pandas DataFrame with columns [date_time, open, high, low, close, volume],
        sorted oldest -> newest, with any exact-duplicate timestamps removed.
    """
    records = []
    for candle in raw_candles:
        records.append({
            # Bybit start_time is in milliseconds → convert to UTC datetime
            "date_time": datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc).replace(tzinfo=None),
            "open":      float(candle[1]),
            "high":      float(candle[2]),
            "low":       float(candle[3]),
            "close":     float(candle[4]),
            "volume":    float(candle[5]),
        })

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Bybit returns newest first — sort to get chronological order.
    # Pagination batches can overlap at the edges (see fetch_candles below),
    # so we also drop any duplicate timestamps here as a safety net before
    # this DataFrame is ever reindexed against an expected_index elsewhere.
    df = df.sort_values("date_time").drop_duplicates(subset="date_time", keep="last")
    df = df.reset_index(drop=True)

    return df


# ── Main fetch function ────────────────────────────────────────────────────────

def fetch_candles(symbol: str, timeframe: str, start_date, end_date: datetime, config: dict) -> pd.DataFrame:
    """
    Fetch all historical OHLCV candles for a symbol from Bybit in batches.

    IMPORTANT — how Bybit's get_kline actually paginates:
    Given a [start, end] window, Bybit returns the MOST RECENT `limit`
    candles inside that window (newest first), not the earliest ones. This
    means the only reliable way to page through a large range is to walk
    the END of the window BACKWARD after each batch — not push the start
    forward. Pushing start forward (as a naive implementation might) barely
    shrinks the effective window near the end date and causes the same
    nearly-identical batch to be re-fetched over and over.

    So this function fetches from the end of the range backward in chunks
    of up to 200 candles, then reverses everything into chronological order
    at the end.

    Args:
        symbol     : coin name from config e.g. "doge"
        timeframe  : e.g. "1m"
        start_date : start date as either a "YYYY-MM-DD" string (used on a
                     symbol's very first run, straight from the yml config)
                     or a datetime/pandas Timestamp (used on incremental
                     runs and gap-fills, computed from the last stored
                     timestamp). Both forms are handled here.
        end_date   : datetime object for end of range
        config     : full config dictionary (for retries, delay, etc.)

    Returns:
        Combined DataFrame of all fetched candles, chronologically sorted,
        with duplicate timestamps removed.
    """
    # Append USDT to make the full trading pair symbol
    full_symbol = f"{symbol.upper()}USDT"

    # Get Bybit interval string
    interval = TIMEFRAME_MAP.get(timeframe)
    if not interval:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # start_date arrives as either a "YYYY-MM-DD" string (first run) or a
    # datetime/Timestamp (incremental run / gap-fill). Normalize both into
    # a tz-aware UTC datetime, same approach as exchange_binance.py.
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = start_date
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    start_sec = int(start_dt.timestamp())
    end_sec   = int(end_date.timestamp())

    # Read retry settings from config
    retries     = config.get("retries", 5)
    retry_delay = config.get("retry_delay", 10)

    # Candle duration in seconds, used to step the window backward by one
    # candle so we don't re-fetch the oldest candle of the previous batch.
    timeframe_seconds_map = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "1d": 86400,
    }
    step_seconds = timeframe_seconds_map.get(timeframe, 60)

    all_candles = []
    current_end_sec = end_sec

    logger.info(f"Fetching Bybit candles for {full_symbol} | {timeframe} | from {start_date} to {end_date}")

    # ── Pagination loop ────────────────────────────────────────────────────────
    # Walk backward from end_sec toward start_sec, one ~200-candle batch at a time.
    while current_end_sec > start_sec:
        attempt = 0
        batch = None

        # ── Retry loop ─────────────────────────────────────────────────────────
        while attempt < retries:
            try:
                batch = fetch_batch(full_symbol, interval, start_sec, current_end_sec)
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

        # If batch is empty or None, there's nothing left in this range.
        if not batch:
            break

        all_candles.extend(batch)

        # Bybit returns this batch NEWEST FIRST, so the OLDEST candle in the
        # batch is the LAST item in the list. Step the window's end back to
        # just before that candle, so the next call asks for everything
        # strictly older than what we already have.
        oldest_candle_in_batch_ms = int(batch[-1][0])
        oldest_candle_in_batch_sec = oldest_candle_in_batch_ms // 1000
        current_end_sec = oldest_candle_in_batch_sec - step_seconds

        logger.info(f"Fetched batch of {len(batch)} candles. Total so far: {len(all_candles)}")

        # Small delay to be respectful to the API rate limits
        time.sleep(0.1)

    if not all_candles:
        logger.warning(f"No candles fetched for {full_symbol}.")
        return pd.DataFrame()

    # Parse all raw candles into a clean, sorted, de-duplicated DataFrame.
    # parse_candles() sorts ascending and drops duplicate timestamps, which
    # also cleans up any small overlap at batch boundaries.
    df = parse_candles(all_candles)
    logger.info(f"Total candles fetched for {full_symbol}: {len(df)}")

    return df