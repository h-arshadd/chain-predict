"""
exchange_bybit.py
-----------------
Handles all Bybit API interactions for fetching OHLCV candle data.
Uses the pybit library.

Key details:
    - Bybit timestamps are in MILLISECONDS
    - Fetches 200 candles per API call (Bybit max)
    - Retries on failure with configurable delay
    - Bybit returns candles NEWEST FIRST, so unlike Binance we paginate
      backward from the end date instead of forward from the start
"""

import time
import logging
from datetime import datetime, timezone
from pybit.unified_trading import HTTP

from crypto_pipeline.data.data_downloader import CANDLE_COLUMNS

logger = logging.getLogger(__name__)

INTERVAL = "1"          # Bybit's code for 1-minute candles
STEP_SECONDS = 60        # length of one candle, used to step the window back


class BybitExchange:

    def __init__(self):
        # No API key needed for public market data
        self.client = HTTP()

    def fetch_batch(self, symbol, start_sec, end_sec):
        """
        Fetch a single batch of up to 1000 raw candles from Bybit.

        Bybit's kline rows have one extra field (turnover) that isn't part
        of our schema, so we only keep as many fields as CANDLE_COLUMNS
        expects — this stays correct automatically if that schema changes.
        """
        response = self.client.get_kline(
            category="linear",
            symbol=symbol,
            interval=INTERVAL,
            start=start_sec * 1000,
            end=end_sec * 1000,
            limit=1000
        )

        if response["retCode"] != 0:
            raise Exception(f"Bybit API error: {response['retMsg']}")

        return [row[:len(CANDLE_COLUMNS)] for row in response["result"]["list"]]

    def fetch_candles(self, symbol, start_date, end_date, config):
        """
        Fetch all raw OHLCV candles for a symbol from Bybit in batches.

        Bybit returns the newest candles first inside a [start, end] window,
        so we page by walking the end of the window backward after each
        batch, then reverse everything into chronological order at the end.

        start_date arrives as either a "YYYY-MM-DD" string (first run) or a
        datetime/Timestamp (incremental run). Both are handled here.

        Returns raw candle lists, oldest to newest, ready for the same
        parse_candles() used for Binance.
        """
        full_symbol = f"{symbol.upper()}USDT"

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

        retries     = config["retries"]
        retry_delay = config["retry_delay"]

        all_candles = []
        current_end_sec = end_sec

        logger.info(f"Fetching Bybit candles for {full_symbol} | from {start_date} to {end_date}")

        while current_end_sec > start_sec:
            attempt = 0
            batch = None

            while attempt < retries:
                try:
                    batch = self.fetch_batch(full_symbol, start_sec, current_end_sec)
                    break

                except Exception as e:
                    attempt += 1
                    logger.warning(f"Attempt {attempt}/{retries} failed for {full_symbol}: {e}")
                    if attempt < retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All {retries} attempts failed for {full_symbol}. Skipping batch.")
                        break

            if not batch:
                break

            all_candles.extend(batch)

            # Batch is newest-first, so the oldest candle is the last item.
            # Step the window's end back to just before it.
            oldest_candle_sec = int(batch[-1][0]) // 1000
            current_end_sec = oldest_candle_sec - STEP_SECONDS

            batch_start = datetime.fromtimestamp(int(batch[-1][0]) / 1000, tz=timezone.utc)
            batch_end   = datetime.fromtimestamp(int(batch[0][0]) / 1000, tz=timezone.utc)
            logger.info(
                f"{full_symbol} | fetched batch of {len(batch)} candles "
                f"({batch_start} -> {batch_end}). Total so far: {len(all_candles)}"
            )
            time.sleep(0.1)

        if not all_candles:
            logger.warning(f"No candles fetched for {full_symbol}.")
            return []

        # Bybit gives candles newest-first — reverse to chronological order
        # so parse_candles() in data_downloader.py gets the same order Binance gives it.
        all_candles.reverse()

        logger.info(f"Total raw candles fetched for {full_symbol}: {len(all_candles)}")
        return all_candles