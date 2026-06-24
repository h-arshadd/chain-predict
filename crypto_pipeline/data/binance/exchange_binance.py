"""
exchange_binance.py
-------------------
Handles all Binance API interactions for fetching OHLCV candle data.
Uses the python-binance library.

Key details:
    - Binance timestamps are in MILLISECONDS
    - Uses futures klines (linear perpetual contracts)
    - Fetches 1000 candles per API call (Binance max)
    - Retries on failure with configurable delay
"""

import time
import logging
from datetime import datetime, timezone
from binance.client import Client

logger = logging.getLogger(__name__)

INTERVAL = Client.KLINE_INTERVAL_1MINUTE


class BinanceExchange:

    def __init__(self):
        # No API key needed for public market data
        self.client = Client()

    def fetch_batch(self, symbol, start_ms, end_ms):
        """
        Fetch a single batch of up to 1000 raw candles from Binance Futures.
        Uses futures_klines (linear perpetual contracts).
        """
        return self.client.get_klines(
            symbol=symbol,
            interval=INTERVAL,
            startTime=start_ms,
            endTime=end_ms,
            limit=1000
        )

    def fetch_candles(self, symbol, start_date, end_date, config):
        """
        Fetch all raw OHLCV candles for a symbol from Binance in batches.

        Since Binance only returns 1000 candles per call, we loop — moving the
        start time forward after each batch — until we reach the end date.

        start_date arrives as either a "YYYY-MM-DD" string (first run) or a
        datetime/Timestamp (incremental run). Both are handled here.

        Returns raw candle lists directly from the Binance API.
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

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms   = int(end_date.timestamp() * 1000)

        retries     = config.get("retries", 5)
        retry_delay = config.get("retry_delay", 10)

        all_candles = []
        current_start_ms = start_ms

        logger.info(f"Fetching Binance candles for {full_symbol} | from {start_date} to {end_date}")

        while current_start_ms < end_ms:
            attempt = 0
            batch = None

            while attempt < retries:
                try:
                    batch = self.fetch_batch(full_symbol, current_start_ms, end_ms)
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
            current_start_ms = batch[-1][0] + 1  # +1ms to avoid re-fetching last candle

            logger.info(f"Fetched batch of {len(batch)} candles. Total so far: {len(all_candles)}")
            time.sleep(0.1)

        if not all_candles:
            logger.warning(f"No candles fetched for {full_symbol}.")
            return []

        logger.info(f"Total raw candles fetched for {full_symbol}: {len(all_candles)}")
        return all_candles