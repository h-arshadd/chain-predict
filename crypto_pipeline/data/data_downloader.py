"""
data_downloader.py
------------------
Core orchestrator for downloading OHLCV candle data.
Handles:
    - Incremental loading (resumes from last stored timestamp)
    - Generating a complete expected date range
    - Detecting and filling missing candles via interpolation or ffill
    - Forward filling zero volumes
    - Calling the correct exchange fetcher (Binance or Bybit)
    - Storing clean data into PostgreSQL
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Set up logger for this module
logger = logging.getLogger(__name__)

# Maps each timeframe to the correct pandas Timedelta for one candle.
# Used when computing actual_start for incremental runs so we step forward
# by exactly one candle interval, not always one minute.
_TIMEFRAME_DELTA = {
    "1m":  pd.Timedelta(minutes=1),
    "5m":  pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h":  pd.Timedelta(hours=1),
    "1d":  pd.Timedelta(days=1),
}


class DataDownloader:
    """
    Orchestrates the full OHLCV data download pipeline for one exchange.

    Handles incremental loading, gap detection and filling, zero-volume
    correction, and insertion into PostgreSQL.
    """

    def __init__(self, config: dict, exchange_fetcher, conn):
        """
        Args:
            config           : parsed yml config dictionary
            exchange_fetcher : module with fetch_candles() function (binance or bybit)
            conn             : active psycopg2 database connection
        """
        self.config = config
        self.exchange_fetcher = exchange_fetcher
        self.conn = conn

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_tz(value):
        """
        Strip timezone info from a datetime-like value, returning it timezone-naive.

        Accepts a python datetime, a pandas Timestamp, or a date string, and
        always returns something safe to mix with other naive datetimes in
        pandas functions like pd.date_range(). Strings and None pass through
        unchanged since pd.date_range / pd.Timestamp handle naive date strings
        fine, and there is nothing to strip from them.

        This exists because pandas raises "Start and end cannot both be
        tz-aware with different timezones" (or behaves inconsistently) if one
        side of a date range is tz-aware and the other isn't, or if they're
        aware with different timezones. Centralizing the stripping logic here
        means every call site gets the same treatment instead of relying on
        scattered isinstance checks that only catch plain `datetime`, not
        pandas Timestamps (which are a subclass, but it's easy to miss a spot).
        """
        if isinstance(value, (datetime, pd.Timestamp)) and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    # ── Date range generation ──────────────────────────────────────────────────

    def generate_expected_index(self, start_date, end_date: datetime, timeframe: str) -> pd.DatetimeIndex:
        """
        Generate a complete continuous datetime index from start to end.
        This is what a perfect dataset with zero gaps looks like.
        All datetimes are timezone-naive for consistency.

        Args:
            start_date : datetime object or date string e.g. "2023-01-01"
            end_date   : datetime object for end of range
            timeframe  : e.g. "1m" -> maps to pandas frequency "1min"

        Returns:
            pd.DatetimeIndex of all expected timestamps
        """
        freq_map = {
            "1m":  "1min",
            "5m":  "5min",
            "15m": "15min",
            "1h":  "1h",
            "1d":  "1D"
        }

        freq = freq_map.get(timeframe)
        if not freq:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Strip timezone info from both ends to keep everything consistent.
        # Handles plain datetimes and pandas Timestamps alike.
        start_date = self._strip_tz(start_date)
        end_date = self._strip_tz(end_date)

        return pd.date_range(start=start_date, end=end_date, freq=freq)

    # ── Missing data handling ──────────────────────────────────────────────────

    def fill_missing_candles(self, df: pd.DataFrame, expected_index: pd.DatetimeIndex, method: str) -> pd.DataFrame:
        """
        Detect and fill missing candles in the dataframe.

        Missing candles happen when there's no trading activity at a given minute.
        ML models need continuous data so we can't leave gaps.

        Two strategies:
            - interpolation : estimate missing values by averaging neighbors
            - ffill         : copy the previous row's values forward

        Args:
            df             : raw dataframe with date_time as index
            expected_index : complete continuous datetime index (no gaps)
            method         : "interpolation" or "ffill"

        Returns:
            DataFrame with all gaps filled
        """
        # Defensive: reindexing fails if the DataFrame's index has duplicate
        # timestamps (can happen if an exchange fetcher returns overlapping
        # candles at batch boundaries). Drop exact duplicates before reindexing
        # so a single upstream hiccup doesn't crash the whole symbol/timeframe.
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(
                f"Found {dup_count} duplicate timestamp(s) in fetched data before "
                f"reindexing — dropping duplicates (keeping last) to avoid a crash."
            )
            df = df[~df.index.duplicated(keep="last")]

        df = df.reindex(expected_index)

        if method == "interpolation":
            df[["open", "high", "low", "close", "volume"]] = (
                df[["open", "high", "low", "close", "volume"]]
                .interpolate(method="linear")
            )
            logger.info("Missing candles filled using linear interpolation.")

        elif method == "ffill":
            df[["open", "high", "low", "close", "volume"]] = (
                df[["open", "high", "low", "close", "volume"]]
                .ffill()
            )
            logger.info("Missing candles filled using forward fill.")

        else:
            raise ValueError(f"Unknown filling method: {method}")

        return df

    def fill_zero_volume(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Handle candles where volume is zero.

        Zero volume means no trades happened during that minute — the
        open/high/low/close values the exchange reports for such a candle are
        not real trade prices, they're just carried-forward placeholders
        (usually the previous close repeated across all four). Since OHLC is
        already a placeholder in that case, we forward-fill the WHOLE row
        (OHLC + volume) for zero-volume candles, not just volume on its own.
        Forward-filling volume alone while leaving OHLC untouched would let a
        real (non-zero-volume) row's prices sit next to a stale/forward-filled
        volume number, which is internally inconsistent for an ML dataset.

        Args:
            df     : dataframe with open/high/low/close/volume columns
            method : currently only "ffill" is supported

        Returns:
            DataFrame with zero-volume rows fully forward-filled (OHLC + volume)
        """
        if method == "ffill":
            zero_volume_mask = df["volume"] == 0
            zero_count = int(zero_volume_mask.sum())

            if zero_count:
                # Mark the whole row as NaN where volume is zero, then forward
                # fill across OHLC + volume together so the row stays internally
                # consistent (same source candle repeated, not a frankenstein
                # mix of a stale volume with a fresh price).
                df.loc[zero_volume_mask, ["open", "high", "low", "close", "volume"]] = np.nan
                df[["open", "high", "low", "close", "volume"]] = (
                    df[["open", "high", "low", "close", "volume"]].ffill()
                )
                logger.info(
                    f"Zero-volume rows ({zero_count}) replaced using forward fill "
                    f"across OHLC + volume together."
                )
            else:
                logger.info("No zero-volume rows found.")

        return df

    # ── Main downloader ────────────────────────────────────────────────────────

    def download(self):
        """
        Main method that orchestrates the full data download pipeline for one exchange.

        Flow for each symbol:
            1. Check last stored timestamp (incremental loading)
            2. Fetch only new candles from last timestamp onwards
            3. Drop the last candle if end_date is "now" (still forming)
            4. Set date_time as DataFrame index
            5. Generate a gapless expected datetime index
            6. Fill any missing candles within fetched range
            7. Fill zero volumes
            8. Reset index
            9. Insert clean data into PostgreSQL
        """
        from crypto_pipeline.utils.db_utils import (
            insert_candles, create_tables,
            get_last_timestamp
        )

        config             = self.config
        exchange           = config["exchange"]
        symbols            = config["symbols"]
        timeframes         = config["time_horizons"]
        config_start_date  = config["start_date"]
        filling_method     = config["filling_missing_method"]
        zero_volume_method = config["fill_zero_volume"]
        config_end_date    = config.get("end_date", "now")

        # Resolve end_date from config.
        # If the config says "now" (or is missing), use the current UTC time.
        # Otherwise parse the "YYYY-MM-DD" string from the config so the pipeline
        # fetches only up to that date — not beyond it.
        # All datetimes are kept timezone-naive (naive-UTC) to stay consistent
        # with psycopg2 TIMESTAMP columns and the rest of the pipeline.
        if config_end_date == "now" or not config_end_date:
            end_date = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        else:
            end_date = datetime.strptime(config_end_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=0, microsecond=0
            )

        logger.info(f"Data fetch end_date resolved to: {end_date} (config value: '{config_end_date}')")

        # Ensure tables exist for exactly the symbols/timeframes in the config
        create_tables(self.conn, exchange, symbols=symbols, timeframes=timeframes)

        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Starting download: {exchange} | {symbol} | {timeframe}")

                try:
                    # ── Step 1: Incremental loading check ─────────────────────────
                    last_timestamp = get_last_timestamp(self.conn, exchange, symbol, timeframe)

                    if last_timestamp:
                        actual_start = last_timestamp + _TIMEFRAME_DELTA[timeframe]
                        actual_start = self._strip_tz(actual_start)
                        logger.info(f"Resuming from last stored timestamp: {actual_start}")
                    else:
                        actual_start = config_start_date
                        logger.info(f"No existing data. Fetching from config start_date: {actual_start}")

                    # Skip if already up to date.
                    # actual_start can be a string (first run) or a
                    # datetime/Timestamp (incremental run) — only compare when
                    # it's actually datetime-like, otherwise this check doesn't
                    # apply and we proceed to fetch.
                    if isinstance(actual_start, (datetime, pd.Timestamp)) and actual_start >= end_date:
                        logger.info(f"Already up to date: {exchange} | {symbol} | {timeframe}. Skipping.")
                        continue

                    # ── Step 2: Fetch new candles ──────────────────────────────────
                    df = self.exchange_fetcher.fetch_candles(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=actual_start,
                        end_date=end_date,
                        config=config
                    )

                    if df is None or df.empty:
                        logger.warning(f"No data returned for {exchange} | {symbol} | {timeframe}")
                        continue

                    # ── Step 3: Drop last candle if still forming ─────────────────
                    # Only drop the last candle when we're fetching up to "now",
                    # because that candle is still open/forming. When end_date is
                    # a fixed historical date from the config, the last candle is
                    # already complete and must be kept.
                    if config_end_date == "now" or not config_end_date:
                        df = df.iloc[:-1]
                        logger.info(f"Dropped last (still-forming) candle. Total candles: {len(df)}")
                    else:
                        logger.info(f"Historical end_date set — keeping all candles. Total candles: {len(df)}")

                    if df.empty:
                        logger.warning(
                            f"No complete candles left after dropping the still-forming "
                            f"candle for {exchange} | {symbol} | {timeframe}."
                        )
                        continue

                    # ── Step 4: Set date_time as index ─────────────────────────────
                    df = df.set_index("date_time")
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                    # ── Step 5: Generate gapless expected index ────────────────────
                    expected_index = self.generate_expected_index(actual_start, end_date, timeframe)

                    # ── Step 6: Fill missing candles ───────────────────────────────
                    df = self.fill_missing_candles(df, expected_index, filling_method)

                    # ── Step 7: Fix zero volumes ───────────────────────────────────
                    df = self.fill_zero_volume(df, zero_volume_method)

                    # ── Step 8: Reset index ────────────────────────────────────────
                    df = df.reset_index().rename(columns={"index": "date_time"})

                    # ── Step 9: Insert into PostgreSQL ─────────────────────────────
                    insert_candles(self.conn, exchange, symbol, timeframe, df)

                    logger.info(f"Completed: {exchange} | {symbol} | {timeframe}")

                except Exception as e:
                    logger.error(f"Failed for {exchange} | {symbol} | {timeframe}: {e}")
                    continue


# ── Module-level convenience function (backward-compatible wrapper) ────────────

def download_data(config: dict, exchange_fetcher, conn):
    """
    Main function that orchestrates the full data download pipeline for one exchange.

    Args:
        config           : parsed yml config dictionary
        exchange_fetcher : module with fetch_candles() function (binance or bybit)
        conn             : active psycopg2 database connection
    """
    downloader = DataDownloader(config=config, exchange_fetcher=exchange_fetcher, conn=conn)
    downloader.download()
