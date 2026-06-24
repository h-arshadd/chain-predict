"""
data_downloader.py
------------------
Core orchestrator for downloading OHLCV candle data.
Handles:
    - Parsing raw candles from the exchange into a DataFrame
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

from crypto_pipeline.utils.db_utils import (
    insert_candles, create_tables, get_last_timestamp, get_candles
)

logger = logging.getLogger(__name__)

TIMEFRAME_DELTA = pd.Timedelta(minutes=1)


class DataDownloader:

    def __init__(self, config, exchange_fetcher, conn):
        self.config = config
        self.exchange_fetcher = exchange_fetcher
        self.conn = conn

    def _strip_tz(self, value):
        if isinstance(value, (datetime, pd.Timestamp)) and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    def parse_candles(self, raw_candles):
        if not raw_candles:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "datetime": datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc).replace(tzinfo=None),
            "open":      float(c[1]),
            "high":      float(c[2]),
            "low":       float(c[3]),
            "close":     float(c[4]),
            "volume":    float(c[5]),
        } for c in raw_candles])

        return df

    def fill_missing_candles(self, df, expected_index, method):
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"Found {dup_count} duplicate timestamp(s) — dropping (keeping last).")
            df = df[~df.index.duplicated(keep="last")]

        df = df.reindex(expected_index)

        if method == "interpolation":
            df[["open", "high", "low", "close", "volume"]] = (
                df[["open", "high", "low", "close", "volume"]].interpolate(method="linear")
            )
            logger.info("Missing candles filled using linear interpolation.")

        elif method == "ffill":
            df[["open", "high", "low", "close", "volume"]] = (
                df[["open", "high", "low", "close", "volume"]].ffill()
            )
            logger.info("Missing candles filled using forward fill.")

        else:
            raise ValueError(f"Unknown filling method: {method}")

        return df

    def fill_zero_volume(self, df, method):
        if method == "ffill":
            zero_volume_mask = df["volume"] == 0
            zero_count = int(zero_volume_mask.sum())

            if zero_count:
                # Only replace volume — never touch OHLC values
                df.loc[zero_volume_mask, "volume"] = np.nan
                df["volume"] = df["volume"].ffill()
                logger.info(f"Zero-volume rows ({zero_count}) volume replaced using forward fill.")
            else:
                logger.info("No zero-volume rows found.")

        return df

    def resample(self, exchange, symbol, timeframe, df=None):
        """
        Resample 1-minute candles into a bigger timeframe (e.g. "5m", "15m", "1h", "1d").

        If df is not given, reads the stored 1m candles for the symbol from the DB.
        Pass df directly to resample data that isn't (or isn't only) in the DB yet.

        Returns a DataFrame with columns: date_time, open, high, low, close, volume
        """
        # if df is None:
        #     df = get_candles(self.conn, exchange, symbol)

        # if df.empty:
        #     logger.warning(f"No 1m data found for {exchange} | {symbol}. Cannot resample.")
        #     return df

        # df = df.set_index("date_time")

        # pandas resample needs "5min" not "5m" — only the minute suffix differs
        pandas_timeframe = timeframe.replace("m", "min") if timeframe.endswith("m") else timeframe

        resampled = df.resample(pandas_timeframe).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        })

        resampled = resampled.reset_index()
        logger.info(f"Resampled {exchange} | {symbol} into {timeframe}: {len(resampled)} candles")
        return resampled

    def clean_candles(self, df, actual_start, end_date, filling_method, zero_volume_method):
        """
        Apply the same cleaning steps used in download() to a raw 1m DataFrame:
        fill missing candles, fill zero volume, convert volume to % change, round.

        df must have a "date_time" column (not yet set as index).
        Returns a cleaned DataFrame with "date_time" as a column again.
        """
        df = df.set_index("date_time")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        expected_index = pd.date_range(start=actual_start, end=end_date - TIMEFRAME_DELTA, freq="1min", name="date_time")
        df = self.fill_missing_candles(df, expected_index, filling_method)
        df = self.fill_zero_volume(df, zero_volume_method)

        # Volume as % change vs previous candle (makes it comparable across coins)
        df["volume"] = df["volume"].pct_change().fillna(0) * 100

        # Rounding removed — store full precision from exchange

        return df.reset_index()

    def get_data(self, exchange, symbol, resample_timeframe, start_date=None, end_date=None, df_1m=False):
        """
        Get the most current data for a symbol, in both 1-minute and the
        requested bigger timeframe, even if the DB hasn't caught up yet.

        Reads whatever 1m data is already in the DB. If it's behind the
        current minute, fetches the missing recent candles directly from
        the exchange (without saving them), cleans them the same way
        download() does, and combines both into one continuous 1m series,
        then resamples that into the timeframe.

        Runtime only — nothing is written to the DB.

        Returns a dict: {"one_min": DataFrame, "resampled": DataFrame}
        """
        # filling_method = self.config["filling_missing_method"]
        # zero_volume_method = self.config["fill_zero_volume"]

        db_df = get_candles(self.conn, exchange, symbol)

        now = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        last_complete_minute = now - TIMEFRAME_DELTA

        db_last_timestamp = db_df["date_time"].max() if not db_df.empty else None

        is_stale = db_last_timestamp is None or db_last_timestamp < last_complete_minute

        if is_stale:
            fetch_start = (db_last_timestamp + TIMEFRAME_DELTA) if db_last_timestamp is not None else self.config["start_date"]
            logger.info(f"DB data for {exchange} | {symbol} is behind. Fetching live gap from {fetch_start} to {now}.")

            try:
                raw_candles = self.exchange_fetcher.fetch_candles(
                    symbol=symbol,
                    start_date=fetch_start,
                    end_date=now,
                    config=self.config
                )
                live_df = self.parse_candles(raw_candles)

                if not live_df.empty:
                    live_df = live_df.iloc[:-1]  # drop the still-forming current-minute candle
                    live_df = self.clean_candles(live_df, fetch_start, now, filling_method, zero_volume_method)

            except Exception as e:
                logger.error(f"Live fetch failed for {exchange} | {symbol}: {e}. Using DB data only.")
                live_df = pd.DataFrame()
        else:
            live_df = pd.DataFrame()

        one_min_df = pd.concat([db_df, live_df], ignore_index=True)
        one_min_df = one_min_df.drop_duplicates(subset="date_time", keep="last").sort_values("date_time")
        one_min_df = one_min_df.reset_index(drop=True)

        resampled_df = self.resample(exchange, symbol, timeframe, df=one_min_df)

        return {"one_min": one_min_df, "resampled": resampled_df}

    def download(self):
        config             = self.config
        exchange           = config["exchange"]
        symbols            = config["symbols"]
        config_start_date  = config["start_date"]
        filling_method     = config["filling_missing_method"]
        zero_volume_method = config["fill_zero_volume"]

        end_date = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        logger.info(f"Data fetch end_date resolved to: {end_date}")

        create_tables(self.conn, exchange, symbols=symbols)

        for symbol in symbols:
            logger.info(f"Starting download: {exchange} | {symbol}")

            try:
                last_timestamp = get_last_timestamp(self.conn, exchange, symbol)

                if last_timestamp:
                    actual_start = self._strip_tz(last_timestamp + TIMEFRAME_DELTA)
                    logger.info(f"Resuming from last stored timestamp: {actual_start}")
                else:
                    actual_start = config_start_date
                    logger.info(f"No existing data. Fetching from config start_date: {actual_start}")

                if isinstance(actual_start, (datetime, pd.Timestamp)) and actual_start >= end_date:
                    logger.info(f"Already up to date: {exchange} | {symbol}. Skipping.")
                    continue

                raw_candles = self.exchange_fetcher.fetch_candles(
                    symbol=symbol,
                    start_date=actual_start,
                    end_date=end_date,
                    config=config
                )

                if not raw_candles:
                    logger.warning(f"No data returned for {exchange} | {symbol}")
                    continue

                df = self.parse_candles(raw_candles)

                if df.empty:
                    logger.warning(f"Empty DataFrame after parsing for {exchange} | {symbol}")
                    continue

                df = df.iloc[:-1]
                logger.info(f"Dropped last (still-forming) candle. Total candles: {len(df)}")

                if df.empty:
                    logger.warning(f"No complete candles left after dropping still-forming candle for {exchange} | {symbol}.")
                    continue

                df = self.clean_candles(df, actual_start, end_date, filling_method, zero_volume_method)

                insert_candles(self.conn, exchange, symbol, df)
                logger.info(f"Completed: {exchange} | {symbol}")

            except Exception as e:
                logger.error(f"Failed for {exchange} | {symbol}: {e}")
                continue