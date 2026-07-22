# crypto_pipeline/data/data_downloader.py

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
    get_db_connection, insert_candles, create_tables, get_last_timestamp, get_candles_from_db
)

logger = logging.getLogger(__name__)

TIMEFRAME_DELTA = pd.Timedelta(minutes=1)
CANDLE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]

DEFAULT_FETCH_CONFIG = {"retries": 5, "retry_delay": 10}

_EXCHANGE_FETCHERS = {}


def _resolve_exchange_fetcher(exchange):
    """
    Map an exchange name string ("binance"/"bybit") to a fresh instance of
    its fetcher class.
    """
    if not _EXCHANGE_FETCHERS:
        from crypto_pipeline.data.binance.exchange_binance import BinanceExchange
        from crypto_pipeline.data.bybit.exchange_bybit import BybitExchange
        _EXCHANGE_FETCHERS["binance"] = BinanceExchange
        _EXCHANGE_FETCHERS["bybit"] = BybitExchange

    try:
        return _EXCHANGE_FETCHERS[exchange]()
    except KeyError:
        raise ValueError(f"Unknown exchange: {exchange!r}. Expected one of {list(_EXCHANGE_FETCHERS)}.")


def parse_candles(raw_candles):
    if not raw_candles:
        return pd.DataFrame()

    df = pd.DataFrame(raw_candles, columns=CANDLE_COLUMNS)
    df["datetime"] = pd.to_datetime(df["datetime"].astype(int), unit="ms", utc=True).dt.tz_localize(None)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df


def resample(timeframe, df):
    """
    Resample 1-minute candles into a bigger timeframe.
    timeframe must be a pandas-native offset string, e.g. "5min", "1h", "1D".

    df must already be indexed by datetime.
    Returns a DataFrame with columns: datetime, open, high, low, close, volume
    """
    resampled_df = df.resample(timeframe).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })

    resampled_df = resampled_df.reset_index()
    return resampled_df


def get_data(exchange, symbol, start_date, end_date, timeframe="1h", config=None, df_1m=False, drop_last_1m=True):
    """
    Get 1-minute and resampled data for a symbol between start_date and end_date.

    Self-contained: picks the right exchange fetcher and opens/closes its own
    DB connection internally, so callers just pass the exchange name as a
    string — no need to import/construct BinanceExchange, BybitExchange,
    DataDownloader, or a DB connection at the call site.

    end_date can be "now" (use the current time) or a fixed datetime.

    Reads whatever 1m data is already in the DB for [start_date, end_date].
    If the DB hasn't caught up to end_date yet, fetches the missing recent
    candles directly from the exchange (without saving them) and combines
    both into one continuous 1m series, then resamples that into the timeframe.

    Runtime only — nothing is written to the DB.

    Args:
        exchange:      "binance" or "bybit"
        symbol:        trading pair
        start_date:    start datetime
        end_date:      end datetime or "now"
        timeframe:     target timeframe for resampling (default "1h")
        config:        dict with at least "retries"/"retry_delay" for the live-gap
                       fetch fallback. Defaults to DEFAULT_FETCH_CONFIG if omitted.
        df_1m:         whether to return 1m data too
        drop_last_1m:  whether to drop the still-forming last 1-minute candle
                       before it's combined into the 1m series (default True).
                       The resampled series always has its still-forming last
                       candle dropped regardless of this flag. Simulator sets
                       this to False so it can see the in-progress 1m candle.

    Returns a dict: {"one_min": DataFrame, "resampled": DataFrame} or {"resampled": DataFrame}
    """
    config = config or DEFAULT_FETCH_CONFIG
    resample_timeframe = timeframe

    if end_date == "now":
        end_date = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)

    last_complete_minute = end_date - TIMEFRAME_DELTA

    conn = get_db_connection()
    try:
        db_last_timestamp = get_last_timestamp(conn, exchange, symbol)

        is_stale = db_last_timestamp is None or db_last_timestamp < last_complete_minute

        if is_stale:
            fetch_start = (db_last_timestamp + TIMEFRAME_DELTA) if db_last_timestamp is not None else start_date
            logger.info(f"DB data for {exchange} | {symbol} is behind. Fetching live gap from {fetch_start} to {end_date}.")

            try:
                exchange_fetcher = _resolve_exchange_fetcher(exchange)
                raw_candles = exchange_fetcher.fetch_candles(
                    symbol=symbol,
                    start_date=fetch_start,
                    end_date=end_date,
                    config=config
                )
                live_df = parse_candles(raw_candles)

                if not live_df.empty and drop_last_1m:
                    live_df = live_df.iloc[:-1]

            except Exception as e:
                logger.error(f"Live fetch failed for {exchange} | {symbol}: {e}. Using DB data only.")
                live_df = pd.DataFrame()
        else:
            live_df = pd.DataFrame()

        db_df = get_candles_from_db(conn, exchange, symbol, start_date, end_date)
    finally:
        conn.close()

    one_min_df = pd.concat([db_df, live_df], ignore_index=True)

    resampled_df = resample(resample_timeframe, df=one_min_df.set_index("datetime"))

    if not resampled_df.empty:
        resampled_df = resampled_df.iloc[:-1]

    logger.info(f"Resampled {exchange} | {symbol} into {resample_timeframe}: {len(resampled_df)} candles")

    if df_1m:
        return {"one_min": one_min_df, "resampled": resampled_df}
    return {"resampled": resampled_df}


class DataDownloader:
    """
    Used for the backfill/incremental download() pipeline only (writes to DB).
    For read-only analysis/plotting, use the standalone get_data() function
    above instead — it doesn't need this class at all.
    """

    def __init__(self, config, exchange_fetcher, conn):
        self.config = config
        self.exchange_fetcher = exchange_fetcher
        self.conn = conn

    def _strip_tz(self, value):
        if isinstance(value, (datetime, pd.Timestamp)) and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

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
                df.loc[zero_volume_mask, "volume"] = np.nan
                df["volume"] = df["volume"].ffill()
                df["volume"] = df["volume"].bfill()
                logger.info(f"Zero-volume rows ({zero_count}) volume replaced using forward fill + backfill.")
            else:
                logger.info("No zero-volume rows found.")

        return df

    def clean_candles(self, df, actual_start, end_date, filling_method, zero_volume_method):
        """
        Apply the same cleaning steps used in download() to a raw 1m DataFrame:
        fill missing candles, fill zero volume, convert volume to % change, round.

        df must have a "datetime" column (not yet set as index).
        Returns a cleaned DataFrame with "datetime" as a column again.
        """
        df = df.set_index("datetime")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        expected_index = pd.date_range(start=actual_start, end=end_date - TIMEFRAME_DELTA, freq="1min", name="datetime")
        df = self.fill_missing_candles(df, expected_index, filling_method)
        df = self.fill_zero_volume(df, zero_volume_method)

        return df.reset_index()

    def download(self):
        config             = self.config
        exchange           = config["exchange"]
        symbols            = config["symbols"]
        config_start_date  = config["start_date"]
        filling_method     = config["filling_missing_method"]
        zero_volume_method = config["fill_zero_volume"]
        time_horizon       = config.get("time_horizon", "1m")

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

                df = parse_candles(raw_candles)

                if df.empty:
                    logger.warning(f"Empty DataFrame after parsing for {exchange} | {symbol}")
                    continue

                df = df.iloc[:-1]
                logger.info(f"Dropped last (still-forming) candle. Total candles: {len(df)}")

                if df.empty:
                    logger.warning(f"No complete candles left after dropping still-forming candle for {exchange} | {symbol}.")
                    continue

                df = self.clean_candles(df, actual_start, end_date, filling_method, zero_volume_method)

                if time_horizon == "1m":
                    insert_candles(self.conn, exchange, symbol, df)
                    logger.info(f"Stored 1m candles for {exchange} | {symbol}")
                else:
                    resampled = resample(time_horizon, df.set_index("datetime"))
                    if resampled.empty:
                        logger.warning(f"Resampled DataFrame is empty for {exchange} | {symbol} | {time_horizon}")
                        continue
                    
                    insert_candles(self.conn, exchange, symbol, resampled)
                    logger.info(f"Stored {time_horizon} candles for {exchange} | {symbol} ({len(resampled)} candles)")

            except Exception as e:
                logger.error(f"Failed for {exchange} | {symbol}: {e}")
                continue