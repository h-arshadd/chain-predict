"""
data_pipeline.py
----------------
Collects market data (OHLCV) from existing data infrastructure.
Handles dynamic timeframe resampling based on config.
"""

import logging
import pandas as pd
from datetime import datetime
from crypto_pipeline.data.data_downloader import get_data

logger = logging.getLogger(__name__)


def _normalize_timeframe(timeframe: str) -> str:
    """
    Normalize timeframe string to pandas-compatible format.
    
    Examples:
        "1h" -> "1h"
        "1H" -> "1h"
        "4h" -> "4h"
        "1d" -> "1D"  (pandas wants uppercase D)
        "1D" -> "1D"
    """
    timeframe = timeframe.lower()
    
    # Handle day/D edge case
    if timeframe.endswith('d'):
        return timeframe.upper()
    
    return timeframe


def collect_market_data(config: dict) -> pd.DataFrame:
    """
    Fetch market data from exchange and resample to configured timeframe.
    
    Args:
        config: ML module config dict with data section
        
    Returns:
        pd.DataFrame: OHLCV data at the specified timeframe
        
    Raises:
        ValueError: If required config fields are missing
    """
    
    data_config = config.get("data", {})
    
    if not data_config.get("enabled"):
        raise ValueError("Data collection is disabled")
    
    # Validate required fields
    required = ["symbol", "exchange", "timeframe", "start_date", "end_date"]
    for field in required:
        if field not in data_config:
            raise ValueError(f"Missing required field in data config: {field}")
    
    symbol = data_config["symbol"]
    exchange = data_config["exchange"].lower()
    timeframe_raw = data_config["timeframe"]
    start_date = data_config["start_date"]
    end_date = data_config["end_date"]
    
    # Parse dates if they're strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    logger.info(f"Fetching {symbol} from {exchange} | {start_date} to {end_date}")
    
    # Fetch 1-minute data
    result = get_data(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    
    df = result["resampled"]  # Already resampled to 1h by default in get_data
    
    # Resample to target timeframe if different from 1h
    timeframe_normalized = _normalize_timeframe(timeframe_raw)
    
    if timeframe_normalized != "1h":
        logger.info(f"Resampling from 1h to {timeframe_normalized}...")
        df = _resample_ohlcv(df, timeframe_normalized)
    
    # Ensure datetime column exists
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError("No datetime column found in market data")
    
    logger.info(f"Market data collected: {len(df)} candles at {timeframe_normalized}")
    return df


def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a different timeframe.
    
    Args:
        df: DataFrame with datetime column and OHLCV
        timeframe: Target timeframe (e.g., "1h", "4h", "1D")
        
    Returns:
        pd.DataFrame: Resampled OHLCV data
    """
    
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have datetime column")
    
    # Set datetime as index if not already
    if df.index.name != "datetime":
        df = df.set_index("datetime")
    
    # Resample using OHLCV aggregation
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    
    # Reset index to get datetime as column again
    resampled = resampled.reset_index()
    
    # Drop incomplete last candle
    resampled = resampled.iloc[:-1]
    
    return resampled