# crypto_pipeline/ml_module/data_pipeline.py

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
        "1d" -> "1D"
        "1D" -> "1D"
        "15m" -> "15min"
        "15min" -> "15min"
    """
    timeframe = timeframe.lower().strip()
    
    if timeframe.endswith('d'):
        return timeframe.upper()
    
    if timeframe.endswith('h'):
        return timeframe
    
    if timeframe.endswith('m'):
        # Convert "15m" to "15min" for pandas compatibility
        return timeframe.replace('m', 'min')
    
    return timeframe


def collect_market_data(config: dict) -> pd.DataFrame:
    """
    Fetch market data from exchange and resample to configured timeframe.

    Note: this always fetches full OHLCV when data.enabled is True.
    config.data.calculate_ohlcv does NOT affect fetching -- indicators,
    patterns, and target generation all need real close/high/low/volume
    to compute from. calculate_ohlcv only controls whether the raw OHLCV
    columns are kept in the final output dataset (see main.py), after
    they've already been used upstream.

    Args:
        config: ML module config dict with data section
        
    Returns:
        pd.DataFrame: OHLCV data at the specified timeframe
        
    Raises:
        ValueError: If required config fields are missing or data collection disabled
    """
    
    data_config = config.get("data", {})
    
    if not data_config.get("enabled"):
        raise ValueError("Data collection is disabled in config")
    
    required = ["symbol", "exchange", "timeframe", "start_date", "end_date"]
    for field in required:
        if field not in data_config:
            raise ValueError(f"Missing required field in data config: {field}")
    
    symbol = data_config["symbol"]
    exchange = data_config["exchange"].lower()
    timeframe_raw = data_config["timeframe"]
    start_date = data_config["start_date"]
    end_date = data_config["end_date"]
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    timeframe_normalized = _normalize_timeframe(timeframe_raw)
    
    logger.info(f"Fetching {symbol} from {exchange} | {start_date} to {end_date} | timeframe: {timeframe_normalized}")
    
    result = get_data(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe_normalized,
    )
    
    df = result["resampled"]
    
    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            raise ValueError("No datetime column found in market data")
    
    logger.info(f"Market data collected: {len(df)} candles at {timeframe_normalized}")
    return df