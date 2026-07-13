# crypto_pipeline/ml_module/ml_utils.py

"""
ml_utils.py
-----------
Utility functions for ML module.
"""

import logging
import yaml
import pandas as pd
from datetime import datetime
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def setup_logging(name: str = "ml_module"):
    """Setup logging for ML module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{name}.log")
        ]
    )


def load_config_yaml(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_target_config(target_config: dict, model_type: str) -> bool:
    """Validate target configuration matches model type."""
    
    if not target_config:
        raise ValueError("Target config is empty")
    
    if model_type == "regression":
        valid_types = ["return", "log_return"]
        target_type = target_config.get("type", "return")
        if target_type not in valid_types:
            raise ValueError(f"Invalid regression target type: {target_type}")
    
    elif model_type == "classification":
        valid_types = ["binary", "threshold"]
        target_type = target_config.get("type", "binary")
        if target_type not in valid_types:
            raise ValueError(f"Invalid classification target type: {target_type}")
    
    return True


def validate_data_config(data_config: dict) -> bool:
    """Validate market data configuration."""
    
    required_fields = ["symbol", "exchange", "timeframe", "start_date", "end_date"]
    
    for field in required_fields:
        if field not in data_config:
            raise ValueError(f"Missing required field in data config: {field}")
    
    valid_exchanges = ["binance", "bybit"]
    if data_config["exchange"].lower() not in valid_exchanges:
        raise ValueError(f"Invalid exchange: {data_config['exchange']}")
    
    return True


def get_sentiment_for_period(source: str, start_date: datetime, end_date: datetime, symbol: str = None) -> pd.DataFrame:
    """
    Fetch aggregated sentiment data from sentiment_clean PostgreSQL schema.
    
    Args:
        source: "reddit", "twitter", "news" - maps to sentiment source
        start_date: Start datetime for query
        end_date: End datetime for query
        symbol: Optional - symbol like "btc", "eth" to filter specific coin
        
    Returns:
        pd.DataFrame with columns [datetime, sen_{SOURCE}] or None if no data found
    """
    
    logger = logging.getLogger(__name__)
    
    # Map symbol to coin table name
    coin_map = {
        "btc": "btc",
        "eth": "eth",
        "doge": "doge",
        "ada": "ada",
        "sol": "sol",
        "ltc": "ltc",
        "mina": "mina",
        "sui": "sui",
    }
    
    if symbol and symbol.lower() in coin_map:
        coin = coin_map[symbol.lower()]
    elif symbol:
        logger.warning(f"Symbol {symbol} not found in sentiment data")
        return None
    else:
        # Default to BTC if no symbol specified
        coin = "btc"
    
    table_name = f"sentiment_clean.{coin}_posts"
    
    try:
        # Open database connection
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        
        # Query sentiment aggregated by hour.
        # sentiment_label (Bullish/Neutral/Bearish) is mapped to 1/0/-1 per
        # post, then averaged per hour -> continuous value in [-1, 1]
        # reflecting how bullish/bearish that hour leaned.
        query = f"""
            SELECT 
                DATE_TRUNC('hour', created_utc) as datetime,
                AVG(
                    CASE LOWER(sentiment_label)
                        WHEN 'bullish' THEN 1
                        WHEN 'neutral' THEN 0
                        WHEN 'bearish' THEN -1
                    END
                ) as sentiment_score,
                COUNT(*) as post_count
            FROM {table_name}
            WHERE created_utc >= %s AND created_utc < %s
            GROUP BY DATE_TRUNC('hour', created_utc)
            ORDER BY datetime
        """
        
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        conn.close()
        
        if df.empty:
            logger.warning(f"No sentiment data found for {coin} between {start_date} and {end_date}")
            return None
        
        # Rename column for consistency
        col_name = f"sen_{source.upper()}"
        df = df.rename(columns={"sentiment_score": col_name})
        df = df.drop(columns=["post_count"])
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        
        logger.info(f"Fetched {len(df)} sentiment records for {coin} from {source}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {source} sentiment: {e}")
        return None