# crypto_pipeline/ml/data_prep/ml_utils.py

"""
ml_utils.py
-----------
Utility functions for ML module.

NOTE: setup_logging(), load_config_yaml(), validate_target_config(), and
validate_data_config() were removed (2026-07-17) -- confirmed unused
anywhere in the codebase (grep across ml/). setup_logging() in
particular was a same-named duplicate of the actually-used
ml/utils/logger.py::setup_logging(); every real call site imports from
there, not from here. validate_target_config() also validated against
a target schema (target.type: "return"/"log_return"/"binary"/
"threshold") that predates the current triple-barrier config shape
(target.horizon/upper_threshold/lower_threshold) and no longer matches
config.yaml's actual target section.
"""

import logging
import pandas as pd
from datetime import datetime
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()


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