# crypto_pipeline/ml_module/sentiment_pipeline.py

"""
sentiment_pipeline.py
---------------------
Collects and encodes sentiment data.
Merges sentiment columns with market data using timestamp alignment.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def collect_sentiment_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Collect sentiment data from configured sources and merge with market data.
    
    Args:
        df: Market OHLCV DataFrame with datetime column
        config: ML module config dict
        
    Returns:
        pd.DataFrame: Market data with sentiment columns merged
    """
    
    sentiment_config = config.get("sentiment", {})
    
    if not sentiment_config.get("enabled"):
        logger.info("Sentiment collection disabled")
        return df
    
    sources = sentiment_config.get("source", [])
    if not sources:
        logger.warning("No sentiment sources configured")
        return df
    
    df_with_sentiment = df.copy()
    
    sentiment_dfs = []
    for source in sources:
        logger.info(f"Collecting sentiment from {source}...")
        
        try:
            source_df = _fetch_sentiment_from_source(source, df)
            if source_df is not None:
                sentiment_dfs.append(source_df)
            else:
                logger.warning(f"No sentiment data returned from {source}")
                
        except Exception as e:
            logger.error(f"Failed to collect sentiment from {source}: {e}")
            continue
    
    if not sentiment_dfs:
        logger.warning("No sentiment data collected from any source")
        return df_with_sentiment
    
    combined_sentiment = pd.concat(sentiment_dfs, axis=1)
    
    encoding = sentiment_config.get("encoding", "numerical")
    combined_sentiment = _encode_sentiment(combined_sentiment, encoding)
    
    df_with_sentiment = _merge_sentiment(df_with_sentiment, combined_sentiment)
    
    logger.info(f"Sentiment data merged: {df_with_sentiment.shape[1]} total columns")
    return df_with_sentiment


def _fetch_sentiment_from_source(source: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch sentiment data from a source (twitter, news, reddit, etc).
    """
    
    logger.info(f"  Fetching {source} sentiment data...")
    
    from sentiment_pipeline.sentiment_model import get_sentiment_for_period
    
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have datetime column for sentiment merge")
    
    start_date = df["datetime"].min()
    end_date = df["datetime"].max()
    
    try:
        sentiment_df = get_sentiment_for_period(
            source=source,
            start_date=start_date,
            end_date=end_date
        )
        
        if sentiment_df is None or sentiment_df.empty:
            logger.warning(f"No sentiment data returned for {source}")
            return None
        
        logger.info(f"  {source}: {len(sentiment_df)} rows")
        return sentiment_df
        
    except Exception as e:
        logger.error(f"Error fetching {source} sentiment: {e}")
        return None


def _encode_sentiment(df: pd.DataFrame, encoding: str) -> pd.DataFrame:
    """
    Encode sentiment columns.
    """
    
    df_encoded = df.copy()
    
    if encoding == "numerical":
        for col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({
                "bullish": 1,
                "neutral": 0,
                "bearish": -1,
            })
            
        logger.info("Sentiment encoded as numerical: bullish=1, neutral=0, bearish=-1")
        
    elif encoding == "onehot":
        for col in df_encoded.columns:
            dummies = pd.get_dummies(
                df_encoded[col],
                prefix=col,
                prefix_sep="_"
            )
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        logger.info("Sentiment encoded as one-hot")
    
    else:
        logger.warning(f"Unknown encoding: {encoding}, returning original")
    
    return df_encoded


def _merge_sentiment(market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentiment data with market data using timestamp alignment.
    """
    
    if "datetime" not in market_df.columns:
        raise ValueError("Market data must have datetime column")
    
    market_indexed = market_df.set_index("datetime")
    
    if isinstance(sentiment_df.index, pd.DatetimeIndex):
        sentiment_indexed = sentiment_df
    else:
        if "datetime" in sentiment_df.columns:
            sentiment_indexed = sentiment_df.set_index("datetime")
        else:
            raise ValueError("Sentiment data must have datetime column or index")
    
    merged = market_indexed.join(sentiment_indexed, how="left")
    
    sentiment_cols = sentiment_df.columns
    for col in sentiment_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(method="ffill")
    
    merged = merged.reset_index()
    
    logger.info(f"Merged sentiment on datetime: {merged.shape}")
    
    return merged