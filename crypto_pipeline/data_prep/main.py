# crypto_pipeline/data_prep/main.py

"""
main.py
-------
Entry point for the ML Module.
Orchestrates configuration loading and the complete data preparation pipeline
for regression and classification tasks.
"""

import logging
import pandas as pd

from crypto_pipeline.data_prep.ml_utils import load_config_yaml
from crypto_pipeline.data_prep.data_pipeline import collect_market_data
from crypto_pipeline.data_prep.feature_pipeline import engineer_features
from crypto_pipeline.data_prep.sentiment_pipeline import collect_sentiment_data
from crypto_pipeline.data_prep.target_pipeline import generate_target


logger = logging.getLogger(__name__)


def run_ml_pipeline(config_path: str) -> pd.DataFrame:
    """
    Execute the complete ML data preparation pipeline.
    
    Steps:
    1. Collect market data (if enabled and calculate_ohlcv is true)
    2. Engineer features (if enabled)
    3. Collect sentiment data (if enabled)
    4. Generate prediction target
    
    Args:
        config_path: Path to ML module config YAML
        
    Returns:
        pd.DataFrame: Merged dataset ready for model training
    """
    
    config = load_config_yaml(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Validate that at least one data source is enabled
    calculate_ohlcv = config.get("data", {}).get("calculate_ohlcv", True)
    features_enabled = config.get("features", {}).get("enabled", False)
    
    if not calculate_ohlcv and not features_enabled:
        raise ValueError("At least one data source must be enabled: either calculate_ohlcv or features")
    
    # Step 1: Market data collection
    df = None
    if config.get("data", {}).get("enabled", False):
        logger.info("Collecting market data...")
        df = collect_market_data(config)
        
        if df is None:
            logger.info("OHLCV calculation disabled, will use features only")
            df = pd.DataFrame()
        else:
            logger.info(f"Market data collected: {len(df)} rows")
    else:
        logger.warning("Market data collection disabled in config")
        df = pd.DataFrame()
    
    # Step 2: Feature engineering
    if config.get("features", {}).get("enabled", False):
        logger.info("Engineering features...")
        df = engineer_features(df, config)
        logger.info(f"Features engineered: {df.shape[1]} columns")
    else:
        logger.info("Features disabled. Using OHLCV data only.")
    
    if df.empty:
        raise ValueError("No data collected after data and feature steps")
    
    # Step 3: Sentiment collection
    if config.get("sentiment", {}).get("enabled", False):
        logger.info("Collecting sentiment data...")
        df = collect_sentiment_data(df, config)
        logger.info(f"Sentiment data merged: {df.shape[1]} columns")
    else:
        logger.info("Sentiment collection disabled.")
    
    # Step 4: Target generation
    logger.info("Generating prediction target...")
    df = generate_target(df, config)
    logger.info(f"Target generated. Final shape: {df.shape}")
    
    logger.info("ML pipeline completed successfully")
    return df


if __name__ == "__main__":
    import os

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config_yaml(config_path)
    df = run_ml_pipeline(config_path)

    # Drop rows with NaN in any column except sentiment (sen_*) columns --
    # a missing/no-post sentiment value shouldn't discard an otherwise
    # valid OHLCV + feature row.
    sentiment_cols = [col for col in df.columns if col.startswith("sen_")]
    required_cols = [col for col in df.columns if col not in sentiment_cols]
    df = df.dropna(subset=required_cols)

    # Output path includes exchange/symbol/model_type so different
    # exchanges (binance/bybit) and target types (regression/classification)
    # never overwrite each other's dataset.csv
    exchange = config["data"]["exchange"]
    symbol = config["data"]["symbol"]
    model_type = config["model_type"]

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", exchange, symbol, model_type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")
    print(f"Final dataset shape: {df.shape} (rows, cols)")