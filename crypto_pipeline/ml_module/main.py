"""
main.py
-------
Entry point for the ML Module.
Orchestrates configuration loading and the complete data preparation pipeline
for regression and classification tasks.
"""

import logging
import yaml
import pandas as pd

from crypto_pipeline.utils.ml_utils import load_config_yaml
from ml_module.data_pipeline import collect_market_data
from ml_module.feature_pipeline import engineer_features
from ml_module.sentiment_pipeline import collect_sentiment_data
from ml_module.target_pipeline import generate_target


logger = logging.getLogger(__name__)


def run_ml_pipeline(config_path: str) -> pd.DataFrame:
    """
    Execute the complete ML data preparation pipeline.
    
    Steps:
    1. Collect market data (if enabled)
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
    
    # Step 1: Market data collection
    df = None
    if config.get("data", {}).get("enabled", False):
        logger.info("Collecting market data...")
        df = collect_market_data(config)
        logger.info(f"Market data collected: {len(df)} rows")
    else:
        logger.warning("Market data collection disabled in config")
        raise ValueError("Market data must be enabled for ML pipeline")
    
    # Step 2: Feature engineering
    if config.get("features", {}).get("enabled", False):
        logger.info("Engineering features...")
        df = engineer_features(df, config)
        logger.info(f"Features engineered: {df.shape[1]} columns")
    else:
        logger.info("Features disabled. Using OHLCV data only.")
    
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
    # Example usage
    config_path = "ml_module/config.yaml"
    df = run_ml_pipeline(config_path)
    
    # Save to CSV for inspection (temporary)
    output_path = "ml_output.csv"
    df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")