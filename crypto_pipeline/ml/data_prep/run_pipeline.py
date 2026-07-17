# crypto_pipeline/ml/data_prep/run_pipeline.py

"""
run_pipeline.py
---------------
Data prep stage functions, called from ml/main.py's run_ml_pipeline().
Renamed from main.py -> run_pipeline.py so it's not confused with
ml/main.py (the ML module's actual entry point) -- this file is not an
entry point itself; it has no __main__ block and isn't meant to be run
directly. It just orchestrates the data preparation steps (market data,
features, sentiment, target) for regression and classification tasks,
one call at a time, from ml/pipeline/dataset_loader.py's load_dataset().
"""

import logging
import pandas as pd

from crypto_pipeline.ml.data_prep.data_pipeline import collect_market_data
from crypto_pipeline.ml.data_prep.feature_pipeline import engineer_features
from crypto_pipeline.ml.data_prep.sentiment_pipeline import collect_sentiment_data
from crypto_pipeline.ml.data_prep.target_pipeline import generate_target


logger = logging.getLogger(__name__)


def run_data_prep_pipeline(config: dict) -> pd.DataFrame:
    """
    Execute the complete data preparation pipeline.

    Steps:
    1. Collect market data (if data.enabled) -- always fetches full OHLCV;
       calculate_ohlcv only controls whether raw OHLCV columns are kept
       in the final output (step 5)
    2. Engineer features (if enabled)
    3. Collect sentiment data (if enabled)
    4. Generate prediction target (needs close/high/low from step 1)
    5. Drop raw OHLCV output columns if calculate_ohlcv is False

    Args:
        config: ml/config.yaml dict (already loaded)

    Returns:
        pd.DataFrame: Merged dataset ready for model training
    """

    # Validate that at least one data source is enabled
    data_enabled = config.get("data", {}).get("enabled", False)
    features_enabled = config.get("features", {}).get("enabled", False)

    if not data_enabled and not features_enabled:
        raise ValueError("At least one data source must be enabled: either data.enabled or features.enabled")

    # Step 1: Market data collection. Always fetches full OHLCV when
    # enabled -- indicators, patterns, and target generation all need
    # real close/high/low/volume to compute from. calculate_ohlcv does
    # NOT skip this fetch; see step 5 for where it actually applies.
    df = None
    if data_enabled:
        logger.info("Collecting market data...")
        df = collect_market_data(config)
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
    logger.info(f"Target generated. Shape before NaN drop: {df.shape}")

    # Drop rows with NaN in any column except sentiment (sen_*) columns --
    # a missing/no-post sentiment value shouldn't discard an otherwise
    # valid OHLCV + feature row. This also clears indicator warm-up NaNs
    # (e.g. the first N-1 rows of a rolling N-period EMA/RSI/etc.), which
    # engineer_features() does not drop on its own.
    sentiment_cols = [col for col in df.columns if col.startswith("sen_")]
    required_cols = [col for col in df.columns if col not in sentiment_cols]
    dropped = len(df) - len(df.dropna(subset=required_cols))
    if dropped:
        logger.info(f"Dropping {dropped} rows with NaN in required (non-sentiment) columns")
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Step 5: strip raw OHLCV columns from the OUTPUT if calculate_ohlcv
    # is False. This runs last, after OHLCV has already done its job
    # feeding indicators/patterns (step 2) and the target (step 4) --
    # calculate_ohlcv=False means "don't include raw price columns in
    # the dataset", not "don't fetch/use OHLCV at all".
    calculate_ohlcv = config.get("data", {}).get("calculate_ohlcv", True)
    if not calculate_ohlcv:
        ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        if ohlcv_cols:
            logger.info(f"calculate_ohlcv=False: dropping raw OHLCV columns from output: {ohlcv_cols}")
            df = df.drop(columns=ohlcv_cols)

    logger.info(f"Data prep pipeline completed successfully. Final shape: {df.shape}")
    return df