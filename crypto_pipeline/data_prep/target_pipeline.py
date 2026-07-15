# crypto_pipeline/ml_module/target_pipeline.py

"""
target_pipeline.py
------------------
Generates prediction targets: log return for regression,
-1/0/1 (threshold-based) for classification.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate prediction target based on model type and config.
    
    Args:
        df: DataFrame with price data (close column required)
        config: ML module config dict with target section
        
    Returns:
        pd.DataFrame: DataFrame with target column added
    """
    
    target_config = config.get("target", {})
    model_type = config.get("model_type", "regression")
    
    if not target_config:
        raise ValueError("No target configuration found")
    
    logger.info(f"Generating target for {model_type} model")
    
    if model_type == "regression":
        df = _generate_regression_target(df, target_config)
    
    elif model_type == "classification":
        df = _generate_classification_target(df, target_config)
    
    elif model_type == "timeseries":
        logger.warning("Timeseries target generation not yet implemented")
        df = _generate_regression_target(df, target_config)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    initial_count = len(df)
    df = df.dropna(subset=["target"])
    final_count = len(df)
    
    if initial_count > final_count:
        logger.info(f"Dropped {initial_count - final_count} rows with NaN targets")
    
    logger.info(f"Target generated: {final_count} valid rows")
    return df


def _generate_regression_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate regression target: log return, current close vs close
    `horizon` candles in the future.
    """
    
    horizon = target_config.get("horizon", 1)
    
    if "close" not in df.columns:
        raise ValueError("close price column required for target generation")
    
    current_close = df["close"]
    future_close = df["close"].shift(-horizon)
    
    df["target"] = np.log(future_close / current_close)
    logger.info(f"Regression target: log return, future close (horizon={horizon})")
    
    if target_config.get("filter_noise", False):
        keep_mask = _noise_keep_mask(df["target"], target_config)
        df = df[keep_mask]
    
    return df


def _generate_classification_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate classification target using Triple Barrier Labeling:
    -1 (down) / 0 (flat) / 1 (up).

    Looking forward up to `horizon` candles from each row, checks whether
    price hits the upper barrier (take-profit) or lower barrier (stop-loss)
    first. If neither is hit within the horizon, label is 0 (flat/timeout).
    """
    
    horizon = target_config.get("horizon", 1)
    upper_threshold = target_config.get("upper_threshold", 0.001)
    lower_threshold = target_config.get("lower_threshold", -0.001)
    
    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Triple barrier labeling requires columns: {required_cols}")
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    
    labels = np.zeros(n)
    labels[:] = np.nan  # rows too close to the end (no full horizon ahead) stay NaN
    
    for i in range(n - horizon):
        entry_price = close[i]
        upper_barrier = entry_price * (1 + upper_threshold)
        lower_barrier = entry_price * (1 + lower_threshold)
        
        label = 0  # default: neither barrier hit within horizon
        for step in range(1, horizon + 1):
            future_high = high[i + step]
            future_low = low[i + step]
            
            hit_upper = future_high >= upper_barrier
            hit_lower = future_low <= lower_barrier
            
            if hit_upper and hit_lower:
                # Both hit in the same candle -- can't tell which came first
                # from OHLC alone, so treat as flat/ambiguous rather than guess.
                label = 0
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break
        
        labels[i] = label
    
    df["target"] = labels
    logger.info(
        f"Classification target: Triple Barrier Labeling (-1/0/1, "
        f"upper={upper_threshold}, lower={lower_threshold}, horizon={horizon})"
    )
    
    return df


def _noise_keep_mask(values: pd.Series, config: dict) -> pd.Series:
    """
    Build a boolean mask marking which rows to KEEP, based on the magnitude
    of `values` (returns or target). True = keep, False = noise, drop it.
    A row is kept if its absolute value is at least noise_threshold.
    """
    
    noise_threshold = config.get("noise_threshold", 0.001)
    mask = values.abs() >= noise_threshold
    logger.info(f"Noise filtering (threshold={noise_threshold}): keeping {mask.sum()}/{len(mask)} rows")
    
    return mask