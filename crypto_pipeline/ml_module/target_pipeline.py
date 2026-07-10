# crypto_pipeline/ml_module/target_pipeline.py

"""
target_pipeline.py
------------------
Generates prediction targets for regression and classification tasks.
Handles return calculation, log returns, noise filtering, and binary classification targets.
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
    Generate regression targets (continuous values).
    """
    
    target_type = target_config.get("type", "return")
    horizon = target_config.get("horizon", 1)
    
    if "close" not in df.columns:
        raise ValueError("close price column required for target generation")
    
    future_close = df["close"].shift(-horizon)
    current_close = df["close"]
    
    if target_type == "return":
        df["target"] = (future_close - current_close) / current_close
        logger.info(f"Regression target: simple return (horizon={horizon})")
        
    elif target_type == "log_return":
        df["target"] = np.log(future_close / current_close)
        logger.info(f"Regression target: log return (horizon={horizon})")
        
    else:
        raise ValueError(f"Unknown regression target type: {target_type}")
    
    if target_config.get("filter_noise", False):
        noise_method = target_config.get("noise_method", "threshold")
        df = _filter_noise(df, noise_method, target_config)
    
    return df


def _generate_classification_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate classification targets (binary labels).
    """
    
    target_type = target_config.get("type", "binary")
    horizon = target_config.get("horizon", 1)
    threshold = target_config.get("threshold", 0.0)
    
    if "close" not in df.columns:
        raise ValueError("close price column required for target generation")
    
    future_close = df["close"].shift(-horizon)
    current_close = df["close"]
    returns = (future_close - current_close) / current_close
    
    if target_type == "binary":
        df["target"] = (returns > threshold).astype(int)
        logger.info(f"Classification target: binary (0/1, threshold={threshold}, horizon={horizon})")
        
    elif target_type == "threshold":
        df["target"] = np.where(
            returns > threshold, 1,
            np.where(returns < -threshold, -1, 0)
        )
        logger.info(f"Classification target: threshold-based (-1/0/1, threshold={threshold}, horizon={horizon})")
        
    else:
        raise ValueError(f"Unknown classification target type: {target_type}")
    
    if target_config.get("filter_noise", False):
        noise_method = target_config.get("noise_method", "threshold")
        df = _filter_noise(df, noise_method, target_config)
    
    return df


def _filter_noise(df: pd.DataFrame, method: str, config: dict) -> pd.DataFrame:
    """
    Filter noisy signals from target.
    """
    
    initial_count = len(df)
    
    if "target" not in df.columns:
        logger.warning("Target column not found, skipping noise filtering")
        return df
    
    if method == "threshold":
        noise_threshold = config.get("noise_threshold", 0.001)
        mask = df["target"].abs() >= noise_threshold
        df = df[mask]
        removed = initial_count - len(df)
        logger.info(f"Noise filtering (threshold={noise_threshold}): removed {removed} rows")
        
    elif method == "zscore":
        zscore_threshold = config.get("zscore_threshold", 3.0)
        targets = df["target"]
        z_scores = np.abs((targets - targets.mean()) / targets.std())
        mask = z_scores <= zscore_threshold
        df = df[mask]
        removed = initial_count - len(df)
        logger.info(f"Noise filtering (z-score={zscore_threshold}): removed {removed} rows")
        
    elif method == "quantile":
        lower_q = config.get("lower_quantile", 0.05)
        upper_q = config.get("upper_quantile", 0.95)
        targets = df["target"]
        lower_bound = targets.quantile(lower_q)
        upper_bound = targets.quantile(upper_q)
        mask = (targets >= lower_bound) & (targets <= upper_bound)
        df = df[mask]
        removed = initial_count - len(df)
        logger.info(f"Noise filtering (quantile {lower_q}-{upper_q}): removed {removed} rows")
        
    else:
        logger.warning(f"Unknown noise filter method: {method}")
    
    return df