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
    
    df_with_target = df.copy()
    
    if model_type == "regression":
        df_with_target = _generate_regression_target(df_with_target, target_config)
    
    elif model_type == "classification":
        df_with_target = _generate_classification_target(df_with_target, target_config)
    
    elif model_type == "timeseries":
        logger.warning("Timeseries target generation not yet implemented")
        df_with_target = _generate_regression_target(df_with_target, target_config)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Handle NaN targets (from missing data or short horizon)
    initial_count = len(df_with_target)
    df_with_target = df_with_target.dropna(subset=["target"])
    final_count = len(df_with_target)
    
    if initial_count > final_count:
        logger.info(f"Dropped {initial_count - final_count} rows with NaN targets")
    
    logger.info(f"Target generated: {final_count} valid rows")
    return df_with_target


def _generate_regression_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate regression targets (continuous values).
    
    Supports:
    - return: (price_t+h - price_t) / price_t
    - log_return: ln(price_t+h / price_t)
    """
    
    df_reg = df.copy()
    
    target_type = target_config.get("type", "return")  # "return" or "log_return"
    horizon = target_config.get("horizon", 1)
    
    if "close" not in df_reg.columns:
        raise ValueError("close price column required for target generation")
    
    # Shift close price forward by horizon
    future_close = df_reg["close"].shift(-horizon)
    current_close = df_reg["close"]
    
    if target_type == "return":
        # Simple return: (P_future - P_current) / P_current
        df_reg["target"] = (future_close - current_close) / current_close
        logger.info(f"Regression target: simple return (horizon={horizon})")
        
    elif target_type == "log_return":
        # Log return: ln(P_future / P_current)
        df_reg["target"] = np.log(future_close / current_close)
        logger.info(f"Regression target: log return (horizon={horizon})")
        
    else:
        raise ValueError(f"Unknown regression target type: {target_type}")
    
    # Apply noise filtering if configured
    if target_config.get("filter_noise", False):
        noise_method = target_config.get("noise_method", "threshold")
        df_reg = _filter_noise(df_reg, noise_method, target_config)
    
    return df_reg


def _generate_classification_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate classification targets (binary labels).
    
    Supports:
    - 0/1: price goes down / up
    - -1/1: price goes down / up (alternative)
    - threshold-based: multi-class with threshold
    """
    
    df_clf = df.copy()
    
    target_type = target_config.get("type", "binary")  # "binary", "threshold"
    horizon = target_config.get("horizon", 1)
    threshold = target_config.get("threshold", 0.0)  # For threshold-based classification
    
    if "close" not in df_clf.columns:
        raise ValueError("close price column required for target generation")
    
    # Calculate return for classification
    future_close = df_clf["close"].shift(-horizon)
    current_close = df_clf["close"]
    returns = (future_close - current_close) / current_close
    
    if target_type == "binary":
        # 0 = down, 1 = up
        df_clf["target"] = (returns > threshold).astype(int)
        logger.info(f"Classification target: binary (0/1, threshold={threshold}, horizon={horizon})")
        
    elif target_type == "threshold":
        # Multi-class based on threshold range
        df_clf["target"] = np.where(
            returns > threshold, 1,
            np.where(returns < -threshold, -1, 0)
        )
        logger.info(f"Classification target: threshold-based (-1/0/1, threshold={threshold}, horizon={horizon})")
        
    else:
        raise ValueError(f"Unknown classification target type: {target_type}")
    
    # Apply noise filtering if configured
    if target_config.get("filter_noise", False):
        noise_method = target_config.get("noise_method", "threshold")
        df_clf = _filter_noise(df_clf, noise_method, target_config)
    
    return df_clf


def _filter_noise(df: pd.DataFrame, method: str, config: dict) -> pd.DataFrame:
    """
    Filter noisy signals from target.
    
    Methods:
    - threshold: remove targets with |value| < noise_threshold
    - zscore: remove targets outside z-score range
    - quantile: remove bottom/top quantile outliers
    """
    
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    
    if "target" not in df_filtered.columns:
        logger.warning("Target column not found, skipping noise filtering")
        return df_filtered
    
    if method == "threshold":
        noise_threshold = config.get("noise_threshold", 0.001)
        mask = df_filtered["target"].abs() >= noise_threshold
        df_filtered = df_filtered[mask]
        removed = initial_count - len(df_filtered)
        logger.info(f"Noise filtering (threshold={noise_threshold}): removed {removed} rows")
        
    elif method == "zscore":
        zscore_threshold = config.get("zscore_threshold", 3.0)
        targets = df_filtered["target"]
        z_scores = np.abs((targets - targets.mean()) / targets.std())
        mask = z_scores <= zscore_threshold
        df_filtered = df_filtered[mask]
        removed = initial_count - len(df_filtered)
        logger.info(f"Noise filtering (z-score={zscore_threshold}): removed {removed} rows")
        
    elif method == "quantile":
        lower_q = config.get("lower_quantile", 0.05)
        upper_q = config.get("upper_quantile", 0.95)
        targets = df_filtered["target"]
        lower_bound = targets.quantile(lower_q)
        upper_bound = targets.quantile(upper_q)
        mask = (targets >= lower_bound) & (targets <= upper_bound)
        df_filtered = df_filtered[mask]
        removed = initial_count - len(df_filtered)
        logger.info(f"Noise filtering (quantile {lower_q}-{upper_q}): removed {removed} rows")
        
    else:
        logger.warning(f"Unknown noise filter method: {method}")
    
    return df_filtered