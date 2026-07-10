"""
feature_pipeline.py
-------------------
Calculates technical indicators and chart patterns based on config.
Merges all features into the market data DataFrame using configured aliases.
"""

import logging
import pandas as pd
from crypto_pipeline.indicators.talib_indicators import *

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate technical indicators and patterns, merge into market data.
    
    Args:
        df: Market OHLCV DataFrame
        config: ML module config dict
        
    Returns:
        pd.DataFrame: Market data with indicator and pattern columns
    """
    
    features_config = config.get("features", {})
    
    if not features_config.get("enabled"):
        logger.info("Features disabled, returning market data unchanged")
        return df
    
    df_with_features = df.copy()
    
    # Calculate indicators
    indicators_config = features_config.get("indicators", {})
    if indicators_config:
        logger.info(f"Calculating {len(indicators_config)} indicator types...")
        df_with_features = _calculate_indicators(df_with_features, indicators_config)
    
    # Calculate patterns
    patterns_config = features_config.get("patterns", {})
    if patterns_config:
        logger.info(f"Calculating {len(patterns_config)} pattern types...")
        df_with_features = _calculate_patterns(df_with_features, patterns_config)
    
    logger.info(f"Features engineered: {df_with_features.shape[1]} total columns")
    return df_with_features


def _calculate_indicators(df: pd.DataFrame, indicators_config: dict) -> pd.DataFrame:
    """
    Calculate all configured technical indicators.
    
    Config format:
        EMA:
          - parameters:
              period: 20
            aliases:
              ema: ind_EMA_20
          - parameters:
              period: 50
            aliases:
              ema: ind_EMA_50
    """
    
    df_ind = df.copy()
    indicator_count = 0
    
    for indicator_name, configs in indicators_config.items():
        indicator_func = _get_indicator_function(indicator_name)
        
        if indicator_func is None:
            logger.warning(f"Indicator {indicator_name} not found, skipping")
            continue
        
        # Each indicator can have multiple parameter sets
        for config_item in configs:
            params = config_item.get("parameters", {})
            aliases = config_item.get("aliases", {})
            
            try:
                result = indicator_func(df_ind, **params)
                
                # Handle multi-output indicators (dict) vs single-output (Series)
                if isinstance(result, dict):
                    for output_key, series in result.items():
                        alias_key = f"{output_key}"
                        alias_name = aliases.get(alias_key, f"ind_{indicator_name}_{output_key}")
                        df_ind[alias_name] = series
                        indicator_count += 1
                else:
                    # Single output
                    alias_name = aliases.get("value", f"ind_{indicator_name}")
                    df_ind[alias_name] = result
                    indicator_count += 1
                
                logger.info(f"  {indicator_name} ({params}) -> {alias_name}")
                
            except Exception as e:
                logger.error(f"Failed to calculate {indicator_name} with params {params}: {e}")
                continue
    
    logger.info(f"Calculated {indicator_count} indicator features")
    return df_ind


def _calculate_patterns(df: pd.DataFrame, patterns_config: dict) -> pd.DataFrame:
    """
    Calculate candlestick patterns based on config.
    
    Config format:
        DOJI:
          aliases:
            pattern: pat_DOJI
        BULLISH_ENGULFING:
          aliases:
            pattern: pat_BULLISH_ENGULFING
    
    Returns binary (0/1) columns indicating pattern presence.
    """
    
    df_pat = df.copy()
    pattern_count = 0
    
    for pattern_name, config_item in patterns_config.items():
        pattern_func = _get_pattern_function(pattern_name)
        
        if pattern_func is None:
            logger.warning(f"Pattern {pattern_name} not found, skipping")
            continue
        
        aliases = config_item.get("aliases", {})
        
        try:
            result = pattern_func(df_pat)
            alias_name = aliases.get("pattern", f"pat_{pattern_name}")
            df_pat[alias_name] = result
            pattern_count += 1
            logger.info(f"  {pattern_name} -> {alias_name}")
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern {pattern_name}: {e}")
            continue
    
    logger.info(f"Calculated {pattern_count} pattern features")
    return df_pat


def _get_indicator_function(name: str):
    """Map indicator name to talib function."""
    
    indicator_map = {
        "EMA": overlap_ema,
        "SMA": overlap_sma,
        "DEMA": overlap_dema,
        "TEMA": overlap_tema,
        "RSI": momentum_rsi,
        "MACD": momentum_macd,
        "BBANDS": overlap_bbands,
        "ATR": volatility_atr,
        "NATR": volatility_natr,
        "ADX": trend_adx,
        "CCI": momentum_cci,
        "CMO": momentum_cmo,
        "MOM": momentum_mom,
        "ROC": momentum_roc,
        "STOCH": momentum_stoch,
        "STOCHF": momentum_stochf,
        "TRIX": trend_trix,
        "WEMA": overlap_wma,
        "TRADERANGE": volatility_traderange,
        "HT_TRENDLINE": overlap_ht_trendline,
    }
    
    return indicator_map.get(name)


def _get_pattern_function(name: str):
    """Map pattern name to talib pattern function."""
    
    pattern_map = {
        "DOJI": pattern_doji,
        "BULLISH_ENGULFING": pattern_bullish_engulfing,
        "BEARISH_ENGULFING": pattern_bearish_engulfing,
        "HAMMER": pattern_hammer,
        "HANGING_MAN": pattern_hanging_man,
        "MORNING_STAR": pattern_morning_star,
        "EVENING_STAR": pattern_evening_star,
        "BULLISH_HARAMI": pattern_bullish_harami,
        "BEARISH_HARAMI": pattern_bearish_harami,
    }
    
    return pattern_map.get(name)