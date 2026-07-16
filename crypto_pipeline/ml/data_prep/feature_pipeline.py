# crypto_pipeline/ml/data_prep/feature_pipeline.py

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
    
    indicators_config = features_config.get("indicators", {})
    if indicators_config:
        logger.info(f"Calculating {len(indicators_config)} indicator types...")
        df = _calculate_indicators(df, indicators_config)
    
    patterns_config = features_config.get("patterns", {})
    if patterns_config:
        logger.info(f"Calculating {len(patterns_config)} pattern types...")
        df = _calculate_patterns(df, patterns_config)
    
    logger.info(f"Features engineered: {df.shape[1]} total columns")
    return df


def _calculate_indicators(df: pd.DataFrame, indicators_config: dict) -> pd.DataFrame:
    """
    Calculate all configured technical indicators.
    """
    
    indicator_count = 0
    
    for indicator_name, configs in indicators_config.items():
        indicator_func = _get_indicator_function(indicator_name)
        
        if indicator_func is None:
            logger.warning(f"Indicator {indicator_name} not found, skipping")
            continue
        
        for config_item in configs:
            params = config_item.get("parameters", {})
            aliases = config_item.get("aliases", {})
            
            try:
                result = indicator_func(df, **params)
                
                if isinstance(result, dict):
                    for output_key, series in result.items():
                        alias_key = f"{output_key}"
                        alias_name = aliases.get(alias_key, f"ind_{indicator_name}_{output_key}")
                        df[alias_name] = series
                        indicator_count += 1
                else:
                    alias_name = aliases.get("value", f"ind_{indicator_name}")
                    df[alias_name] = result
                    indicator_count += 1
                
                logger.info(f"  {indicator_name} ({params}) -> {alias_name}")
                
            except Exception as e:
                logger.error(f"Failed to calculate {indicator_name} with params {params}: {e}")
                continue
    
    logger.info(f"Calculated {indicator_count} indicator features")
    return df


def _calculate_patterns(df: pd.DataFrame, patterns_config: dict) -> pd.DataFrame:
    """
    Calculate candlestick patterns based on config.
    """
    
    pattern_count = 0
    
    for pattern_name, config_item in patterns_config.items():
        pattern_func = _get_pattern_function(pattern_name)
        
        if pattern_func is None:
            logger.warning(f"Pattern {pattern_name} not found, skipping")
            continue
        
        aliases = config_item.get("aliases", {})
        
        try:
            result = pattern_func(df)
            alias_name = aliases.get("pattern", f"pat_{pattern_name}")
            df[alias_name] = result
            pattern_count += 1
            logger.info(f"  {pattern_name} -> {alias_name}")
            
        except Exception as e:
            logger.error(f"Failed to calculate pattern {pattern_name}: {e}")
            continue
    
    logger.info(f"Calculated {pattern_count} pattern features")
    return df


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
        "ADX": momentum_adx,
        "CCI": momentum_cci,
        "CMO": momentum_cmo,
        "MOM": momentum_mom,
        "ROC": momentum_roc,
        "STOCH": momentum_stoch,
        "STOCHF": momentum_stochf,
        "TRIX": momentum_trix,
        "WMA": overlap_wma,
        "HT_TRENDLINE": overlap_ht_trendline,
    }
    
    return indicator_map.get(name)


def _get_pattern_function(name: str):
    """Map pattern name to talib pattern function."""
    
    pattern_map = {
        "DOJI": pattern_cdldoji,
        "ENGULFING": pattern_cdlengulfing,
        "HAMMER": pattern_cdlhammer,
        "HANGING_MAN": pattern_cdlhangingman,
        "MORNING_STAR": pattern_cdlmorningstar,
        "EVENING_STAR": pattern_cdleveningstar,
        "HARAMI": pattern_cdlharami,
        "HARAMI_CROSS": pattern_cdlharamicross,
        "SHOOTING_STAR": pattern_cdlshootingstar,
        "INVERTING_HAMMER": pattern_cdlinvertedhammer,
        "SPINNINGTOP": pattern_cdlspinningtop,
        "PIERCING": pattern_cdlpiercing,
        "DARK_CLOUD_COVER": pattern_cdldarkcloudcover,
        "KICKING": pattern_cdlkicking,
        "BREAKAWAY": pattern_cdlbreakaway,
    }
    
    return pattern_map.get(name)