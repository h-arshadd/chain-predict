"""
signal_helpers.py
-----------------

Helper utilities for signal module.
Indicator registry and calculation.

Location: crypto_pipeline/signals/signal_helpers.py
"""

import pandas as pd

from crypto_pipeline.indicators.talib_indicators import (
    overlap_ema,
    overlap_sma,
    momentum_rsi,
    momentum_macd,
    pattern_cdldoji,
    pattern_cdlengulfing,
)


# ==========================================================
# Indicator Registry
# ==========================================================

INDICATOR_REGISTRY = {
    "SMA": {
        "function": overlap_sma,
        "param_mapper": lambda p: {"period": p["timeperiod"]},
    },
    "EMA": {
        "function": overlap_ema,
        "param_mapper": lambda p: {"period": p["period"]},
    },
    "RSI": {
        "function": momentum_rsi,
        "param_mapper": lambda p: {"period": p["period"]},
    },
    "MACD": {
        "function": momentum_macd,
        "param_mapper": lambda p: {
            "fastperiod": p["fast"],
            "slowperiod": p["slow"],
            "signalperiod": p["signal"],
        },
    },
    "PATTERNS": {"function": None},
}


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    """
    Calculate indicators and assign aliases from config.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    indicator_config : dict
        Indicator configuration dict (all config keys except 'strategy')
    
    Returns
    -------
    pd.DataFrame
        df with indicator columns appended
    """
    data = df.copy()

    for indicator_name, configs in indicator_config.items():
        if indicator_name not in INDICATOR_REGISTRY:
            continue

        # Pattern indicators
        if indicator_name == "PATTERNS":
            for config in configs:
                aliases = config["aliases"]
                
                if "doji" in aliases:
                    data[aliases["doji"]] = pattern_cdldoji(data)
                
                if "bullish_engulfing" in aliases:
                    engulf = pattern_cdlengulfing(data)
                    data[aliases["bullish_engulfing"]] = engulf > 0
            
            continue

        # Standard indicators (SMA, EMA, RSI, MACD, etc.)
        function = INDICATOR_REGISTRY[indicator_name]["function"]
        param_mapper = INDICATOR_REGISTRY[indicator_name].get("param_mapper")

        for config in configs:
            params = config.get("parameters", {})
            aliases = config["aliases"]

            mapped_params = param_mapper(params) if param_mapper else params
            result = function(data, **mapped_params)

            if isinstance(result, dict):
                for key, alias in aliases.items():
                    if key in result:
                        data[alias] = result[key]
            else:
                data[next(iter(aliases.values()))] = result

    return data