"""
main.py
--------

Entry point of the Signal Module.

Responsibilities
----------------
1. Load config.yaml
2. Receive OHLCV data
3. Calculate indicators
4. Assign aliases
5. Merge indicator columns
6. Evaluate strategy conditions
7. Apply rules
8. Return final signal array
"""

from pathlib import Path

import yaml
import pandas as pd

from crypto_pipeline.signals.conditions import evaluate_conditions
from crypto_pipeline.signals.rules import apply_rules

# ------------------------------------------------------------------
# TA-Lib indicator functions
# ------------------------------------------------------------------

from crypto_pipeline.indicators.talib_indicators import (
    overlap_ema,
    momentum_rsi,
    momentum_macd,
    pattern_cdldoji,
    pattern_cdlengulfing,
)


# ==========================================================
# Configuration
# ==========================================================

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config():

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ==========================================================
# Indicator Calculation
# ==========================================================


# ==========================================================
# Indicator Registry
# ==========================================================

INDICATOR_REGISTRY = {
    "EMA": {
        "function": overlap_ema,
        "param_mapper": lambda p: {"period": p["period"]}
    },
    "RSI": {
        "function": momentum_rsi,
        "param_mapper": lambda p: {"period": p["period"]}
    },
    "MACD": {
        "function": momentum_macd,
        "param_mapper": lambda p: {
            "fastperiod": p["fast"],
            "slowperiod": p["slow"],
            "signalperiod": p["signal"]
        }
    },
    "PATTERNS": {"function": None},
}


def calculate_indicators(df: pd.DataFrame, indicator_config: dict):
    """
    Calculate indicators defined in config.yaml using the registry.
    """
    data = df.copy()

    for indicator_name, configs in indicator_config.items():

        if indicator_name not in INDICATOR_REGISTRY:
            continue

        if indicator_name == "PATTERNS":
            for config in configs:
                aliases = config["aliases"]

                if "doji" in aliases:
                    data[aliases["doji"]] = pattern_cdldoji(data)

                if "bullish_engulfing" in aliases:
                    engulf = pattern_cdlengulfing(data)
                    data[aliases["bullish_engulfing"]] = engulf > 0
            continue

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

# ==========================================================
# Signal Generation
# ==========================================================

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals.

    Parameters
    ----------
    df : DataFrame
        OHLCV dataframe containing:
            open
            high
            low
            close
            volume

    Returns
    -------
    pd.Series

        1  -> Buy
        0  -> No Signal
       -1  -> Sell
    """

    config = load_config()

    indicator_keys = {
        k: v
        for k, v in config.items()
        if k != "strategy"
    }

    strategy = config["strategy"]

    # --------------------------------------------------
    # Calculate indicators
    # --------------------------------------------------

    indicator_df = calculate_indicators(
        df,
        indicator_keys
    )

    # --------------------------------------------------
    # Evaluate every condition
    # --------------------------------------------------

    condition_df = evaluate_conditions(
        indicator_df,
        strategy
    )

    # --------------------------------------------------
    # Apply rule engine
    # --------------------------------------------------

    signals = apply_rules(
        condition_df,
        strategy
    )

    return signals


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    from crypto_pipeline.data.data_downloader import get_data
    from datetime import datetime

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]
    
    for exchange in exchanges:
        for symbol in symbols:
            print(f"\n{exchange.upper()} | {symbol.upper()}")
            print("-" * 50)

            result = get_data(
                exchange=exchange,
                symbol=symbol,
                start_date=datetime(2026, 6, 20, 0, 0, 0),
                end_date="now",
            )

            df = result["resampled"]
            signals = generate_signals(df)

            output = pd.DataFrame({
                "datetime": df["datetime"],
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
                "signal": signals
                })
            print(output.tail(20))