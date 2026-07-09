"""
main.py
-------

Entry point of the Signal Module.
Orchestrates config loading, indicator calculation, and signal generation.
"""

from pathlib import Path
import yaml
import pandas as pd

from crypto_pipeline.signals.signal_helpers import calculate_indicators
from crypto_pipeline.signals.conditions import evaluate_conditions
from crypto_pipeline.signals.rules import apply_rules


def load_config(config_path=None) -> dict:
    """Load config from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_config(config: dict) -> tuple:
    """Split config into indicator_config and strategy_config."""
    indicator_config = {k: v for k, v in config.items() if k != "strategy"}
    strategy_config = config["strategy"]
    return indicator_config, strategy_config


def generate_signals(df: pd.DataFrame, config_path: str = None) -> tuple:
    """
    Generate trading signals from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    config_path : str, optional
        Path to signals config.yaml. If not provided, loads from 
        crypto_pipeline/signals/config.yaml

    Returns
    -------
    tuple of (indicator_df, condition_df, signals)
        indicator_df : pd.DataFrame — df with indicator columns appended
        condition_df : pd.DataFrame — one boolean column per strategy condition
        signals : pd.Series — final trading signals (1=Buy, 0=Hold, -1=Sell)
    """
    # Load config and split it internally
    config = load_config(config_path)
    indicator_config, strategy_config = split_config(config)
    
    # Calculate indicators and assign aliases
    indicator_df = calculate_indicators(df, indicator_config)
    
    # Evaluate all strategy conditions
    condition_df = evaluate_conditions(indicator_df, strategy_config)
    
    # Combine conditions into final signals
    signals = apply_rules(condition_df, strategy_config)
    
    return indicator_df, condition_df, signals


# ==========================================================
# Entry Point (config loading and orchestration only)
# ==========================================================

if __name__ == "__main__":

    from crypto_pipeline.data.data_downloader import get_data
    from crypto_pipeline.utils.db_utils import get_db_connection, insert_signals
    from datetime import datetime

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    conn = get_db_connection()

    try:
        for exchange in exchanges:
            for symbol in symbols:

                result = get_data(
                    exchange=exchange,
                    symbol=symbol,
                    start_date=datetime(2026, 4, 1, 0, 0, 0),
                    end_date=datetime(2026, 6, 1, 0, 0, 0)
                )

                df = result["resampled"]

                # Run the full pipeline: generate_signals loads config internally
                indicator_df, condition_df, signals = generate_signals(df)

                # Build output: datetime + ohlcv + indicators + conditions + signal
                #
                # FIX: indicators (and therefore condition_df / signals) are all
                # pre-shifted by 1 bar in talib_indicators.py to avoid lookahead
                # -- i.e. the value in row N was computed from the candle at
                # row N-1. The OHLCV columns below were NOT shifted, so they
                # showed the CURRENT bar's own price sitting in the same row as
                # an indicator/signal value that actually belongs to the
                # PREVIOUS bar. That's what caused signals to appear to line up
                # one row "too early" against the price data.
                #
                # Shifting open/high/low/close/volume by 1 here makes every
                # column in a row refer to the same underlying candle.
                output = pd.DataFrame({
                    "datetime": df["datetime"],
                    "open": df["open"].shift(1),
                    "high": df["high"].shift(1),
                    "low": df["low"].shift(1),
                    "close": df["close"].shift(1),
                    "volume": df["volume"].shift(1),
                })

                # Add all indicator columns
                for col in indicator_df.columns:
                    if col != "datetime":
                        output[col] = indicator_df[col]

                # Add all condition columns
                for col in condition_df.columns:
                    output[col] = condition_df[col]

                # Add final signal
                output["signal"] = signals

                # Drop warm-up rows where indicators aren't fully formed yet
                # (e.g. SMA_20 needs 20 bars before it produces a value)
                output = output.dropna().reset_index(drop=True)

                # Store in DB: signals.{exchange}_{symbol}, full rebuild each run
                insert_signals(conn, exchange, symbol, output)

                print(f"Saved {exchange} {symbol}: {len(output)} rows")

    finally:
        conn.close()