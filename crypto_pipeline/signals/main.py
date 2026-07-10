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
    indicator_config = {k: v for k, v in config.items() if k not in ("strategy", "strategy_name")}
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
    from crypto_pipeline.utils.metadata_utils import (
        get_db_connection as get_metadata_connection,
        get_current_strategy,
    )
    from datetime import datetime

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    # The active strategy's id/name (from metadata.strategy -- the newest
    # row, see create_metadata.py) is stamped onto every signal row below,
    # so each row can be traced back to exactly which strategy produced it.
    metadata_conn = get_metadata_connection()
    try:
        current_strategy = get_current_strategy(metadata_conn)
    finally:
        metadata_conn.close()

    if current_strategy is None:
        raise RuntimeError(
            "No strategy found in metadata.strategy -- run create_metadata.py first."
        )

    strategy_id = current_strategy["strategy_id"]  # not used in table naming; logged only
    strategy_name = current_strategy["strategy_name"]
    print(f"Using strategy_id={strategy_id}, strategy_name={strategy_name!r}")

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

                # Output: just datetime + final signal. OHLCV/indicators/
                # conditions are intermediate values only -- not persisted.
                # strategy_name is NOT a column here -- it's baked into the
                # table name by insert_signals() instead.
                output = pd.DataFrame({
                    "datetime": df["datetime"],
                    "signal": signals,
                })

                # Drop warm-up rows where indicators aren't fully formed yet
                # (e.g. SMA_20 needs 20 bars before it produces a value) --
                # signal will be NaN for those rows since it's derived from
                # conditions on those same not-yet-formed indicators.
                output = output.dropna(subset=["signal"]).reset_index(drop=True)

                # Store in DB: signals.{exchange}_{symbol}_{strategy_name}
                insert_signals(conn, exchange, symbol, strategy_name, output)

                print(f"Saved {exchange} {symbol} (strategy {strategy_name!r}): {len(output)} rows")

    finally:
        conn.close()