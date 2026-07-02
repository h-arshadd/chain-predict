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


def generate_signals(df: pd.DataFrame, indicator_config: dict, strategy_config: dict) -> tuple:
    """
    Generate trading signals from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    indicator_config : dict
        Indicator configuration (load once via load_config + split_config,
        then pass it in here — this function does no config loading itself).
    strategy_config : dict
        Strategy configuration.

    Returns
    -------
    tuple of (indicator_df, condition_df, signals)
        indicator_df : pd.DataFrame — df with indicator columns appended
        condition_df : pd.DataFrame — one boolean column per strategy condition
        signals : pd.Series — final trading signals (1=Buy, 0=Hold, -1=Sell)
    """
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
    from datetime import datetime
    import os

    config = load_config()
    indicator_config, strategy_config = split_config(config)

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    # Create output folder
    output_dir = "signal_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for exchange in exchanges:
        for symbol in symbols:

            result = get_data(
                exchange=exchange,
                symbol=symbol,
                start_date=datetime(2026, 6, 20, 0, 0, 0),
                end_date="now",
            )

            df = result["resampled"]

            # Run the full pipeline: indicators -> conditions -> signals
            indicator_df, condition_df, signals = generate_signals(
                df, indicator_config, strategy_config
            )

            # Build comprehensive output
            output = pd.DataFrame({
                "datetime": df["datetime"],
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"],
                "ind_sma_20": indicator_df["ind_sma_20"],
            })
            
            # Add all conditions
            for col in condition_df.columns:
                output[col] = condition_df[col]
            
            # Add signal
            output["signal"] = signals

            # Drop warm-up rows where indicators aren't fully formed yet
            # (e.g. SMA_20 needs 20 bars before it produces a value)
            output = output.dropna().reset_index(drop=True)

            # Save to CSV in output folder
            csv_filename = os.path.join(output_dir, f"{exchange}_{symbol}_signals.csv")
            output.to_csv(csv_filename, index=False)