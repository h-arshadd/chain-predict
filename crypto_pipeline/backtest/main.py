"""
main.py
-------

Entry point of the Backtest Module.
Loads 1-minute OHLCV data and signal data, runs the vectorized backtest,
and saves the resulting trade ledger per symbol.
"""

import os
from datetime import datetime

import pandas as pd

from crypto_pipeline.backtest.backtest import load_config, run_backtest
from crypto_pipeline.data.data_downloader import get_data

SIGNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "signals", "signal_outputs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backtest_outputs")


if __name__ == "__main__":

    config = load_config()

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for exchange in exchanges:
        for symbol in symbols:

            signal_path = os.path.join(SIGNAL_DIR, f"{exchange}_{symbol}_signals.csv")
            if not os.path.exists(signal_path):
                print(f"Skipping {exchange} {symbol}: no signal file found.")
                continue

            signals = pd.read_csv(signal_path, usecols=["datetime", "signal"], parse_dates=["datetime"])

            # Backtest always runs on 1-minute data, regardless of the
            # timeframe the signals were generated on.
            result = get_data(
                exchange=exchange,
                symbol=symbol,
                start_date=datetime(2026, 6, 20, 0, 0, 0),
                end_date="now",
                df_1m=True,
            )
            ohlcv_1m = result["one_min"]

            backtest_result = run_backtest(ohlcv_1m, signals, config)

            print(
                f"{exchange} {symbol}: "
                f"{backtest_result['total_trades']} trades, "
                f"final balance {backtest_result['final_balance']:.2f}, "
                f"net profit {backtest_result['total_net_profit']:.2f}"
            )

            ledger_path = os.path.join(OUTPUT_DIR, f"{exchange}_{symbol}_trades.csv")
            backtest_result["trade_ledger"].to_csv(ledger_path, index=False)