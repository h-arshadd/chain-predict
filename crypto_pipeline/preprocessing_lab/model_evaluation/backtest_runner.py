# preprocessing_lab/model_evaluation/backtest_runner.py

"""
backtest_runner.py
-------------------
Step 8 (Backtesting) of the task. Reads every (method, target_type, model)
signals.csv already produced by main.py, backtests each one against the
SAME 1-minute OHLCV series and the SAME backtest/config.yaml trading
conditions (fees, slippage, position sizing, stop-loss, take-profit --
loaded once, reused for every experiment, per the task's "must remain
constant across all experiments" requirement), and saves the result.

Kept as its own file, separate from main.py, because:
  - main.py's job is model training + signal generation (Steps 6-7).
  - This file's job is backtesting already-generated signals (Step 8).
  - It reads signals.csv from disk, like main.py reads transformed.csv --
    consistent with how every other stage in this pipeline hands off via
    CSV rather than passing objects in memory.

Run AFTER main.py (needs signals.csv + comparison_table_<target_type>.csv
to already exist):
    python -m crypto_pipeline.preprocessing_lab.model_evaluation.main
    python -m crypto_pipeline.preprocessing_lab.model_evaluation.backtest_runner

For each (method, target_type, model) combo, saves:
    outputs/<method>/<target_type>/<model>/backtest_result.csv
        -- one-row summary (final_balance, total_net_profit, total_trades,
           wins, losses, win_rate), same idea as predictions.csv being the
           per-model file sitting next to signals.csv.
    outputs/<method>/<target_type>/<model>/trade_ledger.csv
        -- full trade-by-trade ledger, kept separate from the summary file
           so the summary stays one row.

Also saves/updates:
    outputs/comparison_backtest.csv
        -- flat method/target_type/model/trading-metrics table, across
           BOTH target types (backtest metrics apply the same way to
           regression and classification signals).
    outputs/comparison_table_regression.csv
        -- main.py's regression table (mae, rmse) with total_return +
           trading columns merged in, in place.
    outputs/comparison_table_classification.csv
        -- main.py's classification table (accuracy, precision) with
           total_return + trading columns merged in, in place.
"""

import os
from datetime import datetime

import yaml
import pandas as pd

from crypto_pipeline.backtest.backtest import load_config as load_backtest_config, run_backtest
from crypto_pipeline.data.data_downloader import get_data


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _parse_date(value):
    """
    ml_module/config.yaml's data.start_date/end_date are plain strings
    ("2025-01-01"). get_data() does datetime arithmetic on end_date
    internally (end_date - TIMEFRAME_DELTA), so it needs real datetime
    objects, not strings -- same conversion backtest/main.py's
    parse_backtest_dates() does for backtest/config.yaml's own dates.
    "now" is left as-is; get_data() resolves it at call time.
    """
    if value == "now":
        return value
    return datetime.strptime(value, "%Y-%m-%d")


def load_ohlcv_1m(ml_config: dict) -> pd.DataFrame:
    """
    1-minute OHLCV for the same symbol/exchange/date-range as
    ml_module/config.yaml, fetched once and reused for every backtest so
    every experiment executes against the identical price series (this is
    what "identical trading conditions" requires on the data side, same as
    backtest_config covers it on the fees/sizing/TP-SL side).
    """
    data_cfg = ml_config["data"]
    result = get_data(
        exchange=data_cfg["exchange"],
        symbol=data_cfg["symbol"],
        start_date=_parse_date(data_cfg["start_date"]),
        end_date=_parse_date(data_cfg["end_date"]),
        timeframe=data_cfg["timeframe"],
        df_1m=True,
    )
    return result["one_min"]


def backtest_one(signals_path: str, ohlcv_1m: pd.DataFrame, backtest_config: dict) -> dict:
    """Load one signals.csv and run it through the backtest engine."""
    signals = pd.read_csv(signals_path, parse_dates=["datetime"])
    return run_backtest(ohlcv_1m, signals, backtest_config)


def merge_trading_metrics(out_dir: str, target_type: str, comparison: pd.DataFrame):
    """
    Reads main.py's comparison_table_<target_type>.csv, merges in
    total_return + trading columns for that target_type, and overwrites
    it in place.
    """
    main_table_path = os.path.join(out_dir, f"comparison_table_{target_type}.csv")
    if not os.path.exists(main_table_path):
        print(f"  (skip merge for {target_type}: {main_table_path} not found -- run main.py first)")
        return

    main_table = pd.read_csv(main_table_path)

    trading_cols = comparison[comparison["target_type"] == target_type][
        ["method", "target_type", "model", "total_net_profit",
         "final_balance", "total_trades", "win_rate"]
    ].rename(columns={"total_net_profit": "total_return"})

    merged = main_table.merge(trading_cols, on=["method", "target_type", "model"], how="left")

    merged.to_csv(main_table_path, index=False)
    print(f"Merged trading metrics into: {main_table_path}")


def run(
    model_eval_config_path: str,
    ml_config_path: str,
    backtest_config_path: str = None,
):
    here = os.path.dirname(os.path.abspath(model_eval_config_path))
    config = load_config(model_eval_config_path)
    ml_config = load_config(ml_config_path)

    methods = config["methods"]
    if isinstance(methods, str):
        methods = [methods]

    target_types = config.get("target_types", ["regression"])
    if isinstance(target_types, str):
        target_types = [target_types]

    out_dir = os.path.join(here, config["output"]["dir"])

    # Loaded once, used for every single (method, target_type, model)
    # backtest below -- this IS what keeps trading conditions constant
    # across all experiments (Step 8 requirement).
    backtest_config = load_backtest_config(backtest_config_path)
    ohlcv_1m = load_ohlcv_1m(ml_config)

    rows = []

    for target_type in target_types:
        for method_name in methods:
            method_dir = os.path.join(out_dir, method_name, target_type)
            if not os.path.isdir(method_dir):
                continue

            model_names = sorted(
                d for d in os.listdir(method_dir)
                if os.path.isdir(os.path.join(method_dir, d))
            )

            for model_name in model_names:
                model_dir = os.path.join(method_dir, model_name)
                signals_path = os.path.join(model_dir, "signals.csv")
                if not os.path.exists(signals_path):
                    print(f"  skip {method_name}/{target_type}/{model_name}: no signals.csv")
                    continue

                print(f"backtesting {method_name} | {target_type} | {model_name}")
                result = backtest_one(signals_path, ohlcv_1m, backtest_config)

                summary = {
                    "method": method_name,
                    "target_type": target_type,
                    "model": model_name,
                    "final_balance": result["final_balance"],
                    "total_net_profit": result["total_net_profit"],
                    "total_trades": result["total_trades"],
                    "wins": result["win_loss"]["wins"],
                    "losses": result["win_loss"]["losses"],
                    "win_rate": result["win_loss"]["win_rate"],
                }
                rows.append(summary)

                # one-row summary, sits next to predictions.csv/signals.csv
                pd.DataFrame([summary]).to_csv(
                    os.path.join(model_dir, "backtest_result.csv"), index=False
                )
                # full trade-by-trade ledger, kept separate so the summary
                # file above stays one row
                result["trade_ledger"].to_csv(
                    os.path.join(model_dir, "trade_ledger.csv"), index=False
                )

    comparison = pd.DataFrame(rows)
    comparison_path = os.path.join(out_dir, "comparison_backtest.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")

    # merge trading metrics into each target_type's own comparison table
    for target_type in target_types:
        merge_trading_metrics(out_dir, target_type, comparison)

    return comparison


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model_eval_config_path = os.path.join(here, "config.yaml")
    ml_config_path = os.path.join(here, "..", "..", "ml_module", "config.yaml")
    run(model_eval_config_path, ml_config_path)