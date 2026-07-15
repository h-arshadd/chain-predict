# crypto_pipeline/stats/stats_runner.py

"""
stats_runner.py
---------------
Step 3 of the Stats task ("implement statistics module capable of
calculating nearly all available statistics"), run across every
(method, target_type, model) combo -- same loop shape as
model_evaluation/backtest_runner.py, one directory level up in the tree.

Reuses backtest_runner.py's own load_config/load_ohlcv_1m/backtest_one
instead of re-implementing them, and calls run_backtest() itself per
combo (in-memory) -- no reading of trade_ledger.csv/signals.csv back off
disk for the stats step. This IS "using the functions given in
model_evaluation" rather than introducing a second, parallel way of
loading signals/OHLCV.

Independent from the Backtesting and ML modules per the task ("module
should remain completely independent") in the sense that stats/ itself
imports nothing from ml_module and only takes a plain dict in
(calculator.compute_stats) -- this runner is just the batch-loop
convenience on top, matching how backtest_runner.py already does its own
batch loop on top of backtest.run_backtest().

Run AFTER model_evaluation/main.py (needs signals.csv per combo to exist):
    python -m crypto_pipeline.preprocessing_lab.model_evaluation.main
    python -m crypto_pipeline.stats.stats_runner

For each (method, target_type, model), saves:
    <out_dir>/<method>/<target_type>/<model>/stats.json
    <out_dir>/<method>/<target_type>/<model>/plots/*.png

Also saves:
    <out_dir>/comparison_stats.csv
        -- flat method/target_type/model table of the same headline
           metrics used elsewhere (sharpe, sortino, calmar, max_drawdown,
           cagr, profit_factor, win_rate, recovery_factor, risk_of_ruin),
           same idea as comparison_backtest.csv.
"""

import os

import yaml
import pandas as pd

from crypto_pipeline.preprocessing_lab.model_evaluation.backtest_runner import (
    load_config as load_yaml,
    load_ohlcv_1m,
    backtest_one,
)
from crypto_pipeline.stats.calculator import compute_stats, save_stats

# Headline metrics for the flat comparison table -- the PDF's explicitly
# named "most important" list. Every other discovered metric still lives
# in each combo's own stats.json; this is just the at-a-glance table.
_HEADLINE_METRICS = [
    "sharpe", "sortino", "calmar", "max_drawdown", "cagr",
    "profit_factor", "win_rate", "recovery_factor", "risk_of_ruin",
]


def run(
    model_eval_config_path: str,
    ml_config_path: str,
    stats_config_path: str = None,
    backtest_config_path: str = None,
):
    here = os.path.dirname(os.path.abspath(model_eval_config_path))
    config = load_yaml(model_eval_config_path)
    ml_config = load_yaml(ml_config_path)

    stats_here = os.path.dirname(os.path.abspath(stats_config_path)) if stats_config_path else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if stats_config_path is None:
        stats_config_path = os.path.join(stats_here, "config.yaml")
    stats_config = load_yaml(stats_config_path)

    methods = config["methods"]
    if isinstance(methods, str):
        methods = [methods]

    target_types = config.get("target_types", ["regression"])
    if isinstance(target_types, str):
        target_types = [target_types]

    signals_dir = os.path.join(here, config["output"]["dir"])
    stats_out_dir = os.path.join(stats_here, stats_config["output"]["dir"])

    # Same idea as backtest_runner.py: loaded/fetched once, reused for
    # every combo below, so every stats run is computed off the identical
    # price series and trading conditions.
    backtest_config = load_yaml(backtest_config_path) if backtest_config_path else \
        _default_backtest_config()
    ohlcv_1m = load_ohlcv_1m(ml_config)

    rows = []

    for target_type in target_types:
        for method_name in methods:
            method_dir = os.path.join(signals_dir, method_name, target_type)
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

                print(f"computing stats: {method_name} | {target_type} | {model_name}")
                backtest_result = backtest_one(signals_path, ohlcv_1m, backtest_config)

                out_dir = os.path.join(stats_out_dir, method_name, target_type, model_name)
                plot_dir = os.path.join(out_dir, "plots")
                stats_dict = compute_stats(backtest_result, stats_config, plot_dir=plot_dir)
                save_stats(stats_dict, os.path.join(out_dir, "stats.json"))

                row = {"method": method_name, "target_type": target_type, "model": model_name}
                row.update({m: stats_dict["metrics"].get(m) for m in _HEADLINE_METRICS})
                rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = os.path.join(stats_out_dir, "comparison_stats.csv")
    os.makedirs(stats_out_dir, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")

    return comparison


def _default_backtest_config():
    from crypto_pipeline.backtest.backtest import load_config as load_backtest_config
    return load_backtest_config()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model_eval_here = os.path.join(
        here, "..", "preprocessing_lab", "model_evaluation"
    )
    model_eval_config_path = os.path.join(model_eval_here, "config.yaml")
    ml_config_path = os.path.join(here, "..", "ml_module", "config.yaml")
    stats_config_path = os.path.join(here, "config.yaml")
    run(model_eval_config_path, ml_config_path, stats_config_path)