# crypto_pipeline/stats/stats_runner.py

"""
stats_runner.py
---------------
Batch driver for the Statistics module. Runs compute_stats() over a set
of backtest results the Backtest module produced -- NOT over ml/
model_evaluation signals. This module stays independent from ml: it
never imports anything from crypto_pipeline.ml.

Core entry point, run(), takes backtest results as plain dicts -- the
same dict run_backtest() itself returns (equity_curve, final_balance,
total_net_profit, total_trades, win_loss, trade_ledger, ...). No
re-deriving equity_curve/trade_summary from a trade ledger here -- if
you already have run_backtest()'s return value, that's the only thing
this needs; hand it over as-is.

For each combo, saves numeric output only -- no PNGs:
    <out_dir>/metrics.json   -- {combo: {...every discovered metric...}}
    <out_dir>/plots.json     -- {combo: {...numeric plot series/tables...}}
    <out_dir>/comparison_stats.csv
        -- flat table of the same headline metrics used elsewhere
           (sharpe, sortino, calmar, max_drawdown, cagr, profit_factor,
           win_rate, recovery_factor, risk_of_ruin) -- kept as the one
           quick-look CSV; everything else is numeric JSON.

Usage (call run() directly with results you already have in memory):
    from crypto_pipeline.stats.stats_runner import run

    # backtest_results: {combo_name: run_backtest()'s return dict}
    backtest_results = {
        "binance_btc": run_backtest(ohlcv_1m, signals, backtest_config),
        "bybit_btc": run_backtest(ohlcv_1m_2, signals_2, backtest_config),
    }
    run(backtest_results, stats_config=stats_config)

Usage (as a script):
    python -m crypto_pipeline.stats.stats_runner
    -- runs the Backtest module itself for every exchange/symbol combo
    (same helpers/loop as backtest/main.py) and feeds the resulting
    backtest results into run(). See the __main__ block below.
"""

import os
import json

import pandas as pd

from crypto_pipeline.stats.calculator import compute_stats
from crypto_pipeline.stats.utils import to_json_safe

# Headline metrics for the flat comparison table -- the PDF's explicitly
# named "most important" list. Every other discovered metric still lives
# in metrics.json; this is just the at-a-glance table.
_HEADLINE_METRICS = [
    "sharpe", "sortino", "calmar", "max_drawdown", "cagr",
    "profit_factor", "win_rate", "recovery_factor", "risk_of_ruin",
]


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_stats_config() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return _load_yaml(os.path.join(here, "config.yaml"))


def run(
    backtest_results: dict,
    stats_config: dict = None,
    stats_out_dir: str = None,
):
    """
    Parameters
    ----------
    backtest_results : dict
        {combo_name: backtest_result}, where backtest_result is exactly
        what run_backtest() returns (needs "equity_curve" at minimum;
        "final_balance"/"total_net_profit"/"total_trades"/"win_loss" are
        used for trade_summary if present). combo_name is whatever label
        the caller wants (e.g. "binance_btc") -- used only as the key in
        metrics.json/plots.json and a column in comparison_stats.csv.
    stats_config : dict, optional
        Loaded from stats/config.yaml if not given.
    stats_out_dir : str, optional
        Where to save metrics.json / plots.json / comparison_stats.csv.
        Defaults to <stats config's output.dir>, next to this file.

    Returns
    -------
    dict: {"metrics": {...}, "plots": {...}, "comparison": DataFrame}
    """
    stats_config = stats_config or _default_stats_config()

    if stats_out_dir is None:
        stats_here = os.path.dirname(os.path.abspath(__file__))
        stats_out_dir = os.path.join(stats_here, stats_config["output"]["dir"])

    all_metrics = {}
    all_plots = {}
    rows = []

    for combo, backtest_result in backtest_results.items():
        if not backtest_result or backtest_result.get("total_trades", 0) == 0:
            print(f"  skip {combo}: no trades")
            continue

        print(f"computing stats: {combo}")
        stats_dict = compute_stats(backtest_result, stats_config)

        all_metrics[combo] = stats_dict["metrics"]
        all_plots[combo] = stats_dict["plots"]

        row = {"combo": combo}
        row.update({m: stats_dict["metrics"].get(m) for m in _HEADLINE_METRICS})
        rows.append(row)

    os.makedirs(stats_out_dir, exist_ok=True)

    metrics_path = os.path.join(stats_out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(to_json_safe(all_metrics), f, indent=2)
    print(f"Saved: {metrics_path}")

    plots_path = os.path.join(stats_out_dir, "plots.json")
    with open(plots_path, "w") as f:
        json.dump(to_json_safe(all_plots), f, indent=2)
    print(f"Saved: {plots_path}")

    comparison = pd.DataFrame(rows)
    comparison_path = os.path.join(stats_out_dir, "comparison_stats.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved: {comparison_path}")

    return {"metrics": all_metrics, "plots": all_plots, "comparison": comparison}


if __name__ == "__main__":
    # Runs the Backtest module itself, per exchange/symbol, straight in
    # memory -- same loop shape and same helpers backtest/main.py already
    # uses (get_data for signal-generation OHLCV, get_1m_data for
    # execution OHLCV, build_signals, run_backtest) -- then hands the
    # resulting backtest_results dict to run(). No CSV, no DB read-back:
    # main.py's insert_trades() write to Postgres is a separate concern
    # from this stats run.
    from crypto_pipeline.backtest.main import (
        parse_backtest_dates,
        get_1m_data,
        build_signals,
    )
    from crypto_pipeline.backtest.backtest import load_config, run_backtest
    from crypto_pipeline.data.data_downloader import get_data

    backtest_config = parse_backtest_dates(load_config())

    exchanges = ["binance", "bybit"]
    symbols = ["doge", "sol", "btc", "eth", "ada", "ltc", "mina", "sui"]

    backtest_results = {}

    for exchange in exchanges:
        for symbol in symbols:
            combo = f"{exchange}_{symbol}"

            hourly_result = get_data(
                exchange=exchange,
                symbol=symbol,
                start_date=backtest_config["start_date"],
                end_date=backtest_config["end_date"],
            )
            ohlcv_1h = hourly_result["resampled"]
            if ohlcv_1h.empty:
                print(f"Skipping {combo}: no hourly data returned.")
                continue

            signals = build_signals(ohlcv_1h)

            ohlcv_1m = get_1m_data(
                exchange=exchange,
                symbol=symbol,
                start_date=backtest_config["start_date"],
                end_date=backtest_config["end_date"],
            )
            if ohlcv_1m.empty:
                print(f"Skipping {combo}: no 1-minute data returned.")
                continue

            backtest_results[combo] = run_backtest(ohlcv_1m, signals, backtest_config)

    run(backtest_results)