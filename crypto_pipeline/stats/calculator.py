# crypto_pipeline/stats/calculator.py

"""
calculator.py
-------------
Main interface for the Statistics Module. Takes the dict run_backtest()
already returns (equity_curve, trade_ledger, etc.) directly -- no CSV
round-trip, no re-deriving anything backtest.py already computed.

    from crypto_pipeline.backtest.backtest import run_backtest
    from crypto_pipeline.stats.calculator import compute_stats

    result = run_backtest(ohlcv_1m, signals, backtest_config)
    stats = compute_stats(result, config)

Public signature and returned dict shape (metrics / trade_summary /
plots) are unchanged from before -- the ml module (ml/evaluation/
evaluator.py) calls compute_stats(backtest_result, stats_config,
plot_dir=plot_dir) and only ever reads stats["metrics"] / stats["trade_summary"],
so it's unaffected by the change below. The only thing that changed is
*what* ends up under stats["plots"]: numeric plot data (JSON-safe dicts of
series/tables) instead of a list of saved .png paths. plot_dir is now just
used as a JSON-safety label passthrough for callers that want to persist
the plot data themselves (see stats_runner.py / save_stats()); no images
are written to it.
"""

import os
import json

from crypto_pipeline.stats import metrics, plots
from crypto_pipeline.stats.utils import equity_to_returns, to_json_safe


def compute_stats(backtest_result: dict, config: dict, plot_dir: str = None) -> dict:
    """
    Parameters
    ----------
    backtest_result : dict
        Whatever run_backtest() returned (needs "equity_curve" at minimum).
    config : dict
        Loaded from stats/config.yaml.
    plot_dir : str, optional
        Unused for file output now (no PNGs are written). Kept only so
        existing callers (e.g. ml/evaluation/evaluator.py) that pass
        plot_dir= don't need to change. Plot *data* is generated whenever
        config["generate_plots"] is true, regardless of this argument.

    Returns
    -------
    dict, JSON-safe:
        {
          "metrics": {...every discovered quantstats stat...},
          "trade_summary": {...pulled straight from backtest_result...},
          "plots": {...numeric data behind each configured plot...},
        }
    """
    equity = backtest_result["equity_curve"]
    returns = equity_to_returns(equity, config.get("resample_freq", "D"))

    computed_metrics = metrics.compute_all_metrics(
        returns=returns,
        equity=equity,
        rf=config.get("risk_free_rate", 0.0),
        periods=config.get("periods_per_year", 252),
        exclude=config.get("exclude_metrics"),
    )

    plot_data = {}
    if config.get("generate_plots", True):
        plot_data = plots.generate_plot_data(returns, equity, config.get("plots", []))

    trade_summary = {
        "final_balance": backtest_result.get("final_balance"),
        "total_net_profit": backtest_result.get("total_net_profit"),
        "total_trades": backtest_result.get("total_trades"),
        "win_loss": backtest_result.get("win_loss"),
    }

    return to_json_safe({
        "metrics": computed_metrics,
        "trade_summary": trade_summary,
        "plots": plot_data,
    })


def save_stats(stats_dict: dict, out_path: str):
    """Saves compute_stats()'s output as a single JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=2)