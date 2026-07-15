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
        Where to save plots. Skipped if not given, even if
        config["generate_plots"] is true.

    Returns
    -------
    dict, JSON-safe:
        {
          "metrics": {...every discovered quantstats stat...},
          "trade_summary": {...pulled straight from backtest_result...},
          "plots": [<paths saved>],
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

    saved_plots = []
    if plot_dir and config.get("generate_plots", True):
        saved_plots = plots.generate_plots(returns, plot_dir, config.get("plots", []))

    trade_summary = {
        "final_balance": backtest_result.get("final_balance"),
        "total_net_profit": backtest_result.get("total_net_profit"),
        "total_trades": backtest_result.get("total_trades"),
        "win_loss": backtest_result.get("win_loss"),
    }

    return to_json_safe({
        "metrics": computed_metrics,
        "trade_summary": trade_summary,
        "plots": saved_plots,
    })


def save_stats(stats_dict: dict, out_path: str):
    """Saves compute_stats()'s output as a single JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=2)