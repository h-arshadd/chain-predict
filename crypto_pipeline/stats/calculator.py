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
import warnings

from crypto_pipeline.stats import metrics, plots
from crypto_pipeline.stats.utils import equity_to_returns, to_json_safe

# quantstats' stats.py raises RuntimeWarning ("Mean of empty slice", "invalid
# value encountered in sqrt"/"scalar divide") for metrics like c_var/sharpe
# when a strategy has too few (or all-winning/all-losing) trades for that
# metric to be meaningful -- NaN is already the correct, expected output in
# that case, so these are just numpy/quantstats being noisy about it, not a
# bug in our data. Scoped to those two modules only, so warnings from our
# own code are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="quantstats")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Approximate trading periods/year for each fallback frequency
# equity_to_returns() may resample down to, so periods_per_year (used to
# annualize sharpe/sortino/calmar/etc.) matches whatever frequency the
# returns series actually ended up at -- NOT always the config's daily
# default. Getting this wrong doesn't change the returns themselves, but
# it does silently mis-scale every annualized ratio.
_PERIODS_PER_YEAR_BY_FREQ = {
    "D": 252,
    "12h": 252 * 2,
    "6h": 252 * 4,
    "1h": 252 * 24,
    "30min": 252 * 24 * 2,
    "15min": 252 * 24 * 4,
    "5min": 252 * 24 * 12,
    "1min": 252 * 24 * 60,
}

# Below this many returns, quantstats' ratios (sharpe/sortino/calmar/
# max_drawdown/etc.) are not statistically meaningful even if they compute
# without error -- e.g. a single return produces near-identical-looking
# "stats" for every strategy regardless of actual performance (this is
# exactly what was happening before this fix, when a ~2-day-old dataset
# collapsed to 1 daily data point under a fixed "D" resample). Rather
# than silently returning numbers that look legitimate but aren't, metrics
# are explicitly nulled out and flagged below this threshold.
MIN_RETURNS_FOR_RELIABLE_METRICS = 10


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
          "metrics": {...every discovered quantstats stat, or all None
                       with "insufficient_data": true if the resampled
                       returns series is too short to be meaningful...},
          "trade_summary": {...pulled straight from backtest_result...},
          "plots": {...numeric data behind each configured plot...},
          "resample_freq_used": the frequency returns were actually
                       resampled at (may differ from config's
                       resample_freq if too few points forced a
                       fallback to a finer frequency -- see
                       equity_to_returns()),
        }
    """
    equity = backtest_result["equity_curve"]
    configured_freq = config.get("resample_freq", "D")
    min_periods = config.get("min_periods_for_stats", MIN_RETURNS_FOR_RELIABLE_METRICS)

    returns, freq_used = equity_to_returns(
        equity,
        resample_freq=configured_freq,
        min_periods=min_periods,
        auto_adjust_freq=config.get("auto_adjust_resample_freq", True),
    )

    periods = _PERIODS_PER_YEAR_BY_FREQ.get(freq_used, config.get("periods_per_year", 252))

    insufficient_data = len(returns) < min_periods

    if insufficient_data:
        # Still discover the metric names (so the DB/JSON schema stays
        # consistent whether or not this run had enough data), but don't
        # compute them -- with this few data points quantstats' ratios
        # would compute without erroring while being statistically
        # meaningless, which is worse than an explicit gap.
        discovered_names = metrics.discover_metrics(exclude=config.get("exclude_metrics"))
        computed_metrics = {name: None for name in discovered_names}
    else:
        computed_metrics = metrics.compute_all_metrics(
            returns=returns,
            equity=equity,
            rf=config.get("risk_free_rate", 0.0),
            periods=periods,
            exclude=config.get("exclude_metrics"),
        )

    plot_data = {}
    if config.get("generate_plots", True) and not insufficient_data:
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
        "resample_freq_used": freq_used,
        "insufficient_data": insufficient_data,
        "returns_count": len(returns),
    })


def save_stats(stats_dict: dict, out_path: str):
    """Saves compute_stats()'s output as a single JSON file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=2)