# crypto_pipeline/stats/metrics.py

"""
metrics.py
----------
Uses Python's `inspect` module to dynamically discover every public
function in quantstats.stats and call it with a returns series, instead
of hardcoding a fixed list of metric names. Adding a new metric later (or
picking up a new one after a quantstats upgrade) needs no changes here --
it's discovered automatically.

Almost every quantstats.stats function takes `returns` (or `series`) as
its first positional argument. The couple of exceptions:
  - max_drawdown(prices)         -- wants a price/equity series, not returns
  - probabilistic_*(series, ...) -- same shape as returns, just named differently
  - anything needing `benchmark` -- skipped, we don't have one (see config.yaml)

So rather than a big if/elif per metric name, we inspect each function's
signature once and route the right input (returns vs. equity) to it.
"""

import inspect
import numpy as np
import pandas as pd
import quantstats as qs


# Functions in quantstats.stats whose first argument is a price/equity
# series rather than a returns series.
_PRICE_INPUT_METRICS = {"max_drawdown"}

# Non-metric exports living in quantstats.stats (type aliases, submodules,
# helpers) -- not callable per-return-series metrics, always skipped.
_NON_METRIC_NAMES = {
    "Literal", "NDArray", "Returns", "distribution", "warn",
    "validate_input", "safe_concat", "drawdown_details",
    "montecarlo", "montecarlo_cagr", "montecarlo_drawdown", "montecarlo_sharpe",
    # These return a full series/DataFrame (one value per date), not a
    # single scalar summary number, so they don't belong in metrics.json --
    # plots.py already covers this ground properly as actual plot data.
    "compsum", "to_drawdown_series", "rolling_sharpe", "rolling_sortino",
    "rolling_volatility", "pct_rank", "remove_outliers", "outliers",
    "implied_volatility", "monthly_returns",
}


def discover_metrics(exclude: list = None) -> dict:
    """
    Inspects quantstats.stats and returns {name: function} for everything
    that looks like a callable, single-series metric.
    """
    exclude = set(exclude or [])
    discovered = {}

    for name, func in inspect.getmembers(qs.stats, inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in _NON_METRIC_NAMES or name in exclude:
            continue

        params = inspect.signature(func).parameters
        first_arg = next(iter(params), None)
        if first_arg not in ("returns", "series", "prices"):
            continue
        # needs a second series (benchmark) we don't have -- skip
        if "benchmark" in params and params["benchmark"].default is inspect.Parameter.empty:
            continue

        discovered[name] = func

    return discovered


def _call_metric(func, returns: pd.Series, equity: pd.Series, rf: float, periods: int):
    """Calls one quantstats.stats function with whatever input/kwargs it accepts."""
    name = func.__name__
    params = inspect.signature(func).parameters

    series = equity if name in _PRICE_INPUT_METRICS else returns

    kwargs = {}
    if "rf" in params:
        kwargs["rf"] = rf
    if "periods" in params:
        kwargs["periods"] = periods

    return func(series, **kwargs)


def compute_all_metrics(returns: pd.Series, equity: pd.Series, rf: float = 0.0,
                         periods: int = 252, exclude: list = None) -> dict:
    """
    Computes every discoverable quantstats.stats metric for one returns
    series. Returns {metric_name: value}. A metric that errors on this
    particular data (e.g. too few points) is recorded as None rather than
    aborting the whole run.
    """
    metrics = discover_metrics(exclude=exclude)
    results = {}

    for name, func in metrics.items():
        try:
            results[name] = _call_metric(func, returns, equity, rf, periods)
        except Exception:
            results[name] = None

    return results