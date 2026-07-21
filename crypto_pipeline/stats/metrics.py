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


def _safe_max_drawdown(equity: pd.Series) -> float:
    """
    Compute max drawdown directly from an equity/price series, bypassing
    quantstats.stats.max_drawdown()'s baseline-guessing heuristic.

    quantstats==0.0.81's max_drawdown() tries to guess whether the first
    value came from its own to_prices() conversion (which defaults to a
    base of 1e5) purely from magnitude: any first_price > 1000 is assumed
    to have that 100,000 baseline, and the "no drawdown" reference point
    is hardcoded to 1e5 rather than the series' own actual starting value
    (see quantstats.stats._get_baseline_value). A real equity curve that
    just happens to start around a normal account balance -- e.g. $10,000
    -- falls squarely in that ">1000" bucket, so quantstats compares
    ~$10,000 against a baseline of $100,000 and reports a drawdown near
    -90% even when the account barely moved. This is a quantstats bug, not
    a property of our data -- confirmed by reproducing it directly against
    _get_baseline_value's source.

    True max drawdown is just: min over time of (equity / running-peak - 1),
    using the series' own actual starting value as the initial peak -- no
    baseline guessing needed.
    """
    if len(equity) == 0:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def _call_metric(func, returns: pd.Series, equity: pd.Series, rf: float, periods: int):
    """Calls one quantstats.stats function with whatever input/kwargs it accepts."""
    name = func.__name__

    # max_drawdown specifically bypasses quantstats entirely -- see
    # _safe_max_drawdown()'s docstring for why.
    if name == "max_drawdown":
        return _safe_max_drawdown(equity)

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