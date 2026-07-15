# crypto_pipeline/stats/utils.py

"""
utils.py
--------
Small shared helpers for the stats module. Kept separate from calculator.py
so JSON-safety isn't tangled up with the actual metric-computation logic.
"""

import numpy as np
import pandas as pd


def to_json_safe(value):
    """
    Recursively convert a value into something json.dump() can handle.

    quantstats functions return a mix of numpy scalars, pandas
    Series/Timestamps, and plain floats -- none of which json.dump()
    accepts directly except plain floats. NaN/inf are also converted to
    None since JSON has no representation for them.
    """
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return {str(k): to_json_safe(v) for k, v in value.to_dict().items()}
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if value is None or isinstance(value, str):
        return value
    # last resort -- anything else (namedtuples, custom objects) as string
    return str(value)


def equity_to_returns(equity_curve: pd.Series, resample_freq: str = "D") -> pd.Series:
    """
    Turn a 1-min (or any sub-daily) equity curve into a resampled returns
    series that quantstats' annualized ratios (periods_per_year=252
    assumes daily) are actually meaningful on.

    equity_curve : datetime-indexed balance series, e.g. run_backtest()'s
        "equity_curve" -- flat between trades, steps at each exit.
    """
    resampled = equity_curve.resample(resample_freq).last().ffill()
    returns = resampled.pct_change().dropna()
    return returns