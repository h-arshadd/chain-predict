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


def equity_to_returns(equity_curve: pd.Series, resample_freq: str = "D",
                       min_periods: int = 10, auto_adjust_freq: bool = True) -> pd.Series:
    """
    Turn a 1-min (or any sub-daily) equity curve into a resampled returns
    series that quantstats' annualized ratios (periods_per_year=252
    assumes daily) are actually meaningful on.

    equity_curve : datetime-indexed balance series, e.g. run_backtest()'s
        "equity_curve" -- flat between trades, steps at each exit.

    IMPORTANT -- the daily-resample trap:
    resample_freq="D" silently collapses an equity curve that only spans
    a few calendar days down to just 1-2 data points (resample("D").last()
    keeps only the final value per calendar day). pct_change().dropna()
    on 1-2 points yields a returns series of length 0 or 1 -- every
    quantstats metric computed on a single return degenerates to near-
    identical, statistically meaningless values across every strategy
    (this is exactly the bug that made every strategy's stats look
    suspiciously alike when the simulator had only ~2 days of history).
    This doesn't raise or warn on its own -- the numbers still *look*
    like real stats, they just aren't.

    To guard against this silently happening again as new strategies get
    reset or as data windows shrink for any other reason:
      - auto_adjust_freq (default True): if resampling at resample_freq
        would yield fewer than min_periods returns, step down through
        progressively finer frequencies (H -> 30min -> 15min -> 5min ->
        1min) until min_periods is met or 1-minute resolution is reached.
        This keeps annualization roughly sane (periods_per_year should be
        adjusted by the caller to match whatever frequency was actually
        used -- see compute_stats(), which now returns the frequency it
        picked).
      - Callers needing the exact configured frequency with no fallback
        can pass auto_adjust_freq=False.

    Returns
    -------
    (returns, freq_used) : tuple[pd.Series, str]
        freq_used is whichever frequency the series was actually resampled
        at (may differ from resample_freq if auto_adjust_freq kicked in).
    """
    candidate_freqs = [resample_freq]
    if auto_adjust_freq:
        # Ordered finest-to-coarsest fallback ladder, only used if the
        # requested frequency doesn't yield enough data points. "D" is not
        # re-added here even if resample_freq wasn't "D" -- we only ever
        # step DOWN to finer resolution, never up to coarser.
        fallback_ladder = ["12h", "6h", "1h", "30min", "15min", "5min", "1min"]
        candidate_freqs += [f for f in fallback_ladder if f != resample_freq]

    freq_used = resample_freq
    returns = pd.Series(dtype=float)
    for freq in candidate_freqs:
        resampled = equity_curve.resample(freq).last().ffill()
        candidate_returns = resampled.pct_change().dropna()
        freq_used = freq
        returns = candidate_returns
        if len(candidate_returns) >= min_periods or not auto_adjust_freq:
            break
        # else: too few points at this frequency -- try the next finer one

    return returns, freq_used