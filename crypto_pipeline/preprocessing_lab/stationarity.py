# preprocessing_lab/preprocessing/stationarity.py

"""
stationarity.py
----------------
Methods that target NON-STATIONARITY directly (unlike scalers, which
only rescale magnitude but leave the underlying trend/drift untouched).
"""

import numpy as np
import pandas as pd


def _frac_diff_weights(d: float, size: int, threshold: float = 1e-4):
    """
    Compute the binomial-series weights for fractional differencing.
    Weights decay toward zero; we cut off once a weight's magnitude drops
    below `threshold` (standard practical approximation, since the true
    series is infinite).
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold or k >= size:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights[::-1])


def apply_fractional_differencing(df: pd.DataFrame, fit_mask=None, d=0.4, threshold=1e-4):
    """
    Fractional Differencing (per column), based on Marcos Lopez de
    Prado's method (Advances in Financial Machine Learning).

    Purpose: ordinary differencing (d=1, i.e. just x[t]-x[t-1]) makes a
    series stationary but throws away ALL memory/trend information --
    the series becomes pure noise-like returns. Fractional differencing
    uses a FRACTIONAL order d (e.g. 0.3, 0.4, 0.5 instead of a full 1.0)
    -- just enough differencing to achieve stationarity while retaining
    as much memory/trend signal as possible. This is THE standard
    technique in quant finance specifically for this trade-off.

    Advantages:
        - Achieves stationarity (passes ADF test) with a much smaller d
          than full differencing typically needs, preserving more
          long-term memory/trend than returns-based (d=1) series.
        - Directly targets the exact problem named in the task doc:
          non-stationarity, while still caring about trend preservation.

    Disadvantages:
        - Requires tuning `d` per series (there's a search process:
          increase d until ADF test passes, stop there -- "minimum
          effective d"). We are NOT auto-tuning it here (kept as a config
          param) -- the analysis step should run the ADF test across a
          few d values and report which is minimal-sufficient. This is
          exactly what belongs in the "Stationarity Analysis" step of the
          project.
        - Loses the first several rows to NaN (weights need `size` prior
          points to compute -- similar to rolling window methods).
        - More computationally expensive and less intuitive than simple
          differencing.

    Suitable models: Especially useful before feeding a series into
    models sensitive to non-stationarity, or before running further
    statistical analysis expecting a stationary process (e.g. many
    classical time-series stats/tests assume stationarity to begin with).

    Preserves trend: YES -- this is the entire point of the method vs.
    plain differencing; higher d = closer to raw series (more memory,
    less stationary), lower d = closer to plain diff (less memory, more
    stationary). d is a dial between the two extremes.
    Improves stationarity: YES -- that is the explicit goal.
    Reversible: NOT simply -- fractional differencing is not exactly
    invertible with a closed-form inverse the way linear scalers are;
    reconstructing the original series requires the full weight series
    and cumulative application, which is out of scope for a simple
    inverse function here. Report should state this explicitly as a
    known limitation.
    """
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    weights = None
    for col in df.columns:
        series = df[col].values
        weights = _frac_diff_weights(d, len(series), threshold)
        w_len = len(weights)
        result = np.full(len(series), np.nan)
        for i in range(w_len - 1, len(series)):
            window = series[i - w_len + 1 : i + 1]
            result[i] = np.dot(weights, window)
        out[col] = result

    fit_info = {
        "method": "fractional_differencing",
        "d": d,
        "threshold": threshold,
        "weight_length": len(weights) if weights is not None else None,
        "note": "first (weight_length - 1) rows are NaN by construction",
    }
    return out, fit_info
# ---------------------------------------------------------------------
# Extra stationarity methods (alongside Fractional Differencing)
# ---------------------------------------------------------------------

def apply_simple_differencing(df: pd.DataFrame, fit_mask=None, order=1):
    """
    Simple/first-order differencing: x_diff[t] = x[t] - x[t-1]
    (order=1 is standard; order=2 means difference-of-differences, rarely needed here)

    Purpose: the "full-strength" version of Fractional Differencing
    (which uses a FRACTIONAL d like 0.3-0.5). This uses d=1 exactly --
    removes ALL memory of the series' level, keeping only the change
    between consecutive points. Directly comparable against Fractional
    Differencing in the report: same family, opposite end of the d-scale.

    Advantages:
        - Extremely effective at achieving stationarity -- almost always
          passes ADF/KPSS after one difference for financial price data.
        - Simple, fast, exact, no tuning needed (no d to choose).

    Disadvantages:
        - Destroys ALL long-term memory/trend information -- the series
          becomes close to noise. This is the exact trade-off Fractional
          Differencing was invented to avoid (see stationarity.py's first
          function for full comparison).
        - Loses the first `order` row(s) to NaN.

    Suitable models: Any model where only short-term change matters, not
    absolute level (e.g. momentum-style strategies). Poor fit if the
    model needs to reason about the actual price/RSI level.

    Preserves trend: NO -- this is the opposite intent of Fractional
    Differencing; nearly all memory is deliberately destroyed.
    Improves stationarity: YES, usually very strongly (often the
    strongest of any method here).
    Reversible: YES via cumulative sum, IF you keep the first original
    value(s) -- x[t] = x[t-1] + x_diff[t]. Not implemented as an inverse
    function here since it requires carrying the seed value separately;
    flagged as a report note.
    """
    out = df.diff(periods=order)
    fit_info = {"method": "simple_differencing", "order": order,
                "note": f"first {order} row(s) are NaN by construction"}
    return out, fit_info


def apply_log_returns(df: pd.DataFrame, fit_mask=None):
    """
    Log returns: log(x[t] / x[t-1]) = log(x[t]) - log(x[t-1])

    Purpose: THE standard transform in quantitative finance for price
    series. Converts a trending, multiplicative price series into an
    additive, roughly stationary "percent change" series -- a 2% move
    looks the same whether price is $1,000 or $100,000.

    Advantages:
        - Very well-behaved statistically (closer to normal than raw
          returns), the default choice in most quant research and
          academic finance.
        - Naturally handles the "different price levels over time"
          problem central to non-stationarity.

    Disadvantages:
        - Requires strictly positive values (fails on MACD-family columns
          which go negative, same caveat as apply_log_transform in
          distribution.py) -- only suitable for open/high/low/close/
          volume/EMA-type columns, not signed indicator columns.
        - Loses the first row to NaN (needs x[t-1]).
        - Like simple differencing, discards the absolute price LEVEL --
          only relative change survives.

    Suitable models: Extremely common preprocessing for any model
    predicting price direction/movement rather than absolute price level
    -- fits this dataset's classification target well conceptually
    (triple-barrier labeling is itself about relative moves, not
    absolute price).

    Preserves trend: PARTIALLY -- local trend (recent momentum) is
    clearly visible in the returns series; long-term absolute trend
    (the fact that price grew from $X to $Y over the year) is removed.
    Improves stationarity: YES, one of the strongest and most standard
    stationarity fixes for price data specifically.
    Reversible: YES via cumulative sum + exp, given a starting price --
    x[t] = x[0] * exp(cumsum(log_returns)). Not implemented as an inverse
    function here since it needs the seed value; noted as a limitation.
    """
    if (df <= 0).any().any():
        cols_with_nonpositive = df.columns[(df <= 0).any()].tolist()
        raise ValueError(
            f"apply_log_returns: columns contain zero/negative values, "
            f"log() is undefined there: {cols_with_nonpositive}. "
            f"Route signed columns (e.g. MACD-family) to a different method."
        )
    out = np.log(df / df.shift(1))
    fit_info = {"method": "log_returns", "note": "first row is NaN by construction"}
    return out, fit_info


def apply_pct_change(df: pd.DataFrame, fit_mask=None):
    """
    Percentage change: (x[t] - x[t-1]) / x[t-1]

    Purpose: same GOAL as log returns (express change as a relative
    percentage, not absolute units) but using a LINEAR formula instead
    of a log formula. Good direct contrast case against log returns in
    the report: linear vs log detrending.

    Advantages:
        - More directly interpretable than log returns ("this went up
          2.3%" is exactly the pct_change value, whereas log returns need
          a small conversion to read as an intuitive percentage).
        - Works on zero values (log returns don't) -- though still
          undefined at x[t-1] == 0 exactly (division by zero).

    Disadvantages:
        - NOT symmetric the way log returns are: a +10% move followed by
          a -10% move does NOT bring you back to the start in pct_change
          terms, but it does (very nearly) in log-return terms. This
          asymmetry is a well-known reason quant finance prefers log
          returns for anything involving compounding across many periods
          -- worth stating explicitly in the report as the key trade-off
          vs apply_log_returns.
        - Loses the first row to NaN.

    Suitable models: Similar use case to log returns -- relative-move
    prediction rather than absolute level. Slightly more intuitive for
    non-technical reporting/interpretation.

    Preserves trend: PARTIALLY -- same as log returns, local trend
    visible, long-term absolute level removed.
    Improves stationarity: YES, similar strength to log returns for this
    kind of price data.
    Reversible: YES via cumulative product, given a starting price --
    not implemented as an inverse function here (needs seed value).
    """
    out = df.pct_change(periods=1)
    fit_info = {"method": "pct_change", "note": "first row is NaN by construction"}
    return out, fit_info


def apply_moving_average_detrend(df: pd.DataFrame, fit_mask=None, window=24):
    """
    Detrending via moving average subtraction:
        x_detrended[t] = x[t] - rolling_mean(x, window)[t]

    Purpose: removes the LOCAL trend/level by subtracting a smoothed
    baseline (default window=24 hours = 1 day for hourly data), while
    keeping short-term fluctuations around that baseline. Different
    mechanism entirely from differencing/returns -- doesn't look at
    point-to-point change, instead asks "how far is this point from its
    recent local average?"

    Advantages:
        - Directly interpretable: positive value = currently above its
          recent local trend, negative = below.
        - Keeps the fluctuation MAGNITUDE in original units (unlike log
          returns which convert to a %-based scale) -- useful if you
          want to reason about deviations in the same units as the raw
          data.
        - Conceptually simple, easy to explain in the report.

    Disadvantages:
        - Choice of window size is a real hyperparameter that changes
          results substantially (short window = removes only very recent
          trend, long window = closer to no detrending at all).
        - Loses the first `window-1` rows to NaN (needs a full window to
          compute the rolling mean).
        - Does not address heteroskedasticity (changing volatility) the
          way rolling z-score does -- only removes LEVEL drift, not
          SPREAD drift.

    Suitable models: Useful for any model that should reason about
    "deviation from recent normal" rather than either absolute level or
    percentage change -- a genuinely different framing from either.

    Preserves trend: LOCAL trend is preserved (fluctuations around the
    baseline are the whole point); LONG-TERM trend is explicitly removed
    (that's what "detrend" means here) -- similar framing to rolling
    z-score in scalers.py, but this only removes the mean, not the
    spread/variance too.
    Improves stationarity: YES for level-drift; does not address
    variance-drift (changing volatility) on its own.
    Reversible: YES if you keep the rolling mean series alongside the
    output -- x[t] = x_detrended[t] + rolling_mean[t]. Not implemented
    as inverse here since it needs the stored rolling mean.
    """
    rolling_mean = df.rolling(window=window, min_periods=window).mean()
    out = df - rolling_mean
    fit_info = {
        "method": "moving_average_detrend",
        "window": window,
        "note": f"first {window - 1} rows are NaN by construction",
    }
    return out, fit_info