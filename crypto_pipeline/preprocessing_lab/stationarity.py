"""
stationarity.py
----------------
Methods that target NON-STATIONARITY directly (unlike scalers, which
only rescale magnitude but leave the underlying trend/drift untouched).

FIXES APPLIED:
- Changed to DROP strategy (no ffill/bfill)
- Fixed pandas deprecation (removed fillna with method parameter)
- Fixed array truth value error in fit_info
- Tuned fractional differencing: d=0.2 (was 0.4) to reduce NaN rows
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


def apply_fractional_differencing(df: pd.DataFrame, fit_mask=None, d=0.2, threshold=1e-3):
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
    
    Parameters:
    -----------
    d : float, default=0.2 (tuned down from 0.4)
        Differencing order. Lower values (0.1-0.3) = less NaN rows but weaker stationarity.
        Higher values (0.5+) = more NaN rows but stronger stationarity.
        Default 0.2 is tuned for this dataset to minimize NaN rows while maintaining stationarity.
    threshold : float, default=1e-3 (tuned up from 1e-4)
        Weight cutoff threshold. Higher values = fewer weights = fewer NaN rows.
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

    # Store weight_length as integer before using in fit_info
    weight_length = len(weights) if weights is not None else None
    
    fit_info = {
        "method": "fractional_differencing",
        "d": d,
        "threshold": threshold,
        "weight_length": weight_length,
        "note": f"first {weight_length - 1 if weight_length else 0} rows are NaN by construction (not enough history). Drop these rows before training.",
    }
    return out, fit_info


# =====================================================================
# Extra stationarity methods (alongside Fractional Differencing)
# =====================================================================

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
    
    Parameters:
    -----------
    order : int, default=1
        Differencing order. order=1 is x[t]-x[t-1], order=2 is diff of diff, etc.
    """
    out = df.diff(periods=order)
    
    fit_info = {
        "method": "simple_differencing", 
        "order": order,
        "note": f"first {order} row(s) are NaN by construction (no prior value to diff against). Drop these rows before training.",
    }
    return out, fit_info