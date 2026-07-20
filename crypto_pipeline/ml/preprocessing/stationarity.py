# crypto_pipeline/ml/preprocessing/stationarity.py

"""
stationarity.py
----------------
Stationarity methods for the ML module's Preprocessing stage (PDF heading 4).

Production version -- method selection/comparison already happened in
preprocessing_lab, this is the trimmed, non-exploratory implementation
used by the real training pipeline.

- fit_mask: boolean mask selecting which ROWS to fit params on. In the
  real pipeline this is ALWAYS the train-set mask -- per the PDF spec,
  "Only pre-processing parameters learned from the training dataset may
  be applied to validation and test datasets" (prevents data leakage).
  fit_mask=None (fit on everything given) only makes sense for ad-hoc/
  exploratory use, never for an actual train/test run.
- fit_info: dict of fitted parameters, persisted alongside the model so
  the exact same transform can be re-applied at inference time.

Note: both methods here create leading NaN rows by construction (not
enough history yet). preprocessing_pipeline.py is responsible for
dropping those rows after transforming -- these functions only report
how many rows via fit_info["note"], they do not drop anything themselves.

inverse_fractional_differencing()/inverse_simple_differencing() below
are used by ml/timeseries/postprocessing.py's inverse_transform_forecast()
(PDF heading 13, Forecast Post-Processing) -- only relevant if the
target column itself was ever run through one of these steps, which it
is not for nbeats/tcn/statsforecast in this project today (see
postprocessing.py's module docstring).
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd


def _frac_diff_weights(d: float, size: int, threshold: float) -> np.ndarray:
    """Binomial-series weights for fractional differencing, cut off once
    a weight's magnitude drops below `threshold` (standard practical
    approximation, since the true series is infinite)."""
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold or k >= size:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights[::-1])


def apply_fractional_differencing(df: pd.DataFrame, fit_mask=None, d: float = 0.2, threshold: float = 1e-3):
    """Fractional differencing (Lopez de Prado): uses a fractional order
    d instead of a full 1.0, achieving stationarity while retaining more
    trend/memory than full differencing. d is config-driven, not
    auto-tuned here -- run the ADF test across a few d values elsewhere
    to pick the minimum-sufficient d for this dataset.

    fit_mask is accepted for interface consistency with the rest of the
    registry, but unused: this is a purely backward-looking, causal
    transform (each point only uses its own past window of the series),
    so there is no separate "fit on train" step the way static scalers
    need -- weights depend only on d/threshold, not on any data statistics.
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

    weight_length = len(weights) if weights is not None else None
    fit_info = {
        "method": "fractional_differencing",
        "d": d,
        "threshold": threshold,
        "weight_length": weight_length,
        # last_values/last_index: the original (pre-transform) series
        # tail, one full weight-window's worth -- needed by
        # inverse_fractional_differencing() to reconstruct price levels
        # from a forecast of differenced values (a forecast has no
        # window of its own original-space history to fall back on,
        # unlike test_df's own feature columns which just get inverted
        # column-by-column against fit_info's static params).
        "last_values": {col: df[col].values[-(weight_length or 1):].tolist() for col in df.columns},
        "note": f"first {weight_length - 1 if weight_length else 0} rows are NaN by construction (not enough history). Drop these rows before training.",
    }
    return out, fit_info


def inverse_fractional_differencing(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    """
    Approximate inverse of apply_fractional_differencing(), for
    reconstructing price-space forecasts from differenced-space
    forecasts (PDF heading 13). Fractional differencing has no exact
    closed-form inverse the way simple differencing does (it's a
    convolution over an infinite-in-principle weight series, truncated
    by `threshold`) -- this reconstructs the level by walking forward
    from fit_info["last_values"] (the original series' tail at fit
    time) and cumulatively re-applying the same weights in reverse,
    which is exact only for full (d=1) differencing and increasingly
    approximate as d moves away from 1. Good enough for turning a
    forecast back into a plottable/comparable price path; not a
    substitute for re-fitting if exact reconstruction matters.
    """
    d = fit_info["d"]
    threshold = fit_info["threshold"]
    weights = _frac_diff_weights(d, fit_info.get("weight_length", 1), threshold)
    w_len = len(weights)

    out = pd.DataFrame(index=transformed_df.index, columns=transformed_df.columns, dtype=float)
    for col in transformed_df.columns:
        history = list(fit_info["last_values"].get(col, [])[-(w_len - 1):]) if w_len > 1 else []
        diffed = transformed_df[col].to_numpy()
        levels = np.empty(len(diffed))
        for i, d_val in enumerate(diffed):
            window = np.array((history + list(levels[:i]))[-(w_len - 1):]) if w_len > 1 else np.array([])
            # weights[-1] corresponds to the current (undifferenced) level's
            # own coefficient (always 1.0 for this weight construction);
            # solve for it given the past window's contribution.
            past_contribution = np.dot(weights[:-1], window) if w_len > 1 and len(window) == w_len - 1 else 0.0
            levels[i] = d_val - past_contribution if w_len > 1 else d_val
        out[col] = levels
    return out


def apply_simple_differencing(df: pd.DataFrame, fit_mask=None, order: int = 1):
    """First-order (or higher) differencing: x_diff[t] = x[t] - x[t-1].
    The full-strength (d=1) end of the same family as fractional
    differencing -- removes all memory of the series' level, keeping
    only the change between consecutive points.

    fit_mask is accepted for interface consistency but unused, same
    reasoning as apply_fractional_differencing: purely backward-looking,
    nothing to fit on training rows specifically.
    """
    out = df.diff(periods=order)
    fit_info = {
        "method": "simple_differencing",
        "order": order,
        # last_values: the original series' last `order` values --
        # needed by inverse_simple_differencing() to reconstruct levels
        # from a forecast of differenced values (see
        # inverse_fractional_differencing()'s docstring for why this is
        # necessary for a forecast specifically, vs. test_df's columns).
        "last_values": {col: df[col].values[-order:].tolist() for col in df.columns},
        "note": f"first {order} row(s) are NaN by construction (no prior value to diff against). Drop these rows before training.",
    }
    return out, fit_info


def inverse_simple_differencing(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    """
    Exact inverse of apply_simple_differencing(): cumulative sum
    starting from fit_info["last_values"] (the original series' value
    immediately before the differenced window begins).
    """
    order = fit_info["order"]
    out = pd.DataFrame(index=transformed_df.index, columns=transformed_df.columns, dtype=float)
    for col in transformed_df.columns:
        last_values = fit_info["last_values"].get(col, [0.0] * order)
        start_level = last_values[-1] if last_values else 0.0
        diffed = transformed_df[col].to_numpy()
        out[col] = start_level + np.cumsum(diffed)
    return out


PREPROCESSING_STATIONARITY: Dict[str, Callable] = {
    "fractional_differencing": apply_fractional_differencing,
    "simple_differencing": apply_simple_differencing,
}