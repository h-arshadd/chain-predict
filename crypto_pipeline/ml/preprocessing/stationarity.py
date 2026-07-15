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
        "note": f"first {weight_length - 1 if weight_length else 0} rows are NaN by construction (not enough history). Drop these rows before training.",
    }
    return out, fit_info


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
        "note": f"first {order} row(s) are NaN by construction (no prior value to diff against). Drop these rows before training.",
    }
    return out, fit_info


PREPROCESSING_STATIONARITY: Dict[str, Callable] = {
    "fractional_differencing": apply_fractional_differencing,
    "simple_differencing": apply_simple_differencing,
}