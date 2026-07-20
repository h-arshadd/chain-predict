# crypto_pipeline/ml/timeseries/postprocessing.py

"""
postprocessing.py
-------------------
Forecast Post-Processing stage (PDF heading 13).

Applies to timeseries REGRESSION forecasts only (nbeats/tcn/
statsforecast, ml/timeseries/registry.py's TS_REGRESSORS) -- the
classification forecaster (sklearn_classifier) predicts discrete
labels directly, nothing to inverse-transform.

For this project, "if preprocessing was applied during training, all
inverse transformations should be applied before comparing forecasts
with actual values" mostly doesn't come up in practice: the target
column (raw close price) is deliberately EXCLUDED from
ml/config.yaml's preprocessing.steps chain for timeseries runs (see
timeseries_pipeline.py's docstring -- only covariate feature columns
get fractional_differencing/robust_scaler/etc, not the target itself,
because nbeats/tcn/statsforecast forecast the raw price directly).
This module exists for the case where a future model in this family
DOES need the target itself pre-transformed (e.g. a plain ARIMA-family
model that assumes stationarity) -- inverse_transform_forecast() below
replays ml/preprocessing's fit_objects (the same ones
preprocessing_pipeline.py already persists) against a 1-column
DataFrame of forecast values, using each method's own inverse_* function.

Also covers heading 13's other three post-processing examples:
    - confidence interval extraction (extract_confidence_intervals())
    - forecast smoothing (smooth_forecast())
    - missing forecast handling (fill_missing_forecast())
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from crypto_pipeline.ml.preprocessing.scalers import PREPROCESSING_SCALERS
from crypto_pipeline.ml.preprocessing.stationarity import PREPROCESSING_STATIONARITY

logger = logging.getLogger(__name__)

# Mirrors preprocessing_pipeline.py's PREPROCESSING_REGISTRY, but
# mapping each method name to its INVERSE function instead of its
# forward one -- built from the same two source-of-truth dicts so a
# new scaler/stationarity method only needs its inverse_* function
# added there, never duplicated here.
_INVERSE_REGISTRY = {
    "standard_scaler": "inverse_standard_scaler",
    "minmax_scaler": "inverse_minmax_scaler",
    "robust_scaler": "inverse_robust_scaler",
    "maxabs_scaler": "inverse_maxabs_scaler",
    "quantile_transformer": "inverse_quantile_transformer",
    "power_transformer": "inverse_power_transformer",
    "fractional_differencing": "inverse_fractional_differencing",
    "simple_differencing": "inverse_simple_differencing",
    # normalizer, rolling_zscore, winsorization have no inverse (see
    # scalers.py/stationarity.py's own fit_info["note"] on each) --
    # deliberately absent here; inverse_transform_forecast() raises a
    # clear error if one of these shows up in fit_objects.
}


def inverse_transform_forecast(forecast: np.ndarray, fit_objects: List[dict], column_name: str = "target") -> np.ndarray:
    """
    Replay the configured preprocessing chain's inverse, in REVERSE
    order, against a forecast array -- the same fit_objects
    preprocessing_pipeline.run_preprocessing() already returns and
    artifact_manager.save_run() already persists per run, so this uses
    the exact fitted params training used, never refitting anything.

    Only relevant if the target column itself went through
    preprocessing.steps (it does not, for nbeats/tcn/statsforecast in
    this project today -- see module docstring). Calling this with
    fit_objects=[] is a no-op, returning `forecast` unchanged.

    Args:
        forecast: 1D array of forecasted (still-transformed-space) values.
        fit_objects: list of {"method": str, "fit_info": dict}, in the
            order originally applied (forward order) -- same shape
            preprocessing_pipeline.py returns/persists. Applied here in
            REVERSE order, since undoing a chain means undoing the last
            step first.
        column_name: the fit_info dict's per-column keys (mean/std/etc)
            are keyed by the original DataFrame's column name(s) --
            since a forecast is a single series, this names that one
            column for the lookup. Must match the column name the
            preprocessing step was originally fit on.

    Returns:
        np.ndarray, same length as `forecast`, in original (price) space.
    """
    if not fit_objects:
        return forecast

    values = pd.DataFrame({column_name: forecast})
    for step in reversed(fit_objects):
        method_name = step["method"]
        fit_info = step["fit_info"]
        inverse_name = _INVERSE_REGISTRY.get(method_name)
        if inverse_name is None:
            raise ValueError(
                f"Preprocessing method '{method_name}' has no inverse transform "
                f"(see ml/preprocessing/scalers.py or stationarity.py's fit_info notes) "
                f"-- forecast cannot be mapped back to original price space."
            )
        # scalers.py/stationarity.py define inverse_* functions at
        # module level, not registered in PREPROCESSING_SCALERS/
        # PREPROCESSING_STATIONARITY (those only hold the forward
        # apply_* functions) -- resolved by name instead.
        inverse_fn = _resolve_inverse_fn(method_name)
        values = inverse_fn(values, fit_info)
        logger.info(f"Inverse-applied preprocessing step: {method_name}")

    return values[column_name].to_numpy()


def _resolve_inverse_fn(method_name: str):
    from crypto_pipeline.ml.preprocessing import scalers, stationarity
    inverse_name = _INVERSE_REGISTRY[method_name]
    module = stationarity if method_name in PREPROCESSING_STATIONARITY else scalers
    fn = getattr(module, inverse_name, None)
    if fn is None:
        raise ValueError(f"{inverse_name}() not found in {module.__name__} -- cannot inverse-transform '{method_name}'")
    return fn


def extract_confidence_intervals(model, n: int, num_samples: int = 100, **predict_kwargs) -> dict:
    """
    Confidence interval extraction (PDF heading 13). Only meaningful
    for probabilistic Darts models (those with a likelihood set, e.g.
    statsforecast's quantile support) -- calls the underlying Darts
    model's predict(num_samples=...) directly (not
    BaseTimeseriesModel.predict(), which always returns a flat 1D
    point-forecast array) to get the sampled distribution, then reduces
    it to lower/median/upper bounds.

    Args:
        model: a trained BaseTimeseriesModel (uses model.model, the
            underlying Darts object, directly).
        n: forecast horizon.
        num_samples: how many samples to draw per forecasted step.
        **predict_kwargs: forwarded to the Darts model's predict()
            (e.g. past_covariates=..., future_covariates=...).

    Returns:
        dict: {"lower": np.ndarray, "median": np.ndarray, "upper": np.ndarray},
        each length n (5th/50th/95th percentile across samples).
    """
    model._require_trained()
    forecast = model.model.predict(n=n, num_samples=num_samples, **predict_kwargs)
    samples = forecast.all_values()  # shape (n, 1, num_samples)
    samples = samples.reshape(n, -1)
    return {
        "lower": np.percentile(samples, 5, axis=1),
        "median": np.percentile(samples, 50, axis=1),
        "upper": np.percentile(samples, 95, axis=1),
    }


def smooth_forecast(forecast: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Forecast smoothing (PDF heading 13) -- simple centered rolling mean
    over the forecast path, to reduce step-to-step noise before
    handing the path to signal generation. window=1 (or forecast
    shorter than window) is a no-op. Edge points use whatever window
    is available (min_periods=1), so the output stays the same length
    as the input -- no NaNs introduced.
    """
    if window <= 1 or len(forecast) <= 1:
        return forecast
    series = pd.Series(forecast)
    smoothed = series.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.to_numpy()


def fill_missing_forecast(forecast: np.ndarray, method: str = "ffill") -> np.ndarray:
    """
    Missing forecast handling (PDF heading 13) -- some Darts models can
    return NaN for a step (e.g. a covariate gap, or an unstable
    statistical fit). Forward-fills by default (carries the last valid
    forecast forward); "interpolate" linearly interpolates between
    valid points instead. Raises if the forecast starts with NaN and
    method="ffill" (nothing valid yet to carry forward).

    Args:
        forecast: 1D array, possibly containing NaN.
        method: "ffill" or "interpolate".
    """
    if not np.isnan(forecast).any():
        return forecast

    series = pd.Series(forecast)
    if method == "ffill":
        if pd.isna(series.iloc[0]):
            raise ValueError(
                "fill_missing_forecast(method='ffill') can't fill a forecast "
                "that starts with NaN -- nothing valid yet to carry forward. "
                "Use method='interpolate' instead."
            )
        filled = series.ffill()
    elif method == "interpolate":
        filled = series.interpolate(limit_direction="both")
    else:
        raise ValueError(f"method must be 'ffill' or 'interpolate', got '{method}'")

    n_filled = series.isna().sum()
    logger.info(f"Filled {n_filled} missing forecast value(s) using method='{method}'")
    return filled.to_numpy()