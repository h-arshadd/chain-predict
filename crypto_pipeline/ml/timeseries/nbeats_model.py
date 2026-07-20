# crypto_pipeline/ml/timeseries/nbeats_model.py

"""
nbeats_model.py
----------------
N-BEATS (darts.models.NBEATSModel) wrapper. Pure deep-learning
forecasting architecture -- no manual differencing/detrending required
beforehand (N-BEATS handles non-stationarity internally), which is why
the raw close price is used as the target series rather than a
pre-computed return (see target_pipeline.py's timeseries branch).

Supports past_covariates natively (indicator/pattern/sentiment
columns), so config-selected features are never wasted. Does NOT
support future_covariates (N-BEATS is a past-covariates-only global
model) -- future_covariates is accepted for interface consistency with
BaseTimeseriesModel but ignored, with a warning if actually given.

Optional target scaling: scale_target=True (see BaseTimeseriesModel's
module docstring) fits a darts Scaler on the target series before
training and transparently inverse-transforms forecasts back to real
price space -- useful here specifically because N-BEATS still trains
on the raw close price's actual scale (unlike the stationarity/scaling
chain in preprocessing.steps, which only touches covariates, not this
target), and an unscaled large-magnitude target (e.g. ~1e5 for a
crypto close price) produces a very large, hard-to-read MSE loss.

historical_forecasts() (one-step and fixed-window walk-forward
evaluation) is inherited unchanged from BaseTimeseriesModel -- it just
calls self.model.historical_forecasts(), which NBEATSModel supports
natively via Darts, including the data_transformers-based scaling path
(see that method's docstring).
"""

import logging
from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import NBEATSModel
except ImportError:
    NBEATSModel = None

logger = logging.getLogger(__name__)


class NBEATSTimeseriesModel(BaseTimeseriesModel):
    """
    Args:
        scale_target: bool, default False -- see BaseTimeseriesModel's
            module docstring. Not a Darts constructor kwarg; popped off
            before the rest of hyperparams are forwarded.
        input_chunk_length: int, required -- how many past steps the
            model looks at to produce a forecast.
        output_chunk_length: int, required -- how many future steps
            it predicts per forward pass (this project uses 1, see
            ml/config.yaml).
        num_stacks, num_blocks, num_layers, layer_widths: architecture
            hyperparams (Darts defaults used if not given).
        n_epochs, batch_size, random_state: training hyperparams.
        Any other kwarg NBEATSModel's constructor accepts.
    """

    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        val_series: Optional["TimeSeries"] = None,
        val_past_covariates: Optional["TimeSeries"] = None,
    ) -> "NBEATSTimeseriesModel":
        if NBEATSModel is None:
            raise ImportError("darts is required for NBEATSTimeseriesModel but is not installed")
        if future_covariates is not None:
            logger.warning("NBEATSTimeseriesModel does not support future_covariates -- ignoring")
        # Fits self._target_scaler on target_series (train-only) and
        # returns the scaled series if scale_target=True; returns
        # target_series unchanged (no-op) otherwise. See
        # BaseTimeseriesModel._fit_and_scale_target().
        scaled_target = self._fit_and_scale_target(target_series)
        # val_series is scaled with the scaler just fit above (train
        # only) -- never refit on val. No-op if scale_target=False or
        # val_series is None. See BaseTimeseriesModel._scale_val().
        scaled_val = self._scale_val(val_series)
        self.model = NBEATSModel(**self.hyperparams)
        self.model.fit(
            scaled_target,
            past_covariates=past_covariates,
            val_series=scaled_val,
            val_past_covariates=val_past_covariates if scaled_val is not None else None,
        )
        return self

    def predict(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        self._require_trained()
        if future_covariates is not None:
            logger.warning("NBEATSTimeseriesModel does not support future_covariates -- ignoring")
        forecast = self.model.predict(n=n, past_covariates=past_covariates)
        # No-op if scale_target=False; inverse-transforms back to real
        # price space otherwise. See BaseTimeseriesModel._inverse_scale().
        return self._inverse_scale(forecast.values().flatten())

    def _darts_model_class(self):
        return NBEATSModel