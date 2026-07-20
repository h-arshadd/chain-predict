# crypto_pipeline/ml/timeseries/tcn_model.py

"""
tcn_model.py
-------------
Temporal Convolutional Network (darts.models.TCNModel) wrapper. Uses
dilated 1D convolutions to capture long-range dependencies -- picked
alongside N-BEATS as a second timeseries regressor since it's
architecturally distinct (convolutional vs fully-connected stacks) and
also supports past_covariates natively.

Raw close price is used as the target series, same as nbeats_model.py
(see target_pipeline.py's timeseries branch for why). Does NOT support
future_covariates (past-covariates-only global model, same as N-BEATS)
-- accepted for interface consistency but ignored, with a warning.

Optional target scaling: scale_target=True (see BaseTimeseriesModel's
module docstring) fits a darts Scaler on the target series before
training and transparently inverse-transforms forecasts back to real
price space -- same reasoning as nbeats_model.py: TCN also trains on
the raw close price's actual scale, and an unscaled large-magnitude
target produces a very large, hard-to-read MSE loss.

historical_forecasts() (one-step and fixed-window walk-forward
evaluation) is inherited unchanged from BaseTimeseriesModel, including
the data_transformers-based scaling path (see that method's docstring).
"""

import logging
from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import TCNModel
except ImportError:
    TCNModel = None

logger = logging.getLogger(__name__)


class TCNTimeseriesModel(BaseTimeseriesModel):
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
        kernel_size, num_filters, dilation_base, num_layers, dropout:
            architecture hyperparams (Darts defaults used if not given).
        n_epochs, batch_size, random_state: training hyperparams.
        Any other kwarg TCNModel's constructor accepts.
    """

    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        val_series: Optional["TimeSeries"] = None,
        val_past_covariates: Optional["TimeSeries"] = None,
    ) -> "TCNTimeseriesModel":
        if TCNModel is None:
            raise ImportError("darts is required for TCNTimeseriesModel but is not installed")
        if future_covariates is not None:
            logger.warning("TCNTimeseriesModel does not support future_covariates -- ignoring")
        # Fits self._target_scaler on target_series (train-only) and
        # returns the scaled series if scale_target=True; returns
        # target_series unchanged (no-op) otherwise. See
        # BaseTimeseriesModel._fit_and_scale_target().
        scaled_target = self._fit_and_scale_target(target_series)
        # val_series is scaled with the scaler just fit above (train
        # only) -- never refit on val. No-op if scale_target=False or
        # val_series is None. See BaseTimeseriesModel._scale_val().
        scaled_val = self._scale_val(val_series)
        self.model = TCNModel(**self.hyperparams)
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
            logger.warning("TCNTimeseriesModel does not support future_covariates -- ignoring")
        forecast = self.model.predict(n=n, past_covariates=past_covariates)
        # No-op if scale_target=False; inverse-transforms back to real
        # price space otherwise. See BaseTimeseriesModel._inverse_scale().
        return self._inverse_scale(forecast.values().flatten())

    def _darts_model_class(self):
        return TCNModel