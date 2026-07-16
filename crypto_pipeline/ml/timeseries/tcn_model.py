# crypto_pipeline/ml/timeseries/tcn_model.py

"""
tcn_model.py
-------------
Temporal Convolutional Network (darts.models.TCNModel) wrapper. Uses
dilated 1D convolutions to capture long-range dependencies -- picked
alongside N-BEATS as the second timeseries model since it's
architecturally distinct (convolutional vs fully-connected stacks) and
also supports past_covariates natively.

Raw close price is used as the target series, same as nbeats_model.py
(see target_pipeline.py's timeseries branch for why).
"""

from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import TCNModel
except ImportError:
    TCNModel = None


class TCNTimeseriesModel(BaseTimeseriesModel):
    """
    Args (forwarded to darts.models.TCNModel):
        input_chunk_length: int, required -- how many past steps the
            model looks at to produce a forecast.
        output_chunk_length: int, required -- how many future steps
            it predicts per forward pass.
        kernel_size, num_filters, dilation_base, num_layers, dropout:
            architecture hyperparams (Darts defaults used if not given).
        n_epochs, batch_size, random_state: training hyperparams.
        Any other kwarg TCNModel's constructor accepts.
    """

    def train(self, target_series: "TimeSeries", past_covariates: Optional["TimeSeries"] = None) -> "TCNTimeseriesModel":
        if TCNModel is None:
            raise ImportError("darts is required for TCNTimeseriesModel but is not installed")
        self.model = TCNModel(**self.hyperparams)
        self.model.fit(target_series, past_covariates=past_covariates)
        return self

    def predict(self, n: int, past_covariates: Optional["TimeSeries"] = None) -> np.ndarray:
        self._require_trained()
        forecast = self.model.predict(n=n, past_covariates=past_covariates)
        return forecast.values().flatten()

    def _darts_model_class(self):
        return TCNModel