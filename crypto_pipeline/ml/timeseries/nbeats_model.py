# crypto_pipeline/ml/timeseries/nbeats_model.py

"""
nbeats_model.py
----------------
N-BEATS (darts.models.NBEATSModel) wrapper. Pure deep-learning
forecasting architecture -- no manual differencing/detrending/scaling
required beforehand (N-BEATS handles non-stationarity internally),
which is why the raw close price is used as the target series rather
than a pre-computed return (see target_pipeline.py's timeseries branch).

Supports past_covariates natively (indicator/pattern/sentiment columns),
so config-selected features are never wasted.
"""

from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import NBEATSModel
except ImportError:
    NBEATSModel = None


class NBEATSTimeseriesModel(BaseTimeseriesModel):
    """
    Args (forwarded to darts.models.NBEATSModel):
        input_chunk_length: int, required -- how many past steps the
            model looks at to produce a forecast.
        output_chunk_length: int, required -- how many future steps
            it predicts per forward pass.
        num_stacks, num_blocks, num_layers, layer_widths: architecture
            hyperparams (Darts defaults used if not given).
        n_epochs, batch_size, random_state: training hyperparams.
        Any other kwarg NBEATSModel's constructor accepts.
    """

    def train(self, target_series: "TimeSeries", past_covariates: Optional["TimeSeries"] = None) -> "NBEATSTimeseriesModel":
        if NBEATSModel is None:
            raise ImportError("darts is required for NBEATSTimeseriesModel but is not installed")
        self.model = NBEATSModel(**self.hyperparams)
        self.model.fit(target_series, past_covariates=past_covariates)
        return self

    def predict(self, n: int, past_covariates: Optional["TimeSeries"] = None) -> np.ndarray:
        self._require_trained()
        forecast = self.model.predict(n=n, past_covariates=past_covariates)
        return forecast.values().flatten()

    def _darts_model_class(self):
        return NBEATSModel