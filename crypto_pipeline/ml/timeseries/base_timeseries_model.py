# crypto_pipeline/ml/timeseries/base_timeseries_model.py

"""
base_timeseries_model.py
-------------------------
Common base class for every Darts-backed timeseries forecasting model
(model_type=timeseries), mirroring the same train()/predict()/save()/
load() interface ml/regressors/base_regressor.py and
ml/classifiers/base_classifier.py already use -- timeseries_pipeline.py
only ever calls these four methods, never branches on which concrete
Darts model class it's holding.

Darts models don't take (X, y) DataFrames like sklearn -- they take
darts.TimeSeries objects, optionally with past_covariates (the
indicator/pattern/sentiment columns). train()/predict() here wrap that
shape difference so the rest of the pipeline (persistence, signals)
never has to import darts.TimeSeries itself.

save()/load() use each Darts model's own .save()/.load() (Darts models
are PyTorch Lightning checkpoints, not plain joblib-picklable objects --
same reasoning base_regressor.py's docstring gives for why deep
learning models need their own serialization).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

try:
    from darts import TimeSeries
except ImportError:
    TimeSeries = None  # darts is an optional dependency; only required if model_type=timeseries is used


class BaseTimeseriesModel(ABC):
    """
    Args:
        **hyperparams: passed straight through to the underlying Darts
            model's constructor by each subclass (see e.g. nbeats_model.py).
            Always includes input_chunk_length and output_chunk_length,
            the two Darts requires from every model in this family.
    """

    def __init__(self, **hyperparams):
        if TimeSeries is None:
            raise ImportError(
                "darts is required for timeseries models (model_type=timeseries) "
                "but is not installed. Install it with: pip install darts"
            )
        self.hyperparams = hyperparams
        self.model = None  # set by train(); the underlying fitted Darts model

    @abstractmethod
    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
    ) -> "BaseTimeseriesModel":
        """
        Fit the model on a target TimeSeries (the close price), optionally
        with past_covariates (indicator/pattern/sentiment columns known
        only up to "now", never into the future -- the correct covariate
        type for technical indicators; see timeseries_pipeline.py).
        Must set self.model and return self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        """
        Forecast n steps ahead from the end of the series train() was
        fit on. Returns a 1D array of length n (the predicted close
        price at each future step), not a TimeSeries -- callers outside
        this package never need to import darts themselves.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the fitted Darts model to `path` via its own .save()."""
        self._require_trained()
        self.model.save(path)

    def load(self, path: str) -> "BaseTimeseriesModel":
        """
        Load a previously-saved model from `path` into this instance.
        Returns self, so both of these work:
            model = SomeTimeseriesModel().load(path)
            model.load(path)  # mutates an existing instance
        """
        model_cls = self._darts_model_class()
        self.model = model_cls.load(path)
        return self

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: predict()/save() called before train() (or load())"
            )

    @abstractmethod
    def _darts_model_class(self):
        """Return the underlying darts.models class (e.g. NBEATSModel), used by load()."""
        raise NotImplementedError


def series_from_dataframe(df: pd.DataFrame, timestamp_column: str, value_columns) -> "TimeSeries":
    """
    Shared helper: build a darts.TimeSeries from a plain DataFrame the
    same way every other stage in this pipeline already works with
    DataFrames. value_columns can be a single column name (target) or a
    list (covariates).
    """
    return TimeSeries.from_dataframe(df, time_col=timestamp_column, value_cols=value_columns)