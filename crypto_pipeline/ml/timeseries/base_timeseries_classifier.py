# crypto_pipeline/ml/timeseries/base_timeseries_classifier.py

"""
base_timeseries_classifier.py
-------------------------------
Common base class for every Darts-backed CLASSIFICATION timeseries
forecasting model (model_type=timeseries, algorithm in
ml/timeseries/registry.py's TS_CLASSIFIERS).

This is a genuinely different model family from
base_timeseries_model.py's regressors (nbeats/tcn/statsforecast):
Darts' classification forecasters (SKLearnClassifierModel,
XGBClassifierModel, LightGBMClassifierModel, CatBoostClassifierModel)
predict discrete class labels directly -- they are lags-based models
(constructor takes lags / lags_past_covariates / lags_future_covariates
+ output_chunk_length, same shape as Darts' RegressionModel family),
not chunk-based global deep learning models, and they don't exist for
nbeats/tcn's architecture. See ml/timeseries/sklearn_classifier_model.py.

Required interface (mirrors base_timeseries_model.py's train()/save()/
load()/historical_forecasts(), plus predict_proba()/classes_ the same
way ml/classifiers/base_classifier.py adds them for row-wise
classifiers):
    train()
    predict()
    predict_proba()
    historical_forecasts()
    save()
    load()

The training pipeline (timeseries_pipeline.py) only ever calls these
methods -- adding a new classification algorithm here only requires a
new BaseTimeseriesClassifier subclass + one line in registry.py.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

try:
    from darts import TimeSeries
except ImportError:
    TimeSeries = None  # darts is an optional dependency; only required if model_type=timeseries is used


class BaseTimeseriesClassifier(ABC):
    """
    Args:
        **hyperparams: passed straight through to the underlying Darts
            classifier model's constructor by each subclass (see e.g.
            sklearn_classifier_model.py). Always includes lags (or
            lags_past_covariates/lags_future_covariates) and
            output_chunk_length -- Darts' lags-based model family
            requires at least one lags-type argument.
    """

    TASK_TYPE = "classification"

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
        future_covariates: Optional["TimeSeries"] = None,
    ) -> "BaseTimeseriesClassifier":
        """
        Fit the model on a target TimeSeries of discrete class labels
        (e.g. the project's -1/0/1 triple-barrier label, see
        target_pipeline.py), optionally with past_covariates and/or
        future_covariates. Must set self.model and return self.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        """
        Forecast n steps ahead. Returns a 1D array of length n, the
        predicted class label at each future step.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        """
        Forecast n steps ahead, returning class probabilities instead
        of the predicted label. Returns a 2D array of shape
        (n, n_classes), columns ordered per self.classes_.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Class labels in the same column order predict_proba() returns them in."""
        raise NotImplementedError

    def historical_forecasts(
        self,
        series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: bool = False,
        train_length: Optional[int] = None,
        last_points_only: bool = True,
    ) -> np.ndarray:
        """
        Walk-forward forecasting over `series` -- same one-step /
        fixed-window modes as BaseTimeseriesModel.historical_forecasts()
        (see that docstring), applied to a classification forecaster.
        """
        self._require_trained()
        kwargs = dict(
            series=series,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            last_points_only=last_points_only,
            verbose=False,
        )
        if past_covariates is not None:
            kwargs["past_covariates"] = past_covariates
        if future_covariates is not None:
            kwargs["future_covariates"] = future_covariates
        if train_length is not None:
            kwargs["train_length"] = train_length

        result = self.model.historical_forecasts(**kwargs)

        if last_points_only:
            return result.values().flatten()
        return [ts.values().flatten() for ts in result]

    def save(self, path: str) -> None:
        """Persist the fitted Darts model to `path` via its own .save()."""
        self._require_trained()
        self.model.save(path)

    def load(self, path: str) -> "BaseTimeseriesClassifier":
        """
        Load a previously-saved model from `path` into this instance.
        Returns self, so both of these work:
            model = SomeTimeseriesClassifier().load(path)
            model.load(path)  # mutates an existing instance
        """
        model_cls = self._darts_model_class()
        self.model = model_cls.load(path)
        return self

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: predict()/predict_proba()/historical_forecasts()/"
                f"save() called before train() (or load())"
            )

    @abstractmethod
    def _darts_model_class(self):
        """Return the underlying darts.models class (e.g. SKLearnClassifierModel), used by load()."""
        raise NotImplementedError