# crypto_pipeline/ml/timeseries/sklearn_classifier_model.py

"""
sklearn_classifier_model.py
------------------------------
Genuine classification forecaster (darts.models.SKLearnClassifierModel)
-- the timeseries family's classification counterpart to nbeats/tcn/
statsforecast's regression. Predicts a discrete class label directly
(the project's -1/0/1 triple-barrier label, see target_pipeline.py),
not a price that gets thresholded afterward.

Unlike NBEATSModel/TCNModel (chunk-based: input_chunk_length/
output_chunk_length), SKLearnClassifierModel is a LAGS-based model,
same family shape as Darts' RegressionModel: it looks back `lags`
steps of the target itself and/or `lags_past_covariates`/
`lags_future_covariates` steps of covariates to predict the label
output_chunk_length steps ahead. Wraps any sklearn-like classifier
underneath (default: LogisticRegression) -- swap `estimator` in
hyperparams for a different one (e.g. a RandomForestClassifier)
without needing a new file, since the estimator itself is just a
constructor kwarg here, same as sklearn-style traditional classifiers
under ml/classifiers/.

predict_proba() uses Darts' likelihood='classprobability' mechanism
(predict_likelihood_parameters=True at predict()-time) -- set once at
construction (see __init__ below), always on, since ml/signals/
timeseries_signals.py needs probabilities to threshold the same way
classification_signals.py does for row-wise classifiers.
"""

import logging
from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_classifier import (
    BaseTimeseriesClassifier,
    TimeSeries,
)

try:
    from darts.models import SKLearnClassifierModel
except ImportError:
    SKLearnClassifierModel = None

logger = logging.getLogger(__name__)


class SKLearnClassifierTimeseriesModel(BaseTimeseriesClassifier):
    """
    Args (forwarded to darts.models.SKLearnClassifierModel):
        lags: int or list[int], required (unless lags_past_covariates/
            lags_future_covariates alone are given) -- how many past
            target steps to look back.
        lags_past_covariates: int or list[int], optional -- past
            covariate lookback (indicator/pattern/sentiment columns).
        lags_future_covariates: int or list[int] or tuple, optional --
            future covariate lookback/lookahead.
        output_chunk_length: int, required -- how many future steps
            it predicts per forward pass (this project uses 1, see
            ml/config.yaml).
        model: an unfitted sklearn-like classifier instance, e.g.
            sklearn.ensemble.RandomForestClassifier() -- optional,
            defaults to Darts' own default (LogisticRegression) if not
            given. This is the "add a new algorithm without a new
            file" lever for this model: swap `model` in
            ml/config.yaml's param_overrides instead of writing a new
            wrapper class, same as ml/classifiers/*.py wrap different
            sklearn estimators.
        random_state: int, optional.
        Any other kwarg SKLearnClassifierModel's constructor accepts.
    """

    def __init__(self, **hyperparams):
        # likelihood='classprobability' is required for predict_proba()
        # to work (Darts' predict_likelihood_parameters mechanism) --
        # always set here rather than left to config, since this
        # wrapper's predict_proba() always needs it.
        hyperparams.setdefault("likelihood", "classprobability")
        super().__init__(**hyperparams)

    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        val_series: Optional["TimeSeries"] = None,
        val_past_covariates: Optional["TimeSeries"] = None,
    ) -> "SKLearnClassifierTimeseriesModel":
        if SKLearnClassifierModel is None:
            raise ImportError("darts is required for SKLearnClassifierTimeseriesModel but is not installed")
        # val_series/val_past_covariates accepted for interface
        # consistency with BaseTimeseriesModel.train() (so
        # timeseries_pipeline.py can call every algorithm the same way)
        # but ignored here -- this wraps an sklearn-style lags-based
        # classifier, not an epoch-trained PyTorch Lightning model, so
        # there's no val_loss-driven early stopping/LR-scheduling for a
        # validation set to feed.
        if val_series is not None:
            logger.warning(
                "SKLearnClassifierTimeseriesModel does not use val_series "
                "(not an epoch-trained model) -- ignoring"
            )
        self.model = SKLearnClassifierModel(**self.hyperparams)
        self.model.fit(
            target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        return self

    def predict(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        self._require_trained()
        forecast = self.model.predict(
            n=n, past_covariates=past_covariates, future_covariates=future_covariates
        )
        return forecast.values().flatten()

    def predict_proba(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        self._require_trained()
        forecast = self.model.predict(
            n=n,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            predict_likelihood_parameters=True,
        )
        # Darts returns one probability column per class, named
        # "<component>_<class>_p" -- values() preserves that column
        # order, which already matches self.classes_ (both come from
        # the same underlying sklearn estimator's classes_ order).
        return forecast.values()

    @property
    def classes_(self) -> np.ndarray:
        """
        Class labels, in the order predict_proba()'s columns come back
        in. SKLearnClassifierModel wraps the underlying sklearn-like
        estimator (LogisticRegression by default) as self.model.model
        in current Darts versions; older/alternate versions may name
        it differently, so this falls back to scanning the wrapper's
        own attributes for anything exposing classes_ rather than
        hardcoding one private attribute path.
        """
        self._require_trained()
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "classes_"):
            return np.asarray(inner.classes_)
        if hasattr(self.model, "classes_"):
            return np.asarray(self.model.classes_)
        raise AttributeError(
            f"Could not find classes_ on the trained {type(self.model).__name__} "
            f"or its wrapped estimator -- check the installed darts version's "
            f"SKLearnClassifierModel internals."
        )

    def _darts_model_class(self):
        return SKLearnClassifierModel