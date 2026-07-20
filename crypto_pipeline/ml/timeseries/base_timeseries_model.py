# crypto_pipeline/ml/timeseries/base_timeseries_model.py

"""
base_timeseries_model.py
-------------------------
Common base class for every Darts-backed REGRESSION timeseries
forecasting model (model_type=timeseries, algorithm in
ml/timeseries/registry.py's TS_REGRESSORS -- nbeats, tcn,
statsforecast), mirroring the same train()/predict()/save()/load()
interface ml/regressors/base_regressor.py and
ml/classifiers/base_classifier.py already use, plus
historical_forecasts() (PDF heading 10 -- walk-forward evaluation).

For genuine classification forecasters (algorithm in TS_CLASSIFIERS,
e.g. the sklearn-style lags classifier) see
base_timeseries_classifier.py instead -- darts' classifier models
predict discrete labels + probabilities via a different constructor
shape (lags / lags_past_covariates / lags_future_covariates instead of
input_chunk_length / output_chunk_length), so they get their own base
class rather than being shoehorned into this one.

Darts models don't take (X, y) DataFrames like sklearn -- they take
darts.TimeSeries objects, optionally with past_covariates and/or
future_covariates. train()/predict()/historical_forecasts() here wrap
that shape difference so the rest of the pipeline (persistence,
signals) never has to import darts.TimeSeries itself.

Optional target scaling (opt-in via scale_target=True, off by default):
nbeats/tcn train on the raw close price (see target_pipeline.py's
timeseries branch -- these models are deliberately NOT run through
preprocessing.steps' scaling chain, which only touches covariate
feature columns). For a large-magnitude series like a crypto close
price (e.g. ~1e5), an unscaled MSE loss lands around 1e10, which is
harmless to correctness but makes the loss curve unreadable and can
make gradient scale worse than it needs to be. When scale_target=True,
train() fits a darts.dataprocessing.transformers.Scaler (MinMaxScaler
by default) on the target series ONLY (same train-only-fit discipline
as every other scaler in this project), trains the underlying Darts
model on the SCALED series, and predict()/historical_forecasts()
inverse-transform the forecast back to real price space before
returning -- so nothing downstream (signals, evaluation, CSV dumps)
ever sees a scaled value; the scaler is entirely invisible outside this
class. The fitted scaler is persisted alongside the model checkpoint by
save() and restored by load(), so a reloaded model still forecasts in
real price space. statsforecast (a local statistical model, not deep
learning) and the classifier family don't use this -- see each of
their own files.

save()/load() use each Darts model's own .save()/.load() (Darts models
are PyTorch Lightning checkpoints or pickled statistical models, not
plain joblib-picklable objects -- same reasoning base_regressor.py's
docstring gives for why deep learning models need their own
serialization), plus the target scaler (if any) pickled to a sibling
path.
"""

from abc import ABC, abstractmethod
from typing import Optional
import os

import numpy as np
import pandas as pd

try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler as DartsScaler
except ImportError:
    TimeSeries = None  # darts is an optional dependency; only required if model_type=timeseries is used
    DartsScaler = None


class BaseTimeseriesModel(ABC):
    """
    Args:
        scale_target: bool, default False. If True, train() fits a
            darts Scaler on the target series (train-only) and trains
            the underlying model on the scaled series; predict()/
            historical_forecasts() transparently inverse-transform
            back to real price space (see module docstring). Popped
            off before the remaining hyperparams are forwarded to the
            underlying Darts model's constructor -- it is NOT a Darts
            constructor kwarg.
        **hyperparams: passed straight through to the underlying Darts
            model's constructor by each subclass (see e.g.
            nbeats_model.py). Global models (nbeats/tcn) always include
            input_chunk_length and output_chunk_length; local
            statistical models (statsforecast) take neither -- see each
            model's own docstring for what it actually requires.
    """

    # Every concrete subclass forecasts a continuous value (the raw
    # close price) -- target_pipeline.py checks this attribute (via
    # registry.TS_REGRESSORS membership) to decide which target shape
    # to generate. Kept as a plain class attribute (not computed) so
    # it's inspectable without instantiating the model.
    TASK_TYPE = "regression"

    def __init__(self, scale_target: bool = False, **hyperparams):
        if TimeSeries is None:
            raise ImportError(
                "darts is required for timeseries models (model_type=timeseries) "
                "but is not installed. Install it with: pip install darts"
            )
        self.hyperparams = hyperparams
        self.scale_target = scale_target
        self.model = None  # set by train(); the underlying fitted Darts model
        self._target_scaler = None  # set by train() if scale_target=True; a fitted darts Scaler

    def _fit_and_scale_target(self, target_series: "TimeSeries") -> "TimeSeries":
        """
        If scale_target=True, fits self._target_scaler on
        `target_series` (train only -- called once, from train(), on
        exactly the series train() was given) and returns the scaled
        series to actually pass to the underlying Darts model's
        .fit(). If scale_target=False, returns target_series unchanged
        and leaves self._target_scaler as None. Called by each
        subclass's train() in place of using target_series directly.
        """
        if not self.scale_target:
            return target_series
        self._target_scaler = DartsScaler()
        return self._target_scaler.fit_transform(target_series)

    def _scale_val(self, val_series: Optional["TimeSeries"]) -> Optional["TimeSeries"]:
        """
        Apply the ALREADY-FITTED self._target_scaler (fit on train only,
        by _fit_and_scale_target()) to a validation series, for the
        val_series passed to the underlying Darts model's .fit() during
        early-stopping/LR-scheduling. Never fits/refits anything itself
        -- val must never influence the scaler's own statistics, same
        train-only-fit discipline as everywhere else in this project.
        No-op (returns val_series unchanged, including None) if
        scale_target=False.
        """
        if val_series is None or self._target_scaler is None:
            return val_series
        return self._target_scaler.transform(val_series)

    def _inverse_scale(self, forecast: np.ndarray) -> np.ndarray:
        """
        Inverse-transform a plain 1D forecast array back to real price
        space using self._target_scaler (a no-op, returning `forecast`
        unchanged, if scale_target=False or the scaler hasn't been fit
        yet). Called by each subclass's predict() only -- NOT used by
        historical_forecasts(), which uses Darts' own
        data_transformers mechanism instead (see that method's
        docstring for why: it correctly handles retrain=True re-fitting
        the scaler per rolling window, which a single frozen scaler
        fit once in train() cannot). predict() has no such per-window
        refitting concern (it always uses the one frozen self.model /
        self._target_scaler from train()), so the simpler manual
        inverse here is correct for it.
        """
        if self._target_scaler is None:
            return forecast
        # darts.Scaler works on TimeSeries, not bare arrays -- wrap the
        # forecast in a throwaway TimeSeries (a plain integer range
        # index is fine here, inverse_transform() only cares about
        # values, not timestamps), inverse-transform, unwrap.
        wrapped = TimeSeries.from_values(np.asarray(forecast).reshape(-1, 1))
        return self._target_scaler.inverse_transform(wrapped).values().flatten()

    @abstractmethod
    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        val_series: Optional["TimeSeries"] = None,
        val_past_covariates: Optional["TimeSeries"] = None,
    ) -> "BaseTimeseriesModel":
        """
        Fit the model on a target TimeSeries (the close price), optionally
        with past_covariates (indicator/pattern/sentiment columns known
        only up to "now" -- the correct covariate type for technical
        indicators) and/or future_covariates (columns whose values are
        known ahead of time over the forecast horizon -- statistical
        models like statsforecast use these; nbeats/tcn ignore
        future_covariates since they only support past_covariates).

        val_series / val_past_covariates: optional validation TimeSeries
        (chronologically between train and test -- see
        train_test_split.py's val_df), passed straight through to the
        underlying Darts model's own .fit(val_series=...,
        val_past_covariates=...) so PyTorch Lightning's built-in early
        stopping / LR-scheduling (on val_loss) actually has something to
        watch. Ignored (no-op) by algorithms that don't support it (e.g.
        statsforecast). If scale_target=True, subclasses must scale
        val_series with the ALREADY-FITTED train scaler (via
        _scale_val()) before handing it to Darts -- never re-fit on val.

        Must set self.model and return self.
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
        Forecast n steps ahead from the end of the series train() was
        fit on. Returns a 1D array of length n (the predicted close
        price at each future step), not a TimeSeries -- callers outside
        this package never need to import darts themselves.
        """
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
        Walk-forward forecasting over `series` (PDF heading 10 --
        producing many forecasts across the test period instead of one
        single forecast path, so evaluation isn't anchored at a single
        point). Thin wrapper around Darts' own
        ForecastingModel.historical_forecasts(), which every concrete
        model here inherits from.

        Two forecasting modes this project uses, both expressed through
        forecast_horizon/stride/retrain -- nothing else changes:
            one-step forecasting:    forecast_horizon=1, stride=1
                (re-forecast one step ahead at every point, rolling
                forward through the series)
            fixed window forecasting: train_length=N, retrain=True/False
                (each forecast uses a fixed-size window of history
                rather than the whole series -- retrain=True re-fits on
                that window each time, retrain=False reuses the
                already-trained model and just slides the input window)

        `series` must ALWAYS be passed in real (unscaled) price space,
        regardless of scale_target -- if scale_target=True, Darts'
        native data_transformers mechanism is used instead of manually
        inverse-transforming the output: when retrain=True, Darts
        re-fits the target scaler on each rolling/expanding window
        internally (the same discipline used everywhere else in this
        project -- a scaler fit once upfront and never refit would
        leak stale scale statistics into windows a fresh fit wasn't
        meant to see), and when retrain=False the already-fit scaler
        (self._target_scaler, from train()) is reused and just
        transforms/inverse-transforms around the frozen model, exactly
        matching how predict() handles the retrain=False case. This is
        Darts' documented mechanism for this exact situation, not a
        manual reimplementation of it.

        Args:
            series: the full TimeSeries to walk forward over (typically
                train+test concatenated, or just test -- Darts handles
                slicing internally based on train_length/retrain).
                Always real price space, never pre-scaled.
            past_covariates / future_covariates: same shape as
                train()/predict(), covering `series`'s full span.
            forecast_horizon: how many steps ahead each individual
                forecast looks (this project uses 1, per
                output_chunk_length=1 -- see ml/config.yaml).
            stride: how many steps to move forward between forecasts.
            retrain: if True, re-fit the model (and, if scale_target=True,
                the target scaler) at every step (or every `stride`
                steps) using the available history up to that point;
                if False, reuse the already-trained model (self.model,
                from train()) and only re-predict.
            train_length: if set, use only the last `train_length`
                points of history for each retrain (rolling/fixed
                window instead of an ever-expanding one). Only takes
                effect when retrain=True.
            last_points_only: if True, return only the final point of
                each forecast_horizon-step forecast (one array, one
                value per stride step) -- the right shape for
                forecast_horizon=1. If False, every full forecast path
                is kept (a list of TimeSeries) -- forecast_horizon > 1 use.

        Returns:
            np.ndarray of predicted close prices, one per historical
            forecast point (when last_points_only=True), already in
            real price space; a list of np.ndarray forecast paths
            otherwise.
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
        if self.scale_target:
            # Darts' documented mechanism for this: pass the UNSCALED
            # series plus a transformer to apply/inverse internally,
            # rather than us scaling series ourselves beforehand. Reuse
            # the scaler fit during train() as the starting point --
            # Darts re-fits its own internal copy per window when
            # retrain=True, and reuses it as-is when retrain=False (see
            # docstring above); either way self._target_scaler itself
            # is never mutated here.
            scaler = self._target_scaler if self._target_scaler is not None else DartsScaler()
            kwargs["data_transformers"] = {"series": scaler}

        result = self.model.historical_forecasts(**kwargs)

        if last_points_only:
            return result.values().flatten()
        return [ts.values().flatten() for ts in result]

    def save(self, path: str) -> None:
        """
        Persist the fitted Darts model to `path` via its own .save(),
        plus the fitted target scaler (if scale_target=True) pickled to
        a sibling path (`path` + ".target_scaler.pkl") -- both are
        needed to reload a model that still forecasts in real price
        space; the model checkpoint alone only knows the SCALED series
        it was trained on.
        """
        self._require_trained()
        self.model.save(path)
        if self._target_scaler is not None:
            import pickle
            with open(f"{path}.target_scaler.pkl", "wb") as f:
                pickle.dump(self._target_scaler, f)

    def load(self, path: str) -> "BaseTimeseriesModel":
        """
        Load a previously-saved model from `path` into this instance,
        plus its target scaler from the sibling path (if one exists --
        i.e. if this model was saved with scale_target=True) so
        predict()/historical_forecasts() keep returning real price
        space after reloading. Returns self, so both of these work:
            model = SomeTimeseriesModel().load(path)
            model.load(path)  # mutates an existing instance
        """
        model_cls = self._darts_model_class()
        self.model = model_cls.load(path)
        scaler_path = f"{path}.target_scaler.pkl"
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, "rb") as f:
                self._target_scaler = pickle.load(f)
            self.scale_target = True
        return self

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: predict()/historical_forecasts()/save() called before train() (or load())"
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