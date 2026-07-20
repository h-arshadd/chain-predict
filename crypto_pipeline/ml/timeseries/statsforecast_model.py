# crypto_pipeline/ml/timeseries/statsforecast_model.py

"""
statsforecast_model.py
------------------------
Auto-ARIMA (darts.models.AutoARIMA, backed by Nixtla's
`statsforecast` package) wrapper. Third timeseries regressor alongside
N-BEATS and TCN -- picked as the classical-statistics counterpart to
those two deep-learning models: no training epochs, fits an ARIMA order
automatically per-series via an information criterion, and is fast
enough to retrain at every step of historical_forecasts() (relevant for
fixed-window / walk-forward evaluation, PDF heading 10).

Unlike NBEATSModel/TCNModel (global, chunk-based, past_covariates
only), AutoARIMA is a LOCAL, univariate model -- it takes
NO input_chunk_length/output_chunk_length (Auto-ARIMA has no such
notion; n is only ever a predict()-time argument).

Run UNIVARIATE here on purpose -- no past_covariates, no
future_covariates. AutoARIMA's future_covariates would need columns
whose values are genuinely knowable ahead of the forecast horizon
(e.g. calendar/event features -- hour-of-day, day-of-week, days-until-
next-FOMC), which this project doesn't have. This project's feature
set (ind_*/pat_*/sen_* plus close/volume, ml/config.yaml's
features.feature_columns_extra) is entirely derived from the target
series itself, so passing it as future_covariates would either leak
the target (close is literally in feature_columns_extra) or -- once
that's excluded -- still be unknowable in advance. Both past_covariates
and future_covariates are accepted below for interface consistency
with BaseTimeseriesModel/timeseries_pipeline.py, but ignored with a
warning if given.

Raw close price is used as the target series, same as nbeats_model.py/
tcn_model.py (see target_pipeline.py's timeseries branch).
"""

import logging
from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import AutoARIMA as StatsForecastAutoARIMA
    _import_error = None
except ImportError as _e:
    StatsForecastAutoARIMA = None
    # Keep the ORIGINAL exception instead of discarding it -- "darts is
    # required but not installed" is only one possible cause of this
    # import failing; it can just as easily be a missing/incompatible
    # sub-dependency (e.g. the `statsforecast` package itself, or a
    # version mismatch between it and this darts version) even when
    # darts itself imports fine elsewhere (proven by nbeats/tcn working
    # in the same run). Surfacing the real message is the only way to
    # tell those apart instead of guessing.
    _import_error = _e

logger = logging.getLogger(__name__)


class StatsForecastTimeseriesModel(BaseTimeseriesModel):
    """
    Args (forwarded to darts.models.AutoARIMA, which
    forwards them on to statsforecast.models.AutoARIMA):
        season_length: int -- seasonal period, e.g. 24 for hourly data
            with a daily seasonal pattern. Defaults to statsforecast's
            own default (1, no seasonality) if not given.
        Any other kwarg statsforecast.models.AutoARIMA accepts (e.g.
            max_p, max_q, max_d, stepwise).

    No input_chunk_length/output_chunk_length here -- unlike
    nbeats/tcn, Auto-ARIMA has no such concept. n (how many steps ahead
    to forecast) is only ever a predict()-time argument.

    Univariate only -- see module docstring for why past_covariates/
    future_covariates are ignored rather than used.
    """

    def train(
        self,
        target_series: "TimeSeries",
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
        val_series: Optional["TimeSeries"] = None,
        val_past_covariates: Optional["TimeSeries"] = None,
    ) -> "StatsForecastTimeseriesModel":
        if StatsForecastAutoARIMA is None:
            raise ImportError(
                "Failed to import darts.models.AutoARIMA. "
                f"Underlying error: {type(_import_error).__name__}: {_import_error}"
            ) from _import_error
        # val_series/val_past_covariates accepted for interface
        # consistency with BaseTimeseriesModel.train() (so
        # timeseries_pipeline.py can call every algorithm the same way)
        # but ignored here -- Auto-ARIMA is a classical statistical
        # model with no epochs/gradient training, so there's no
        # val_loss-driven early stopping for a validation set to feed.
        if val_series is not None:
            logger.warning(
                "StatsForecastTimeseriesModel does not use val_series "
                "(not an epoch-trained model) -- ignoring"
            )
        if past_covariates is not None:
            logger.warning(
                "StatsForecastTimeseriesModel does not support past_covariates "
                "(it's a local univariate model) -- ignoring"
            )
        if future_covariates is not None:
            logger.warning(
                "StatsForecastTimeseriesModel runs univariate -- this project has "
                "no columns that are legitimately known ahead of the forecast "
                "horizon (see module docstring), so future_covariates is ignored"
            )
        # output_chunk_length is read by timeseries_pipeline.py as the
        # forecast horizon n (predict()-time argument), not a real
        # AutoARIMA constructor kwarg -- see this class's docstring.
        # Popped here rather than at the pipeline level since nbeats/tcn
        # DO take it as a real constructor kwarg.
        arima_kwargs = {k: v for k, v in self.hyperparams.items() if k != "output_chunk_length"}
        self.model = StatsForecastAutoARIMA(**arima_kwargs)
        self.model.fit(target_series)
        return self

    def predict(
        self,
        n: int,
        past_covariates: Optional["TimeSeries"] = None,
        future_covariates: Optional["TimeSeries"] = None,
    ) -> np.ndarray:
        self._require_trained()
        if past_covariates is not None:
            logger.warning(
                "StatsForecastTimeseriesModel does not support past_covariates "
                "(it's a local univariate model) -- ignoring"
            )
        if future_covariates is not None:
            logger.warning(
                "StatsForecastTimeseriesModel runs univariate -- this project has "
                "no columns that are legitimately known ahead of the forecast "
                "horizon (see module docstring), so future_covariates is ignored"
            )
        forecast = self.model.predict(n=n)
        return forecast.values().flatten()

    def _darts_model_class(self):
        return StatsForecastAutoARIMA