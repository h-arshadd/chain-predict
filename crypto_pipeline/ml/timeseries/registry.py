# crypto_pipeline/ml/timeseries/registry.py

"""
registry.py
-----------
Maps a config string (ml/config.yaml's model.algorithm) to a timeseries
model class. Same mechanism as ml/regressors/registry.py and
ml/classifiers/registry.py -- see those files' docstrings.
timeseries_pipeline.py never hardcodes "if algorithm == 'nbeats'"
anywhere; it just calls build_ts_regressor(name, **params) or
build_ts_classifier(name, **params).

Two registries, not one -- same split as ml/regressors/registry.py vs
ml/classifiers/registry.py, applied within model_type=timeseries:

    TS_REGRESSORS  -- forecast the continuous close price directly
                       (nbeats, tcn, statsforecast). Built on
                       BaseTimeseriesModel (base_timeseries_model.py).
    TS_CLASSIFIERS -- forecast a discrete class label directly
                       (sklearn_classifier -- Darts' lags-based
                       classification family). Built on
                       BaseTimeseriesClassifier
                       (base_timeseries_classifier.py).

Both live under model_type: timeseries (not a separate model_type) --
this project's target_pipeline.py picks which target shape to generate
(raw close price vs triple-barrier label) by checking which registry
the configured algorithm is in, the same way main.py already checks
`algorithm in REGRESSORS` vs `algorithm in DL_REGRESSORS` to route
between traditional and deep learning regressors.

To add a new REGRESSION algorithm: write a new BaseTimeseriesModel
subclass under ml/timeseries/, add one line to TS_REGRESSORS, done.
To add a new CLASSIFICATION algorithm: write a new
BaseTimeseriesClassifier subclass, add one line to TS_CLASSIFIERS,
done. timeseries_pipeline.py does not change either way.
"""

from typing import Dict, Type, Union

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel
from crypto_pipeline.ml.timeseries.base_timeseries_classifier import BaseTimeseriesClassifier
from crypto_pipeline.ml.timeseries.nbeats_model import NBEATSTimeseriesModel
from crypto_pipeline.ml.timeseries.tcn_model import TCNTimeseriesModel
from crypto_pipeline.ml.timeseries.statsforecast_model import StatsForecastTimeseriesModel
from crypto_pipeline.ml.timeseries.sklearn_classifier_model import SKLearnClassifierTimeseriesModel

# Note: every class here imports cleanly even if darts isn't installed
# (see base_timeseries_model.py's lazy-import handling) -- the
# ImportError only fires once you actually try to build one of these
# models. lstm/gru are NOT here -- they live in ml/deep_learning/
# (lstm.py, gru.py) and are selected via model_type: regression /
# model_type: classification instead, same as mlp -- see
# ml/deep_learning/registry.py.
TS_REGRESSORS: Dict[str, Type[BaseTimeseriesModel]] = {
    "nbeats": NBEATSTimeseriesModel,
    "tcn": TCNTimeseriesModel,
    "statsforecast": StatsForecastTimeseriesModel,
}

TS_CLASSIFIERS: Dict[str, Type[BaseTimeseriesClassifier]] = {
    "sklearn_classifier": SKLearnClassifierTimeseriesModel,
}

# Combined view, used only where the pipeline genuinely doesn't care
# which family an algorithm belongs to (e.g. "is this name registered
# at all" validation) -- model building always goes through the
# specific build_ts_regressor()/build_ts_classifier() below, never this.
TS_MODELS: Dict[str, Type] = {**TS_REGRESSORS, **TS_CLASSIFIERS}


def build_ts_regressor(algorithm: str, **hyperparams) -> BaseTimeseriesModel:
    """
    Instantiate a timeseries REGRESSION model by config name.

    Args:
        algorithm: key into TS_REGRESSORS, e.g. "nbeats", "tcn", "statsforecast"
        **hyperparams: forwarded to the model's constructor. nbeats/tcn
            require input_chunk_length and output_chunk_length;
            statsforecast requires neither (see its own docstring).

    Returns:
        An untrained BaseTimeseriesModel subclass instance (call .train() next).
    """
    if algorithm not in TS_REGRESSORS:
        raise ValueError(
            f"Unknown timeseries regression algorithm '{algorithm}'. "
            f"Available: {sorted(TS_REGRESSORS.keys())}"
        )
    return TS_REGRESSORS[algorithm](**hyperparams)


def build_ts_classifier(algorithm: str, **hyperparams) -> BaseTimeseriesClassifier:
    """
    Instantiate a timeseries CLASSIFICATION model by config name.

    Args:
        algorithm: key into TS_CLASSIFIERS, e.g. "sklearn_classifier"
        **hyperparams: forwarded to the model's constructor (lags,
            lags_past_covariates, lags_future_covariates,
            output_chunk_length, model, random_state).

    Returns:
        An untrained BaseTimeseriesClassifier subclass instance (call .train() next).
    """
    if algorithm not in TS_CLASSIFIERS:
        raise ValueError(
            f"Unknown timeseries classification algorithm '{algorithm}'. "
            f"Available: {sorted(TS_CLASSIFIERS.keys())}"
        )
    return TS_CLASSIFIERS[algorithm](**hyperparams)


def build_timeseries_model(algorithm: str, **hyperparams) -> Union[BaseTimeseriesModel, BaseTimeseriesClassifier]:
    """
    Instantiate a timeseries model by config name, regardless of family
    (regression or classification) -- looks in TS_REGRESSORS first,
    then TS_CLASSIFIERS. Kept for callers that don't yet know/care
    which family `algorithm` belongs to (e.g. a quick lookup); anywhere
    the family matters (training, target generation), use
    build_ts_regressor()/build_ts_classifier() directly instead, same
    as main.py branches on `algorithm in REGRESSORS` vs
    `algorithm in DL_REGRESSORS`.
    """
    if algorithm in TS_REGRESSORS:
        return build_ts_regressor(algorithm, **hyperparams)
    if algorithm in TS_CLASSIFIERS:
        return build_ts_classifier(algorithm, **hyperparams)
    raise ValueError(
        f"Unknown timeseries algorithm '{algorithm}'. "
        f"Available regressors: {sorted(TS_REGRESSORS.keys())}, "
        f"classifiers: {sorted(TS_CLASSIFIERS.keys())}"
    )