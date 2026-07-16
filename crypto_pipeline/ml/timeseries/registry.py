# crypto_pipeline/ml/timeseries/registry.py

"""
registry.py
-----------
Maps a config string (ml/config.yaml's model.algorithm) to a timeseries
model class. Same mechanism as ml/regressors/registry.py and
ml/classifiers/registry.py -- see those files' docstrings.
timeseries_pipeline.py never hardcodes "if algorithm == 'nbeats'"
anywhere; it just calls build_timeseries_model(name, **params).

To add a new algorithm: write a new BaseTimeseriesModel subclass under
ml/timeseries/, add one line here, done -- timeseries_pipeline.py does
not change.
"""

from typing import Dict, Type

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel
from crypto_pipeline.ml.timeseries.nbeats_model import NBEATSTimeseriesModel
from crypto_pipeline.ml.timeseries.tcn_model import TCNTimeseriesModel

# Note: both classes import cleanly even if darts isn't installed (see
# base_timeseries_model.py's lazy-import handling) -- the ImportError
# only fires once you actually try to build one of these models.
TS_MODELS: Dict[str, Type[BaseTimeseriesModel]] = {
    "nbeats": NBEATSTimeseriesModel,
    "tcn": TCNTimeseriesModel,
}


def build_timeseries_model(algorithm: str, **hyperparams) -> BaseTimeseriesModel:
    """
    Instantiate a timeseries model by config name.

    Args:
        algorithm: key into TS_MODELS, e.g. "nbeats"
        **hyperparams: forwarded to the model's constructor. Must
            include input_chunk_length and output_chunk_length (every
            Darts model in this family requires both); everything else
            is architecture/training-specific, see each model's own
            docstring.

    Returns:
        An untrained BaseTimeseriesModel subclass instance (call .train() next).
    """
    if algorithm not in TS_MODELS:
        raise ValueError(
            f"Unknown timeseries algorithm '{algorithm}'. "
            f"Available: {sorted(TS_MODELS.keys())}"
        )
    return TS_MODELS[algorithm](**hyperparams)