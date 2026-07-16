# crypto_pipeline/ml/deep_learning/registry.py

"""
registry.py
-----------
Maps a config string (ml/config.yaml's model.algorithm) to a deep
learning model class. Same mechanism as ml/regressors/registry.py and
ml/classifiers/registry.py -- see those files' docstrings. A future
deep-learning pipeline never hardcodes "if algorithm == 'lstm'"
anywhere; it just calls build_dl_regressor(name, **params) or
build_dl_classifier(name, **params).

To add a new architecture: write a new BaseRegressorNetwork /
BaseClassifierNetwork subclass under ml/deep_learning/, add one line
here, done.
"""

from typing import Dict, Type

from crypto_pipeline.ml.deep_learning.base_network import BaseClassifierNetwork, BaseNetwork
from crypto_pipeline.ml.deep_learning.mlp import MLPClassifierModel, MLPRegressorModel
from crypto_pipeline.ml.deep_learning.lstm import LSTMClassifierModel, LSTMRegressorModel
from crypto_pipeline.ml.deep_learning.gru import GRUClassifierModel, GRURegressorModel

DL_REGRESSORS: Dict[str, Type[BaseNetwork]] = {
    "mlp": MLPRegressorModel,
    "lstm": LSTMRegressorModel,
    "gru": GRURegressorModel,
}

DL_CLASSIFIERS: Dict[str, Type[BaseClassifierNetwork]] = {
    "mlp": MLPClassifierModel,
    "lstm": LSTMClassifierModel,
    "gru": GRUClassifierModel,
}


def build_dl_regressor(algorithm: str, **hyperparams) -> BaseNetwork:
    """
    Instantiate a deep learning regressor by config name.

    Args:
        algorithm: key into DL_REGRESSORS, e.g. "lstm"
        **hyperparams: forwarded to the model's constructor (hidden_layers,
            hidden_units, activation, dropout, batch_norm, optimizer,
            learning_rate, scheduler, scheduler_params, batch_size,
            epochs, early_stopping_patience, random_seed).

    Returns:
        An untrained BaseRegressorNetwork subclass instance (call .train() next).
    """
    if algorithm not in DL_REGRESSORS:
        raise ValueError(
            f"Unknown deep learning regression algorithm '{algorithm}'. "
            f"Available: {sorted(DL_REGRESSORS.keys())}"
        )
    return DL_REGRESSORS[algorithm](**hyperparams)


def build_dl_classifier(algorithm: str, **hyperparams) -> BaseClassifierNetwork:
    """
    Instantiate a deep learning classifier by config name.

    Args:
        algorithm: key into DL_CLASSIFIERS, e.g. "gru"
        **hyperparams: same set as build_dl_regressor().

    Returns:
        An untrained BaseClassifierNetwork subclass instance (call .train() next).
    """
    if algorithm not in DL_CLASSIFIERS:
        raise ValueError(
            f"Unknown deep learning classification algorithm '{algorithm}'. "
            f"Available: {sorted(DL_CLASSIFIERS.keys())}"
        )
    return DL_CLASSIFIERS[algorithm](**hyperparams)