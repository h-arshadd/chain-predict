# crypto_pipeline/ml/regressors/base_regressor.py

"""
base_regressor.py
------------------
Common base class for every regression model (PDF heading 5).

Required interface, per the PDF:
    train()
    predict()
    save()
    load()

regression_pipeline.py only ever calls these four methods -- it never
checks "if algorithm == 'xgboost' ..." or otherwise branches on which
concrete model it's holding. That's the whole point of this base class:
a new algorithm can be added anywhere under ml/regressors/ (plus one
line in registry.py) without touching the pipeline at all.

save()/load() here are the shared, default implementation (joblib dump
of the fitted estimator + hyperparams + class name, for a sanity check
on load). Every regressor in this package uses this default as-is --
none of them override it. A future non-sklearn-style model (e.g. a
PyTorch/TensorFlow deep learning model under PDF heading 7, not part of
this heading) would be the case where save()/load() actually need
overriding, since joblib isn't the right serialization format there.
"""

from abc import ABC, abstractmethod
from typing import Optional

import joblib
import numpy as np
import pandas as pd


class BaseRegressor(ABC):
    """
    Args:
        **hyperparams: passed straight through to the underlying
            estimator's constructor by each subclass (see e.g.
            random_forest.py). Never inspected or special-cased here --
            keeping this base class algorithm-agnostic is what lets
            registry.py build any of them the same way.
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None  # set by train(); the underlying fitted estimator

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseRegressor":
        """
        Fit the model on training data. Must set self.model and return
        self (so `model = SomeRegressor(**params).train(X, y)` chains).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict on X. Per PDF heading 8, regression prediction output is
        just the predicted value -- a 1D array, one prediction per row
        of X, in the same row order as X.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the fitted model + hyperparams to `path` via joblib."""
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: cannot save before train() has been called"
            )
        joblib.dump(
            {
                "class_name": type(self).__name__,
                "model": self.model,
                "hyperparams": self.hyperparams,
            },
            path,
        )

    def load(self, path: str) -> "BaseRegressor":
        """
        Load a previously-saved model from `path` into this instance.
        Returns self, so both of these work:
            model = SomeRegressor().load(path)
            model.load(path)  # mutates an existing instance
        """
        state = joblib.load(path)
        saved_class = state.get("class_name")
        if saved_class != type(self).__name__:
            raise ValueError(
                f"Model file at {path} was saved as '{saved_class}', "
                f"but you're loading it into a '{type(self).__name__}'. "
                f"Load it with the matching class instead."
            )
        self.model = state["model"]
        self.hyperparams = state["hyperparams"]
        return self

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: predict() called before train() (or load())"
            )