# crypto_pipeline/ml/classifiers/base_classifier.py

"""
base_classifier.py
--------------------
Common base class for every classification model (PDF heading 6).

Required interface, per the PDF:
    train()
    predict()
    predict_proba()
    save()
    load()

Same contract as ml/regressors/base_regressor.py, plus predict_proba()
-- classification_pipeline.py only ever calls these five methods, never
branching on which concrete algorithm it's holding. To add a new
algorithm: write a new BaseClassifier subclass under ml/classifiers/,
add one line to registry.py, done -- the pipeline does not change.

save()/load() are the shared, default implementation (joblib dump of
the fitted estimator + hyperparams + class name). None of the
classifiers in this package override it.
"""

from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """
    Args:
        **hyperparams: passed straight through to the underlying
            estimator's constructor by each subclass. Never inspected
            or special-cased here -- keeping this base class
            algorithm-agnostic is what lets registry.py build any of
            them the same way.
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None  # set by train(); the underlying fitted estimator

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseClassifier":
        """
        Fit the model on training data. Must set self.model and return
        self (so `model = SomeClassifier(**params).train(X, y)` chains).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for X. Per PDF heading 8, classification
        prediction output is the predicted class -- a 1D array, one
        label per row of X, in the same row order as X.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for X. Returns a 2D array of shape
        (n_rows, n_classes), columns ordered per self.classes_ (see
        that property below) -- one probability distribution per row
        of X, in the same row order as X.
        """
        raise NotImplementedError

    @property
    def classes_(self) -> np.ndarray:
        """
        Class labels in the same column order predict_proba() returns
        them in. Needed downstream (PDF heading 9's signal generation,
        e.g. "bullish probability > threshold") to know which
        predict_proba() column corresponds to which class.
        """
        self._require_trained()
        return self.model.classes_

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

    def load(self, path: str) -> "BaseClassifier":
        """
        Load a previously-saved model from `path` into this instance.
        Returns self, so both of these work:
            model = SomeClassifier().load(path)
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
                f"{type(self).__name__}: called before train() (or load())"
            )