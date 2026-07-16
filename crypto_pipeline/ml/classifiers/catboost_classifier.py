# crypto_pipeline/ml/classifiers/catboost_classifier.py

"""
CatBoost classifier wrapper. catboost is an optional dependency, lazily
imported inside train() -- see ml/regressors/xgboost_regressor.py's
docstring for why. Named catboost_classifier.py so it doesn't shadow
the real `catboost` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class CatBoostClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostClassifierModel":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError(
                "algorithm='catboost' requires the catboost package: pip install catboost"
            ) from e
        # CatBoost is noisy (prints training iterations) by default --
        # keep it quiet unless the caller explicitly asked for verbosity.
        params = {"verbose": False, **self.hyperparams}
        self.model = CatBoostClassifier(**params)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        # CatBoost's predict() returns a column vector (n, 1) for
        # classification, unlike sklearn's flat (n,) -- flatten it so
        # this wrapper's output shape matches every other classifier's.
        preds = self.model.predict(X.values)
        return np.asarray(preds).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)