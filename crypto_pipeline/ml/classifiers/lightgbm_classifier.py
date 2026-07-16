# crypto_pipeline/ml/classifiers/lightgbm_classifier.py

"""
LightGBM classifier wrapper. lightgbm is an optional dependency, lazily
imported inside train() -- see ml/regressors/xgboost_regressor.py's
docstring for why. Named lightgbm_classifier.py so it doesn't shadow
the real `lightgbm` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class LightGBMClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMClassifierModel":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError(
                "algorithm='lightgbm' requires the lightgbm package: pip install lightgbm"
            ) from e
        self.model = LGBMClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)