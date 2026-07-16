# crypto_pipeline/ml/classifiers/xgboost_classifier.py

"""
XGBoost classifier wrapper. xgboost is an optional dependency, lazily
imported inside train() -- see ml/regressors/xgboost_regressor.py's
docstring for why. Named xgboost_classifier.py so it doesn't shadow the
real `xgboost` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class XGBoostClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostClassifierModel":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "algorithm='xgboost' requires the xgboost package: pip install xgboost"
            ) from e
        self.model = XGBClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)