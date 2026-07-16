# crypto_pipeline/ml/classifiers/logistic_regression.py

"""Logistic Regression classifier."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LogisticRegressionModel":
        self.model = LogisticRegression(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)