# crypto_pipeline/ml/classifiers/naive_bayes.py

"""
Naive Bayes classifier. Uses GaussianNB specifically -- this pipeline's
features (indicators, patterns, OHLCV, scaled/differenced values) are
continuous, not counts or booleans, so GaussianNB (assumes per-feature
Gaussian likelihood) is the right variant here, not MultinomialNB
(count data, e.g. word frequencies) or BernoulliNB (binary features).
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class NaiveBayesModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NaiveBayesModel":
        self.model = GaussianNB(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)