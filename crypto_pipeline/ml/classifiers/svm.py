# crypto_pipeline/ml/classifiers/svm.py

"""
Support Vector Machine classifier.

sklearn's SVC only computes predict_proba() if probability=True was
passed at construction (it fits an extra internal cross-validated
calibration step to do so, which is why it defaults to False upstream).
This wrapper does NOT set that (or any other param) itself -- every
hyperparam, including probability, comes straight from ml/config.yaml's
model.params, same as every other model here. If you use predict_proba()
with this model, set probability: true in config yourself.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class SVMModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SVMModel":
        self.model = SVC(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)