# crypto_pipeline/ml/classifiers/random_forest.py

"""Random Forest classifier (bagged decision trees)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class RandomForestClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestClassifierModel":
        self.model = RandomForestClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)