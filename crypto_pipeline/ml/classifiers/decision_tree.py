# crypto_pipeline/ml/classifiers/decision_tree.py

"""Decision Tree classifier."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class DecisionTreeClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "DecisionTreeClassifierModel":
        self.model = DecisionTreeClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)