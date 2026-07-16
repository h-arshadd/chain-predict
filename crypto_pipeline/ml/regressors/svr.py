# crypto_pipeline/ml/regressors/svr.py

"""Support Vector Regressor."""

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class SVRModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SVRModel":
        self.model = SVR(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)