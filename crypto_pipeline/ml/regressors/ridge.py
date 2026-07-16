# crypto_pipeline/ml/regressors/ridge.py

"""Ridge (L2-regularized) regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class RidgeRegressionModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeRegressionModel":
        self.model = Ridge(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)