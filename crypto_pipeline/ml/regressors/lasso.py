# crypto_pipeline/ml/regressors/lasso.py

"""Lasso (L1-regularized) regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class LassoRegressionModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LassoRegressionModel":
        self.model = Lasso(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)