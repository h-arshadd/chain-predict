# crypto_pipeline/ml/regressors/linear_regression.py

"""Ordinary least squares. No regularization -- see ridge.py/lasso.py/elasticnet.py for that."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class LinearRegressionModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionModel":
        self.model = LinearRegression(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)