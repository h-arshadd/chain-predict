# crypto_pipeline/ml/regressors/elasticnet.py

"""ElasticNet (combined L1 + L2 regularized) regression."""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class ElasticNetRegressionModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ElasticNetRegressionModel":
        self.model = ElasticNet(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)