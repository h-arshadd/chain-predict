# crypto_pipeline/ml/regressors/random_forest.py

"""Random Forest regression (bagged decision trees)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class RandomForestRegressorModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestRegressorModel":
        self.model = RandomForestRegressor(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)