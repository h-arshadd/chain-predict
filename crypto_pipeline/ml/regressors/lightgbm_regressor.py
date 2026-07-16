# crypto_pipeline/ml/regressors/lightgbm_regressor.py

"""
LightGBM Regressor wrapper. lightgbm is an optional dependency, lazily
imported inside train() -- see xgboost_regressor.py's docstring for why.
Named lightgbm_regressor.py so it doesn't shadow the real `lightgbm` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class LightGBMRegressorModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMRegressorModel":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError(
                "algorithm='lightgbm' requires the lightgbm package: pip install lightgbm"
            ) from e
        self.model = LGBMRegressor(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)