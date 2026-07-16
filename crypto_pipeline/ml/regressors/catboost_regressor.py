# crypto_pipeline/ml/regressors/catboost_regressor.py

"""
CatBoost Regressor wrapper. catboost is an optional dependency, lazily
imported inside train() -- see xgboost_regressor.py's docstring for why.
Named catboost_regressor.py so it doesn't shadow the real `catboost` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class CatBoostRegressorModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostRegressorModel":
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "algorithm='catboost' requires the catboost package: pip install catboost"
            ) from e
        # CatBoost is noisy (prints training iterations) by default --
        # keep it quiet unless the caller explicitly asked for verbosity.
        params = {"verbose": False, **self.hyperparams}
        self.model = CatBoostRegressor(**params)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)