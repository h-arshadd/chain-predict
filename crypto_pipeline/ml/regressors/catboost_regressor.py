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
        # No params are injected here -- every hyperparam (including
        # verbose, allow_writing_files, etc.) comes straight from
        # ml/config.yaml's model.params, same as every other model in
        # this package. CatBoost is noisy and writes a catboost_info/
        # directory by default; set verbose: false / allow_writing_files:
        # false in config yourself if you don't want that.
        self.model = CatBoostRegressor(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)