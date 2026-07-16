# crypto_pipeline/ml/regressors/xgboost_regressor.py

"""
XGBoost Regressor wrapper.

xgboost is an optional dependency -- imported lazily inside train()/
predict() rather than at module load time, so importing this file (or
the rest of the regressors package via registry.py) never fails just
because xgboost isn't installed. It only becomes a hard error if you
actually try to use this specific algorithm.

Note: this file is named xgboost_regressor.py, not xgboost.py, so it
doesn't shadow the real `xgboost` package on sys.path.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class XGBoostRegressorModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostRegressorModel":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "algorithm='xgboost' requires the xgboost package: pip install xgboost"
            ) from e
        self.model = XGBRegressor(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)