# crypto_pipeline/ml/regressors/extra_trees.py

"""Extra Trees regression (extremely randomized trees -- random split thresholds, not just random subsets)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor


class ExtraTreesRegressorModel(BaseRegressor):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ExtraTreesRegressorModel":
        self.model = ExtraTreesRegressor(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict(X.values)