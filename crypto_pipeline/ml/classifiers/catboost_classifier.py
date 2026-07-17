# crypto_pipeline/ml/classifiers/catboost_classifier.py

"""
CatBoost classifier wrapper. catboost is an optional dependency, lazily
imported inside train() -- see ml/regressors/xgboost_regressor.py's
docstring for why. Named catboost_classifier.py so it doesn't shadow
the real `catboost` package.
"""

import numpy as np
import pandas as pd

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class CatBoostClassifierModel(BaseClassifier):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CatBoostClassifierModel":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError(
                "algorithm='catboost' requires the catboost package: pip install catboost"
            ) from e
        # No params are injected here -- every hyperparam (including
        # loss_function, verbose, allow_writing_files, etc.) comes
        # straight from ml/config.yaml's model.params, same as every
        # other model in this package. CatBoost defaults to binary
        # Logloss unless loss_function is set explicitly, which crashes
        # on this pipeline's 3-class {-1, 0, 1} triple-barrier target --
        # see ml/config.yaml's classification.catboost block, where
        # loss_function is set to MultiClass. CatBoost is noisy and
        # writes a catboost_info/ directory by default; set verbose:
        # false / allow_writing_files: false in config yourself if you
        # don't want that.
        self.model = CatBoostClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_train.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        # CatBoost's predict() returns a column vector (n, 1) for
        # classification, unlike sklearn's flat (n,) -- flatten it so
        # this wrapper's output shape matches every other classifier's.
        preds = self.model.predict(X.values)
        return np.asarray(preds).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)