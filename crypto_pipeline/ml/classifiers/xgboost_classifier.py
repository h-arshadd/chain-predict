# crypto_pipeline/ml/classifiers/xgboost_classifier.py

"""
XGBoost classifier wrapper. xgboost is an optional dependency, lazily
imported inside train() -- see ml/regressors/xgboost_regressor.py's
docstring for why. Named xgboost_classifier.py so it doesn't shadow the
real `xgboost` package.
"""

import joblib
import numpy as np
import pandas as pd

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier


class XGBoostClassifierModel(BaseClassifier):
    """
    objective/num_class (multiclass vs binary) come from ml/config.yaml's
    model.params, same as every other hyperparam -- that part is fully
    config-driven.

    What config CANNOT fix: XGBoost's sklearn API independently requires
    class labels to be contiguous integers starting at 0 (0, 1, 2, ...),
    regardless of objective/num_class. This pipeline's classification
    targets are triple-barrier labels in {-1, 0, 1} (see
    ml/data_prep/target_pipeline.py), which XGBoost rejects directly
    ("Invalid classes inferred from unique values of `y`"). There's no
    config knob for this -- sklearn's own estimators remap labels
    internally, XGBoost's sklearn wrapper doesn't -- so it's handled
    here: map to 0..n_classes-1 before fit(), map predictions back to
    the original labels after.
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostClassifierModel":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "algorithm='xgboost' requires the xgboost package: pip install xgboost"
            ) from e
        self._orig_classes = np.sort(y_train.unique())
        label_to_index = {label: idx for idx, label in enumerate(self._orig_classes)}
        y_mapped = y_train.map(label_to_index).values

        self.model = XGBClassifier(**self.hyperparams)
        self.model.fit(X_train.values, y_mapped)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        mapped_preds = self.model.predict(X.values)
        return self._orig_classes[mapped_preds]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return self.model.predict_proba(X.values)

    @property
    def classes_(self) -> np.ndarray:
        self._require_trained()
        return self._orig_classes

    def save(self, path: str) -> None:
        """
        Same as BaseClassifier.save(), plus _orig_classes -- the
        -1/0/1 <-> 0/1/2 label mapping computed in train(). Without
        saving this, a reloaded model has no way to map XGBoost's
        internal 0/1/2 predictions back to the original labels.
        """
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__}: cannot save before train() has been called"
            )
        joblib.dump(
            {
                "class_name": type(self).__name__,
                "model": self.model,
                "hyperparams": self.hyperparams,
                "orig_classes": self._orig_classes,
            },
            path,
        )

    def load(self, path: str) -> "XGBoostClassifierModel":
        """Same as BaseClassifier.load(), plus restoring _orig_classes."""
        state = joblib.load(path)
        saved_class = state.get("class_name")
        if saved_class != type(self).__name__:
            raise ValueError(
                f"Model file at {path} was saved as '{saved_class}', "
                f"but you're loading it into a '{type(self).__name__}'. "
                f"Load it with the matching class instead."
            )
        self.model = state["model"]
        self.hyperparams = state["hyperparams"]
        self._orig_classes = state["orig_classes"]
        return self