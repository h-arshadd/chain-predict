# preprocessing_lab/model_evaluation/models.py

"""
models.py
---------
Simple (non-deep-learning) model registry for both regression and
classification. Same pattern as preprocessing_lab/registry.py: a plain
dict mapping a name to a constructor. To add a model, add one line here.

xgboost and lightgbm are optional -- if either isn't installed, its
entries are just left out of the registry (main.py loops over whatever's
in the dict, so nothing else needs to change).
"""

from typing import Callable, Dict

from sklearn.linear_model import LinearRegression, LogisticRegression

REGRESSION_MODELS: Dict[str, Callable] = {
    "linear_regression": lambda: LinearRegression(),
}

CLASSIFICATION_MODELS: Dict[str, Callable] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
}

try:
    from xgboost import XGBRegressor, XGBClassifier

    REGRESSION_MODELS["xgboost"] = lambda: XGBRegressor(
        n_estimators=100, max_depth=4, random_state=42
    )
    CLASSIFICATION_MODELS["xgboost"] = lambda: XGBClassifier(
        n_estimators=100, max_depth=4, random_state=42, eval_metric="mlogloss"
    )
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor, LGBMClassifier

    REGRESSION_MODELS["lightgbm"] = lambda: LGBMRegressor(
        n_estimators=100, max_depth=4, random_state=42, verbose=-1
    )
    CLASSIFICATION_MODELS["lightgbm"] = lambda: LGBMClassifier(
        n_estimators=100, max_depth=4, random_state=42, verbose=-1
    )
except ImportError:
    pass