# crypto_pipeline/ml/classifiers/registry.py

"""
registry.py
-----------
Maps a config string (ml/config.yaml's model.algorithm) to a classifier
class. Same mechanism as ml/regressors/registry.py -- see that file's
docstring. classification_pipeline.py never hardcodes an algorithm
name; it just calls build_classifier(name, **params).
"""

from typing import Dict, Type

from crypto_pipeline.ml.classifiers.base_classifier import BaseClassifier
from crypto_pipeline.ml.classifiers.logistic_regression import LogisticRegressionModel
from crypto_pipeline.ml.classifiers.decision_tree import DecisionTreeClassifierModel
from crypto_pipeline.ml.classifiers.random_forest import RandomForestClassifierModel
from crypto_pipeline.ml.classifiers.extra_trees import ExtraTreesClassifierModel
from crypto_pipeline.ml.classifiers.svm import SVMModel
from crypto_pipeline.ml.classifiers.knn import KNNModel
from crypto_pipeline.ml.classifiers.naive_bayes import NaiveBayesModel
from crypto_pipeline.ml.classifiers.xgboost_classifier import XGBoostClassifierModel
from crypto_pipeline.ml.classifiers.lightgbm_classifier import LightGBMClassifierModel
from crypto_pipeline.ml.classifiers.catboost_classifier import CatBoostClassifierModel

# Note: xgboost/lightgbm/catboost classes import cleanly even if those
# packages aren't installed (see their lazy-import docstrings) -- the
# ImportError only fires if you actually pick one of those algorithms.
CLASSIFIERS: Dict[str, Type[BaseClassifier]] = {
    "logistic_regression": LogisticRegressionModel,
    "decision_tree": DecisionTreeClassifierModel,
    "random_forest": RandomForestClassifierModel,
    "extra_trees": ExtraTreesClassifierModel,
    "svm": SVMModel,
    "knn": KNNModel,
    "naive_bayes": NaiveBayesModel,
    "xgboost": XGBoostClassifierModel,
    "lightgbm": LightGBMClassifierModel,
    "catboost": CatBoostClassifierModel,
}


def build_classifier(algorithm_name: str, **hyperparams) -> BaseClassifier:
    """
    Instantiate a classifier by config name.

    Args:
        algorithm_name: key into CLASSIFIERS, e.g. "random_forest"
        **hyperparams: forwarded to the model's constructor, and from
            there straight into the underlying estimator -- ml/config.yaml's
            model.params dict is unpacked into this.

    Note: this parameter is named `algorithm_name` (not `algorithm`)
    specifically because some models' own hyperparams include a key
    called "algorithm" (e.g. KNN's `algorithm: auto` sklearn param).
    If this function's own parameter were also named `algorithm`,
    `build_classifier(algorithm, **hyperparams)` would raise
    "got multiple values for argument 'algorithm'" whenever hyperparams
    contained that key.

    Returns:
        An untrained BaseClassifier subclass instance (call .train() next).
    """
    if algorithm_name not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classification algorithm '{algorithm_name}'. "
            f"Available: {sorted(CLASSIFIERS.keys())}"
        )
    return CLASSIFIERS[algorithm_name](**hyperparams)