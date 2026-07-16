# crypto_pipeline/ml/regressors/registry.py

"""
registry.py
-----------
Maps a config string (ml/config.yaml's model.algorithm) to a regressor
class. This is the ONLY place that knows every available algorithm's
name -- regression_pipeline.py never hardcodes "if algorithm == 'xgboost'"
anywhere; it just calls build_regressor(name, **params).

To add a new algorithm: write a new BaseRegressor subclass under
ml/regressors/, add one line here, done -- regression_pipeline.py does
not change. This is the literal mechanism behind the PDF's "Additional
algorithms may be added without modifying the pipeline" and "no
hardcoded models" requirements.
"""

from typing import Dict, Type

from crypto_pipeline.ml.regressors.base_regressor import BaseRegressor
from crypto_pipeline.ml.regressors.linear_regression import LinearRegressionModel
from crypto_pipeline.ml.regressors.ridge import RidgeRegressionModel
from crypto_pipeline.ml.regressors.lasso import LassoRegressionModel
from crypto_pipeline.ml.regressors.elasticnet import ElasticNetRegressionModel
from crypto_pipeline.ml.regressors.random_forest import RandomForestRegressorModel
from crypto_pipeline.ml.regressors.extra_trees import ExtraTreesRegressorModel
from crypto_pipeline.ml.regressors.svr import SVRModel
from crypto_pipeline.ml.regressors.xgboost_regressor import XGBoostRegressorModel
from crypto_pipeline.ml.regressors.lightgbm_regressor import LightGBMRegressorModel
from crypto_pipeline.ml.regressors.catboost_regressor import CatBoostRegressorModel

# Note: xgboost/lightgbm/catboost classes import cleanly even if those
# packages aren't installed (see their lazy-import docstrings) -- the
# ImportError only fires if you actually pick one of those algorithms.
REGRESSORS: Dict[str, Type[BaseRegressor]] = {
    "linear_regression": LinearRegressionModel,
    "ridge": RidgeRegressionModel,
    "lasso": LassoRegressionModel,
    "elasticnet": ElasticNetRegressionModel,
    "random_forest": RandomForestRegressorModel,
    "extra_trees": ExtraTreesRegressorModel,
    "svr": SVRModel,
    "xgboost": XGBoostRegressorModel,
    "lightgbm": LightGBMRegressorModel,
    "catboost": CatBoostRegressorModel,
}


def build_regressor(algorithm: str, **hyperparams) -> BaseRegressor:
    """
    Instantiate a regressor by config name.

    Args:
        algorithm: key into REGRESSORS, e.g. "random_forest"
        **hyperparams: forwarded to the model's constructor, and from
            there straight into the underlying estimator (see e.g.
            random_forest.py) -- ml/config.yaml's model.params dict is
            unpacked into this.

    Returns:
        An untrained BaseRegressor subclass instance (call .train() next).
    """
    if algorithm not in REGRESSORS:
        raise ValueError(
            f"Unknown regression algorithm '{algorithm}'. "
            f"Available: {sorted(REGRESSORS.keys())}"
        )
    return REGRESSORS[algorithm](**hyperparams)