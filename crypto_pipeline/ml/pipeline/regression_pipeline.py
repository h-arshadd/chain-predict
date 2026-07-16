# crypto_pipeline/ml/pipeline/regression_pipeline.py

"""
regression_pipeline.py
-----------------------
Orchestrates PDF headings 1-5 for a regression run: dataset loading,
feature selection, train/test split, preprocessing, and model training,
using whichever algorithm ml/config.yaml's model.algorithm names (via
ml/regressors/registry.py).

Routing between regression and classification is AUTOMATIC, driven by
data_prep/config.yaml's model_type -- not a separate switch you flip
here. That file already decides regression vs classification once, at
the source: model_type controls which target target_pipeline.py
generates (continuous log-return for regression, -1/0/1 triple-barrier
labels for classification). A dataset built for one target type isn't
meaningfully usable by the other model type, so there is exactly one
place this gets decided, and both dataset_loader.py (via its debug CSV
path) and this pipeline read the same field rather than letting it be
set twice and risk disagreeing.

Concretely: run_regression_pipeline() reads data_prep_config["model_type"]
and raises immediately if it isn't "regression" -- pointing you at
classification_pipeline.py instead. There's no model_type field in
ml/config.yaml to keep in sync; it isn't duplicated here, same pattern
dataset_loader.py already uses for exchange/symbol/model_type.

This module stops at "trained model + raw test-set predictions" --
PDF headings 8-11 (standardized prediction formatting, signal
generation, evaluation, full experiment persistence) are separate,
not-yet-built stages that would consume this pipeline's output dict.
"""

import logging

import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing
from crypto_pipeline.ml.regressors.registry import build_regressor

logger = logging.getLogger(__name__)


def run_regression_pipeline(ml_config_path: str, data_prep_config_path: str) -> dict:
    """
    Run the full regression pipeline through model training.

    Args:
        ml_config_path: path to ml/config.yaml
        data_prep_config_path: path to data_prep/config.yaml

    Returns:
        dict with keys:
            model: trained BaseRegressor instance (ready for .predict()/.save())
            predictions: np.ndarray, raw predicted values for test_df, same
                row order as test_df
            y_test: pd.Series, true target values for test_df (for scoring)
            feature_columns: list[str], order used for training/inference
            split_info: dict from train_test_split.split_dataset() (train/test
                date ranges etc, per PDF heading 3's record-keeping requirement)
            fit_objects: list from preprocessing_pipeline.run_preprocessing()
                (the fitted scalers/transforms, to persist alongside the model)
            algorithm: str, the model.algorithm name used
    """

    ml_config = _load_yaml(ml_config_path)
    data_prep_config = _load_yaml(data_prep_config_path)

    model_type = data_prep_config.get("model_type")
    if model_type != "regression":
        raise ValueError(
            f"run_regression_pipeline() requires data_prep_config['model_type'] == "
            f"'regression', got '{model_type}'. Use classification_pipeline.py for "
            f"a classification dataset instead -- model_type is set once in "
            f"data_prep/config.yaml and drives which target was generated, so it "
            f"can't be overridden here."
        )

    # Headings 1-4: load, select features, split, preprocess.
    df = load_dataset(ml_config_path, data_prep_config_path)
    selected = select_features(df, ml_config)
    feature_columns = selected["feature_columns"]
    target_column = selected["target_column"]

    split_info = split_dataset(df, ml_config, timestamp_column=selected["timestamp_column"])

    preprocessed = run_preprocessing(
        split_info["train_df"], split_info["test_df"], feature_columns, ml_config
    )
    train_df = preprocessed["train_df"]
    test_df = preprocessed["test_df"]

    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    # Heading 5: model training. Which algorithm + hyperparams is entirely
    # config-driven -- this function contains no model-specific logic at all.
    model_config = ml_config.get("model", {})
    algorithm = model_config.get("algorithm")
    if not algorithm:
        raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'random_forest')")
    params = model_config.get("params", {}) or {}

    logger.info(f"Training regressor: algorithm={algorithm}, params={params}")
    model = build_regressor(algorithm, **params)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    logger.info(f"Regression training complete: {len(predictions)} test predictions generated")

    return {
        "model": model,
        "predictions": predictions,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "split_info": split_info,
        "fit_objects": preprocessed["fit_objects"],
        "algorithm": algorithm,
    }


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)