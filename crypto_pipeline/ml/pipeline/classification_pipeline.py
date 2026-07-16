# crypto_pipeline/ml/pipeline/classification_pipeline.py

"""
classification_pipeline.py
----------------------------
Orchestrates PDF headings 1-4 and 6 for a classification run: dataset
loading, feature selection, train/test split, preprocessing, and model
training, using whichever algorithm ml/config.yaml's model.algorithm
names (via ml/classifiers/registry.py).

Mirrors regression_pipeline.py exactly, including the routing rule --
see that file's module docstring for the full rationale. Short version:
data_prep/config.yaml's model_type decides regression vs classification
once, at the source (it controls which target target_pipeline.py
generates), so this pipeline reads that same field and refuses to run
against a non-classification dataset rather than taking a separate
switch that could disagree with it.

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
from crypto_pipeline.ml.classifiers.registry import build_classifier

logger = logging.getLogger(__name__)


def run_classification_pipeline(ml_config_path: str, data_prep_config_path: str) -> dict:
    """
    Run the full classification pipeline through model training.

    Args:
        ml_config_path: path to ml/config.yaml
        data_prep_config_path: path to data_prep/config.yaml

    Returns:
        dict with keys:
            model: trained BaseClassifier instance (ready for .predict()/
                .predict_proba()/.save())
            predictions: np.ndarray, predicted class labels for test_df,
                same row order as test_df
            probabilities: np.ndarray, shape (n_test_rows, n_classes),
                class probabilities for test_df (columns ordered per
                model.classes_)
            y_test: pd.Series, true class labels for test_df (for scoring)
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
    if model_type != "classification":
        raise ValueError(
            f"run_classification_pipeline() requires data_prep_config['model_type'] == "
            f"'classification', got '{model_type}'. Use regression_pipeline.py for "
            f"a regression dataset instead -- model_type is set once in "
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

    # Heading 6: model training. Which algorithm + hyperparams is entirely
    # config-driven -- this function contains no model-specific logic at all.
    model_config = ml_config.get("model", {})
    algorithm = model_config.get("algorithm")
    if not algorithm:
        raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'random_forest')")
    params = model_config.get("params", {}) or {}

    logger.info(f"Training classifier: algorithm={algorithm}, params={params}")
    model = build_classifier(algorithm, **params)
    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    logger.info(f"Classification training complete: {len(predictions)} test predictions generated")

    return {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "split_info": split_info,
        "fit_objects": preprocessed["fit_objects"],
        "algorithm": algorithm,
    }


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)