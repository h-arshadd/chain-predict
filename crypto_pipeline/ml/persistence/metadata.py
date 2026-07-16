# crypto_pipeline/ml/persistence/metadata.py

"""
metadata.py
-----------
Builds the experiment config dict that gets written alongside every
trained model (PDF heading 11: "Each trained model shall include a
configuration describing the complete experiment").

One shared builder, build_metadata(), used by every pipeline
(regression, classification, and -- once wired up -- deep learning).
The sections that must differ by model type (per your lead's
requirement: "regressors will have different info and classifiers will
have different info") are isolated into three small functions:
    _regressor_model_info()   -- traditional regression models
    _classifier_model_info()  -- traditional classification models
    _deep_learning_model_info() -- MLP/LSTM/GRU, either task type

build_metadata() picks the right one based on `model_kind`, so callers
never have to hand-assemble the model_info section themselves, and a
new model kind is one new `_..._model_info()` function plus one line in
build_metadata() -- nothing else changes.

Everything OUTSIDE model_info (dataset info, feature info, data split,
preprocessing, evaluation) is identical across all three, since none of
that depends on which algorithm was used.
"""

from typing import Optional

import numpy as np
import pandas as pd


def build_metadata(
    model_kind: str,
    data_prep_config: dict,
    feature_columns: list,
    target_column: str,
    timestamp_column: str,
    split_info: dict,
    preprocessing_config: dict,
    fit_objects: list,
    algorithm: str,
    hyperparams: dict,
    row_counts: dict,
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
    classes: Optional[np.ndarray] = None,
) -> dict:
    """
    Assemble the full experiment config for one trained model.

    Args:
        model_kind: "regressor" | "classifier" | "deep_learning_regressor"
            | "deep_learning_classifier" -- picks which model_info
            section shape gets used.
        data_prep_config: ml/data_prep/config.yaml dict, for dataset info
            (symbol/exchange/timeframe/date range) and feature
            engineering config (indicators/patterns/sentiment params).
        feature_columns: list[str], from feature_selector.select_features(),
            order preserved (PDF heading 2's requirement).
        target_column: str
        timestamp_column: str
        split_info: dict from train_test_split.split_dataset()
            (train_start/train_end/test_start/test_end/test_size).
        preprocessing_config: ml_config["preprocessing"] dict (the
            configured steps -- scaling method, stationarity method).
        fit_objects: list from preprocessing_pipeline.run_preprocessing()
            (fitted scaler/transform params -- persisted separately as
            the actual pickled objects, but their method names + params
            are echoed here for readability).
        algorithm: str, e.g. "random_forest", "xgboost", "lstm".
        hyperparams: dict, the model's hyperparameters as passed to
            train() (for deep learning models this includes
            hidden_layers, dropout, optimizer, etc; for traditional
            models whatever was forwarded to the sklearn-style constructor).
        row_counts: dict, e.g.:
            {
              "total_rows": 8760,
              "train_rows": 7008, "test_rows": 1752,
              "dropped_rows_train": 0, "dropped_rows_test": 0,
              "n_features": 42,
            }
            Everything a reader would need to sanity-check the run at a
            glance without re-running the pipeline -- how many rows went
            in, how many made it through preprocessing on each side of
            the split, how many features were used. Pass whatever subset
            you have; missing keys are simply omitted (not padded with
            null), so this stays honest about what was actually tracked.
        train_metrics / val_metrics / test_metrics: dict or None, filled
            in once heading 10 (Evaluation) runs. Left as {} if not yet
            available, so this function can be called right after
            training and updated later without changing shape.
        classes: np.ndarray or None -- required for classifier /
            deep_learning_classifier kinds (the label set), ignored for
            regressor kinds.

    Returns:
        dict, the full config to be written to
        artifacts/configs/{run_id}.yaml (or .json) by artifact_manager.py
    """
    if model_kind not in _MODEL_INFO_BUILDERS:
        raise ValueError(
            f"Unknown model_kind '{model_kind}'. Available: {sorted(_MODEL_INFO_BUILDERS.keys())}"
        )

    dataset_info = _dataset_info(data_prep_config, row_counts)
    feature_info = _feature_info(data_prep_config, feature_columns, target_column, timestamp_column)
    data_split = _data_split(split_info, row_counts)
    preprocessing = _preprocessing_info(preprocessing_config, fit_objects)
    model_info = _MODEL_INFO_BUILDERS[model_kind](algorithm, hyperparams, classes)
    evaluation = {
        "train_metrics": train_metrics or {},
        "val_metrics": val_metrics or {},
        "test_metrics": test_metrics or {},
    }

    return {
        "model_kind": model_kind,
        # run_summary: everything a reader would want at a glance without
        # opening the dataset/features/model sections below -- one flat
        # block, not nested, since this is meant to be skimmed.
        "run_summary": {
            "dataset_name": dataset_info["dataset_name"],
            "symbol": dataset_info["symbol"],
            "exchange": dataset_info["exchange"],
            "timeframe": dataset_info["timeframe"],
            "start_date": dataset_info["start_date"],
            "end_date": dataset_info["end_date"],
            "total_rows": dataset_info["total_rows"],
            "train_period": f"{data_split['train_start']} -> {data_split['train_end']}",
            "train_rows": data_split["train_rows"],
            "test_period": f"{data_split['test_start']} -> {data_split['test_end']}",
            "test_rows": data_split["test_rows"],
            "n_features": feature_info["n_features"],
            "model_type": model_info["model_type"],
            "algorithm": algorithm,
        },
        "dataset": dataset_info,
        "features": feature_info,
        "data_split": data_split,
        "preprocessing": preprocessing,
        "model": model_info,
        "evaluation": evaluation,
    }


# ----------------------------------------------------------------------
# Sections shared by every model kind
# ----------------------------------------------------------------------
def _dataset_info(data_prep_config: dict, row_counts: dict) -> dict:
    data_cfg = data_prep_config.get("data", {})
    symbol = data_cfg.get("symbol")
    exchange = data_cfg.get("exchange")
    timeframe = data_cfg.get("timeframe")
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")

    return {
        # dataset_name: synthesized, not a separate config field anywhere
        # upstream -- {exchange}_{symbol}_{timeframe} is the same naming
        # shape dataset_loader.py's debug CSV path already uses
        # (base_dir/exchange/symbol/model_type/), so this is recognizable
        # rather than inventing a new convention.
        "dataset_name": "_".join(str(p) for p in (exchange, symbol, timeframe) if p),
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "model_type": data_prep_config.get("model_type"),
        "total_rows": row_counts.get("total_rows"),
    }


def _feature_info(data_prep_config: dict, feature_columns: list, target_column: str, timestamp_column: str) -> dict:
    return {
        "feature_columns": list(feature_columns),
        "n_features": len(feature_columns),
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "feature_engineering_config": {
            "indicators": data_prep_config.get("features", {}).get("indicators", {}),
            "patterns": data_prep_config.get("features", {}).get("patterns", {}),
            "sentiment": data_prep_config.get("sentiment", {}),
            "target": data_prep_config.get("target", {}),
        },
    }


def _data_split(split_info: dict, row_counts: dict) -> dict:
    def _iso(value):
        return value.isoformat() if isinstance(value, pd.Timestamp) else value

    return {
        "train_start": _iso(split_info.get("train_start")),
        "train_end": _iso(split_info.get("train_end")),
        "train_rows": row_counts.get("train_rows"),
        "test_start": _iso(split_info.get("test_start")),
        "test_end": _iso(split_info.get("test_end")),
        "test_rows": row_counts.get("test_rows"),
        "test_size": split_info.get("test_size"),
        # Rows dropped by preprocessing_pipeline.run_preprocessing()'s
        # trailing dropna() (stationarity methods leave leading NaNs) --
        # train_rows/test_rows above are already POST-drop, so this is
        # what explains any gap against the pre-split row counts.
        "dropped_rows_train": row_counts.get("dropped_rows_train"),
        "dropped_rows_test": row_counts.get("dropped_rows_test"),
    }


def _preprocessing_info(preprocessing_config: dict, fit_objects: list) -> dict:
    return {
        "steps": preprocessing_config.get("steps", []),
        "fitted_methods": [obj["method"] for obj in fit_objects],
    }


# ----------------------------------------------------------------------
# model_info sections -- this is the part that differs by model kind
# ----------------------------------------------------------------------
def _regressor_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Traditional regression model (ml/regressors/*) -- no classes, no output_dim."""
    return {
        "model_type": "regressor",
        "algorithm": algorithm,
        "hyperparameters": hyperparams,
        "random_seed": hyperparams.get("random_state"),
        "serialization_format": "joblib",
    }


def _classifier_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Traditional classification model (ml/classifiers/*) -- includes the class label set."""
    return {
        "model_type": "classifier",
        "algorithm": algorithm,
        "hyperparameters": hyperparams,
        "random_seed": hyperparams.get("random_state"),
        "classes": classes.tolist() if classes is not None else None,
        "serialization_format": "joblib",
    }


def _deep_learning_regressor_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Deep learning regressor (ml/deep_learning/*, BaseNetwork) -- architecture + training config, no classes."""
    return {
        "model_type": "deep_learning_regressor",
        "algorithm": algorithm,
        "architecture": _architecture_info(hyperparams),
        "training": _training_info(hyperparams),
        "random_seed": hyperparams.get("random_seed"),
        "serialization_format": "pytorch_checkpoint",
    }


def _deep_learning_classifier_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Deep learning classifier (ml/deep_learning/*, BaseClassifierNetwork) -- architecture + training config + classes."""
    return {
        "model_type": "deep_learning_classifier",
        "algorithm": algorithm,
        "architecture": _architecture_info(hyperparams),
        "training": _training_info(hyperparams),
        "random_seed": hyperparams.get("random_seed"),
        "classes": classes.tolist() if classes is not None else None,
        "serialization_format": "pytorch_checkpoint",
    }


def _architecture_info(hyperparams: dict) -> dict:
    """Shared by both deep learning model_info builders -- the network-shape hyperparams."""
    return {
        "hidden_layers": hyperparams.get("hidden_layers"),
        "hidden_units": hyperparams.get("hidden_units"),
        "activation": hyperparams.get("activation"),
        "dropout": hyperparams.get("dropout"),
        "batch_norm": hyperparams.get("batch_norm"),
    }


def _training_info(hyperparams: dict) -> dict:
    """Shared by both deep learning model_info builders -- the training-loop hyperparams."""
    return {
        "optimizer": hyperparams.get("optimizer"),
        "learning_rate": hyperparams.get("learning_rate"),
        "scheduler": hyperparams.get("scheduler"),
        "scheduler_params": hyperparams.get("scheduler_params"),
        "batch_size": hyperparams.get("batch_size"),
        "epochs": hyperparams.get("epochs"),
        "early_stopping_patience": hyperparams.get("early_stopping_patience"),
        "loss": hyperparams.get("loss"),
    }


_MODEL_INFO_BUILDERS = {
    "regressor": _regressor_model_info,
    "classifier": _classifier_model_info,
    "deep_learning_regressor": _deep_learning_regressor_model_info,
    "deep_learning_classifier": _deep_learning_classifier_model_info,
}