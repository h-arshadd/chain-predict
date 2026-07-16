# crypto_pipeline/ml/persistence/metadata.py

"""
metadata.py
-----------
Builds the per-stage experiment config dicts that get written alongside
every trained model (PDF heading 11: "Each trained model shall include a
configuration describing the complete experiment").

One yaml per pipeline stage, instead of a single combined file -- each
stage's builder only knows about that stage's own inputs, so a new field
in (say) preprocessing never touches the data_prep or model builders:

    build_data_prep_metadata()    -- data_prep/ folder: data/features/
                                      sentiment/target config, dataset info
    build_split_metadata()        -- train_test_split.py: train/test date
                                      ranges, row counts, test_size
    build_preprocessing_metadata() -- preprocessing/ folder: configured
                                      steps + which methods actually fit
    build_model_metadata()        -- regressors/classifiers/deep_learning
                                      folders: algorithm, hyperparams,
                                      architecture (isolated per model_kind,
                                      same as before)
    build_evaluation_metadata()   -- evaluation/ folder: train/val/test metrics

artifact_manager.py writes each of these to its own file under
artifacts/configs/{run_id}/; this module only assembles the dicts.
"""

from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# data_prep/ -- dataset + feature engineering + sentiment + target config
# ----------------------------------------------------------------------
def build_data_prep_metadata(data_prep_config: dict, row_counts: dict) -> dict:
    """
    Everything that came out of the data_prep/ folder: what data was
    pulled (symbol/exchange/timeframe/date range), how features were
    engineered (indicators/patterns), sentiment config, and target
    generation config (horizon/thresholds) -- data_prep/config.yaml
    verbatim plus the resulting row count.

    Args:
        data_prep_config: ml/data_prep/config.yaml dict
        row_counts: dict from the pipeline, uses "total_rows"

    Returns:
        dict to be written to artifacts/configs/{run_id}/data_prep.yaml
    """
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
        "model_type": data_prep_config.get("model_type"),
        "data": {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "calculate_ohlcv": data_cfg.get("calculate_ohlcv"),
        },
        "features": data_prep_config.get("features", {}),
        "sentiment": data_prep_config.get("sentiment", {}),
        "target": data_prep_config.get("target", {}),
        "total_rows": row_counts.get("total_rows"),
    }


# ----------------------------------------------------------------------
# train_test_split.py -- chronological split dates + row counts
# ----------------------------------------------------------------------
def build_split_metadata(split_info: dict, row_counts: dict) -> dict:
    """
    Train/test split record (PDF heading 3's required fields): training
    start/end date, test start/end date, plus row counts on each side
    both before and after preprocessing's trailing dropna().

    Args:
        split_info: dict from train_test_split.split_dataset()
        row_counts: dict from the pipeline (train_rows/test_rows are
            POST preprocessing-drop; dropped_rows_train/test explain
            the gap against the pre-drop split)

    Returns:
        dict to be written to artifacts/configs/{run_id}/split.yaml
    """
    def _iso(value):
        return value.isoformat() if isinstance(value, pd.Timestamp) else value

    return {
        "test_size": split_info.get("test_size"),
        "train": {
            "start_date": _iso(split_info.get("train_start")),
            "end_date": _iso(split_info.get("train_end")),
            "rows": row_counts.get("train_rows"),
            "dropped_rows": row_counts.get("dropped_rows_train"),
        },
        "test": {
            "start_date": _iso(split_info.get("test_start")),
            "end_date": _iso(split_info.get("test_end")),
            "rows": row_counts.get("test_rows"),
            "dropped_rows": row_counts.get("dropped_rows_test"),
        },
    }


# ----------------------------------------------------------------------
# preprocessing/ -- configured steps + which methods actually fit
# ----------------------------------------------------------------------
def build_preprocessing_metadata(
    feature_columns: list,
    target_column: str,
    timestamp_column: str,
    preprocessing_config: dict,
    fit_objects: list,
) -> dict:
    """
    Feature selection (PDF heading 2) + preprocessing (PDF heading 4)
    config: which columns were used, and which preprocessing steps were
    configured vs actually fit.

    Args:
        feature_columns: list[str], from feature_selector.select_features(),
            order preserved
        target_column: str
        timestamp_column: str
        preprocessing_config: ml_config["preprocessing"] dict (the
            configured steps -- scaling method, stationarity method)
        fit_objects: list from preprocessing_pipeline.run_preprocessing()
            (fitted scaler/transform params -- persisted separately as
            the actual pickled objects, but their method names are
            echoed here for readability)

    Returns:
        dict to be written to artifacts/configs/{run_id}/preprocessing.yaml
    """
    return {
        "feature_columns": list(feature_columns),
        "n_features": len(feature_columns),
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "steps": preprocessing_config.get("steps", []),
        "fitted_methods": [obj["method"] for obj in fit_objects],
    }


# ----------------------------------------------------------------------
# regressors/ / classifiers/ / deep_learning/ -- algorithm + hyperparams
# ----------------------------------------------------------------------
def build_model_metadata(
    model_kind: str,
    algorithm: str,
    hyperparams: dict,
    classes: Optional[np.ndarray] = None,
) -> dict:
    """
    Model config (PDF headings 5-7): which algorithm, what hyperparams,
    and (for deep learning) architecture/training-loop settings. Shape
    differs by model_kind, same split as before:
        _regressor_model_info()            -- traditional regressors
        _classifier_model_info()           -- traditional classifiers
        _deep_learning_regressor_model_info() -- MLP/LSTM/GRU regressors
        _deep_learning_classifier_model_info() -- MLP/LSTM/GRU classifiers
        _timeseries_model_info()           -- Darts-backed models (NBEATS/TCN)

    Args:
        model_kind: "regressor" | "classifier" | "deep_learning_regressor"
            | "deep_learning_classifier" | "timeseries"
        algorithm: str, e.g. "random_forest", "xgboost", "lstm"
        hyperparams: dict, as passed to train() (for deep learning models
            this includes hidden_layers, dropout, optimizer, etc; for
            traditional models whatever was forwarded to the
            sklearn-style constructor)
        classes: np.ndarray or None -- required for classifier /
            deep_learning_classifier kinds (the label set), ignored for
            regressor kinds

    Returns:
        dict to be written to artifacts/configs/{run_id}/model.yaml
    """
    if model_kind not in _MODEL_INFO_BUILDERS:
        raise ValueError(
            f"Unknown model_kind '{model_kind}'. Available: {sorted(_MODEL_INFO_BUILDERS.keys())}"
        )
    return _MODEL_INFO_BUILDERS[model_kind](algorithm, hyperparams, classes)


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


def _timeseries_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Darts-backed timeseries model (ml/timeseries/*, e.g. NBEATSModel, TCNModel) -- no classes, uses Darts' own checkpoint format."""
    return {
        "model_type": "timeseries",
        "algorithm": algorithm,
        "input_chunk_length": hyperparams.get("input_chunk_length"),
        "output_chunk_length": hyperparams.get("output_chunk_length"),
        "hyperparameters": hyperparams,
        "random_seed": hyperparams.get("random_state"),
        "serialization_format": "darts_checkpoint",
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
    "timeseries": _timeseries_model_info,
}


# ----------------------------------------------------------------------
# evaluation/ -- train/val/test metrics
# ----------------------------------------------------------------------
def build_evaluation_metadata(
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
) -> dict:
    """
    Evaluation results (PDF heading 10). Left as {} for any split not yet
    available, so this can be called right after training and updated
    later without changing shape.

    Returns:
        dict to be written to artifacts/configs/{run_id}/evaluation.yaml
    """
    return {
        "train_metrics": train_metrics or {},
        "val_metrics": val_metrics or {},
        "test_metrics": test_metrics or {},
    }