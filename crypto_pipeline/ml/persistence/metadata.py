# crypto_pipeline/ml/persistence/metadata.py

"""
metadata.py
-----------
Builds the per-stage experiment config dicts that get written alongside
every trained model (PDF heading 11: "Each trained model shall include a
configuration describing the complete experiment").

This is a RECORD of what ran (dates, split boundaries, row counts,
which preprocessing/model settings were used, signal counts, and a
handful of summary metrics) -- it's meant to be readable at a glance
and reusable later for inference. It intentionally does NOT hold any
of the heavy computation (no per-day quantstats series, no equity
curves, no trade ledgers) -- that stuff already lives in
pipeline_out/{algorithm}/07_metrics.csv and 07_trade_ledger.csv if you
need the full detail.

One builder per pipeline stage -- each only knows about that stage's
own inputs, so a new field in (say) preprocessing never touches the
data_prep or model builders:

    build_data_prep_metadata()    -- data/features/sentiment/target
                                      config (from ml/config.yaml), dataset info
    build_split_metadata()        -- train_test_split.py: train/test/(validation)
                                      date ranges, row counts, test_size
    build_preprocessing_metadata() -- preprocessing/ folder: configured
                                      steps + which methods actually fit
    build_model_metadata()        -- regressors/classifiers/deep_learning
                                      folders: algorithm, hyperparams,
                                      architecture (isolated per model_kind,
                                      same as before)
    build_evaluation_metadata()   -- evaluation/ folder: a short scalar
                                      summary (ml metrics + trade summary
                                      + a few named trading metrics), not
                                      the full quantstats output

artifact_manager.py merges all five into one run_config.json under
artifacts/configs/{run_id}/; this module only assembles the dicts.
"""

from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# data_prep/ -- dataset + feature engineering + sentiment + target config
# ----------------------------------------------------------------------
def build_data_prep_metadata(ml_config: dict, row_counts: dict, target_counts: Optional[dict] = None) -> dict:
    """
    Everything that drove data prep: what data was pulled
    (symbol/exchange/timeframe/date range), how features were engineered
    (indicators/patterns), sentiment config, and target generation
    config (horizon/thresholds) -- read straight from ml/config.yaml
    plus the resulting row count and target label distribution.

    Args:
        ml_config: ml/config.yaml dict
        row_counts: dict from the pipeline, uses "total_rows"
        target_counts: optional dict of {label: count} for the
            triple-barrier target column (e.g. {"-1": 1200, "0": 3400,
            "1": 1150}) -- how many rows fall into each class before
            train/test split. None if not computed by the caller.

    Returns:
        dict to be written to artifacts/configs/{run_id}/data_prep.yaml
    """
    data_cfg = ml_config.get("data", {})
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
        "model_type": ml_config.get("model_type"),
        "data": {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "calculate_ohlcv": data_cfg.get("calculate_ohlcv"),
        },
        "features": ml_config.get("features", {}),
        "sentiment": ml_config.get("sentiment", {}),
        "target": ml_config.get("target", {}),
        "total_rows": row_counts.get("total_rows"),
        "target_counts": target_counts or {},
    }


# ----------------------------------------------------------------------
# train_test_split.py -- chronological split dates + row counts
# ----------------------------------------------------------------------
def build_split_metadata(split_info: dict, row_counts: dict) -> dict:
    """
    Train/test(/validation) split record (PDF heading 3's required
    fields): start/end date and row count for each split.

    Args:
        split_info: dict from train_test_split.split_dataset(). If/when
            a validation split is added there, this picks it up
            automatically as long as it uses the same
            val_start/val_end/val_size key naming as train/test -- no
            other caller needs to change.
        row_counts: dict from the pipeline (train_rows/test_rows/
            val_rows are POST preprocessing-drop; dropped_rows_train/
            test/val explain the gap against the pre-drop split)

    Returns:
        dict to be written to artifacts/configs/{run_id}/split.yaml
    """
    def _iso(value):
        return value.isoformat() if isinstance(value, pd.Timestamp) else value

    result = {
        "test_size": split_info.get("test_size"),
        "train_size": split_info.get("train_size"),
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

    # Only added if a validation split actually exists upstream --
    # split_dataset() doesn't produce one today, so this stays out of
    # the written config until it does (no empty/null placeholder
    # clutter in the meantime).
    if split_info.get("val_start") is not None or row_counts.get("val_rows") is not None:
        result["validation"] = {
            "start_date": _iso(split_info.get("val_start")),
            "end_date": _iso(split_info.get("val_end")),
            "rows": row_counts.get("val_rows"),
            "dropped_rows": row_counts.get("dropped_rows_val"),
        }

    return result


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
    requested_hyperparams: Optional[dict] = None,
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
            | "deep_learning_classifier" | "timeseries" | "timeseries_classifier"
        algorithm: str, e.g. "random_forest", "xgboost", "lstm"
        hyperparams: dict -- the model's COMPLETE effective parameter
            set (for sklearn-style models: model.model.get_params(),
            i.e. every override PLUS every library default that
            actually got used; for deep learning/timeseries models:
            same as requested_hyperparams, since those don't expose a
            generic get_params()). Written as "hyperparameters".
        requested_hyperparams: dict or None -- just the keys explicitly
            set in ml_config's param_overrides for this run (i.e. what
            _params_for() returned), kept separately so it's easy to
            see what you actually configured vs what the library filled
            in. Written as "configured_overrides". Defaults to
            `hyperparams` if not given (keeps old call sites working).
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
    if requested_hyperparams is None:
        requested_hyperparams = hyperparams
    info = _MODEL_INFO_BUILDERS[model_kind](algorithm, hyperparams, classes)
    info["configured_overrides"] = requested_hyperparams
    return info


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
    """Darts-backed timeseries REGRESSION model (ml/timeseries/*, TS_REGRESSORS -- NBEATSModel, TCNModel, StatsForecastAutoARIMA) -- no classes, uses Darts' own checkpoint format."""
    return {
        "model_type": "timeseries",
        "algorithm": algorithm,
        "input_chunk_length": hyperparams.get("input_chunk_length"),
        "output_chunk_length": hyperparams.get("output_chunk_length"),
        "hyperparameters": hyperparams,
        "random_seed": hyperparams.get("random_state"),
        "serialization_format": "darts_checkpoint",
    }


def _timeseries_classifier_model_info(algorithm: str, hyperparams: dict, classes: Optional[np.ndarray]) -> dict:
    """Darts-backed timeseries CLASSIFICATION model (ml/timeseries/*, TS_CLASSIFIERS -- e.g. SKLearnClassifierModel) -- lags-based, not chunk-based, includes the class label set."""
    return {
        "model_type": "timeseries_classifier",
        "algorithm": algorithm,
        "lags": hyperparams.get("lags"),
        "lags_past_covariates": hyperparams.get("lags_past_covariates"),
        "lags_future_covariates": hyperparams.get("lags_future_covariates"),
        "output_chunk_length": hyperparams.get("output_chunk_length"),
        "hyperparameters": hyperparams,
        "random_seed": hyperparams.get("random_state"),
        "classes": classes.tolist() if classes is not None else None,
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
    "timeseries_classifier": _timeseries_classifier_model_info,
}


# ----------------------------------------------------------------------
# evaluation/ -- short scalar summary (NOT the full quantstats dump)
# ----------------------------------------------------------------------

# The trading metrics worth keeping in the config as a quick-glance
# summary -- same names evaluator.py's _METRIC_NAME_MAP already maps
# onto compute_stats()'s actual quantstats keys. Everything else
# quantstats returns (rolling_sharpe, pct_rank, per-day series, etc.)
# is computation, not config, and is skipped here -- the full dict is
# still available in pipeline_out/{algorithm}/07_metrics.csv.
_SUMMARY_TRADING_METRICS = (
    "comp", "sharpe", "sortino", "calmar", "max_drawdown", "profit_factor", "win_rate",
)


def build_evaluation_metadata(
    ml_metrics: Optional[dict] = None,
    trading_metrics: Optional[dict] = None,
    trade_summary: Optional[dict] = None,
    signal_counts: Optional[dict] = None,
) -> dict:
    """
    Evaluation results (PDF heading 10) -- a short, scalar-only summary
    meant to be read at a glance and reused for inference later, not
    the full backtest/quantstats computation.

    Args:
        ml_metrics: dict from compute_regression_metrics() /
            compute_classification_metrics() (already scalar-only,
            e.g. {"mae": ..., "rmse": ...} or {"accuracy": ..., ...})
        trading_metrics: the full "metrics" dict from
            stats.calculator.compute_stats() (evaluation["trading_metrics"]);
            only the named scalars in _SUMMARY_TRADING_METRICS are
            pulled out of it, everything else (rolling/per-day series)
            is dropped
        trade_summary: the "trade_summary" dict from compute_stats()
            (final_balance, total_net_profit, total_trades, win_loss)
            -- already scalar-only, kept as-is
        signal_counts: dict from signals.signal_utils.signal_counts(),
            e.g. {"Buy": 40, "Sell": 35, "Hold": 900}

    Returns:
        dict to be written to artifacts/configs/{run_id}/evaluation.yaml
    """
    trading_metrics = trading_metrics or {}
    return {
        "ml_metrics": ml_metrics or {},
        "trading_metrics_summary": {
            k: trading_metrics.get(k) for k in _SUMMARY_TRADING_METRICS if k in trading_metrics
        },
        "trade_summary": trade_summary or {},
        "signal_counts": signal_counts or {},
    }