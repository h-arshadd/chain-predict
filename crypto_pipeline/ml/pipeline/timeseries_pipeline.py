# crypto_pipeline/ml/pipeline/timeseries_pipeline.py

"""
timeseries_pipeline.py
------------------------
Orchestrates PDF headings 1-12 for a timeseries run (Darts-backed
models, ml/timeseries/*: NBEATSModel, TCNModel): dataset loading,
feature selection, train/test split, preprocessing (of covariates only),
model training, prediction, signal generation, evaluation, full
model/experiment persistence, and centralized logging -- same shape as
regression_pipeline.py / classification_pipeline.py, adapted for
Darts' TimeSeries input and forecast-path output.

Routing is the same as the other two pipelines: driven by
ml/config.yaml's model_type -- run_timeseries_pipeline() reads
ml_config["model_type"] and raises immediately if it isn't "timeseries".

Key differences from regression/classification, all because Darts
models don't take flat (X, y) rows:
  - dataset_loader.load_dataset() and train_test_split.split_dataset()
    are reused UNCHANGED (dataset loading and chronological split don't
    care what shape the model eventually wants).
  - preprocessing_pipeline.run_preprocessing() is reused, but only for
    the covariate feature columns -- N-BEATS/TCN don't require
    stationarity transforms on the TARGET itself (they handle
    non-stationarity internally), so only feature_columns are scaled,
    the target ("close") is left as the raw price Darts expects.
  - After preprocessing, train_df/test_df are converted to
    darts.TimeSeries (target_series + past_covariates) via
    base_timeseries_model.series_from_dataframe().
  - predictor.py's generate_predictions() (row-by-row X_test) doesn't
    apply -- generate_timeseries_predictions() (n-steps-ahead forecast)
    is used instead.
  - There is exactly ONE forecast path per run (anchored at the end of
    train), not one prediction per test row -- signals.timeseries_signals
    produces one Buy/Sell/Hold signal for that path, and evaluation
    compares the forecast against the corresponding first-n rows of
    test_df's actual close prices.

algorithm routing: model.algorithm is looked up in
ml/timeseries/registry.py's TS_MODELS -- there is no traditional/deep
learning split here, every model in this family is Darts-backed.
model_kind is always "timeseries", threaded through to
build_model_metadata() same as the other two pipelines.

This module runs the full PDF pipeline end to end: every run writes its
config + fitted preprocessing objects + model weights to artifacts/ via
artifact_manager.save_run(), so any run can be reloaded later with
model_loader.load_run(run_id) for inference.

Evaluation (heading 10) needs 1-minute OHLCV for the test period plus
backtest/config.yaml + stats/config.yaml, same pattern as
ml_config_path, not hardcoded.

Logging (heading 12): setup_logging() is called once, at the very top,
before dataset loading starts. The whole body runs inside a try/except
so any error at any stage is logged (with traceback) before propagating.
"""

import logging

import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.pipeline.predictor import generate_timeseries_predictions
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing
from crypto_pipeline.ml.timeseries.registry import TS_MODELS, build_timeseries_model
from crypto_pipeline.ml.timeseries.base_timeseries_model import series_from_dataframe
from crypto_pipeline.ml.signals.timeseries_signals import generate_timeseries_signals
from crypto_pipeline.ml.evaluation.evaluator import evaluate_model
from crypto_pipeline.backtest.backtest import load_config as load_backtest_config
from crypto_pipeline.ml.persistence.metadata import (
    build_data_prep_metadata,
    build_split_metadata,
    build_preprocessing_metadata,
    build_model_metadata,
    build_evaluation_metadata,
)
from crypto_pipeline.ml.persistence.artifact_manager import make_run_id, save_run, ARTIFACTS_DIR
from crypto_pipeline.ml.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def run_timeseries_pipeline(
    ml_config_path: str,
    ohlcv_1m: pd.DataFrame,
    backtest_config_path: str = None,
    stats_config_path: str = None,
    plot_dir: str = None,
    artifacts_dir: str = ARTIFACTS_DIR,
    run_id: str = None,
) -> dict:
    """
    Run the full timeseries pipeline through model training, forecasting,
    signal generation, evaluation, and persistence.

    Args:
        ml_config_path: path to ml/config.yaml (single config file --
            controls data prep, features, split, preprocessing, model,
            signals, and evaluation)
        ohlcv_1m: 1-minute OHLCV DataFrame covering the test period,
            passed straight through to evaluator.evaluate_model() for
            backtest execution. Same as regression/classification_pipeline.
        backtest_config_path: path to backtest/config.yaml. Defaults to
            backtest.backtest.load_config()'s own default location if
            not given.
        stats_config_path: path to stats/config.yaml. Defaults to
            stats/config.yaml next to stats/calculator.py if not given.
        plot_dir: optional directory to save quantstats plots into --
            skipped if not given.
        artifacts_dir: root artifacts/ folder (PDF heading 11).
        run_id: identifier for this run's config/model/log files.
            Defaults to artifact_manager.make_run_id(algorithm) if not given.

    Returns:
        dict with keys:
            model: trained BaseTimeseriesModel instance
            prediction_result: dict from
                predictor.generate_timeseries_predictions() (forecast,
                last_known_close, n_predictions)
            signals: np.ndarray of str, length 1 (one Buy/Sell/Hold
                signal for the whole forecast path) from
                signals.timeseries_signals.generate_timeseries_signals()
            evaluation: dict from evaluator.evaluate_model() -- ml_metrics
                (MAE/MSE/RMSE against the actual test-period close
                prices), trading_metrics, trade_summary, backtest_result
            run_id: str, this run's identifier
            artifact_paths: dict from artifact_manager.save_run()
            feature_columns: list[str], the covariate columns used
            algorithm: str, the model.algorithm name used
            model_kind: str, always "timeseries"
    """

    ml_config = _load_yaml(ml_config_path)

    algorithm_for_run_id = ml_config.get("model", {}).get("algorithm", "unknown")
    resolved_run_id = run_id or make_run_id(algorithm_for_run_id)
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"Timeseries pipeline starting: run_id={resolved_run_id}, log file={log_path}")

    try:
        model_type = ml_config.get("model_type")
        if model_type != "timeseries":
            raise ValueError(
                f"run_timeseries_pipeline() requires ml_config['model_type'] == "
                f"'timeseries', got '{model_type}'. Use regression_pipeline.py or "
                f"classification_pipeline.py instead -- model_type is set once in "
                f"ml/config.yaml and can't be overridden here."
            )

        # Headings 1-2: load, select features. train_test_split.split_dataset()
        # is reused unchanged -- chronological split doesn't care what
        # shape the model eventually wants the data in.
        df = load_dataset(ml_config)
        selected = select_features(df, ml_config)
        feature_columns = selected["feature_columns"]
        target_column = selected["target_column"]
        timestamp_column = selected["timestamp_column"]

        split_info = split_dataset(df, ml_config, timestamp_column=timestamp_column)
        train_df = split_info["train_df"]
        test_df = split_info["test_df"]

        # Heading 4: preprocessing runs on the COVARIATE columns only --
        # N-BEATS/TCN handle non-stationarity in the target internally
        # (see target_pipeline.py's timeseries branch), so the target
        # column itself is deliberately excluded from the configured
        # preprocessing chain and stays the raw close price.
        preprocessed = run_preprocessing(train_df, test_df, feature_columns, ml_config)
        train_df = preprocessed["train_df"]
        test_df = preprocessed["test_df"]

        row_counts = {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "dropped_rows_train": preprocessed["dropped_rows"]["train"],
            "dropped_rows_test": preprocessed["dropped_rows"]["test"],
        }
        logger.info(f"Row counts: {row_counts}")

        # Convert to darts.TimeSeries. target_series is the training
        # portion's close price; past_covariates is the training
        # portion's feature columns, aligned on the same timestamp axis.
        target_series = series_from_dataframe(train_df, timestamp_column, target_column)
        past_covariates = (
            series_from_dataframe(train_df, timestamp_column, feature_columns)
            if feature_columns else None
        )

        # Heading 5/7: model training. Which algorithm + hyperparams is
        # entirely config-driven, same as regression/classification_pipeline.
        # There's only one registry here (TS_MODELS) -- every model in
        # this family is Darts-backed, no traditional/deep-learning split.
        model_config = ml_config.get("model", {})
        algorithm = model_config.get("algorithm")
        if not algorithm:
            raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'nbeats')")
        params = model_config.get("params", {}) or {}

        if algorithm not in TS_MODELS:
            raise ValueError(
                f"Unknown timeseries algorithm '{algorithm}'. "
                f"Available: {sorted(TS_MODELS.keys())}"
            )
        model_kind = "timeseries"
        logger.info(f"Training timeseries model: algorithm={algorithm}, params={params}")
        model = build_timeseries_model(algorithm, **params)
        model.train(target_series, past_covariates=past_covariates)

        # Heading 8: forecast n steps ahead, where n = output_chunk_length
        # (how far the model was configured to look ahead per predict()
        # call). Forecast covariates (if any) come from test_df, since
        # they're known values (already-computed indicators) covering
        # the forecast horizon, not future-unknown data.
        n = params.get("output_chunk_length")
        if not n:
            raise ValueError("ml/config.yaml's model.params must set output_chunk_length")
        forecast_covariates = (
            series_from_dataframe(test_df, timestamp_column, feature_columns)
            if feature_columns else None
        )
        last_known_close = float(train_df[target_column].iloc[-1])
        prediction_result = generate_timeseries_predictions(
            model, n=n, last_known_close=last_known_close, past_covariates=forecast_covariates
        )

        # Heading 9: convert the forecast path into a Buy/Sell/Hold signal.
        signals = generate_timeseries_signals(prediction_result, ml_config)

        logger.info(
            f"Timeseries pipeline complete: {prediction_result['n_predictions']}-step "
            f"forecast, signal generated"
        )

        # Heading 10: evaluate the trained model. y_true is the actual
        # close price for the first n rows of test_df (the same n steps
        # the forecast covers); y_pred is the forecast itself.
        y_true = test_df[target_column].iloc[:n].to_numpy()
        y_pred = prediction_result["forecast"][: len(y_true)]
        signal_timestamps = test_df[timestamp_column].iloc[:1]  # one signal, anchored at test start

        backtest_config = load_backtest_config(backtest_config_path)
        stats_config = _load_yaml(stats_config_path) if stats_config_path else _default_stats_config()

        evaluation = evaluate_model(
            task_type="timeseries",
            y_true=y_true,
            y_pred=y_pred,
            signals=signals,
            signal_timestamps=signal_timestamps,
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
            plot_dir=plot_dir,
            run_id=algorithm,
        )

        # Heading 11: full model/experiment persistence, split per stage
        # same as regression/classification_pipeline.
        metadata = {
            "data_prep": build_data_prep_metadata(
                ml_config=ml_config,
                row_counts=row_counts,
            ),
            "split": build_split_metadata(
                split_info=split_info,
                row_counts=row_counts,
            ),
            "preprocessing": build_preprocessing_metadata(
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
                preprocessing_config=ml_config.get("preprocessing", {}),
                fit_objects=preprocessed["fit_objects"],
            ),
            "model": build_model_metadata(
                model_kind=model_kind,
                algorithm=algorithm,
                hyperparams=params,
            ),
            "evaluation": build_evaluation_metadata(
                test_metrics={**evaluation["ml_metrics"], **evaluation["trading_metrics"]},
            ),
        }
        artifact_paths = save_run(
            run_id=resolved_run_id,
            metadata=metadata,
            fit_objects=preprocessed["fit_objects"],
            base_dir=artifacts_dir,
            model_save_fn=model.save,
        )
        logger.info(f"Run persisted: run_id={resolved_run_id}, artifacts at {artifact_paths['run_dir']}")

        return {
            "model": model,
            "prediction_result": prediction_result,
            "signals": signals,
            "evaluation": evaluation,
            "run_id": resolved_run_id,
            "artifact_paths": artifact_paths,
            "feature_columns": feature_columns,
            "algorithm": algorithm,
            "model_kind": model_kind,
        }

    except Exception:
        logger.exception(f"Timeseries pipeline failed: run_id={resolved_run_id}")
        raise


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_stats_config() -> dict:
    """Fallback if stats_config_path isn't given -- mirrors regression_pipeline.py's own default."""
    return {}