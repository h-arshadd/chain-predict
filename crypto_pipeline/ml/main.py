# crypto_pipeline/ml/main.py

"""
main.py
-------
Top-level entry point for the ML Module (PDF headings 1-12, end to end).

This didn't exist before: regression_pipeline.py / classification_pipeline.py
/ timeseries_pipeline.py each define a run_*_pipeline() function, but
nothing in the codebase ever called them -- this file is that caller.

Routing is automatic, driven by ml/data_prep/config.yaml's model_type,
exactly the same field each pipeline file already gates on internally:

    model_type: regression     -> regressors/deep_learning regressors
    model_type: classification -> classifiers/deep_learning classifiers
    model_type: timeseries     -> darts (nbeats/tcn)

Unlike calling run_regression_pipeline() etc. directly, this file does
NOT treat the pipeline as one black box. It calls the exact same
underlying stage functions each pipeline file already calls internally
(load_dataset, select_features, split_dataset, run_preprocessing, the
matching train/predict/signal functions, evaluate_model, save_run) IN
THE SAME ORDER, but writes a CSV after every stage into pipeline_out/ so
you can see exactly what each stage did -- no logic is different from
the real pipeline, this is just the real pipeline with a to_csv() call
added between each step.

Usage (as a script):

    python -m crypto_pipeline.ml.main --ohlcv-1m path/to/ohlcv_1m.csv

Or import run_ml_pipeline() directly from your own driver code:

    from crypto_pipeline.ml.main import run_ml_pipeline
    result = run_ml_pipeline(ohlcv_1m=my_df)

ohlcv_1m (1-minute OHLCV covering the test period) is the one input this
project's zip has no automatic way to fetch -- the real pipeline pulls
it from Postgres via crypto_pipeline.utils.db_utils.get_candles_from_db(),
which isn't part of what was handed to this file, so it's taken as a
plain argument/CSV path here instead, same as the three pipeline files
already require it as a parameter rather than fetching it themselves.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.pipeline.predictor import generate_predictions, generate_timeseries_predictions
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing

from crypto_pipeline.ml.regressors.registry import REGRESSORS, build_regressor
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS, build_classifier
from crypto_pipeline.ml.deep_learning.registry import (
    DL_REGRESSORS, build_dl_regressor,
    DL_CLASSIFIERS, build_dl_classifier,
)
from crypto_pipeline.ml.timeseries.registry import TS_MODELS, build_timeseries_model
from crypto_pipeline.ml.timeseries.base_timeseries_model import series_from_dataframe

from crypto_pipeline.ml.signals.regression_signals import generate_regression_signals
from crypto_pipeline.ml.signals.classification_signals import generate_classification_signals
from crypto_pipeline.ml.signals.timeseries_signals import generate_timeseries_signals
from crypto_pipeline.ml.signals.signal_utils import signal_counts

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

# Where every stage's inspection CSV gets written. Doesn't affect
# artifacts/ (that's still owned by artifact_manager.save_run(), see
# heading 11) -- this is purely a "let me see what happened" folder.
PIPELINE_OUT_DIR = "pipeline_out"


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_stats_config() -> dict:
    stats_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "stats", "config.yaml"
    )
    return _load_yaml(stats_config_path)


def run_ml_pipeline(
    ml_config_path: str = "ml/config.yaml",
    data_prep_config_path: str = "ml/data_prep/config.yaml",
    ohlcv_1m: pd.DataFrame = None,
    backtest_config_path: str = None,
    stats_config_path: str = None,
    plot_dir: str = None,
    artifacts_dir: str = ARTIFACTS_DIR,
    run_id: str = None,
    pipeline_out_dir: str = PIPELINE_OUT_DIR,
) -> dict:
    """
    Run the full ML pipeline end to end, routed by model_type, writing a
    CSV after every stage into pipeline_out_dir/ along the way.

    Args:
        ml_config_path: path to ml/config.yaml
        data_prep_config_path: path to ml/data_prep/config.yaml -- its
            model_type field decides regression vs classification vs
            timeseries at every branching point below.
        ohlcv_1m: 1-minute OHLCV DataFrame (datetime, open, high, low,
            close) covering the test period, needed for backtest
            execution (PDF heading 10). Required.
        backtest_config_path: path to backtest/config.yaml. Defaults to
            backtest.backtest.load_config()'s own default location.
        stats_config_path: path to stats/config.yaml. Defaults to
            stats/config.yaml next to stats/calculator.py.
        plot_dir: optional directory to save quantstats plot data into.
        artifacts_dir: root artifacts/ folder (PDF heading 11).
        run_id: identifier for this run's config/model/log files.
        pipeline_out_dir: folder every stage's inspection CSV gets
            written to (default "pipeline_out"). Separate from
            artifacts_dir -- this is just for you to look at, nothing
            downstream reads it back.

    Returns:
        dict, same shape as run_regression_pipeline() / etc. -- model,
        prediction_result, signals, evaluation, run_id, artifact_paths,
        feature_columns, algorithm, model_kind, plus model-type-specific
        keys (y_test/split_info/fit_objects for regression/classification).
    """
    if ohlcv_1m is None:
        raise ValueError(
            "ohlcv_1m is required (1-minute OHLCV DataFrame covering the test "
            "period, needed for backtest execution) -- there is no default "
            "fetch for it in this codebase, pass it in explicitly."
        )

    os.makedirs(pipeline_out_dir, exist_ok=True)

    def _dump(name: str, df: pd.DataFrame):
        path = os.path.join(pipeline_out_dir, name)
        df.to_csv(path, index=False)
        logger.info(f"[pipeline_out] wrote {path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return path

    ml_config = _load_yaml(ml_config_path)
    data_prep_config = _load_yaml(data_prep_config_path)

    model_type = data_prep_config.get("model_type")
    if model_type not in ("regression", "classification", "timeseries"):
        raise ValueError(
            f"Unknown model_type '{model_type}' in {data_prep_config_path}. "
            f"Expected one of: regression, classification, timeseries"
        )

    algorithm_for_run_id = ml_config.get("model", {}).get("algorithm", "unknown")
    resolved_run_id = run_id or make_run_id(algorithm_for_run_id)
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"ML pipeline starting: run_id={resolved_run_id}, model_type={model_type}, log file={log_path}")

    try:
        # ---- Heading 1: dataset loading -----------------------------
        df = load_dataset(ml_config_path, data_prep_config_path)
        _dump("01_dataset.csv", df)

        # ---- Heading 2: feature selection ---------------------------
        selected = select_features(df, ml_config, data_prep_config)
        feature_columns = selected["feature_columns"]
        target_column = selected["target_column"]
        timestamp_column = selected["timestamp_column"]
        _dump(
            "02_selected_columns.csv",
            df[[timestamp_column] + feature_columns + [target_column]],
        )

        # ---- Heading 3: train/test split -----------------------------
        split_info = split_dataset(df, ml_config, timestamp_column=timestamp_column)
        _dump("03_train.csv", split_info["train_df"])
        _dump("03_test.csv", split_info["test_df"])

        # ---- Heading 4: preprocessing (feature_columns only) ---------
        preprocessed = run_preprocessing(
            split_info["train_df"], split_info["test_df"], feature_columns, ml_config
        )
        train_df = preprocessed["train_df"]
        test_df = preprocessed["test_df"]
        _dump("04_train_preprocessed.csv", train_df)
        _dump("04_test_preprocessed.csv", test_df)

        row_counts = {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "dropped_rows_train": preprocessed["dropped_rows"]["train"],
            "dropped_rows_test": preprocessed["dropped_rows"]["test"],
        }
        logger.info(f"Row counts: {row_counts}")

        model_config = ml_config.get("model", {})
        algorithm = model_config.get("algorithm")
        if not algorithm:
            raise ValueError("ml/config.yaml must set model.algorithm")
        params = model_config.get("params", {}) or {}

        y_test = None
        classes = None

        # ---- Headings 5-7: model training (branches by model_type) ---
        if model_type == "regression":
            X_train, y_train = train_df[feature_columns], train_df[target_column]
            X_test, y_test = test_df[feature_columns], test_df[target_column]

            if algorithm in REGRESSORS:
                model_kind = "regressor"
                model = build_regressor(algorithm, **params)
            elif algorithm in DL_REGRESSORS:
                model_kind = "deep_learning_regressor"
                model = build_dl_regressor(algorithm, **params)
            else:
                raise ValueError(
                    f"Unknown regression algorithm '{algorithm}'. "
                    f"Available traditional: {sorted(REGRESSORS.keys())}, "
                    f"deep learning: {sorted(DL_REGRESSORS.keys())}"
                )
            model.train(X_train, y_train)

            prediction_result = generate_predictions(model, X_test, task_type="regression")
            signals = generate_regression_signals(prediction_result, ml_config)

            predictions_df = pd.DataFrame({
                timestamp_column: test_df[timestamp_column],
                "actual": y_test,
                "predicted": prediction_result["predictions"],
            })
            signal_timestamps = test_df[timestamp_column]

        elif model_type == "classification":
            X_train, y_train = train_df[feature_columns], train_df[target_column]
            X_test, y_test = test_df[feature_columns], test_df[target_column]

            if algorithm in CLASSIFIERS:
                model_kind = "classifier"
                model = build_classifier(algorithm, **params)
            elif algorithm in DL_CLASSIFIERS:
                model_kind = "deep_learning_classifier"
                model = build_dl_classifier(algorithm, **params)
            else:
                raise ValueError(
                    f"Unknown classification algorithm '{algorithm}'. "
                    f"Available traditional: {sorted(CLASSIFIERS.keys())}, "
                    f"deep learning: {sorted(DL_CLASSIFIERS.keys())}"
                )
            model.train(X_train, y_train)

            prediction_result = generate_predictions(model, X_test, task_type="classification")
            signals = generate_classification_signals(prediction_result, ml_config)
            classes = np.asarray(model.classes_)

            predictions_df = pd.DataFrame({
                timestamp_column: test_df[timestamp_column],
                "actual": y_test,
                "predicted": prediction_result["predictions"],
            })
            for i, cls in enumerate(prediction_result["classes"]):
                predictions_df[f"prob_{cls}"] = prediction_result["probabilities"][:, i]
            signal_timestamps = test_df[timestamp_column]

        else:  # timeseries
            if algorithm not in TS_MODELS:
                raise ValueError(
                    f"Unknown timeseries algorithm '{algorithm}'. Available: {sorted(TS_MODELS.keys())}"
                )
            model_kind = "timeseries"
            n = params.get("output_chunk_length")
            if not n:
                raise ValueError("ml/config.yaml's model.params must set output_chunk_length")

            train_df[timestamp_column] = pd.to_datetime(train_df[timestamp_column])
            test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column])

            target_series = series_from_dataframe(train_df, timestamp_column, target_column)
            past_covariates = (
                series_from_dataframe(train_df, timestamp_column, feature_columns)
                if feature_columns else None
            )
            model = build_timeseries_model(algorithm, **params)
            model.train(target_series, past_covariates=past_covariates)

            forecast_covariates = (
                series_from_dataframe(test_df, timestamp_column, feature_columns)
                if feature_columns else None
            )
            last_known_close = float(train_df[target_column].iloc[-1])
            prediction_result = generate_timeseries_predictions(
                model, n=n, last_known_close=last_known_close, past_covariates=forecast_covariates
            )
            signals = generate_timeseries_signals(prediction_result, ml_config)

            n_pred = prediction_result["n_predictions"]
            predictions_df = pd.DataFrame({
                timestamp_column: test_df[timestamp_column].iloc[:n_pred].reset_index(drop=True),
                "actual": test_df[target_column].iloc[:n_pred].reset_index(drop=True),
                "predicted": prediction_result["forecast"],
            })
            signal_timestamps = test_df[timestamp_column].iloc[:1]

        _dump("05_predictions.csv", predictions_df)

        # ---- Heading 9: signal generation -----------------------------
        if model_type == "timeseries":
            signals_df = pd.DataFrame({
                timestamp_column: [signal_timestamps.iloc[0]],
                "signal": signals,
            })
        else:
            signals_df = predictions_df.copy()
            signals_df["signal"] = signals
        _dump("06_signals.csv", signals_df)
        logger.info(f"Signal counts: {signal_counts(signals)}")

        # ---- Heading 10: evaluation (ML metrics + backtest + stats) ---
        backtest_config = load_backtest_config(backtest_config_path)
        stats_config = _load_yaml(stats_config_path) if stats_config_path else _default_stats_config()

        evaluation = evaluate_model(
            task_type=model_type,
            y_true=predictions_df["actual"].to_numpy(),
            y_pred=predictions_df["predicted"].to_numpy(),
            signals=signals,
            signal_timestamps=pd.to_datetime(signal_timestamps),
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
            plot_dir=plot_dir,
            run_id=algorithm,
        )

        metrics_row = {**evaluation["ml_metrics"], **evaluation["trading_metrics"]}
        _dump("07_metrics.csv", pd.DataFrame([metrics_row]))
        _dump("07_trade_ledger.csv", evaluation["backtest_result"]["trade_ledger"])

        # ---- Heading 11: full model/experiment persistence ------------
        metadata = {
            "data_prep": build_data_prep_metadata(
                data_prep_config=data_prep_config,
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
                classes=classes,
            ),
            "evaluation": build_evaluation_metadata(
                test_metrics=metrics_row,
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

        result = {
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
        if model_type in ("regression", "classification"):
            result.update({
                "y_test": y_test,
                "split_info": split_info,
                "fit_objects": preprocessed["fit_objects"],
            })
        return result

    except Exception:
        logger.exception(f"ML pipeline failed: run_id={resolved_run_id}")
        raise


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the full ML pipeline end to end, stage CSVs included.")
    parser.add_argument("--ml-config", default="ml/config.yaml", help="Path to ml/config.yaml")
    parser.add_argument(
        "--data-prep-config", default="ml/data_prep/config.yaml",
        help="Path to ml/data_prep/config.yaml (its model_type field decides routing)",
    )
    parser.add_argument(
        "--ohlcv-1m", required=True,
        help="Path to a CSV of 1-minute OHLCV (columns: datetime, open, high, low, close) "
             "covering the test period, for backtest execution.",
    )
    parser.add_argument("--backtest-config", default=None, help="Path to backtest/config.yaml")
    parser.add_argument("--stats-config", default=None, help="Path to stats/config.yaml")
    parser.add_argument("--plot-dir", default=None, help="Optional directory for quantstats plot data")
    parser.add_argument("--artifacts-dir", default=ARTIFACTS_DIR, help="Root artifacts/ folder")
    parser.add_argument("--run-id", default=None, help="Optional explicit run_id")
    parser.add_argument(
        "--pipeline-out-dir", default=PIPELINE_OUT_DIR,
        help="Folder every stage's inspection CSV gets written to (default: pipeline_out)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ohlcv_1m = pd.read_csv(args.ohlcv_1m)
    ohlcv_1m["datetime"] = pd.to_datetime(ohlcv_1m["datetime"])

    result = run_ml_pipeline(
        ml_config_path=args.ml_config,
        data_prep_config_path=args.data_prep_config,
        ohlcv_1m=ohlcv_1m,
        backtest_config_path=args.backtest_config,
        stats_config_path=args.stats_config,
        plot_dir=args.plot_dir,
        artifacts_dir=args.artifacts_dir,
        run_id=args.run_id,
        pipeline_out_dir=args.pipeline_out_dir,
    )

    print(f"run_id: {result['run_id']}")
    print(f"algorithm: {result['algorithm']}")
    print(f"ML metrics: {result['evaluation']['ml_metrics']}")
    print(f"Trading metrics: {result['evaluation']['trading_metrics']}")
    print(f"Artifacts written to: {result['artifact_paths']['run_dir']}")
    print(f"Stage-by-stage CSVs written to: {args.pipeline_out_dir}/")