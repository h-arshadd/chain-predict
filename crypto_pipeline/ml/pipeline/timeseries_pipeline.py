# crypto_pipeline/ml/pipeline/timeseries_pipeline.py

"""
timeseries_pipeline.py
------------------------
Orchestrates PDF headings 1-13 for a timeseries run (Darts-backed
models, ml/timeseries/*). Mirrors regression_pipeline.py /
classification_pipeline.py's shape exactly, including the
run_*_pipeline() / run_*_algorithm() split:

    run_timeseries_pipeline()   -- headings 1-4 (load, select features,
        split, preprocess) done ONCE, then delegates to
        run_timeseries_algorithm() below. Standalone entry point for
        training a single timeseries algorithm.
    run_timeseries_algorithm()  -- headings 5-13 (train, predict, signal,
        evaluate, persist) for ONE algorithm, given data that's already
        been loaded/split/preprocessed by the caller. This is what
        main.py's run_ml_pipeline() calls directly, once per algorithm
        in ml_config["model"]["algorithms"]["timeseries"], reusing the
        SAME loaded/split/preprocessed dataset across all of them --
        same reasoning run_regression_algorithm() /
        run_classification_algorithm() give in their own docstrings.

Two families under model_type: timeseries (see ml/timeseries/registry.py):
    TS_REGRESSORS  (nbeats, tcn, statsforecast) -- forecast the raw
        close price directly. target_pipeline.py generates a raw-price
        target for these.
    TS_CLASSIFIERS (sklearn_classifier) -- forecast a discrete
        -1/0/1 triple-barrier label directly. Same label target
        classification_pipeline.py uses.
Which family model.algorithm belongs to is resolved once near the top
of run_timeseries_algorithm() and threaded through prediction shape,
signal generation, evaluation metrics, and persisted metadata --
model_kind ("timeseries" vs "timeseries_classifier") is what lets
persistence/model_loader.py reconstruct the right class later, same
role model_kind plays in regression/classification_pipeline.py.

Differences from regression/classification, all because Darts models
take TimeSeries, not flat (X, y) rows:
  - dataset_loader.load_dataset() / train_test_split.split_dataset()
    reused unchanged.
  - preprocessing_pipeline.run_preprocessing() runs on the covariate
    feature columns only -- the target itself (raw price or label)
    isn't scaled/differenced for the models in this project today (see
    ml/timeseries/postprocessing.py's module docstring for what to do
    if a future model needs that).
  - train_df/test_df get converted to darts.TimeSeries.
  - predictor.generate_timeseries_predictions() (one anchored n-step
    forecast) or generate_timeseries_historical_predictions()
    (walk-forward, PDF heading 10) replaces generate_predictions().

forecast_mode (ml/config.yaml's model.params.forecast_mode, default
"anchored") picks between the two:
    "anchored"   -- single forecast path from the end of train.
    "historical" -- many forecasts walking across train+test via
                    historical_forecasts(); one-step
                    (forecast_horizon=1, stride=1) or fixed-window
                    (train_length=N) depending on
                    model.params.historical_forecasts.

Logging (heading 12): setup_logging() runs once at the top of
run_timeseries_algorithm(); the whole body runs inside a try/except so
any failure is logged with traceback before propagating.
"""

import logging
import os

import numpy as np
import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.pipeline.predictor import (
    generate_timeseries_predictions,
    generate_timeseries_historical_predictions,
)
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing
from crypto_pipeline.ml.timeseries.registry import TS_REGRESSORS, TS_CLASSIFIERS, build_ts_regressor, build_ts_classifier
from crypto_pipeline.ml.timeseries.base_timeseries_model import series_from_dataframe
from crypto_pipeline.ml.timeseries.postprocessing import smooth_forecast, fill_missing_forecast
from crypto_pipeline.ml.signals.timeseries_signals import generate_timeseries_signals
from crypto_pipeline.ml.signals.signal_utils import signal_counts as _signal_counts
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
        ohlcv_1m: 1-minute OHLCV DataFrame (datetime, open, high, low,
            close) covering the test period -- passed straight through
            to evaluator.evaluate_model() for backtest execution.
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
        Same dict shape as run_timeseries_algorithm() -- see its
        docstring for the key list.
    """
    ml_config = _load_yaml(ml_config_path)

    model_type = ml_config.get("model_type")
    if model_type != "timeseries":
        raise ValueError(
            f"run_timeseries_pipeline() requires ml_config['model_type'] == "
            f"'timeseries', got '{model_type}'. Use regression_pipeline.py or "
            f"classification_pipeline.py instead -- model_type is set once in "
            f"ml/config.yaml and can't be overridden here."
        )

    # Headings 1-4: load, select features, split, preprocess. Standalone
    # callers of this function need this done once, here -- callers that
    # already have this data (e.g. main.py training several timeseries
    # algorithms against the SAME dataset) should call
    # run_timeseries_algorithm() directly instead of going through this loader.
    df = load_dataset(ml_config)
    selected = select_features(df, ml_config)
    feature_columns = selected["feature_columns"]
    target_column = selected["target_column"]
    timestamp_column = selected["timestamp_column"]

    split_info = split_dataset(df, ml_config, timestamp_column=timestamp_column)

    preprocessed = run_preprocessing(
        split_info["train_df"], split_info["test_df"], feature_columns, ml_config,
        val_df=split_info.get("val_df"),
    )
    # split_info["val_df"] was raw/untransformed -- replace it with the
    # preprocessed version so run_timeseries_algorithm()'s
    # split_info.get("val_df") read below sees val data transformed the
    # same way as train_df/test_df.
    split_info["val_df"] = preprocessed["val_df"]

    row_counts = {
        "total_rows": len(df),
        "train_rows": len(preprocessed["train_df"]),
        "test_rows": len(preprocessed["test_df"]),
        "dropped_rows_train": preprocessed["dropped_rows"]["train"],
        "dropped_rows_test": preprocessed["dropped_rows"]["test"],
    }

    model_config = ml_config.get("model", {})
    algorithm = model_config.get("algorithm")
    if not algorithm:
        raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'nbeats', 'sklearn_classifier')")
    params = dict(model_config.get("params", {}) or {})

    return run_timeseries_algorithm(
        algorithm=algorithm,
        params=params,
        ml_config=ml_config,
        train_df=preprocessed["train_df"],
        test_df=preprocessed["test_df"],
        feature_columns=feature_columns,
        target_column=target_column,
        timestamp_column=timestamp_column,
        split_info=split_info,
        fit_objects=preprocessed["fit_objects"],
        row_counts=row_counts,
        ohlcv_1m=ohlcv_1m,
        backtest_config_path=backtest_config_path,
        stats_config_path=stats_config_path,
        plot_dir=plot_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
    )


def run_timeseries_algorithm(
    algorithm: str,
    params: dict,
    ml_config: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    timestamp_column: str,
    split_info: dict,
    fit_objects: list,
    row_counts: dict,
    ohlcv_1m: pd.DataFrame,
    backtest_config_path: str = None,
    stats_config_path: str = None,
    backtest_config: dict = None,
    stats_config: dict = None,
    plot_dir: str = None,
    artifacts_dir: str = ARTIFACTS_DIR,
    models_dir: str = None,
    run_id: str = None,
    requested_hyperparams: dict = None,
    effective_hyperparams_fn=None,
) -> dict:
    """
    Train + forecast + signal + evaluate + persist ONE timeseries
    algorithm, given data that's already been loaded/split/preprocessed
    (headings 1-4) by the caller.

    This is the part of run_timeseries_pipeline() (headings 5-13) that
    doesn't care where train_df/test_df came from -- pulled out into its
    own function so a caller training several timeseries algorithms
    against the SAME dataset (e.g. main.py's run_ml_pipeline(), one call
    per algorithm in ml_config["model"]["algorithms"]["timeseries"])
    doesn't have to reload/re-split/re-preprocess the dataset once per
    algorithm just to reach this logic. Mirrors
    run_regression_algorithm() / run_classification_algorithm() exactly
    -- see those files' docstrings for the fuller rationale.

    Args:
        algorithm: str, e.g. "nbeats" -- looked up in
            ml/timeseries/registry.py's TS_REGRESSORS first, then
            TS_CLASSIFIERS.
        params: dict, hyperparameters passed to the model constructor,
            PLUS the two pipeline-level knobs forecast_mode and
            historical_forecasts (popped off before the Darts
            constructor is built -- see module docstring).
        ml_config: the loaded ml/config.yaml dict.
        train_df / test_df: preprocessed train/test DataFrames (output
            of preprocessing_pipeline.run_preprocessing(), covariate
            columns only -- the target column itself is untouched, see
            module docstring).
        feature_columns / target_column / timestamp_column: str/list,
            from preprocessing.feature_selector.select_features().
        split_info: dict from train_test_split.split_dataset().
        fit_objects: list from run_preprocessing() (fitted scalers, for
            persistence).
        row_counts: dict (total_rows/train_rows/test_rows/dropped_rows_*),
            for metadata.
        ohlcv_1m: 1-minute OHLCV DataFrame for backtest execution.
        backtest_config_path / stats_config_path: as in
            run_timeseries_pipeline().
        backtest_config / stats_config: optional pre-loaded config
            dicts -- if given, used directly instead of re-reading the
            *_config_path off disk (same as regression/classification's
            algorithm-level functions; lets main.py load these ONCE
            outside its per-algorithm loop).
        plot_dir: optional quantstats plot directory.
        artifacts_dir: root artifacts/ folder.
        models_dir: root models/ folder, forwarded to
            artifact_manager.save_run() -- defaults to save_run()'s own
            default if not given.
        run_id: this run's identifier. Defaults to
            make_run_id(algorithm) if not given.
        requested_hyperparams: optional dict, the subset of params
            explicitly configured -- passed to build_model_metadata()
            as "configured_overrides". Defaults to `params` if not given.
        effective_hyperparams_fn: optional callable(model, params) -> dict,
            called with the fitted model and requested params AFTER
            training. Darts models have no generic get_params() the way
            sklearn estimators do, so this typically just returns
            `params` back unchanged -- present only for call-site
            symmetry with run_regression_algorithm()/
            run_classification_algorithm(), which do use it meaningfully.

    Returns:
        dict with keys:
            model: trained BaseTimeseriesModel or BaseTimeseriesClassifier
            prediction_result: dict from predictor.py (anchored or historical)
            signals: np.ndarray of str (Buy/Sell/Hold)
            evaluation: dict from evaluator.evaluate_model()
            run_id: str, this run's identifier
            artifact_paths: dict from artifact_manager.save_run()
            feature_columns: list[str], the covariate columns used
            algorithm: str, the model.algorithm name used
            model_kind: str, "timeseries" (regressor) or "timeseries_classifier"
            task_type: str, "timeseries_regression" or "timeseries_classification"
    """
    resolved_run_id = run_id or make_run_id(algorithm)
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"Timeseries pipeline starting: run_id={resolved_run_id}, log file={log_path}")

    try:
        logger.info(f"Row counts: {row_counts}")

        if algorithm in TS_CLASSIFIERS:
            is_classifier, model_kind, task_type = True, "timeseries_classifier", "timeseries_classification"
        elif algorithm in TS_REGRESSORS:
            is_classifier, model_kind, task_type = False, "timeseries", "timeseries_regression"
        else:
            raise ValueError(
                f"Unknown timeseries algorithm '{algorithm}'. "
                f"Available regressors: {sorted(TS_REGRESSORS.keys())}, "
                f"classifiers: {sorted(TS_CLASSIFIERS.keys())}"
            )

        # forecast_mode / historical_forecasts are pipeline-level knobs,
        # not Darts constructor kwargs -- popped off params before
        # building the model (heading 10's one-step / fixed-window
        # modes, see predictor.generate_timeseries_historical_predictions()).
        params = dict(params)
        forecast_mode = params.pop("forecast_mode", "anchored")
        hf_config = params.pop("historical_forecasts", {}) or {}

        # val_df (chronologically between train and test) is only
        # present when ml/config.yaml's split.val_size > 0 -- see
        # train_test_split.split_dataset(). None otherwise, same as it
        # always was, so this is fully backward compatible with a
        # val_size-less config.
        val_df = split_info.get("val_df")

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[timestamp_column] = pd.to_datetime(train_df[timestamp_column])
        test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column])
        if val_df is not None:
            val_df = val_df.copy()
            val_df[timestamp_column] = pd.to_datetime(val_df[timestamp_column])
            full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        else:
            full_df = pd.concat([train_df, test_df], ignore_index=True)

        def to_series(frame, cols):
            return series_from_dataframe(frame, timestamp_column, cols) if cols else None

        target_series = series_from_dataframe(train_df, timestamp_column, target_column)
        train_past_cov = to_series(train_df, feature_columns)
        # future_covariates must cover the forecast horizon too (Darts
        # requirement) -- built from train+val+test combined (or
        # train+test if there's no val split).
        full_future_cov = to_series(full_df, feature_columns)

        # val_series / val_past_covariates: only built if a validation
        # split exists in config. Passed through to nbeats/tcn's train()
        # so Darts' own early stopping / LR-scheduling (on val_loss) has
        # something to watch -- ignored (no-op) by algorithms that don't
        # support it (statsforecast, sklearn_classifier), same shape as
        # every other pipeline-level knob here.
        if val_df is not None:
            val_series = series_from_dataframe(val_df, timestamp_column, target_column)
            val_past_cov = to_series(val_df, feature_columns)
        else:
            val_series = None
            val_past_cov = None

        # Heading 5/7: model training. Which algorithm + hyperparams is
        # entirely config-driven -- this function contains no
        # model-specific logic at all. algorithm is looked up in
        # TS_REGRESSORS first, then TS_CLASSIFIERS; whichever matches
        # decides model_kind, but both expose the same train()/
        # predict()/save() interface (plus predict_proba()/classes_ for
        # classifiers), so nothing below this block branches on which
        # kind it is beyond is_classifier.
        logger.info(f"Training timeseries model: algorithm={algorithm}, params={params}")
        model = (build_ts_classifier if is_classifier else build_ts_regressor)(algorithm, **params)
        model.train(
            target_series,
            past_covariates=train_past_cov,
            future_covariates=full_future_cov,
            val_series=val_series,
            val_past_covariates=val_past_cov,
        )

        # Heading 8: prediction -- "anchored" (single n-step forecast
        # from the end of train) or "historical" (walk-forward across
        # train+test, one-step or fixed-window).
        n = params.get("output_chunk_length", 1)
        if forecast_mode == "anchored":
            # IMPORTANT: model.predict(n=n, ...) is called below WITHOUT
            # series= (predictor.generate_timeseries_predictions() never
            # passes it), so per Darts' own docs ("Predict the n time
            # step following the end of the training series") the
            # forecast is anchored at the end of whatever series
            # train() was fit on -- i.e. the end of train_df -- NOT the
            # end of val_df or test_df, regardless of whether a
            # validation split exists. past_covariates therefore must
            # cover input_chunk_length steps ending at TRAIN's last
            # timestamp (extended forward through the forecast horizon
            # via future covariates), not test's. Slicing from test_df
            # alone (the pre-val-split behavior) or from val/test's
            # boundary (an earlier, incorrect version of this fix) both
            # give Darts the wrong window here.
            # input_chunk_length (nbeats/tcn) or lags/lags_past_covariates
            # (sklearn_classifier) -- whichever this algorithm's params
            # actually define -- is how far back past_covariates must
            # extend before train's end. statsforecast has neither (0,
            # no lookback needed -- Auto-ARIMA has no such notion).
            lookback = max(
                params.get("input_chunk_length", 0),
                params.get("lags", 0),
                params.get("lags_past_covariates", 0),
            )
            train_end_idx = len(full_df) - len(val_df) - len(test_df) if val_df is not None else len(full_df) - len(test_df)
            cov_start_idx = max(train_end_idx - lookback, 0)
            test_past_cov = to_series(full_df.iloc[cov_start_idx:], feature_columns)
            last_known_close = None if is_classifier else float(train_df[target_column].iloc[-1])
            prediction_result = generate_timeseries_predictions(
                model, n=n, last_known_close=last_known_close,
                past_covariates=test_past_cov, future_covariates=full_future_cov,
            )
        elif forecast_mode == "historical":
            full_target_series = to_series(full_df, target_column)
            full_past_cov = to_series(full_df, feature_columns)
            prediction_result = generate_timeseries_historical_predictions(
                model, series=full_target_series,
                past_covariates=full_past_cov, future_covariates=full_future_cov,
                forecast_horizon=hf_config.get("forecast_horizon", 1),
                stride=hf_config.get("stride", 1),
                retrain=hf_config.get("retrain", False),
                train_length=hf_config.get("train_length"),
            )
        else:
            raise ValueError(
                f"model.params.forecast_mode must be 'anchored' or 'historical', got '{forecast_mode}'"
            )

        # Heading 13: forecast post-processing -- missing-forecast
        # handling always applied if NaNs are present; smoothing only
        # if configured, and only for regression forecasts (a smoothed
        # class label makes no sense). Inverse scaling/differencing is
        # not needed here since the target column bypasses
        # preprocessing.steps for this project's models (see module
        # docstring) -- ml/timeseries/postprocessing.py's
        # inverse_transform_forecast() is available for a future model
        # that does need it.
        forecast = prediction_result["forecast"]
        if np.isnan(forecast).any():
            forecast = fill_missing_forecast(forecast)
        smoothing_config = (ml_config.get("postprocessing", {}) or {}).get("smoothing", {})
        if not is_classifier and smoothing_config.get("enabled"):
            forecast = smooth_forecast(forecast, window=smoothing_config.get("window", 3))
        prediction_result["forecast"] = forecast

        # Heading 9: Buy/Sell/Hold signal. Only meaningful for
        # "anchored" mode's single forecast path -- "historical" mode
        # produces many forecasts for evaluation, not one tradeable
        # decision, so it gets an all-Hold signal array instead (no
        # trades taken, evaluation still runs on the ML metrics).
        n_pred = prediction_result["n_predictions"]
        if forecast_mode == "anchored":
            signals = generate_timeseries_signals(prediction_result, ml_config)
            signal_timestamps = test_df[timestamp_column].iloc[:1]
            y_true = test_df[target_column].iloc[:n_pred].to_numpy()
            y_pred = forecast[: len(y_true)]
        else:
            signals = np.array(["Hold"] * n_pred)
            signal_timestamps = full_df[timestamp_column].iloc[-n_pred:]
            y_true = full_df[target_column].iloc[-n_pred:].to_numpy()
            y_pred = forecast

        logger.info(
            f"Timeseries pipeline complete: {n_pred} predictions generated "
            f"(forecast_mode={forecast_mode}), signals generated"
        )

        # Heading 10: evaluate the trained model. Same pre-loaded-config
        # pattern as run_regression_algorithm()/run_classification_algorithm() --
        # main.py loads these ONCE outside its per-algorithm loop and
        # passes the same dict into every algorithm's call here.
        resolved_backtest_config = (
            backtest_config if backtest_config is not None else load_backtest_config(backtest_config_path)
        )
        resolved_stats_config = (
            stats_config if stats_config is not None else
            (_load_yaml(stats_config_path) if stats_config_path else _default_stats_config())
        )

        evaluation = evaluate_model(
            task_type=task_type,
            y_true=y_true,
            y_pred=y_pred,
            signals=signals,
            signal_timestamps=pd.to_datetime(signal_timestamps),
            ohlcv_1m=ohlcv_1m,
            backtest_config=resolved_backtest_config,
            stats_config=resolved_stats_config,
            plot_dir=plot_dir,
            run_id=algorithm,
        )

        # Heading 11: full model/experiment persistence, same split-by-stage
        # shape as regression/classification_pipeline. classes comes from
        # the trained model itself (model.classes_) for the classifier
        # family, same reasoning classification_pipeline.py gives --
        # None for the regression family (no notion of classes).
        classes = model.classes_ if is_classifier else None
        metadata = {
            "data_prep": build_data_prep_metadata(ml_config=ml_config, row_counts=row_counts),
            "split": build_split_metadata(split_info=split_info, row_counts=row_counts),
            "preprocessing": build_preprocessing_metadata(
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
                preprocessing_config=ml_config.get("preprocessing", {}),
                fit_objects=fit_objects,
            ),
            "model": build_model_metadata(
                model_kind=model_kind,
                algorithm=algorithm,
                hyperparams=effective_hyperparams_fn(model, params) if effective_hyperparams_fn is not None else params,
                requested_hyperparams=requested_hyperparams if requested_hyperparams is not None else params,
                classes=classes,
            ),
            "evaluation": build_evaluation_metadata(
                ml_metrics=evaluation["ml_metrics"],
                trading_metrics=evaluation["trading_metrics"],
                trade_summary=evaluation["trade_summary"],
                signal_counts=_signal_counts(signals),
            ),
        }
        save_run_kwargs = dict(
            run_id=resolved_run_id,
            metadata=metadata,
            fit_objects=fit_objects,
            base_dir=artifacts_dir,
            model_save_fn=model.save,
        )
        if models_dir is not None:
            save_run_kwargs["models_dir"] = models_dir
        artifact_paths = save_run(**save_run_kwargs)
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
            "task_type": task_type,
        }

    except Exception:
        # Heading 12: "Errors and exceptions" must be logged.
        logger.exception(f"Timeseries pipeline failed: run_id={resolved_run_id}")
        raise


def _default_stats_config() -> dict:
    """
    Loads stats/config.yaml from its own default location, same pattern
    as regression_pipeline.py / classification_pipeline.py's own
    _default_stats_config().
    """
    stats_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "stats", "config.yaml"
    )
    return _load_yaml(stats_config_path)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)