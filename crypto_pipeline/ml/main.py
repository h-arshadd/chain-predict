# crypto_pipeline/ml/main.py

"""
main.py
-------
Top-level entry point for the ML Module (PDF headings 1-12, end to end).

This didn't exist before: regression_pipeline.py / classification_pipeline.py
/ timeseries_pipeline.py each define a run_*_pipeline() function, but
nothing in the codebase ever called them -- this file is that caller.

Routing is automatic, driven by ml/config.yaml's model_type,
exactly the same field each pipeline file already gates on internally:

    model_type: regression     -> regressors/deep_learning regressors (mlp/lstm/gru)
    model_type: classification -> classifiers/deep_learning classifiers (mlp/lstm/gru)
    model_type: timeseries     -> darts (nbeats/tcn/statsforecast, or sklearn_classifier)

Unlike calling run_regression_pipeline() etc. directly, this file does
NOT treat the pipeline as one black box. It calls the exact same
underlying stage functions each pipeline file already calls internally
(load_dataset, select_features, split_dataset, run_preprocessing, then
delegates headings 5-13 to run_regression_algorithm() /
run_classification_algorithm() / run_timeseries_algorithm()) IN THE
SAME ORDER, but writes a CSV for each MAIN module's output into
pipeline_out/ -- dataset (data_prep), predictions (model), signals
(signals), metrics + trade ledger (evaluation) -- no logic is different
from the real pipeline, this is just the real pipeline with a to_csv()
call added after the modules whose output is actually worth inspecting.
Intermediate/debug steps (selected feature columns, raw train/test
split, preprocessed train/test) are NOT dumped here; that detail already
lives in run_config.json's split/preprocessing sections if you need it.

Runs every algorithm registered for the current model_type, not just
one -- see "Which algorithms run" below.

1-minute execution data (ohlcv_1m) is fetched here the same way
backtest/main.py fetches it: straight from Postgres via
crypto_pipeline.utils.db_utils.get_candles_from_db(), using the same
exchange/symbol/start_date/end_date already sitting in
ml/config.yaml's data: section. Nothing to pass in by hand --
if you want to run against different data, edit that config, not the
call site.

Which algorithms run
---------------------
ml/config.yaml's model.algorithms is REQUIRED and is the only thing
that decides what trains this run -- there is no "nothing set -> run
everything registered" fallback any more. Config is the single source
of truth: whatever is listed in model.algorithms is exactly what runs,
for regression, classification, and timeseries alike. Leaving it unset
raises a clear error naming what's available for the current
model_type (see ml/regressors/registry.py, ml/classifiers/registry.py,
ml/deep_learning/registry.py, ml/timeseries/registry.py) instead of
silently training things you didn't ask for.

Each algorithm gets its own run_id (artifacts/models/pipeline_out all
keyed by it) and its own pipeline_out/{algorithm}/ subfolder, so
per-stage CSVs from different algorithms never collide. One algorithm
failing (e.g. a missing optional dependency like xgboost) logs the
error and continues on to the next one rather than aborting the whole
run -- see run_ml_pipeline()'s return value for a per-algorithm summary.

Usage (as a script):

    python -m crypto_pipeline.ml.main

Or import run_ml_pipeline() directly from your own driver code:

    from crypto_pipeline.ml.main import run_ml_pipeline
    result = run_ml_pipeline()
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.pipeline.regression_pipeline import run_regression_algorithm
from crypto_pipeline.ml.pipeline.classification_pipeline import run_classification_algorithm
from crypto_pipeline.ml.pipeline.timeseries_pipeline import run_timeseries_algorithm
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing

from crypto_pipeline.ml.regressors.registry import REGRESSORS, build_regressor
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS, build_classifier
from crypto_pipeline.ml.deep_learning.registry import (
    DL_REGRESSORS, build_dl_regressor,
    DL_CLASSIFIERS, build_dl_classifier,
)
from crypto_pipeline.ml.timeseries.registry import TS_REGRESSORS, TS_CLASSIFIERS

from crypto_pipeline.ml.signals.signal_utils import signal_counts

from crypto_pipeline.backtest.backtest import load_config as load_backtest_config
from crypto_pipeline.utils.db_utils import get_db_connection, get_candles_from_db
from crypto_pipeline.ml.persistence.artifact_manager import make_run_id, ARTIFACTS_DIR, MODELS_DIR
from crypto_pipeline.ml.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# Where every stage's inspection CSV gets written. Doesn't affect
# artifacts/ or models/ (those are still owned by artifact_manager.py,
# see heading 11) -- this is purely a "let me see what happened" folder.
# Anchored to _ML_DIR (defined right below) rather than a bare relative
# string, for the same reason artifact_manager.py anchors ARTIFACTS_DIR/
# MODELS_DIR: a relative "pipeline_out" resolves against whatever
# directory the process is launched from, so running this as
# `python -m crypto_pipeline.ml.main` from the repo root instead of
# ml/ wrote pipeline_out/ outside the ml project entirely.

# model_type -> (traditional registry, traditional builder, deep-learning
# registry, deep-learning builder). Used to build the full "run every
# registered algorithm" list without hardcoding it in three places.
_TRADITIONAL_REGISTRIES = {
    "regression": (REGRESSORS, build_regressor, DL_REGRESSORS, build_dl_regressor),
    "classification": (CLASSIFIERS, build_classifier, DL_CLASSIFIERS, build_dl_classifier),
}


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Directory main.py itself lives in (crypto_pipeline/ml/) -- used so the
# default config path below resolves correctly regardless of which
# directory the command is launched from (python -m crypto_pipeline.ml.main
# from anywhere still finds crypto_pipeline/ml/config.yaml).
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ML_CONFIG_PATH = os.path.join(_ML_DIR, "config.yaml")
PIPELINE_OUT_DIR = os.path.join(_ML_DIR, "pipeline_out")


def _default_stats_config() -> dict:
    stats_config_path = os.path.join(_ML_DIR, "..", "stats", "config.yaml")
    return _load_yaml(stats_config_path)


def _resolve_algorithms(model_type: str, ml_config: dict) -> list:
    """
    Which algorithms to train this run.

    ml_config["model"]["algorithms"] is a dict keyed by model_type --
    {"regression": [...], "classification": [...], "timeseries": [...]}
    -- not a single flat list. This is deliberate: model_type can be
    changed independently of which algorithms are configured for it,
    and a flat list would silently be the wrong algorithms (or raise a
    confusing "unknown algorithm" error) the moment you switch model_type
    without also hand-editing the list. Keying by model_type means every
    mode always has its own explicit, ready-to-go list -- switching
    model_type just picks a different key, nothing to remember to edit.

    ml_config["model"]["algorithms"][model_type] is REQUIRED and must be
    non-empty for the model_type currently in use -- config is the
    single source of truth for what runs. There is no "if not set, run
    every registered algorithm" fallback: that silently trained things
    (e.g. mlp) that were never explicitly asked for just because they
    happened to be registered.
    """
    algorithms_config = ml_config.get("model", {}).get("algorithms", {})
    explicit = algorithms_config.get(model_type) if isinstance(algorithms_config, dict) else None

    if not explicit:
        if model_type == "timeseries":
            available = sorted(TS_REGRESSORS.keys()) + sorted(TS_CLASSIFIERS.keys())
        else:
            traditional, _, deep_learning, _ = _TRADITIONAL_REGISTRIES[model_type]
            available = sorted(traditional.keys()) + sorted(deep_learning.keys())
        raise ValueError(
            f"ml/config.yaml's model.algorithms.{model_type} is not set (or is "
            f"empty), but ml/config.yaml's model_type is "
            f"'{model_type}'. Nothing runs implicitly any more -- set "
            f"model.algorithms.{model_type} to an explicit list. Available for "
            f"'{model_type}': {available}"
        )

    return list(explicit)


def _params_for(algorithm: str, model_type: str, ml_config: dict) -> dict:
    """
    Hyperparameters for one algorithm: ml_config["model"]["param_overrides"]
    [model_type][algorithm] if given, otherwise {} (the model class's own/
    sklearn's defaults apply). Nested by model_type (not a flat
    {algorithm: {...}} dict) because several algorithm names are shared
    between regression and classification (random_forest, extra_trees,
    xgboost, lightgbm, catboost) and sklearn's own defaults for those
    differ by task (e.g. RandomForestRegressor's default max_features is
    1.0, RandomForestClassifier's is "sqrt") -- a flat dict couldn't hold
    both at once.
    """
    overrides = ml_config.get("model", {}).get("param_overrides", {})
    return overrides.get(model_type, {}).get(algorithm, {}) or {}


def _effective_hyperparams(model, requested_params: dict) -> dict:
    """
    What the model actually trained with, not just what was overridden
    in config. `requested_params` (from _params_for()) only holds the
    keys explicitly set in ml_config's param_overrides -- anything left
    out falls back to the underlying estimator's own defaults, and
    those defaults are otherwise invisible in the saved config.

    For sklearn-style estimators (regressors/classifiers -- anything
    with model.model.get_params()) this pulls the complete effective
    parameter set straight off the fitted estimator. For everything
    else (deep learning nets, Darts timeseries models -- no generic
    get_params()) requested_params IS the full picture already: any
    remaining defaults live inside that model's own _build_network()/
    constructor and aren't introspectable generically, so there's
    nothing more to add here.
    """
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "get_params"):
        try:
            return inner_model.get_params()
        except Exception:
            logger.warning(
                f"{type(model).__name__}.model.get_params() failed -- "
                f"falling back to the requested param_overrides only",
                exc_info=True,
            )
    return dict(requested_params)


def _fetch_ohlcv_1m(ml_config: dict) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV straight from Postgres, same call pattern
    backtest/main.py's get_1m_data() uses -- exchange/symbol/start_date/
    end_date come from ml/config.yaml's data: section, so there's
    nothing to pass in separately; edit that config to point at
    different data.
    """
    data_config = ml_config["data"]
    conn = get_db_connection()
    try:
        return get_candles_from_db(
            conn,
            data_config["exchange"],
            data_config["symbol"],
            data_config["start_date"],
            data_config["end_date"],
        )
    finally:
        conn.close()


def run_ml_pipeline(
    ml_config_path: str = DEFAULT_ML_CONFIG_PATH,
    ohlcv_1m: pd.DataFrame = None,
    backtest_config_path: str = None,
    stats_config_path: str = None,
    plot_dir: str = None,
    artifacts_dir: str = ARTIFACTS_DIR,
    models_dir: str = MODELS_DIR,
    pipeline_out_dir: str = PIPELINE_OUT_DIR,
) -> dict:
    """
    Run the full ML pipeline end to end for every algorithm registered
    for the current model_type (or ml_config["model"]["algorithms"] if
    set), routed by model_type, writing a CSV after every stage into
    pipeline_out_dir/{algorithm}/ along the way.

    Args:
        ml_config_path: path to ml/config.yaml -- the single config
            file. Its model_type field decides regression vs
            classification vs timeseries at every branching point
            below, and its data: section (exchange/symbol/start_date/
            end_date) is what ohlcv_1m is fetched with if not supplied
            directly.
        ohlcv_1m: 1-minute OHLCV DataFrame (datetime, open, high, low,
            close) covering the test period, needed for backtest
            execution (PDF heading 10). Optional -- if not given, it's
            fetched from Postgres the same way backtest/main.py does,
            using ml_config's data: section.
        backtest_config_path: path to backtest/config.yaml. Defaults to
            backtest.backtest.load_config()'s own default location.
        stats_config_path: path to stats/config.yaml. Defaults to
            stats/config.yaml next to stats/calculator.py.
        plot_dir: optional directory to save quantstats plot data into.
        artifacts_dir: root artifacts/ folder (PDF heading 11).
        models_dir: root models/ folder (trained model + preprocessing
            files -- kept separate from artifacts_dir's configs).
        pipeline_out_dir: root folder every stage's inspection CSV gets
            written to, one subfolder per algorithm (default
            "pipeline_out"). Separate from artifacts_dir/models_dir --
            this is just for you to look at, nothing downstream reads
            it back.

    Returns:
        dict keyed by algorithm name, each value either the same shape
        run_regression_pipeline() etc. return (model, prediction_result,
        signals, evaluation, run_id, artifact_paths, feature_columns,
        algorithm, model_kind, plus model-type-specific keys), or
        {"error": str(exception)} if that algorithm failed -- one bad
        algorithm (e.g. a missing optional dependency) doesn't stop the
        others from running.
    """
    ml_config = _load_yaml(ml_config_path)

    model_type = ml_config.get("model_type")
    if model_type not in ("regression", "classification", "timeseries"):
        raise ValueError(
            f"Unknown model_type '{model_type}' in {ml_config_path}. "
            f"Expected one of: regression, classification, timeseries"
        )

    if ohlcv_1m is None:
        ohlcv_1m = _fetch_ohlcv_1m(ml_config)
    if ohlcv_1m.empty:
        raise ValueError(
            "No 1-minute OHLCV data returned for the exchange/symbol/date range in "
            f"{ml_config_path}'s data: section -- nothing to backtest against."
        )

    algorithms = _resolve_algorithms(model_type, ml_config)
    logger.info(f"ML pipeline starting: model_type={model_type}, algorithms={algorithms}")

    # ---- Heading 1: dataset loading (shared across every algorithm) --
    # load_dataset/select_features/split_dataset/run_preprocessing don't
    # depend on which algorithm trains on the result, so they run once,
    # not once per algorithm.
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
    train_df = preprocessed["train_df"]
    test_df = preprocessed["test_df"]
    # split_info["val_df"] was raw/untransformed -- replace it with the
    # preprocessed version so every downstream split_info.get("val_df")
    # read (regression/classification/timeseries pipelines) sees val
    # data transformed the same way as train_df/test_df, not raw
    # features the model was never trained on the scale of.
    split_info["val_df"] = preprocessed["val_df"]

    row_counts = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "dropped_rows_train": preprocessed["dropped_rows"]["train"],
        "dropped_rows_test": preprocessed["dropped_rows"]["test"],
    }
    logger.info(f"Row counts: {row_counts}")

    # How many rows fall into each target class (e.g. -1/0/1 for the
    # triple-barrier label) over the full dataset, before the
    # train/test split -- written into data_prep metadata below so a
    # later inference run can see the class balance the model trained on.
    # Classification-only: regression's target is a continuous float
    # (log return), so value_counts() on it would just be ~1 per unique
    # value -- not a meaningful distribution, so it's skipped there.
    if model_type == "classification":
        target_counts = {str(k): int(v) for k, v in df[target_column].value_counts().items()}
    else:
        target_counts = {}
    logger.info(f"Target counts: {target_counts}")

    backtest_config = load_backtest_config(backtest_config_path)
    stats_config = _load_yaml(stats_config_path) if stats_config_path else _default_stats_config()

    results = {}
    for algorithm in algorithms:
        try:
            results[algorithm] = _run_one_algorithm(
                algorithm=algorithm,
                model_type=model_type,
                ml_config=ml_config,
                df=df,
                train_df=train_df,
                test_df=test_df,
                preprocessed=preprocessed,
                split_info=split_info,
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=timestamp_column,
                row_counts=row_counts,
                target_counts=target_counts,
                ohlcv_1m=ohlcv_1m,
                backtest_config=backtest_config,
                stats_config=stats_config,
                plot_dir=plot_dir,
                artifacts_dir=artifacts_dir,
                models_dir=models_dir,
                pipeline_out_dir=pipeline_out_dir,
            )
        except Exception as exc:
            logger.exception(f"Algorithm '{algorithm}' failed, continuing with the rest")
            results[algorithm] = {"error": str(exc)}

    succeeded = [a for a, r in results.items() if "error" not in r]
    failed = [a for a, r in results.items() if "error" in r]
    logger.info(f"ML pipeline finished: {len(succeeded)} succeeded {succeeded}, {len(failed)} failed {failed}")

    return results


def _run_one_algorithm(
    algorithm: str,
    model_type: str,
    ml_config: dict,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessed: dict,
    split_info: dict,
    feature_columns: list,
    target_column: str,
    timestamp_column: str,
    row_counts: dict,
    target_counts: dict,
    ohlcv_1m: pd.DataFrame,
    backtest_config: dict,
    stats_config: dict,
    plot_dir: str,
    artifacts_dir: str,
    models_dir: str,
    pipeline_out_dir: str,
) -> dict:
    """Train/predict/signal/evaluate/persist one algorithm end to end."""

    data_cfg = ml_config.get("data", {})
    resolved_run_id = make_run_id(
        algorithm,
        symbol=data_cfg.get("symbol"),
        exchange=data_cfg.get("exchange"),
        model_type=model_type,
        horizon=ml_config.get("target", {}).get("horizon"),
    )
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"Training '{algorithm}': run_id={resolved_run_id}, model_type={model_type}, log file={log_path}")

    algo_out_dir = os.path.join(pipeline_out_dir, model_type, algorithm)
    os.makedirs(algo_out_dir, exist_ok=True)

    def _dump(name: str, dump_df: pd.DataFrame):
        path = os.path.join(algo_out_dir, name)
        dump_df.to_csv(path, index=False)
        logger.info(f"[pipeline_out] wrote {path} ({dump_df.shape[0]} rows, {dump_df.shape[1]} cols)")
        return path

    # Only the main-module outputs get dumped here (data_prep's dataset,
    # the model's predictions, signals, and evaluation) -- the
    # intermediate per-stage debug CSVs (selected columns, raw train/test
    # split, preprocessed train/test) were dropped since they just spam
    # pipeline_out/ without adding anything you can't already see in
    # 01_dataset.csv + run_config.json's split/preprocessing sections.
    _dump("01_dataset.csv", df)

    params = _params_for(algorithm, model_type, ml_config)
    y_test = None
    classes = None

    # ---- Headings 5-11: model training, prediction, signals,
    # evaluation, and persistence (branches by model_type). For
    # regression/classification this delegates to
    # regression_pipeline.run_regression_algorithm() /
    # classification_pipeline.run_classification_algorithm() -- the
    # exact same per-algorithm logic run_regression_pipeline() /
    # run_classification_pipeline() use standalone, just given data
    # that's already been loaded/split/preprocessed once (above) rather
    # than reloading it per algorithm. main.py itself only adds the
    # pipeline_out/ CSV dumps around that shared logic; it doesn't
    # reimplement training/evaluation/persistence.
    if model_type == "regression":
        algo_result = run_regression_algorithm(
            algorithm=algorithm,
            params=params,
            ml_config=ml_config,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            split_info=split_info,
            fit_objects=preprocessed["fit_objects"],
            row_counts=row_counts,
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
            plot_dir=plot_dir,
            artifacts_dir=artifacts_dir,
            models_dir=models_dir,
            run_id=resolved_run_id,
            target_counts=target_counts,
            requested_hyperparams=params,
            effective_hyperparams_fn=_effective_hyperparams,
        )
        model = algo_result["model"]
        model_kind = algo_result["model_kind"]
        prediction_result = algo_result["prediction_result"]
        signals = algo_result["signals"]
        evaluation = algo_result["evaluation"]
        artifact_paths = algo_result["artifact_paths"]
        y_test = algo_result["y_test"]

        predictions_df = pd.DataFrame({
            timestamp_column: test_df[timestamp_column],
            "actual": y_test,
            "predicted": prediction_result["predictions"],
        })
        signal_timestamps = test_df[timestamp_column]

    elif model_type == "classification":
        algo_result = run_classification_algorithm(
            algorithm=algorithm,
            params=params,
            ml_config=ml_config,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            split_info=split_info,
            fit_objects=preprocessed["fit_objects"],
            row_counts=row_counts,
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
            plot_dir=plot_dir,
            artifacts_dir=artifacts_dir,
            models_dir=models_dir,
            run_id=resolved_run_id,
            target_counts=target_counts,
            requested_hyperparams=params,
            effective_hyperparams_fn=_effective_hyperparams,
        )
        model = algo_result["model"]
        model_kind = algo_result["model_kind"]
        prediction_result = algo_result["prediction_result"]
        signals = algo_result["signals"]
        evaluation = algo_result["evaluation"]
        artifact_paths = algo_result["artifact_paths"]
        y_test = algo_result["y_test"]
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
        algo_result = run_timeseries_algorithm(
            algorithm=algorithm,
            params=params,
            ml_config=ml_config,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            target_column=target_column,
            timestamp_column=timestamp_column,
            split_info=split_info,
            fit_objects=preprocessed["fit_objects"],
            row_counts=row_counts,
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
            plot_dir=plot_dir,
            artifacts_dir=artifacts_dir,
            models_dir=models_dir,
            run_id=resolved_run_id,
            requested_hyperparams=params,
            effective_hyperparams_fn=_effective_hyperparams,
        )
        model = algo_result["model"]
        model_kind = algo_result["model_kind"]
        prediction_result = algo_result["prediction_result"]
        signals = algo_result["signals"]
        evaluation = algo_result["evaluation"]
        artifact_paths = algo_result["artifact_paths"]
        classes = getattr(model, "classes_", None) if model_kind == "timeseries_classifier" else None

        n_pred = prediction_result["n_predictions"]
        # "anchored" forecast_mode (the default) is anchored at the start
        # of test_df; "historical" mode's forecasts instead line up with
        # the LAST n_pred rows of train+test combined (see
        # timeseries_pipeline.py's run_timeseries_algorithm() for why).
        forecast_mode = (params or {}).get("forecast_mode", "anchored")
        if forecast_mode == "anchored":
            actual_slice = test_df.iloc[:n_pred]
            signal_timestamps = test_df[timestamp_column].iloc[:1]
        else:
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            actual_slice = combined_df.iloc[-n_pred:]
            signal_timestamps = actual_slice[timestamp_column]

        predictions_df = pd.DataFrame({
            timestamp_column: actual_slice[timestamp_column].reset_index(drop=True),
            "actual": actual_slice[target_column].reset_index(drop=True),
            "predicted": prediction_result["forecast"],
        })

    _dump("05_predictions.csv", predictions_df)

    # ---- Heading 9: signal generation ---------------------------------
    if model_type == "timeseries":
        signals_df = pd.DataFrame({
            timestamp_column: [signal_timestamps.iloc[0]] if len(signals) == 1 else signal_timestamps.reset_index(drop=True),
            "signal": signals,
        })
    else:
        signals_df = predictions_df.copy()
        signals_df["signal"] = signals
    _dump("06_signals.csv", signals_df)
    algo_signal_counts = signal_counts(signals)
    logger.info(f"[{algorithm}] Signal counts: {algo_signal_counts}")

    # regression/classification/timeseries: evaluation + persistence
    # (heading 10/11) already happened inside run_regression_algorithm() /
    # run_classification_algorithm() / run_timeseries_algorithm() above --
    # `evaluation` and `artifact_paths` are already sitting in algo_result,
    # reused as-is rather than redone here.

    # 07_metrics.csv keeps the FULL computation (ml metrics + every
    # quantstats key) for anyone who wants to dig into one run -- the
    # run_config.json written below only gets the short summary.
    metrics_row = {**evaluation["ml_metrics"], **evaluation["trading_metrics"]}
    _dump("07_metrics.csv", pd.DataFrame([metrics_row]))
    _dump("07_trade_ledger.csv", evaluation["backtest_result"]["trade_ledger"])

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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full ML pipeline end to end for every algorithm registered "
                     "for the current model_type, stage CSVs included."
    )
    parser.add_argument("--ml-config", default=DEFAULT_ML_CONFIG_PATH, help="Path to ml/config.yaml")
    parser.add_argument(
        "--ohlcv-1m", default=None,
        help="Optional path to a CSV of 1-minute OHLCV (columns: datetime, open, high, low, "
             "close). If omitted (the default), it's fetched from Postgres using "
             "ml_config's data: section, same as backtest/main.py.",
    )
    parser.add_argument("--backtest-config", default=None, help="Path to backtest/config.yaml")
    parser.add_argument("--stats-config", default=None, help="Path to stats/config.yaml")
    parser.add_argument("--plot-dir", default=None, help="Optional directory for quantstats plot data")
    parser.add_argument("--artifacts-dir", default=ARTIFACTS_DIR, help="Root artifacts/ folder")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Root models/ folder")
    parser.add_argument(
        "--pipeline-out-dir", default=PIPELINE_OUT_DIR,
        help="Root folder every stage's inspection CSV gets written to, one subfolder per "
             "algorithm (default: pipeline_out)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ohlcv_1m = None
    if args.ohlcv_1m:
        ohlcv_1m = pd.read_csv(args.ohlcv_1m)
        ohlcv_1m["datetime"] = pd.to_datetime(ohlcv_1m["datetime"])

    results = run_ml_pipeline(
        ml_config_path=args.ml_config,
        ohlcv_1m=ohlcv_1m,
        backtest_config_path=args.backtest_config,
        stats_config_path=args.stats_config,
        plot_dir=args.plot_dir,
        artifacts_dir=args.artifacts_dir,
        models_dir=args.models_dir,
        pipeline_out_dir=args.pipeline_out_dir,
    )

    for algorithm, result in results.items():
        if "error" in result:
            print(f"{algorithm}: FAILED - {result['error']}")
        else:
            print(f"{algorithm}: run_id={result['run_id']}")