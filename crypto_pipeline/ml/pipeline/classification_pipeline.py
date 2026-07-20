# crypto_pipeline/ml/pipeline/classification_pipeline.py

"""
classification_pipeline.py
----------------------------
Orchestrates PDF headings 1-4, 6-12 for a classification run: dataset
loading, feature selection, train/test split, preprocessing, model
training, prediction, signal generation, evaluation (both ML metrics
and trading strategy backtest), full model/experiment persistence, and
centralized logging, using whichever algorithm ml/config.yaml's
model.algorithm names (via ml/classifiers/registry.py for traditional
models, or ml/deep_learning/registry.py for mlp/lstm/gru).

Mirrors regression_pipeline.py exactly, including the routing rule --
see that file's module docstring for the full rationale. Short version:
ml/config.yaml's model_type decides regression vs classification once,
at the source (it controls which target target_pipeline.py generates),
so this pipeline reads that same field and refuses to run against a
non-classification dataset rather than taking a separate switch that
could disagree with it.

algorithm routing (traditional vs deep learning): model.algorithm is
looked up in ml/classifiers/registry.py's CLASSIFIERS first; if it's
not there, ml/deep_learning/registry.py's DL_CLASSIFIERS is checked
instead. Either way the rest of the pipeline is unchanged -- every
model exposes the same train()/predict()/predict_proba()/save()/load()
interface, so nothing past this point cares which registry it came
from. model_kind ("classifier" vs "deep_learning_classifier") is
threaded through to build_model_metadata() so persistence/model_loader.py can
reconstruct the right class later.

This module runs the full PDF pipeline end to end: every run writes its
config + fitted preprocessing objects + model weights to artifacts/ via
artifact_manager.save_run(), so any run can be reloaded later with
model_loader.load_run(run_id) for inference.

Evaluation (heading 10) needs 1-minute OHLCV for the test period plus
backtest/config.yaml + stats/config.yaml, since ml/evaluation/evaluator.py
invokes the real Backtesting and Statistics modules directly -- these
are function args here (ohlcv_1m/backtest_config_path/
stats_config_path), same pattern as ml_config_path, not hardcoded.

Logging (heading 12): setup_logging() is called once, at the very top,
before dataset loading starts -- every stage below already logs through
its own module-level logger, and those all propagate up to the root
logger this configures, so this one call centralizes logging for the
entire run. The whole body runs inside a try/except so any error at any
stage is logged (with traceback) before propagating, per heading 12's
"Errors and exceptions" requirement.
"""

import logging
import os

import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.pipeline.predictor import generate_predictions
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS, build_classifier
from crypto_pipeline.ml.deep_learning.registry import DL_CLASSIFIERS, build_dl_classifier
from crypto_pipeline.ml.signals.classification_signals import generate_classification_signals
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


def run_classification_pipeline(
    ml_config_path: str,
    ohlcv_1m: pd.DataFrame,
    backtest_config_path: str = None,
    stats_config_path: str = None,
    plot_dir: str = None,
    artifacts_dir: str = ARTIFACTS_DIR,
    run_id: str = None,
) -> dict:
    """
    Run the full classification pipeline through model training,
    prediction, signal generation, evaluation, and persistence.

    Args:
        ml_config_path: path to ml/config.yaml (single config file --
            controls data prep, features, split, preprocessing, model,
            signals, and evaluation)
        ohlcv_1m: 1-minute OHLCV DataFrame (datetime, open, high, low,
            close) covering the test period -- passed straight through
            to evaluator.evaluate_model() for backtest execution. Not
            loaded here: this pipeline has no data-fetching logic of its
            own for 1-minute execution data (see backtest/main.py's
            get_1m_data() for how a caller typically obtains this).
        backtest_config_path: path to backtest/config.yaml. Defaults to
            backtest.backtest.load_config()'s own default location if
            not given.
        stats_config_path: path to stats/config.yaml. Defaults to
            stats/config.yaml next to stats/calculator.py if not given.
        plot_dir: optional directory to save quantstats plots into --
            skipped if not given.
        artifacts_dir: root artifacts/ folder (PDF heading 11) -- passed
            straight to artifact_manager.save_run(). Defaults to
            "artifacts", created if it doesn't exist yet.
        run_id: identifier for this run's config/model/log files. Defaults
            to artifact_manager.make_run_id(algorithm) (a UTC timestamp
            + algorithm slug) if not given.

    Returns:
        dict with keys:
            model: trained model instance -- BaseClassifier for
                traditional algorithms, or BaseClassifierNetwork for
                mlp/lstm/gru (both expose the same train()/predict()/
                predict_proba()/save() interface)
            prediction_result: dict from predictor.generate_predictions()
                (PDF heading 8's standardized format -- predictions/
                probabilities/classes/n_predictions)
            signals: np.ndarray of str, Buy/Sell/Hold per test row, from
                signals.classification_signals.generate_classification_signals()
                (PDF heading 9), same row order as test_df
            evaluation: dict from evaluator.evaluate_model() (PDF heading
                10) -- ml_metrics (Accuracy/Precision/Recall),
                trading_metrics (every quantstats metric), trade_summary,
                backtest_result
            run_id: str, this run's identifier (as used for artifacts/ and logs/)
            artifact_paths: dict from artifact_manager.save_run() --
                run_dir/config_path/preprocessing_path/model_path (PDF
                heading 11: model weights + config + preprocessing
                objects, all persisted and reloadable via
                model_loader.load_run(run_id))
            y_test: pd.Series, true class labels for test_df (for scoring)
            feature_columns: list[str], order used for training/inference
            split_info: dict from train_test_split.split_dataset() (train/test
                date ranges etc, per PDF heading 3's record-keeping requirement)
            fit_objects: list from preprocessing_pipeline.run_preprocessing()
                (the fitted scalers/transforms, to persist alongside the model)
            algorithm: str, the model.algorithm name used
            model_kind: str, "classifier" or "deep_learning_classifier"
    """

    ml_config = _load_yaml(ml_config_path)

    model_type = ml_config.get("model_type")
    if model_type != "classification":
        raise ValueError(
            f"run_classification_pipeline() requires ml_config['model_type'] == "
            f"'classification', got '{model_type}'. Use regression_pipeline.py for "
            f"a regression dataset instead -- model_type is set once in "
            f"ml/config.yaml and drives which target was generated, so it "
            f"can't be overridden here."
        )

    # Headings 1-4: load, select features, split, preprocess. Standalone
    # callers of this function need this done once, here -- callers that
    # already have this data (e.g. main.py training several algorithms
    # against the SAME dataset) should call run_classification_algorithm()
    # directly instead of going through this loader.
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
    # preprocessed version so the val_df read further down (X_val/y_val)
    # sees val data transformed the same way as train_df/test_df.
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
        raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'random_forest')")
    params = model_config.get("params", {}) or {}

    return run_classification_algorithm(
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


def run_classification_algorithm(
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
    target_counts: dict = None,
    requested_hyperparams: dict = None,
    effective_hyperparams_fn=None,
) -> dict:
    """
    Train + predict + signal + evaluate + persist ONE classification
    algorithm, given data that's already been loaded/split/preprocessed
    (headings 1-4) by the caller.

    This is the part of run_classification_pipeline() (headings 5-11)
    that doesn't care where train_df/test_df came from -- pulled out
    into its own function so a caller training several algorithms
    against the SAME dataset (e.g. main.py's run_ml_pipeline(), one call
    per algorithm in ml_config["model"]["algorithms"]) doesn't have to
    reload/re-split/re-preprocess the dataset once per algorithm just to
    reach this logic. run_classification_pipeline() itself is a thin
    wrapper around this: it does headings 1-4 once, then calls this.

    Args mirror run_regression_algorithm() in regression_pipeline.py --
    see that docstring for the shared arguments. Only differences:
    algorithm/params are looked up against ml/classifiers/registry.py
    and ml/deep_learning/registry.py's classifier tables instead, and
    the returned/persisted model_kind is "classifier" or
    "deep_learning_classifier".

    Returns:
        Same dict shape as run_classification_pipeline() -- see its
        docstring for the key list.
    """
    resolved_run_id = run_id or make_run_id(algorithm)
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"Classification pipeline starting: run_id={resolved_run_id}, log file={log_path}")

    try:
        logger.info(f"Row counts: {row_counts}")

        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_test, y_test = test_df[feature_columns], test_df[target_column]

        # val_df (chronologically between train and test) is only
        # present when ml/config.yaml's split.val_size > 0 -- see
        # train_test_split.split_dataset(). None otherwise, same as it
        # always was, so this is fully backward compatible with a
        # val_size-less config.
        val_df = split_info.get("val_df")
        if val_df is not None:
            X_val, y_val = val_df[feature_columns], val_df[target_column]
        else:
            X_val, y_val = None, None

        # Heading 6/7: model training. Which algorithm + hyperparams is
        # entirely config-driven -- this function contains no
        # model-specific logic at all. algorithm is looked up in the
        # traditional registry first, then the deep learning registry;
        # whichever matches decides model_kind, but both expose the same
        # train()/predict()/predict_proba()/save() interface, so nothing
        # below this block branches on which kind it is.
        if algorithm in CLASSIFIERS:
            model_kind = "classifier"
            logger.info(f"Training classifier: algorithm={algorithm}, params={params}")
            model = build_classifier(algorithm, **params)
            # Traditional sklearn-style classifiers' train() only takes
            # (X_train, y_train) -- no validation-set concept (no
            # epochs/early stopping), so X_val/y_val are never passed here.
            model.train(X_train, y_train)
        elif algorithm in DL_CLASSIFIERS:
            model_kind = "deep_learning_classifier"
            logger.info(f"Training deep learning classifier: algorithm={algorithm}, params={params}")
            model = build_dl_classifier(algorithm, **params)
            # X_val/y_val (None if no split.val_size configured) drive
            # early stopping + ReduceLROnPlateau inside
            # deep_learning/trainer.py's train_network() -- see
            # deep_learning/base_network.py's train().
            model.train(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(
                f"Unknown classification algorithm '{algorithm}'. "
                f"Available traditional: {sorted(CLASSIFIERS.keys())}, "
                f"deep learning: {sorted(DL_CLASSIFIERS.keys())}"
            )

        # Heading 8: standardized prediction format (shared with regression,
        # traditional models, and deep learning models alike).
        prediction_result = generate_predictions(model, X_test, task_type="classification")

        # Heading 9: convert predictions into Buy/Sell/Hold signals. Thresholds
        # come entirely from ml/config.yaml -- this pipeline has no signal logic
        # of its own, only the model-agnostic prediction_result to hand off.
        signals = generate_classification_signals(prediction_result, ml_config)

        logger.info(
            f"Classification pipeline complete: {prediction_result['n_predictions']} test "
            f"predictions, signals generated"
        )

        # Heading 10: evaluate the trained model -- ML metrics (reported only,
        # never used for selection) plus a real backtest + stats run on the
        # generated signals.
        # Use a pre-loaded config dict if the caller has one (e.g.
        # main.py loads these ONCE outside its per-algorithm loop and
        # passes the same dict into every algorithm's call here, rather
        # than this function re-reading the same YAML file off disk once
        # per algorithm) -- otherwise load from the given path, same as
        # a standalone caller of run_regression_pipeline()/
        # run_classification_pipeline() would.
        resolved_backtest_config = (
            backtest_config if backtest_config is not None else load_backtest_config(backtest_config_path)
        )
        resolved_stats_config = (
            stats_config if stats_config is not None else
            (_load_yaml(stats_config_path) if stats_config_path else _default_stats_config())
        )

        evaluation = evaluate_model(
            task_type="classification",
            y_true=y_test.to_numpy(),
            y_pred=prediction_result["predictions"],
            signals=signals,
            signal_timestamps=test_df[timestamp_column],
            ohlcv_1m=ohlcv_1m,
            backtest_config=resolved_backtest_config,
            stats_config=resolved_stats_config,
            plot_dir=plot_dir,
            run_id=algorithm,
        )

        # Heading 11: full model/experiment persistence. Each pipeline
        # stage gets its own config dict, all sourced from the same
        # ml_config, so a run can still be inspected stage by stage.
        # test_metrics comes straight from this run's own evaluation,
        # not re-derived. classes comes from the trained model itself
        # (model.classes_, per heading 6/7's required classifier
        # interface -- traditional and deep learning classifiers both
        # expose it), not re-derived from y_test -- the label SET a
        # classifier learned during train() is what matters for
        # reproducibility, not just whatever happens to appear in this
        # particular test split. model_kind is whichever registry
        # matched above -- this is what lets model_loader.load_run()
        # reconstruct a deep learning model from the DL registry instead
        # of the traditional one later.
        metadata = {
            "data_prep": build_data_prep_metadata(
                ml_config=ml_config,
                row_counts=row_counts,
                target_counts=target_counts,
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
                fit_objects=fit_objects,
            ),
            "model": build_model_metadata(
                model_kind=model_kind,
                algorithm=algorithm,
                hyperparams=effective_hyperparams_fn(model, params) if effective_hyperparams_fn is not None else params,
                requested_hyperparams=requested_hyperparams if requested_hyperparams is not None else params,
                classes=model.classes_,
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
            "y_test": y_test,
            "feature_columns": feature_columns,
            "split_info": split_info,
            "fit_objects": fit_objects,
            "algorithm": algorithm,
            "model_kind": model_kind,
        }

    except Exception:
        # Heading 12: "Errors and exceptions" must be logged. logger.exception()
        # records the full traceback to both the console and this run's log
        # file, then the exception is re-raised unchanged -- this pipeline
        # never swallows an error, it only makes sure it's on record before
        # the caller sees it.
        logger.exception(f"Classification pipeline failed: run_id={resolved_run_id}")
        raise


def _default_stats_config() -> dict:
    """
    Loads stats/config.yaml from its own default location, same pattern
    as backtest.backtest.load_config()'s config_path=None default --
    used when the caller doesn't pass stats_config_path explicitly.
    """
    stats_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "stats", "config.yaml"
    )
    return _load_yaml(stats_config_path)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)