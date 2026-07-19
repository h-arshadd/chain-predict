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
from crypto_pipeline.ml.evaluation.evaluator import evaluate_model
from crypto_pipeline.ml.evaluation.classification_metrics import compute_classification_metrics
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
            val_metrics: dict from compute_classification_metrics() on the
                validation split (Accuracy/Precision/Recall), or None if
                split.val_size wasn't set in ml/config.yaml. For mlp/lstm/
                gru this is the same signal that already drove early
                stopping during training; for traditional models it's
                purely a reported number, since they don't take a
                validation set during training.
            run_id: str, this run's identifier (as used for artifacts/ and logs/)
            artifact_paths: dict from artifact_manager.save_run() --
                run_dir/config_path/preprocessing_path/model_path (PDF
                heading 11: model weights + config + preprocessing
                objects, all persisted and reloadable via
                model_loader.load_run(run_id))
            y_test: pd.Series, true class labels for test_df (for scoring)
            y_val: pd.Series or None, true class labels for val_df (None
                if split.val_size wasn't set in ml/config.yaml)
            feature_columns: list[str], order used for training/inference
            split_info: dict from train_test_split.split_dataset() (train/test
                date ranges etc, per PDF heading 3's record-keeping requirement)
            fit_objects: list from preprocessing_pipeline.run_preprocessing()
                (the fitted scalers/transforms, to persist alongside the model)
            algorithm: str, the model.algorithm name used
            model_kind: str, "classifier" or "deep_learning_classifier"
    """

    ml_config = _load_yaml(ml_config_path)

    # Heading 12: centralized logging, configured once per run before any
    # other stage does anything -- same pattern as regression_pipeline.py.
    # run_id is resolved here so the SAME id names this run's log file
    # (logs/{run_id}.log), its artifacts (artifacts/configs/{run_id}.yaml),
    # and everything else about the run.
    algorithm_for_run_id = ml_config.get("model", {}).get("algorithm", "unknown")
    resolved_run_id = run_id or make_run_id(algorithm_for_run_id)
    log_path = setup_logging(run_id=resolved_run_id)
    logger.info(f"Classification pipeline starting: run_id={resolved_run_id}, log file={log_path}")

    try:
        model_type = ml_config.get("model_type")
        if model_type != "classification":
            raise ValueError(
                f"run_classification_pipeline() requires ml_config['model_type'] == "
                f"'classification', got '{model_type}'. Use regression_pipeline.py for "
                f"a regression dataset instead -- model_type is set once in "
                f"ml/config.yaml and drives which target was generated, so it "
                f"can't be overridden here."
            )

        # Headings 1-4: load, select features, split, preprocess.
        df = load_dataset(ml_config)
        selected = select_features(df, ml_config)
        feature_columns = selected["feature_columns"]
        target_column = selected["target_column"]

        split_info = split_dataset(df, ml_config, timestamp_column=selected["timestamp_column"])

        preprocessed = run_preprocessing(
            split_info["train_df"], split_info["test_df"], feature_columns, ml_config,
            val_df=split_info["val_df"],
        )
        train_df = preprocessed["train_df"]
        val_df = preprocessed["val_df"]  # None if split.val_size wasn't set in config
        test_df = preprocessed["test_df"]
        has_val = val_df is not None

        # Row-count bookkeeping for metadata.build_data_prep_metadata()
        # and build_split_metadata() (heading 11 + your lead's "total
        # rows, training rows from where to where, everything possible"
        # requirement) -- captured here since this is the one place both
        # the pre-split total and the post-drop train/val/test counts are
        # all in scope together. val_rows/dropped_rows_val are only added
        # when a validation split actually exists, so build_split_metadata()
        # doesn't write an empty "validation" section for train/test-only runs.
        row_counts = {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "dropped_rows_train": preprocessed["dropped_rows"]["train"],
            "dropped_rows_test": preprocessed["dropped_rows"]["test"],
        }
        if has_val:
            row_counts["val_rows"] = len(val_df)
            row_counts["dropped_rows_val"] = preprocessed["dropped_rows"]["val"]
        logger.info(f"Row counts: {row_counts}")

        X_train, y_train = train_df[feature_columns], train_df[target_column]
        X_test, y_test = test_df[feature_columns], test_df[target_column]
        X_val, y_val = (val_df[feature_columns], val_df[target_column]) if has_val else (None, None)

        # Heading 6/7: model training. Which algorithm + hyperparams is
        # entirely config-driven -- this function contains no
        # model-specific logic at all. algorithm is looked up in the
        # traditional registry first, then the deep learning registry;
        # whichever matches decides model_kind, but both expose the same
        # train()/predict()/predict_proba()/save() interface, so nothing
        # below this block branches on which kind it is.
        model_config = ml_config.get("model", {})
        algorithm = model_config.get("algorithm")
        if not algorithm:
            raise ValueError("ml/config.yaml must set model.algorithm (e.g. 'random_forest')")
        params = model_config.get("params", {}) or {}

        if algorithm in CLASSIFIERS:
            model_kind = "classifier"
            logger.info(f"Training classifier: algorithm={algorithm}, params={params}")
            model = build_classifier(algorithm, **params)
            # Traditional classifiers (sklearn/xgboost/etc wrappers) only
            # expose train(X_train, y_train) today -- the validation split
            # still exists and is still reported below, it just isn't fed
            # into training for these algorithms.
            model.train(X_train, y_train)
        elif algorithm in DL_CLASSIFIERS:
            model_kind = "deep_learning_classifier"
            logger.info(f"Training deep learning classifier: algorithm={algorithm}, params={params}")
            model = build_dl_classifier(algorithm, **params)
            # X_val/y_val (None if split.val_size wasn't set) drive early
            # stopping and ReduceLROnPlateau inside base_network.py/
            # trainer.py -- without them the model just trains for the
            # full configured epoch count with no early stopping signal.
            model.train(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(
                f"Unknown classification algorithm '{algorithm}'. "
                f"Available traditional: {sorted(CLASSIFIERS.keys())}, "
                f"deep learning: {sorted(DL_CLASSIFIERS.keys())}"
            )

        # Validation metrics (reported only, same "never used to pick the
        # best model" rule the PDF applies to ml_metrics on test -- model
        # SELECTION there is evaluator.select_best_model() on trading
        # metrics; here val is what the DL models above already used
        # internally for early stopping, this is just surfacing the same
        # number for the run's records).
        val_metrics = None
        if has_val:
            val_predictions = model.predict(X_val)
            val_metrics = compute_classification_metrics(y_val.to_numpy(), val_predictions)
            logger.info(f"Validation metrics: {val_metrics}")

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
        backtest_config = load_backtest_config(backtest_config_path)
        stats_config = _load_yaml(stats_config_path) if stats_config_path else _default_stats_config()

        evaluation = evaluate_model(
            task_type="classification",
            y_true=y_test.to_numpy(),
            y_pred=prediction_result["predictions"],
            signals=signals,
            signal_timestamps=test_df[selected["timestamp_column"]],
            ohlcv_1m=ohlcv_1m,
            backtest_config=backtest_config,
            stats_config=stats_config,
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
            ),
            "split": build_split_metadata(
                split_info=split_info,
                row_counts=row_counts,
            ),
            "preprocessing": build_preprocessing_metadata(
                feature_columns=feature_columns,
                target_column=target_column,
                timestamp_column=selected["timestamp_column"],
                preprocessing_config=ml_config.get("preprocessing", {}),
                fit_objects=preprocessed["fit_objects"],
            ),
            "model": build_model_metadata(
                model_kind=model_kind,
                algorithm=algorithm,
                hyperparams=params,
                classes=model.classes_,
            ),
            # build_evaluation_metadata() takes ml_metrics/trading_metrics
            # separately (pre-existing bug fixed here: this used to call it
            # with a `test_metrics=` kwarg the function doesn't accept,
            # which would raise on every run). val_metrics has no
            # dedicated slot in build_evaluation_metadata() yet, so it's
            # folded into ml_metrics under a "val_" prefix -- reported
            # alongside test's ml_metrics without disturbing the existing
            # accuracy/precision/recall keys test already uses.
            "evaluation": build_evaluation_metadata(
                ml_metrics={
                    **evaluation["ml_metrics"],
                    **({f"val_{k}": v for k, v in val_metrics.items()} if has_val else {}),
                },
                trading_metrics=evaluation["trading_metrics"],
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
            "val_metrics": val_metrics,  # None if split.val_size wasn't set in config
            "run_id": resolved_run_id,
            "artifact_paths": artifact_paths,
            "y_test": y_test,
            "y_val": y_val,  # None if split.val_size wasn't set in config
            "feature_columns": feature_columns,
            "split_info": split_info,
            "fit_objects": preprocessed["fit_objects"],
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