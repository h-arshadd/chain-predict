# crypto_pipeline/ml/inference/inference_check.py

"""
inference_check.py
-------------------
Sanity check: does a saved model reproduce the predictions it made
during training, when reloaded from disk and run again on the same
test data?

This is NOT new inference on new/live data -- it's a pipeline
correctness check. It rebuilds the dataset/split/preprocessing exactly
the way ml/main.py did for a given run (same ml/config.yaml, same
deterministic chronological split, same preprocessing steps re-fit on
the same train rows), loads that run's saved model via
persistence.model_loader.load_run(), predicts on the rebuilt test set,
and compares the result row-by-row against the predictions already
saved in pipeline_out/{algorithm}/05_predictions.csv from the original
training run.

Why rebuilding is safe (not cheating): train_test_split.py's split is
a pure positional/chronological cutoff (no randomness), and
preprocessing_pipeline.run_preprocessing() re-fits deterministically on
train_df every call. So re-running load_dataset -> select_features ->
split_dataset -> run_preprocessing against the same config.yaml
reproduces bit-identical train/test feature matrices -- there is no
separate "apply an already-fitted transform to new data" entry point
in this codebase to call instead, so this is the faithful way to
reproduce inference without adding new preprocessing-application code.

If this check passes (predictions match, or match within a tiny
floating-point tolerance for regression), it confirms:
    - the saved model file actually is what trained
    - load_run() reconstructs it correctly
    - predict() is deterministic and reproducible
That's what "the pipeline works" means here.

Usage:
    # Check a single run:
    python -m crypto_pipeline.ml.inference.inference_check <run_id>

    e.g.
    python -m crypto_pipeline.ml.inference.inference_check binance_btc_regression_h4_random_forest

    # Check every algorithm configured for both regression and
    # classification (ml/config.yaml's model.algorithms), writing each
    # one's 08_inference_check.csv into its own pipeline_out/{algorithm}/:
    python -m crypto_pipeline.ml.inference.inference_check --all

Exits with a non-zero status code if any row fails to match (or any
run errors out in --all mode), so this can be used as a CI/smoke-test
gate as well as run by hand.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from crypto_pipeline.ml.pipeline.dataset_loader import load_dataset
from crypto_pipeline.ml.pipeline.train_test_split import split_dataset
from crypto_pipeline.ml.preprocessing.feature_selector import select_features
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import run_preprocessing
from crypto_pipeline.ml.persistence.model_loader import load_run
from crypto_pipeline.ml.persistence.artifact_manager import make_run_id
from crypto_pipeline.ml.regressors.registry import REGRESSORS
from crypto_pipeline.ml.classifiers.registry import CLASSIFIERS
from crypto_pipeline.ml.deep_learning.registry import DL_REGRESSORS, DL_CLASSIFIERS
from crypto_pipeline.ml.utils.logger import setup_logging

logger = logging.getLogger(__name__)

_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ML_CONFIG_PATH = os.path.join(_ML_DIR, "config.yaml")
PIPELINE_OUT_DIR = os.path.join(_ML_DIR, "pipeline_out")

# Regression predictions are floats reproduced through the same model
# object, not re-fit -- any difference should be pure floating-point
# noise, not a real mismatch. Classification predictions are labels
# and must match exactly.
REGRESSION_TOLERANCE = 1e-6


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_dataset_for_model_type(ml_config: dict, model_type: str) -> dict:
    """
    Load + split + preprocess the dataset for one model_type, the same
    way ml/main.py does once per run_ml_pipeline() call -- reused here
    so batch mode builds each model_type's dataset exactly once instead
    of once per algorithm (data_prep + preprocessing don't depend on
    which algorithm trains on the result).

    model_type is forced into a COPY of ml_config (never mutates the
    caller's dict) since data_prep/target_pipeline.py reads
    ml_config["model_type"] to decide which target column to generate
    (continuous log-return for regression, triple-barrier -1/0/1 labels
    for classification) -- regression and classification are genuinely
    different datasets, not just different models on the same data.
    """
    run_config = {**ml_config, "model_type": model_type}

    df = load_dataset(run_config)
    selected = select_features(df, run_config)
    timestamp_column = selected["timestamp_column"]
    target_column = selected["target_column"]
    feature_columns = selected["feature_columns"]

    split_info = split_dataset(df, run_config, timestamp_column=timestamp_column)
    train_df, test_df = split_info["train_df"], split_info["test_df"]

    return {
        "run_config": run_config,
        "train_df": train_df,
        "test_df": test_df,
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "feature_columns": feature_columns,
    }


def run_inference_check(
    run_id: str,
    ml_config_path: str = DEFAULT_ML_CONFIG_PATH,
    pipeline_out_dir: str = PIPELINE_OUT_DIR,
    _prebuilt_dataset: dict = None,
) -> dict:
    """
    Reload a trained run's model, re-predict on the same rebuilt test
    set, and compare against that run's originally-saved predictions.

    Args:
        run_id: the run identifier used for both models/{run_id}/ and
            artifacts/configs/{run_id}/ (see persistence.artifact_manager
            .make_run_id()), e.g. "binance_btc_regression_h4_random_forest".
        ml_config_path: path to ml/config.yaml -- must be the SAME config
            the original run used (data range, split, preprocessing,
            feature selection all have to match, or the rebuilt test set
            won't line up with the saved predictions). Ignored if
            _prebuilt_dataset is given.
        pipeline_out_dir: root pipeline_out/ folder (see ml/main.py) --
            original_predictions is read from
            {pipeline_out_dir}/{algorithm}/05_predictions.csv here.
        _prebuilt_dataset: internal -- pass the dict returned by
            _build_dataset_for_model_type() to skip rebuilding the
            dataset (used by run_all_inference_checks() so every
            algorithm of the same model_type shares one build instead
            of each re-running load_dataset/select_features/split/
            preprocessing from scratch).

    Returns:
        dict:
            run_id: str, echoed back
            algorithm: str, from the loaded run's metadata
            model_type: str, "regression" or "classification"
            n_rows: int, rows compared
            n_matched: int, rows within tolerance / exact match
            n_mismatched: int, rows that differ
            max_abs_diff: float or None, regression only
            comparison_df: pd.DataFrame, one row per test-set timestamp,
                columns: timestamp, actual, predicted_original,
                predicted_inference, match (bool), abs_diff (regression only)
            output_path: str, where comparison_df was written
    """
    run = load_run(run_id)
    metadata = run["metadata"]
    model = run["model"]
    feature_columns = run["feature_columns"]

    algorithm = metadata["model"]["algorithm"]
    model_kind = metadata["model"]["model_type"]
    task_type = "classification" if "classifier" in model_kind else "regression"

    logger.info(f"Loaded run '{run_id}': algorithm={algorithm}, model_kind={model_kind}")

    # ---- Rebuild (or reuse) the exact same test set the original run trained/predicted on ----
    if _prebuilt_dataset is not None:
        built = _prebuilt_dataset
    else:
        ml_config = _load_yaml(ml_config_path)
        built = _build_dataset_for_model_type(ml_config, task_type)

    train_df, test_df = built["train_df"], built["test_df"]
    timestamp_column = built["timestamp_column"]
    target_column = built["target_column"]

    preprocessed = run_preprocessing(train_df, test_df, feature_columns, built["run_config"])
    test_df = preprocessed["test_df"]

    missing = set(feature_columns) - set(test_df.columns)
    if missing:
        raise ValueError(
            f"Rebuilt test set is missing feature columns the model was trained "
            f"on: {sorted(missing)}. Is ml_config_path the SAME config.yaml the "
            f"original run used?"
        )

    X_test = test_df[feature_columns]

    # ---- Re-predict with the reloaded model ----
    inference_predictions = model.predict(X_test)

    # ---- Load the original run's saved predictions to compare against ----
    original_predictions_path = os.path.join(pipeline_out_dir, task_type, algorithm, "05_predictions.csv")
    if not os.path.exists(original_predictions_path):
        raise FileNotFoundError(
            f"No original predictions found at {original_predictions_path} -- "
            f"run ml/main.py for this config first (it writes 05_predictions.csv)."
        )
    original_predictions_df = pd.read_csv(original_predictions_path)
    if timestamp_column not in original_predictions_df.columns:
        raise ValueError(
            f"{original_predictions_path} has no '{timestamp_column}' column -- "
            f"unexpected file format, can't align rows."
        )

    comparison_df = pd.DataFrame({
        timestamp_column: test_df[timestamp_column].values,
        "actual": test_df[target_column].values,
        "predicted_inference": inference_predictions,
    })

    # Align on timestamp (not just position) -- catches any silent
    # row-count/order drift between the original run and this rebuild
    # instead of comparing misaligned rows.
    original_predictions_df[timestamp_column] = original_predictions_df[timestamp_column].astype(str)
    comparison_df[timestamp_column] = comparison_df[timestamp_column].astype(str)
    merged = comparison_df.merge(
        original_predictions_df[[timestamp_column, "predicted"]].rename(
            columns={"predicted": "predicted_original"}
        ),
        on=timestamp_column,
        how="outer",
        indicator=True,
    )

    unmatched_rows = merged[merged["_merge"] != "both"]
    if len(unmatched_rows) > 0:
        raise ValueError(
            f"{len(unmatched_rows)} rows didn't line up by timestamp between the "
            f"rebuilt test set and {original_predictions_path} -- the rebuilt "
            f"test set doesn't match the original run's. Is ml_config_path the "
            f"SAME config.yaml the original run used (same data range/split)?"
        )
    merged = merged.drop(columns=["_merge"])

    if task_type == "regression":
        merged["abs_diff"] = (merged["predicted_original"] - merged["predicted_inference"]).abs()
        merged["match"] = merged["abs_diff"] <= REGRESSION_TOLERANCE
        max_abs_diff = float(merged["abs_diff"].max())
    else:
        merged["match"] = merged["predicted_original"] == merged["predicted_inference"]
        max_abs_diff = None

    n_rows = len(merged)
    n_matched = int(merged["match"].sum())
    n_mismatched = n_rows - n_matched

    os.makedirs(os.path.join(pipeline_out_dir, task_type, algorithm), exist_ok=True)
    output_path = os.path.join(pipeline_out_dir, task_type, algorithm, "08_inference_check.csv")
    merged.to_csv(output_path, index=False)

    logger.info(
        f"Inference check for run_id='{run_id}': {n_matched}/{n_rows} rows matched"
        + (f", max_abs_diff={max_abs_diff:.10f}" if max_abs_diff is not None else "")
    )
    if n_mismatched:
        logger.warning(f"{n_mismatched}/{n_rows} rows did NOT match -- see {output_path}")
    else:
        logger.info(f"All rows matched -- pipeline reproduces its own predictions. See {output_path}")

    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "model_type": task_type,
        "n_rows": n_rows,
        "n_matched": n_matched,
        "n_mismatched": n_mismatched,
        "max_abs_diff": max_abs_diff,
        "comparison_df": merged,
        "output_path": output_path,
    }


def _resolve_algorithms_for_type(ml_config: dict, model_type: str) -> list:
    """
    Same source of truth ml/main.py's _resolve_algorithms() uses --
    ml_config["model"]["algorithms"][model_type], required and
    non-empty. No "run everything registered" fallback here either:
    this only checks runs that config.yaml actually asked to train.
    """
    algorithms_config = ml_config.get("model", {}).get("algorithms", {})
    explicit = algorithms_config.get(model_type) if isinstance(algorithms_config, dict) else None
    if not explicit:
        traditional = REGRESSORS if model_type == "regression" else CLASSIFIERS
        deep_learning = DL_REGRESSORS if model_type == "regression" else DL_CLASSIFIERS
        raise ValueError(
            f"ml/config.yaml's model.algorithms.{model_type} is not set (or is empty) -- "
            f"nothing to check. Available: {sorted(traditional.keys()) + sorted(deep_learning.keys())}"
        )
    return list(explicit)


def run_all_inference_checks(
    ml_config_path: str = DEFAULT_ML_CONFIG_PATH,
    pipeline_out_dir: str = PIPELINE_OUT_DIR,
) -> dict:
    """
    Run the inference check for every algorithm configured for BOTH
    regression and classification (ml/config.yaml's
    model.algorithms.regression and model.algorithms.classification),
    writing each one's 08_inference_check.csv into its own
    pipeline_out/{algorithm}/ folder -- same per-algorithm folder
    layout ml/main.py already uses for 01_dataset.csv..07_*.csv.

    Regression and classification each get their dataset rebuilt once
    (data_prep + preprocessing don't depend on which algorithm trains
    on the result) and every algorithm of that model_type reuses it,
    rather than rebuilding per algorithm.

    One algorithm's model missing/failing to load logs the error and
    continues to the rest, same "don't let one bad run stop the others"
    behavior as ml/main.py's run_ml_pipeline().

    Returns:
        dict keyed by run_id, each value either the same dict
        run_inference_check() returns, or {"error": str(exception)}.
    """
    ml_config = _load_yaml(ml_config_path)
    data_cfg = ml_config.get("data", {})
    horizon = ml_config.get("target", {}).get("horizon")

    results = {}
    for model_type in ("regression", "classification"):
        algorithms = _resolve_algorithms_for_type(ml_config, model_type)
        logger.info(f"Inference check starting: model_type={model_type}, algorithms={algorithms}")

        try:
            built = _build_dataset_for_model_type(ml_config, model_type)
        except Exception:
            logger.exception(f"Failed to build dataset for model_type={model_type}, skipping its algorithms")
            for algorithm in algorithms:
                run_id = make_run_id(
                    algorithm, symbol=data_cfg.get("symbol"), exchange=data_cfg.get("exchange"),
                    model_type=model_type, horizon=horizon,
                )
                results[run_id] = {"error": f"dataset build failed for model_type={model_type}"}
            continue

        for algorithm in algorithms:
            run_id = make_run_id(
                algorithm, symbol=data_cfg.get("symbol"), exchange=data_cfg.get("exchange"),
                model_type=model_type, horizon=horizon,
            )
            try:
                results[run_id] = run_inference_check(
                    run_id=run_id,
                    pipeline_out_dir=pipeline_out_dir,
                    _prebuilt_dataset=built,
                )
            except Exception as exc:
                logger.exception(f"Inference check failed for run_id='{run_id}', continuing with the rest")
                results[run_id] = {"error": str(exc)}

    succeeded = [r for r, v in results.items() if "error" not in v]
    failed = [r for r, v in results.items() if "error" in v]
    logger.info(
        f"Inference check batch finished: {len(succeeded)} succeeded {succeeded}, "
        f"{len(failed)} failed {failed}"
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reload trained model(s) and verify they reproduce their own saved predictions."
    )
    parser.add_argument(
        "run_id", nargs="?", default=None,
        help="Run identifier, e.g. binance_btc_regression_h4_random_forest. "
             "Omit and pass --all to check every configured regression + classification algorithm.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check every algorithm configured for both regression and classification "
             "(ml/config.yaml's model.algorithms), instead of a single run_id.",
    )
    parser.add_argument(
        "--ml-config-path", default=DEFAULT_ML_CONFIG_PATH,
        help="Path to ml/config.yaml (must match the config the run(s) were originally trained with)",
    )
    parser.add_argument(
        "--pipeline-out-dir", default=PIPELINE_OUT_DIR,
        help="Root pipeline_out/ folder (default: ml/pipeline_out)",
    )
    args = parser.parse_args()

    if not args.all and not args.run_id:
        parser.error("Provide a run_id, or pass --all to check every configured algorithm.")

    if args.all:
        setup_logging(run_id="inference_check_all")
        results = run_all_inference_checks(
            ml_config_path=args.ml_config_path,
            pipeline_out_dir=args.pipeline_out_dir,
        )
        any_failed = False
        for run_id, result in results.items():
            if "error" in result:
                print(f"\nrun_id={run_id}: FAILED -- {result['error']}")
                any_failed = True
                continue
            print(
                f"\nrun_id={run_id} algorithm={result['algorithm']} model_type={result['model_type']}: "
                f"{result['n_matched']}/{result['n_rows']} rows matched"
                + (f", max_abs_diff={result['max_abs_diff']:.10f}" if result['max_abs_diff'] is not None else "")
            )
            print(f"  -> {result['output_path']}")
            if result["n_mismatched"] > 0:
                any_failed = True
        if any_failed:
            sys.exit(1)
        return

    setup_logging(run_id=f"inference_check_{args.run_id}")

    result = run_inference_check(
        run_id=args.run_id,
        ml_config_path=args.ml_config_path,
        pipeline_out_dir=args.pipeline_out_dir,
    )

    print(
        f"\nrun_id={result['run_id']} algorithm={result['algorithm']}: "
        f"{result['n_matched']}/{result['n_rows']} rows matched"
        + (f", max_abs_diff={result['max_abs_diff']:.10f}" if result['max_abs_diff'] is not None else "")
    )
    print(f"Full comparison written to {result['output_path']}")

    if result["n_mismatched"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()