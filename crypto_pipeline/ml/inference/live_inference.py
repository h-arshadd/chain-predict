# crypto_pipeline/ml/inference/live_inference.py

"""
live_inference.py
------------------
Load a trained run's model, run it on real data (the model's own
symbol/exchange/timeframe, from its training start_date through now --
NOT the held-out test split inference_check.py replays), convert
predictions to signals, and upsert into ml.model_signals -- one row per
(run_id, datetime), same table every run_id writes to.

Meant to be run on a schedule (Task Scheduler / cron, see
run_live_inference.bat). Each run just re-predicts the whole range and
upserts -- save_model_signals() already upserts on (run_id, datetime),
so re-writing bars that haven't changed costs nothing and needs no
separate "what's already saved" bookkeeping.

Only real difference from inference_check.py's dataset build: this does
NOT call load_dataset()/generate_target(), because target generation
shifts `horizon` bars into the future and drops every row with no
future bar to look at yet -- which would always throw away the most
recent bars, exactly the ones we want a signal for. Everything else
(load model, load fitted preprocessing, predict, convert to signal) is
the same as training/inference_check.

Usage:
    python -m crypto_pipeline.ml.inference.live_inference <run_id>
    python -m crypto_pipeline.ml.inference.live_inference --all
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from crypto_pipeline.ml.data_prep.data_pipeline import collect_market_data
from crypto_pipeline.ml.data_prep.feature_pipeline import engineer_features
from crypto_pipeline.ml.data_prep.sentiment_pipeline import collect_sentiment_data
from crypto_pipeline.ml.persistence.model_loader import load_run
from crypto_pipeline.ml.preprocessing.preprocessing_pipeline import apply_saved_preprocessing
from crypto_pipeline.ml.pipeline.predictor import generate_predictions
from crypto_pipeline.ml.signals.classification_signals import generate_classification_signals
from crypto_pipeline.ml.signals.regression_signals import generate_regression_signals
from crypto_pipeline.ml.signals.signal_utils import signals_to_int
from crypto_pipeline.ml.utils.logger import setup_logging
from crypto_pipeline.utils.db_utils import get_db_connection, save_model_signals, list_ml_run_configs

logger = logging.getLogger(__name__)

# Same regression/classification-only restriction ml.model_signals and
# api/repos/ml_repo.py already enforce -- timeseries is a different
# pipeline (forward-looking forecast, not a per-bar signal) and was
# never wired into this table.
_SUPPORTED_MODEL_KINDS = {"regressor", "classifier", "deep_learning_regressor", "deep_learning_classifier"}


def run_live_inference(run_id: str, conn=None) -> dict:
    """
    Load run_id's model, run it on data from its own training
    start_date through now, and upsert the resulting signal series into
    ml.model_signals.

    Returns:
        dict: run_id, n_rows written, skipped (True if this run_id's
        model_kind is timeseries -- not supported here).
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_db_connection()
    try:
        run = load_run(run_id, conn=conn)
        metadata = run["metadata"]
        model = run["model"]
        fit_objects = run["fit_objects"]
        feature_columns = run["feature_columns"]

        model_kind = metadata["model"].get("model_type")
        if model_kind not in _SUPPORTED_MODEL_KINDS:
            logger.info(f"[{run_id}] model_kind={model_kind!r} is timeseries -- not supported here, skipping.")
            return {"run_id": run_id, "n_rows": 0, "skipped": True}

        task_type = "classification" if "classifier" in model_kind else "regression"
        data_prep_config = metadata["data_prep"]

        # Data + features (+ sentiment) only -- same config the run was
        # trained with, end_date pushed to now instead of the original
        # training end_date. No target generation (see module docstring).
        #
        # data.enabled forced True (not read from data_prep_config):
        # metadata.build_data_prep_metadata() rebuilds the "data" block
        # field-by-field (symbol/exchange/timeframe/start_date/end_date/
        # calculate_ohlcv) and never included "enabled" in that list, so
        # it's simply never in ml.run_configs to read back -- collect_
        # market_data() would see it missing and refuse to run. A run_id
        # existing in ml.run_configs at all already proves data
        # collection was enabled when it trained, so this sets it rather
        # than trying to read a value that was never persisted.
        run_config = {
            **data_prep_config,
            "data": {**data_prep_config.get("data", {}), "enabled": True, "end_date": datetime.now(timezone.utc)},
        }
        df = collect_market_data(run_config)
        if run_config.get("features", {}).get("enabled", False):
            df = engineer_features(df, run_config)
        if run_config.get("sentiment", {}).get("enabled", False):
            df = collect_sentiment_data(df, run_config)

        # Same NaN-drop rule the real training dataset uses: sen_*
        # columns may legitimately be NaN, everything else must not be
        # (clears indicator warm-up NaNs at the start of the range).
        sentiment_cols = [c for c in df.columns if c.startswith("sen_")]
        required_cols = [c for c in df.columns if c not in sentiment_cols]
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"[{run_id}] Data is missing feature columns this run was trained on: {missing}")

        # Apply the run's already-fitted preprocessing (never re-fit --
        # see apply_saved_preprocessing()'s own docstring).
        preprocessed = apply_saved_preprocessing(df, feature_columns, fit_objects)
        preprocessed = preprocessed.dropna(subset=feature_columns).reset_index(drop=True)

        prediction_result = generate_predictions(model, preprocessed[feature_columns], task_type=task_type)

        signals_config = {"signals": _load_signals_config()}
        if task_type == "classification":
            signals = generate_classification_signals(prediction_result, signals_config)
        else:
            signals = generate_regression_signals(prediction_result, signals_config)

        signals_df = pd.DataFrame({
            "datetime": preprocessed["datetime"].reset_index(drop=True),
            "signal": signals_to_int(signals),
        })

        save_model_signals(conn, run_id, signals_df)
        logger.info(f"[{run_id}] Wrote {len(signals_df)} signal row(s) to ml.model_signals.")
        return {"run_id": run_id, "n_rows": len(signals_df), "skipped": False}
    finally:
        if owns_conn:
            conn.close()


def _load_signals_config() -> dict:
    """ml/config.yaml's signals: section (Buy/Sell/Hold thresholds) -- not
    persisted per-run, so read live off disk, same as ml/main.py does."""
    import os
    import yaml
    ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(ml_dir, "config.yaml")) as f:
        return yaml.safe_load(f).get("signals", {})


def run_all_live_inference(conn=None) -> dict:
    """Run live inference for every run_id currently in ml.run_configs.
    One run_id failing logs the error and continues with the rest."""
    owns_conn = conn is None
    if owns_conn:
        conn = get_db_connection()
    try:
        run_ids = [
            (config or {}).get("run_summary", {}).get("run_id")
            for config in list_ml_run_configs(conn)
        ]
        run_ids = [r for r in run_ids if r]

        results = {}
        for run_id in run_ids:
            try:
                results[run_id] = run_live_inference(run_id, conn=conn)
            except Exception as exc:
                logger.exception(f"Live inference failed for run_id='{run_id}', continuing with the rest")
                results[run_id] = {"error": str(exc)}
        return results
    finally:
        if owns_conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run inference and save signals to ml.model_signals.")
    parser.add_argument("run_id", nargs="?", default=None, help="A single run_id to run.")
    parser.add_argument("--all", action="store_true", help="Run every run_id in ml.run_configs.")
    args = parser.parse_args()

    if not args.all and not args.run_id:
        parser.error("Provide a run_id, or pass --all.")

    if args.all:
        setup_logging(run_id="live_inference_all")
        results = run_all_live_inference()
        any_failed = False
        for run_id, result in results.items():
            if "error" in result:
                print(f"run_id={run_id}: FAILED -- {result['error']}")
                any_failed = True
            elif result.get("skipped"):
                print(f"run_id={run_id}: skipped (timeseries)")
            else:
                print(f"run_id={run_id}: {result['n_rows']} row(s) written")
        sys.exit(1 if any_failed else 0)

    setup_logging(run_id=f"live_inference_{args.run_id}")
    result = run_live_inference(args.run_id)
    if result.get("skipped"):
        print(f"run_id={args.run_id}: skipped (timeseries)")
    else:
        print(f"run_id={args.run_id}: {result['n_rows']} row(s) written")


if __name__ == "__main__":
    main()