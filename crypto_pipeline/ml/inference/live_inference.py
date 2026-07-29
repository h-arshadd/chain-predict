# crypto_pipeline/ml/inference/live_inference.py

"""
live_inference.py
------------------
Load a trained run's model, run it on real data (the model's own
symbol/exchange/timeframe) and upsert the resulting signals into
ml.model_signals -- one row per (run_id, datetime), same table every
run_id writes to.

Meant to be run on a schedule (Task Scheduler / cron, see
run_live_inference.bat), often every few minutes. Each run resumes from
where this run_id last left off (see get_last_signal_timestamp), so a
scheduled run only fetches/predicts the small new gap since last time --
EXCEPT the very first run for a given run_id (or after retraining wipes
its signals), which backfills the model's *entire* trained history in
one go. The backfill matters because Strategy Builder lets the user
pick any historical date range to backtest against, and
resolve_component_signal() silently fills any bar missing a signal row
with 0/Hold -- so every bar back to the model's training start_date
needs a real signal on file, not just recent ones.

save_model_signals() upserts on (run_id, datetime), so this is safe to
run on a tight schedule with no risk of duplicating or corrupting rows
either way.

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
from crypto_pipeline.utils.db_utils import (
    get_db_connection, save_model_signals, list_ml_run_configs, get_last_signal_timestamp,
)

logger = logging.getLogger(__name__)

# Same regression/classification-only restriction ml.model_signals and
# api/repos/ml_repo.py already enforce -- timeseries is a different
# pipeline (forward-looking forecast, not a per-bar signal) and was
# never wired into this table.
_SUPPORTED_MODEL_KINDS = {"regressor", "classifier", "deep_learning_regressor", "deep_learning_classifier"}

# Every talib_indicators.py wrapper takes its window length under one of
# these parameter names (see crypto_pipeline/indicators/talib_indicators.py --
# "period" for most, "timeperiod" for a few, "fastperiod"/"slowperiod" for
# MACD/APO-style ones). Patterns (DOJI, ENGULFING, etc.) take no period at
# all and are simply absent from any config_item's "parameters".
_PERIOD_PARAM_NAMES = ("period", "timeperiod", "slowperiod", "fastperiod", "signalperiod")


def _max_indicator_period(data_prep_config: dict, default: int = 50) -> int:
    """
    Largest indicator lookback period configured for this run, so the
    live-inference lookback window (see run_live_inference) is always
    wide enough for that indicator's warm-up regardless of what this
    specific run_id was configured with. Falls back to `default` if no
    period-like parameter is found (e.g. features disabled, or only
    pattern indicators configured).
    """
    indicators_config = data_prep_config.get("features", {}).get("indicators", {})
    periods = [
        int(config_item["parameters"][name])
        for configs in indicators_config.values()
        for config_item in configs
        for name in _PERIOD_PARAM_NAMES
        if name in config_item.get("parameters", {})
    ]
    return max(periods) if periods else default


def run_live_inference(run_id: str, conn=None) -> dict:
    """
    Load run_id's model, run it on data from either just after its last
    saved signal (normal scheduled run) or its full training start_date
    (first run / after retraining), through now, and upsert the
    resulting signal series into ml.model_signals.

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
        # start_date resumes from where this run_id left off in
        # ml.model_signals, instead of blindly re-scoring the model's
        # entire history on every scheduled run:
        #   - If this run_id already has signal rows, resume from just
        #     after the last one, minus a warm-up buffer (indicators
        #     like EMA-50 need bars *before* their first signal date to
        #     compute correctly -- dropna() below trims the buffer rows
        #     back off before they're written).
        #   - If this run_id has NO signal rows yet (first run, or after
        #     a fresh training run), fall back to the model's own
        #     training start_date and backfill the *entire* range in one
        #     go. This matters because Strategy Builder lets the user
        #     pick any historical date range to backtest against
        #     (assemble.py's resolve_component_signal reindexes onto
        #     whatever range ohlcv covers and fills missing signal rows
        #     with 0/Hold) -- so every bar back to training start_date
        #     needs a real signal in the table, not just recent ones.
        # Either way, save_model_signals() upserts on (run_id, datetime),
        # so this is safe to re-run on a schedule with no risk of
        # duplicating or corrupting existing rows.
        lookback_bars = _max_indicator_period(data_prep_config)
        timeframe_str = data_prep_config.get("data", {}).get("timeframe", "1h")
        try:
            bar_duration = pd.Timedelta(timeframe_str)
            if pd.isna(bar_duration):
                raise ValueError(f"unparseable timeframe: {timeframe_str!r}")
        except (ValueError, TypeError):
            bar_duration = pd.Timedelta(hours=1)
        warmup_buffer = max(bar_duration * lookback_bars * 3, pd.Timedelta(days=1))

        last_signal_ts = get_last_signal_timestamp(conn, run_id)
        training_start = data_prep_config.get("data", {}).get("start_date")

        if last_signal_ts is not None:
            # Resume: rewind by the warm-up buffer so indicators have
            # enough prior bars, but never go earlier than training_start
            # (no point re-fetching data from before the model's own
            # training window even existed).
            #
            # Normalize both sides to naive UTC before comparing/
            # subtracting -- ml.model_signals.datetime is a plain
            # TIMESTAMP column (naive) so last_signal_ts is naive in
            # practice, but don't assume that blindly; this is the same
            # class of bug fixed in data_downloader.get_data().
            last_signal_ts = pd.Timestamp(last_signal_ts)
            if last_signal_ts.tzinfo is not None:
                last_signal_ts = last_signal_ts.tz_convert("UTC").tz_localize(None)

            resume_start = last_signal_ts - warmup_buffer
            if training_start is not None:
                training_start_ts = pd.Timestamp(training_start)
                if training_start_ts.tzinfo is not None:
                    training_start_ts = training_start_ts.tz_convert("UTC").tz_localize(None)
                resume_start = max(resume_start, training_start_ts)
            fetch_start = resume_start.to_pydatetime()
            logger.info(f"[{run_id}] Resuming from last signal at {last_signal_ts}, fetching from {fetch_start}.")
        else:
            # First run for this run_id -- backfill the full range so
            # Strategy Builder has signals for the model's entire
            # trained history, not just recent bars.
            fetch_start = training_start if training_start is not None else (
                datetime.now(timezone.utc) - warmup_buffer
            )
            logger.info(f"[{run_id}] No existing signals -- backfilling from training start_date {fetch_start}.")

        run_config = {
            **data_prep_config,
            "data": {
                **data_prep_config.get("data", {}),
                "enabled": True,
                "start_date": fetch_start,
                "end_date": datetime.now(timezone.utc),
            },
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