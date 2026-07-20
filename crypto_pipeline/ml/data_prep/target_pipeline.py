# crypto_pipeline/ml/data_prep/target_pipeline.py

"""
target_pipeline.py
------------------
Generates prediction targets: log return for regression,
-1/0/1 (threshold-based) for classification, raw close price for
timeseries REGRESSION models (Darts-backed models that forecast the
series directly, e.g. nbeats/tcn/statsforecast), or the same -1/0/1
triple-barrier label as classification for timeseries CLASSIFICATION
models (Darts-backed classifiers that forecast a discrete label
directly, e.g. sklearn_classifier).

Which of the two timeseries target shapes gets generated is picked by
config.yaml's target.timeseries_task ("regression" or
"classification") -- an explicit field, read ONCE per run, same level
as model_type itself deciding regression vs classification. This is
deliberately NOT inferred from model.algorithm: main.py loads the
dataset (including target generation) ONCE and then loops over every
algorithm in model.algorithms.timeseries, the same way it does for
model_type=regression/classification -- target shape can't depend on
which individual algorithm happens to run, or a mixed
algorithms.timeseries list (some regressors, some classifiers) would
have no single correct target to generate up front. Every algorithm
listed under model.algorithms.timeseries for a given run must belong
to the family named by target.timeseries_task (validated below) -- a
run trains either timeseries regressors or timeseries classifiers, not
both at once, same as model_type=regression and model_type=classification
already can't be mixed in one run today.
"""

import logging
import pandas as pd
import numpy as np

from crypto_pipeline.ml.timeseries.registry import TS_REGRESSORS, TS_CLASSIFIERS

logger = logging.getLogger(__name__)


def generate_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate prediction target based on model type and config.
    
    Args:
        df: DataFrame with price data (close column required)
        config: ML module config dict with target section
        
    Returns:
        pd.DataFrame: DataFrame with target column added
    """
    
    target_config = config.get("target", {})
    model_type = config.get("model_type", "regression")
    
    if not target_config:
        raise ValueError("No target configuration found")
    
    logger.info(f"Generating target for {model_type} model")
    
    if model_type == "regression":
        df = _generate_regression_target(df, target_config)
    
    elif model_type == "classification":
        df = _generate_classification_target(df, target_config)
    
    elif model_type == "timeseries":
        if _resolve_timeseries_task(config) == "classification":
            df = _generate_classification_target(df, target_config)
        else:
            df = _generate_timeseries_target(df, target_config)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    initial_count = len(df)
    df = df.dropna(subset=["target"])
    final_count = len(df)
    
    if initial_count > final_count:
        logger.info(f"Dropped {initial_count - final_count} rows with NaN targets")
    
    logger.info(f"Target generated: {final_count} valid rows")
    return df


def _resolve_timeseries_task(config: dict) -> str:
    """
    "regression" or "classification" -- which target shape to generate
    for model_type=timeseries, from ml/config.yaml's
    target.timeseries_task. Required, explicit, read once per run (not
    inferred from model.algorithm -- see module docstring for why).

    Also validates every algorithm listed in
    model.algorithms.timeseries actually belongs to the declared
    family (TS_REGRESSORS for "regression", TS_CLASSIFIERS for
    "classification"), so a config mistake (e.g. timeseries_task:
    regression with sklearn_classifier in the algorithms list) is
    caught here, at target-generation time, rather than surfacing
    later as a silently-wrong target shape fed into a classifier or a
    price target fed into a model expecting labels. Skipped gracefully
    if model.algorithms.timeseries isn't set yet at this point in the
    pipeline (main.py's _resolve_algorithms() does its own check
    later; this is a best-effort early catch, not the only guard).
    """
    task = config.get("target", {}).get("timeseries_task")
    if task not in ("regression", "classification"):
        raise ValueError(
            "ml/config.yaml's target.timeseries_task must be set to 'regression' "
            "(nbeats/tcn/statsforecast -- forecasts the raw close price) or "
            "'classification' (sklearn_classifier -- forecasts a -1/0/1 label) "
            f"for model_type=timeseries, got {task!r}."
        )

    algorithms = config.get("model", {}).get("algorithms", {}).get("timeseries")
    if algorithms:
        expected_registry = TS_CLASSIFIERS if task == "classification" else TS_REGRESSORS
        mismatched = [a for a in algorithms if a not in expected_registry]
        if mismatched:
            other_family = "TS_REGRESSORS" if task == "classification" else "TS_CLASSIFIERS"
            raise ValueError(
                f"ml/config.yaml's target.timeseries_task is '{task}', but "
                f"model.algorithms.timeseries includes {mismatched}, which "
                f"belong to {other_family}. A single timeseries run trains "
                f"either regressors or classifiers, not both -- either change "
                f"timeseries_task, or remove {mismatched} from "
                f"model.algorithms.timeseries."
            )

    return task


def _generate_regression_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate regression target: log return, current close vs close
    `horizon` candles in the future.
    """
    
    horizon = target_config.get("horizon", 1)
    
    if "close" not in df.columns:
        raise ValueError("close price column required for target generation")
    
    current_close = df["close"]
    future_close = df["close"].shift(-horizon)
    
    df["target"] = np.log(future_close / current_close)
    logger.info(f"Regression target: log return, future close (horizon={horizon})")
    
    if target_config.get("filter_noise", False):
        keep_mask = _noise_keep_mask(df["target"], target_config)
        df = df[keep_mask]
    
    return df


def _generate_classification_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate classification target using Triple Barrier Labeling:
    -1 (down) / 0 (flat) / 1 (up).

    Looking forward up to `horizon` candles from each row, checks whether
    price hits the upper barrier (take-profit) or lower barrier (stop-loss)
    first. If neither is hit within the horizon, label is 0 (flat/timeout).

    Shared by model_type=classification AND model_type=timeseries when
    target.timeseries_task=classification (see _resolve_timeseries_task()
    above) -- same label shape either way, since a timeseries classifier
    forecasts this exact label directly rather than a per-row prediction
    from features.
    """
    
    horizon = target_config.get("horizon", 1)
    upper_threshold = target_config.get("upper_threshold", 0.001)
    lower_threshold = target_config.get("lower_threshold", -0.001)
    
    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Triple barrier labeling requires columns: {required_cols}")
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    
    labels = np.zeros(n)
    labels[:] = np.nan  # rows too close to the end (no full horizon ahead) stay NaN
    
    for i in range(n - horizon):
        entry_price = close[i]
        upper_barrier = entry_price * (1 + upper_threshold)
        lower_barrier = entry_price * (1 + lower_threshold)
        
        label = 0  # default: neither barrier hit within horizon
        for step in range(1, horizon + 1):
            future_high = high[i + step]
            future_low = low[i + step]
            
            hit_upper = future_high >= upper_barrier
            hit_lower = future_low <= lower_barrier
            
            if hit_upper and hit_lower:
                # Both hit in the same candle -- can't tell which came first
                # from OHLC alone, so treat as flat/ambiguous rather than guess.
                label = 0
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break
        
        labels[i] = label
    
    df["target"] = labels
    logger.info(
        f"Classification target: Triple Barrier Labeling (-1/0/1, "
        f"upper={upper_threshold}, lower={lower_threshold}, horizon={horizon})"
    )
    
    return df


def _generate_timeseries_target(df: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    """
    Generate timeseries REGRESSION target: the raw close price itself,
    unshifted. Used when target.timeseries_task=regression (nbeats,
    tcn, statsforecast -- see registry.py's TS_REGRESSORS).

    Unlike regression (log return) and classification (triple-barrier
    label), these Darts models forecast the actual series values
    directly and handle non-stationarity/scaling internally -- there's
    no need to pre-compute a return here, and doing so would just mean
    converting predictions back to a price before signal generation
    anyway. The target column is simply a copy of "close"; horizon is
    not applied here since these models take how-many-steps-ahead as a
    predict()-time argument (output_chunk_length / n), not something
    baked into the target column itself.
    """

    if "close" not in df.columns:
        raise ValueError("close price column required for target generation")

    df["target"] = df["close"]
    logger.info("Timeseries target: raw close price (no return/label transform)")

    return df


def _noise_keep_mask(values: pd.Series, config: dict) -> pd.Series:
    """
    Build a boolean mask marking which rows to KEEP, based on the magnitude
    of `values` (returns or target). True = keep, False = noise, drop it.
    A row is kept if its absolute value is at least noise_threshold.
    """
    
    noise_threshold = config.get("noise_threshold", 0.001)
    mask = values.abs() >= noise_threshold
    logger.info(f"Noise filtering (threshold={noise_threshold}): keeping {mask.sum()}/{len(mask)} rows")
    
    return mask