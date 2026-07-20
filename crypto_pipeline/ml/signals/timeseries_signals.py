# crypto_pipeline/ml/signals/timeseries_signals.py

"""
timeseries_signals.py
-----------------------
Timeseries signal logic (PDF heading 9), for Darts-backed models
(ml/timeseries/*).

Two shapes, one per family in ml/timeseries/registry.py:

    task_type == "timeseries_regression" (nbeats/tcn/statsforecast):
        predicted % change (last known close -> forecast) > buy_threshold  -> Buy
        predicted % change (last known close -> forecast) < sell_threshold -> Sell
        otherwise                                                          -> Hold
    Reduces predictor.py's forecast PATH (n future close prices) down
    to the % change from the last known close to the FINAL forecasted
    step, same as before.

    task_type == "timeseries_classification" (sklearn_classifier):
        bullish probability > bullish_threshold -> Buy
        bearish probability > bearish_threshold -> Sell
        otherwise                               -> Hold
    Same thresholding shape as classification_signals.py, applied to
    the LAST step of the forecasted probability path (the classifier
    forecasts a path of class probabilities the same way the
    regressors forecast a path of prices).

Thresholds are read from ml/config.yaml via signal_utils.get_thresholds()
-- signals.timeseries for the regression shape (unchanged),
signals.classification (the same section row-wise classifiers already
use) for the classification shape, since it's the identical
bullish/bearish-threshold shape either way.
"""

import logging

import numpy as np

from crypto_pipeline.ml.signals.signal_utils import get_thresholds, label_from_thresholds, signal_counts

logger = logging.getLogger(__name__)


def generate_timeseries_signals(prediction_result: dict, ml_config: dict) -> np.ndarray:
    """
    Convert a timeseries forecast (regression price path, or
    classification probability path) into a Buy/Sell/Hold signal.

    Args:
        prediction_result: dict from predictor.generate_timeseries_predictions()
            with task_type in ("timeseries_regression", "timeseries_classification").
        ml_config: ml/config.yaml dict. Expects (regression shape):

            signals:
              timeseries:
                buy_threshold: 0.002    # predicted % change above this -> Buy
                sell_threshold: -0.002  # predicted % change below this -> Sell

        or (classification shape, same section classification_signals.py uses):

            signals:
              classification:
                bullish_class: 1
                bearish_class: -1
                bullish_threshold: 0.6
                bearish_threshold: 0.6

    Returns:
        np.ndarray of str, shape (1,) -- one signal for the whole
        forecast path (there is exactly one anchor point per predict()
        call, unlike regression/classification which produce one signal
        per row of a test set).
    """
    task_type = prediction_result["task_type"]
    if task_type == "timeseries_regression":
        signal = _regression_signal(prediction_result, ml_config)
    elif task_type == "timeseries_classification":
        signal = _classification_signal(prediction_result, ml_config)
    else:
        raise ValueError(
            f"generate_timeseries_signals() requires a timeseries prediction_result "
            f"(task_type 'timeseries_regression' or 'timeseries_classification'), "
            f"got task_type='{task_type}'."
        )

    signals = np.array([signal])
    logger.info(f"Timeseries signal generated ({task_type}): {signal_counts(signals)}")
    return signals


def _regression_signal(prediction_result: dict, ml_config: dict) -> str:
    thresholds = get_thresholds(ml_config, "timeseries")
    buy_threshold = thresholds.get("buy_threshold")
    sell_threshold = thresholds.get("sell_threshold")
    if buy_threshold is None or sell_threshold is None:
        raise ValueError(
            "ml/config.yaml's signals.timeseries section must set both "
            "buy_threshold and sell_threshold"
        )
    if sell_threshold >= buy_threshold:
        raise ValueError(
            f"signals.timeseries.sell_threshold ({sell_threshold}) must be less than "
            f"buy_threshold ({buy_threshold}), otherwise Buy/Sell ranges overlap or invert"
        )

    last_known_close = prediction_result["last_known_close"]
    forecast = prediction_result["forecast"]
    final_forecast = forecast[-1]

    predicted_pct_change = (final_forecast - last_known_close) / last_known_close

    return label_from_thresholds(
        predicted_pct_change,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        buy_comparison="greater",
        sell_comparison="less",
    )


def _classification_signal(prediction_result: dict, ml_config: dict) -> str:
    thresholds = get_thresholds(ml_config, "classification")
    bullish_class = thresholds.get("bullish_class", 1)
    bearish_class = thresholds.get("bearish_class", -1)
    bullish_threshold = thresholds.get("bullish_threshold")
    bearish_threshold = thresholds.get("bearish_threshold")
    if bullish_threshold is None or bearish_threshold is None:
        raise ValueError(
            "ml/config.yaml's signals.classification section must set both "
            "bullish_threshold and bearish_threshold"
        )

    classes = np.asarray(prediction_result["classes"])
    probabilities = prediction_result["probabilities"]  # shape (n, n_classes)
    final_probabilities = probabilities[-1]  # last forecasted step's distribution

    bullish_idx = _class_index(classes, bullish_class, "bullish_class")
    bearish_idx = _class_index(classes, bearish_class, "bearish_class")
    bullish_p = final_probabilities[bullish_idx]
    bearish_p = final_probabilities[bearish_idx]

    buy_hit = bullish_p > bullish_threshold
    sell_hit = bearish_p > bearish_threshold
    if buy_hit and sell_hit:
        raise ValueError(
            f"Forecast triggers both Buy and Sell (bullish_prob={bullish_p:.4f} > "
            f"{bullish_threshold}, bearish_prob={bearish_p:.4f} > {bearish_threshold}) -- "
            f"thresholds are misconfigured (they should not both be this low)."
        )
    return "Buy" if buy_hit else "Sell" if sell_hit else "Hold"


def _class_index(classes: np.ndarray, label, label_name: str) -> int:
    matches = np.where(classes == label)[0]
    if len(matches) == 0:
        raise ValueError(
            f"signals.classification.{label_name}={label!r} not found in classifier's "
            f"classes_={classes.tolist()}. Check that the configured class label "
            f"matches the actual target encoding."
        )
    return int(matches[0])