# crypto_pipeline/ml/signals/timeseries_signals.py

"""
timeseries_signals.py
-----------------------
Timeseries signal logic (PDF heading 9), for Darts-backed models
(ml/timeseries/*, e.g. NBEATSModel, TCNModel):

    predicted % change (last known close -> forecast) > buy_threshold  -> Buy
    predicted % change (last known close -> forecast) < sell_threshold -> Sell
    otherwise                                                          -> Hold

Unlike regression_signals.py (one predicted return per row) and
classification_signals.py (one probability per row), a timeseries
model's predict() returns a forecast PATH -- n future close prices from
one anchor point (see ml/timeseries/base_timeseries_model.py). This
module reduces that path to the same single-number-per-threshold shape
signal_utils.label_from_thresholds() already expects: the % change from
the last known close to the final forecasted step. Thresholds are read
from ml/config.yaml via signal_utils.get_thresholds(), never hardcoded
here -- same as every other signals module.
"""

import logging

import numpy as np

from crypto_pipeline.ml.signals.signal_utils import get_thresholds, label_from_thresholds, signal_counts

logger = logging.getLogger(__name__)


def generate_timeseries_signals(prediction_result: dict, ml_config: dict) -> np.ndarray:
    """
    Convert a forecasted price path into a Buy/Sell/Hold signal.

    Args:
        prediction_result: dict from predictor.generate_timeseries_predictions()
            with task_type == "timeseries". Uses:
                "last_known_close": float, the close price the forecast
                    is anchored from (the last row of the input series)
                "forecast": np.ndarray, the n predicted future close
                    prices, in chronological order
        ml_config: ml/config.yaml dict. Expects:

            signals:
              timeseries:
                buy_threshold: 0.002    # predicted % change above this -> Buy
                sell_threshold: -0.002  # predicted % change below this -> Sell

    Returns:
        np.ndarray of str, shape (1,) -- one signal for the whole
        forecast path (there is exactly one anchor point per predict()
        call, unlike regression/classification which produce one signal
        per row of a test set).
    """
    if prediction_result["task_type"] != "timeseries":
        raise ValueError(
            f"generate_timeseries_signals() requires a timeseries prediction_result, "
            f"got task_type='{prediction_result['task_type']}'. "
            f"Use generate_regression_signals() or generate_classification_signals() instead."
        )

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

    signal = label_from_thresholds(
        predicted_pct_change,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        buy_comparison="greater",
        sell_comparison="less",
    )
    signals = np.array([signal])

    logger.info(
        f"Timeseries signal generated (predicted_pct_change={predicted_pct_change:.5f}, "
        f"buy_threshold={buy_threshold}, sell_threshold={sell_threshold}): {signal_counts(signals)}"
    )
    return signals