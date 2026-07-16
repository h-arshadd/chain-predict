# crypto_pipeline/ml/signals/regression_signals.py

"""
regression_signals.py
-----------------------
Regression signal logic (PDF heading 9):

    Predicted return > threshold  -> Buy
    Predicted return < threshold  -> Sell
    Otherwise                     -> Hold

Consumes predictor.generate_predictions()'s output dict directly
(task_type == "regression", so predictions is the predicted return
array and probabilities is None) -- never touches the model itself.
Thresholds are read from ml/config.yaml via signal_utils.get_thresholds(),
never hardcoded here.
"""

import logging

import numpy as np

from crypto_pipeline.ml.signals.signal_utils import get_thresholds, label_from_thresholds, signal_counts

logger = logging.getLogger(__name__)


def generate_regression_signals(prediction_result: dict, ml_config: dict) -> np.ndarray:
    """
    Convert predicted returns into Buy/Sell/Hold signals.

    Args:
        prediction_result: dict from predictor.generate_predictions()
            with task_type == "regression" (uses only the
            "predictions" key -- predicted return per row).
        ml_config: ml/config.yaml dict. Expects:

            signals:
              regression:
                buy_threshold: 0.002   # predicted return above this -> Buy
                sell_threshold: -0.002 # predicted return below this -> Sell

    Returns:
        np.ndarray of str, shape (n_rows,), one of "Buy"/"Sell"/"Hold"
        per row, same row order as prediction_result["predictions"].
    """
    if prediction_result["task_type"] != "regression":
        raise ValueError(
            f"generate_regression_signals() requires a regression prediction_result, "
            f"got task_type='{prediction_result['task_type']}'. "
            f"Use generate_classification_signals() for classification predictions."
        )

    thresholds = get_thresholds(ml_config, "regression")
    buy_threshold = thresholds.get("buy_threshold")
    sell_threshold = thresholds.get("sell_threshold")
    if buy_threshold is None or sell_threshold is None:
        raise ValueError(
            "ml/config.yaml's signals.regression section must set both "
            "buy_threshold and sell_threshold"
        )
    if sell_threshold >= buy_threshold:
        raise ValueError(
            f"signals.regression.sell_threshold ({sell_threshold}) must be less than "
            f"buy_threshold ({buy_threshold}), otherwise Buy/Sell ranges overlap or invert"
        )

    predicted_returns = prediction_result["predictions"]

    signals = np.array([
        label_from_thresholds(
            value,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            buy_comparison="greater",
            sell_comparison="less",
        )
        for value in predicted_returns
    ])

    logger.info(
        f"Regression signals generated (buy_threshold={buy_threshold}, "
        f"sell_threshold={sell_threshold}): {signal_counts(signals)}"
    )
    return signals