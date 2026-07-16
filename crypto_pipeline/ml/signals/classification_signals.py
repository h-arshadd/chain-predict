# crypto_pipeline/ml/signals/classification_signals.py

"""
classification_signals.py
----------------------------
Classification signal logic (PDF heading 9):

    Bullish probability > threshold  -> Buy
    Bearish probability > threshold  -> Sell
    Otherwise                        -> Hold

Consumes predictor.generate_predictions()'s output dict directly
(task_type == "classification", uses "probabilities" + "classes" --
never "predictions" directly, since thresholding the probability is
what the PDF specifies, not the argmax class) -- never touches the
model itself. Thresholds are read from ml/config.yaml via
signal_utils.get_thresholds(), never hardcoded here.

Label convention: this project's target_pipeline (data_prep) encodes
triple-barrier labels as -1 (bearish) / 0 (neutral) / 1 (bullish) --
see ml/data_prep/config.yaml's target section. bullish_class /
bearish_class default to 1 / -1 accordingly, but are config-driven
(not hardcoded) so a differently-labeled dataset still works.
"""

import logging

import numpy as np

from crypto_pipeline.ml.signals.signal_utils import get_thresholds, label_from_thresholds, signal_counts

logger = logging.getLogger(__name__)


def generate_classification_signals(prediction_result: dict, ml_config: dict) -> np.ndarray:
    """
    Convert class probabilities into Buy/Sell/Hold signals.

    Args:
        prediction_result: dict from predictor.generate_predictions()
            with task_type == "classification" (uses "probabilities"
            and "classes" -- shape (n_rows, n_classes) and (n_classes,)).
        ml_config: ml/config.yaml dict. Expects:

            signals:
              classification:
                bullish_class: 1        # label value that means "bullish"
                bearish_class: -1       # label value that means "bearish"
                bullish_threshold: 0.6  # bullish probability above this -> Buy
                bearish_threshold: 0.6  # bearish probability above this -> Sell

    Returns:
        np.ndarray of str, shape (n_rows,), one of "Buy"/"Sell"/"Hold"
        per row, same row order as prediction_result["probabilities"].
    """
    if prediction_result["task_type"] != "classification":
        raise ValueError(
            f"generate_classification_signals() requires a classification prediction_result, "
            f"got task_type='{prediction_result['task_type']}'. "
            f"Use generate_regression_signals() for regression predictions."
        )

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

    probabilities = prediction_result["probabilities"]
    classes = np.asarray(prediction_result["classes"])

    bullish_idx = _class_index(classes, bullish_class, "bullish_class")
    bearish_idx = _class_index(classes, bearish_class, "bearish_class")

    bullish_probs = probabilities[:, bullish_idx]
    bearish_probs = probabilities[:, bearish_idx]

    signals = []
    for bullish_p, bearish_p in zip(bullish_probs, bearish_probs):
        # Two independent probability columns, not one shared axis like
        # regression's single value -- can't reuse label_from_thresholds()'s
        # single-value comparison directly, so Buy/Sell are each checked
        # against their own probability + threshold, then combined the
        # same way (both true -> misconfigured, since a low enough pair
        # of thresholds could make both bullish and bearish probability
        # exceed their threshold at once).
        buy_hit = bullish_p > bullish_threshold
        sell_hit = bearish_p > bearish_threshold
        if buy_hit and sell_hit:
            raise ValueError(
                f"Row triggers both Buy and Sell (bullish_prob={bullish_p:.4f} > "
                f"{bullish_threshold}, bearish_prob={bearish_p:.4f} > {bearish_threshold}) -- "
                f"thresholds are misconfigured (they should not both be this low)."
            )
        signals.append("Buy" if buy_hit else "Sell" if sell_hit else "Hold")

    signals = np.array(signals)

    logger.info(
        f"Classification signals generated (bullish_threshold={bullish_threshold}, "
        f"bearish_threshold={bearish_threshold}): {signal_counts(signals)}"
    )
    return signals


def _class_index(classes: np.ndarray, label, label_name: str) -> int:
    matches = np.where(classes == label)[0]
    if len(matches) == 0:
        raise ValueError(
            f"signals.classification.{label_name}={label!r} not found in model.classes_={classes.tolist()}. "
            f"Check that the configured class label matches the actual target encoding."
        )
    return int(matches[0])