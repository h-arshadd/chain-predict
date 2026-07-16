# crypto_pipeline/ml/signals/signal_utils.py

"""
signal_utils.py
----------------
Shared building blocks for signal generation (PDF heading 9).

Both regression_signals.py and classification_signals.py return the
same three labels via the same threshold-comparison pattern -- this
file is the one place that:
    - defines what the three signal labels actually are (SIGNALS)
    - turns a threshold comparison into a label (label_from_thresholds())
    - reads the configured thresholds out of ml/config.yaml (get_thresholds())

Neither regression_signals.py nor classification_signals.py import
anything from ml/regressors/, ml/classifiers/, or ml/deep_learning/ --
they only ever see the standardized dict predictor.generate_predictions()
already produced. That's what "signal generation must remain
independent from the model implementations" means in code: this whole
folder has zero knowledge of which algorithm produced the numbers it's
thresholding.
"""

from typing import Dict

import numpy as np

BUY = "Buy"
SELL = "Sell"
HOLD = "Hold"
SIGNALS = (BUY, SELL, HOLD)


def label_from_thresholds(value: float, buy_threshold: float, sell_threshold: float,
                           buy_comparison: str = "greater", sell_comparison: str = "greater") -> str:
    """
    Turn one numeric value (a predicted return, or a class probability)
    into a Buy/Sell/Hold label by comparing it against two configurable
    thresholds. Both regression_signals.py (predicted return vs a single
    symmetric threshold) and classification_signals.py (two separate
    probability thresholds, one per class) reduce to this same
    comparison shape.

    Args:
        value: the number being thresholded (predicted return, or a
            probability in [0, 1])
        buy_threshold: value must exceed this to trigger Buy
        sell_threshold: value must exceed (or fall below, depending on
            sell_comparison) this to trigger Sell
        buy_comparison: "greater" (value > buy_threshold) -- Buy is
            always "greater" in both the PDF's regression and
            classification examples, kept as a parameter only so this
            function doesn't hardcode an assumption its two callers
            both happen to share today.
        sell_comparison: "greater" (value > sell_threshold, e.g. a
            bearish probability) or "less" (value < sell_threshold, e.g.
            a predicted return below a negative threshold)

    Returns:
        One of BUY, SELL, HOLD.
    """
    buy_hit = value > buy_threshold if buy_comparison == "greater" else value < buy_threshold
    if sell_comparison == "greater":
        sell_hit = value > sell_threshold
    elif sell_comparison == "less":
        sell_hit = value < sell_threshold
    else:
        raise ValueError(f"sell_comparison must be 'greater' or 'less', got '{sell_comparison}'")

    if buy_hit and sell_hit:
        raise ValueError(
            f"value={value} triggers both Buy and Sell thresholds "
            f"(buy_threshold={buy_threshold}, sell_threshold={sell_threshold}) -- "
            f"thresholds are misconfigured (they must not overlap)."
        )
    if buy_hit:
        return BUY
    if sell_hit:
        return SELL
    return HOLD


def get_thresholds(ml_config: dict, section: str) -> Dict[str, float]:
    """
    Read the configured thresholds for one signal section out of
    ml/config.yaml, e.g.:

        signals:
          regression:
            buy_threshold: 0.002
            sell_threshold: -0.002
          classification:
            bullish_threshold: 0.6
            bearish_threshold: 0.6

    Args:
        ml_config: ml/config.yaml dict
        section: "regression" or "classification"

    Returns:
        dict of threshold_name -> float, exactly as configured (this
        function doesn't know the specific key names either section
        uses -- regression_signals.py / classification_signals.py read
        the keys they need out of the returned dict).
    """
    signals_config = ml_config.get("signals", {})
    section_config = signals_config.get(section, {})
    if not section_config:
        raise ValueError(
            f"ml/config.yaml is missing signals.{section} thresholds. "
            f"Add a 'signals: {section}: ...' section with the required threshold keys."
        )
    return section_config


def signal_counts(signals: np.ndarray) -> Dict[str, int]:
    """Quick summary of how many Buy/Sell/Hold signals were generated -- useful for logging."""
    return {label: int(np.sum(signals == label)) for label in SIGNALS}