# preprocessing_lab/model_evaluation/signal_conversion.py

"""
signal_conversion.py
---------------------
Converts model predictions into trading signals (1=long, -1=short, 0=hold).
Same logic is used for every preprocessing method and every model, so the
preprocessing method/model stays the only experimental variable (task
requirement: "Ensure that all preprocessing methods are evaluated using
identical signal-generation logic").

classification predictions are already -1/0/1 (Triple Barrier Labeling) --
passed straight through as the signal, no extra step needed.

regression predictions are a predicted log-return -- turned into a signal
with the SAME threshold ml_module/config.yaml already uses to build the
classification target (target.upper_threshold / target.lower_threshold),
not a separate number invented here. This keeps "how big a move counts as
worth trading" consistent between regression and classification.
"""

import pandas as pd
import numpy as np


def predictions_to_signals(
    datetimes: pd.Series,
    predictions: np.ndarray,
    target_type: str,
    upper_threshold: float = None,
    lower_threshold: float = None,
) -> pd.DataFrame:
    """
    Build a (datetime, signal) DataFrame from raw model predictions.

    Parameters
    ----------
    datetimes : pd.Series
        Datetime for each prediction (same order/length as predictions).
    predictions : np.ndarray
        Raw model output. Classification: -1/0/1. Regression: predicted
        log-return (float).
    target_type : str
        "regression" or "classification".
    upper_threshold, lower_threshold : float
        Only used for regression. Predicted return > upper_threshold -> long,
        < lower_threshold -> short, otherwise hold. Pass ml_module's own
        target.upper_threshold / target.lower_threshold here so the same
        bar is used everywhere.

    Returns
    -------
    pd.DataFrame with columns: datetime, signal
    """
    predictions = np.asarray(predictions)

    if target_type == "classification":
        # Triple Barrier Labeling already outputs -1/0/1 -- that IS the signal.
        signal = predictions.astype(int)

    elif target_type == "regression":
        if upper_threshold is None or lower_threshold is None:
            raise ValueError("regression signal conversion requires upper_threshold and lower_threshold")

        signal = np.zeros(len(predictions), dtype=int)
        signal[predictions > upper_threshold] = 1
        signal[predictions < lower_threshold] = -1

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return pd.DataFrame({"datetime": pd.Series(datetimes).reset_index(drop=True), "signal": signal})