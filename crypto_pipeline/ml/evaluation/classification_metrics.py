# crypto_pipeline/ml/evaluation/classification_metrics.py

"""
classification_metrics.py
---------------------------
Machine Learning Evaluation for classification models (PDF heading 10).

Just the three metrics the PDF names -- Accuracy, Precision, Recall.
Computed immediately after training, reported alongside the trading
metrics, but per the PDF never used to pick the "best" model (that's
evaluator.select_best_model(), driven entirely by trading_metrics).

Precision/recall use average="weighted" since targets here are the
project's -1/0/1 (bearish/neutral/bullish) triple-barrier labels, not
binary -- weighted averaging accounts for class imbalance across all
three classes without requiring the caller to name a "positive" class.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Args:
        y_true: true class labels, shape (n_rows,)
        y_pred: predicted class labels, shape (n_rows,) -- e.g.
            prediction_result["predictions"] from predictor.py

    Returns:
        dict: {"accuracy": float, "precision": float, "recall": float}
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }