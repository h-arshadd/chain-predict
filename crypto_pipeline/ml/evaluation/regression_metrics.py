# crypto_pipeline/ml/evaluation/regression_metrics.py

"""
regression_metrics.py
----------------------
Machine Learning Evaluation for regression models (PDF heading 10).

Just the three metrics the PDF names -- MAE, MSE, RMSE. Computed
immediately after training, reported alongside the trading metrics, but
per the PDF never used to pick the "best" model (that's
evaluator.select_best_model(), driven entirely by trading_metrics).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Args:
        y_true: true target values, shape (n_rows,)
        y_pred: predicted values, shape (n_rows,) -- e.g.
            prediction_result["predictions"] from predictor.py

    Returns:
        dict: {"mae": float, "mse": float, "rmse": float}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
    }