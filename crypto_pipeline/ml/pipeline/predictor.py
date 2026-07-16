# crypto_pipeline/ml/pipeline/predictor.py

"""
predictor.py
------------
Prediction stage (PDF heading 8).

Lives in ml/pipeline/ alongside dataset_loader.py and
train_test_split.py -- the PDF's recommended tree doesn't give
Prediction its own folder, it's a pipeline stage between "Model
Training" and "Signal Generation", same as those two files.

Takes an already-trained model (any BaseRegressor, BaseClassifier,
BaseNetwork, or BaseClassifierNetwork -- traditional or deep learning,
doesn't matter which) plus the test dataset, and returns predictions in
one standardized format. This is the ONE place that calls
model.predict() / model.predict_proba() for the prediction stage --
regression_pipeline.py / classification_pipeline.py currently call
predict() inline themselves (heading 5/6 training output), but any
downstream stage (signal generation, evaluation, heading 9-10) should
consume THIS module's output, not call the model directly, so the
format is guaranteed identical regardless of model type.

Per the PDF:
    Regression      -> Predicted value
    Classification  -> Predicted class, Class probabilities
"The prediction interface should remain identical regardless of model
type" -- this is enforced by generate_predictions() always returning
the same dict shape, with probabilities=None for regression instead of
a different return type/signature.

Timeseries (Darts-backed models, ml/timeseries/*) does NOT go through
generate_predictions(): Darts models forecast n steps forward from
wherever train() left off (model.predict(n=...)), there is no X_test
DataFrame to predict row-by-row against, so the same function signature
doesn't apply. generate_timeseries_predictions() below is the
timeseries equivalent -- same standardized-dict philosophy, different
shape because the input shape is genuinely different, not because
timeseries is treated as a lesser case.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_predictions(model, X_test: pd.DataFrame, task_type: str) -> dict:
    """
    Generate standardized predictions for the test dataset.

    Args:
        model: a trained model exposing .predict() (and .predict_proba()
            + .classes_ if task_type == "classification"). Works
            identically whether `model` is a traditional model
            (BaseRegressor/BaseClassifier) or a deep learning model
            (BaseNetwork/BaseClassifierNetwork) -- both expose the same
            method names, so this function never branches on which kind
            of object `model` actually is, only on task_type.
        X_test: pd.DataFrame, test-set features, same column order used
            during training
        task_type: "regression" or "classification"

    Returns:
        dict, same keys regardless of task_type:
            task_type: str, echoed back
            predictions: np.ndarray
                regression: predicted value, shape (n_rows,)
                classification: predicted class label, shape (n_rows,)
            probabilities: np.ndarray or None
                regression: always None (no probabilities for regression)
                classification: shape (n_rows, n_classes), columns
                    ordered per `classes`
            classes: np.ndarray or None
                regression: always None
                classification: model.classes_, the label each
                    probabilities column corresponds to
            n_predictions: int, len(predictions), for a quick sanity
                check against len(X_test)
    """
    if task_type not in ("regression", "classification"):
        raise ValueError(f"task_type must be 'regression' or 'classification', got '{task_type}'")

    if task_type == "regression":
        result = _predict_regression(model, X_test)
    else:
        result = _predict_classification(model, X_test)

    logger.info(
        f"Generated {result['n_predictions']} {task_type} predictions "
        f"for {len(X_test)} test rows"
    )
    return result


def _predict_regression(model, X_test: pd.DataFrame) -> dict:
    predictions = model.predict(X_test)
    return {
        "task_type": "regression",
        "predictions": predictions,
        "probabilities": None,
        "classes": None,
        "n_predictions": len(predictions),
    }


def _predict_classification(model, X_test: pd.DataFrame) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    return {
        "task_type": "classification",
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": classes,
        "n_predictions": len(predictions),
    }


def generate_timeseries_predictions(model, n: int, last_known_close: float, past_covariates=None) -> dict:
    """
    Generate a standardized prediction result for a timeseries model
    (ml/timeseries/*, e.g. NBEATSTimeseriesModel, TCNTimeseriesModel).

    Args:
        model: a trained BaseTimeseriesModel exposing .predict(n, past_covariates)
        n: int, how many steps ahead to forecast (typically
            ml/config.yaml's model.params.output_chunk_length)
        last_known_close: float, the close price the forecast is
            anchored from (the last row of whatever series train() was
            fit on) -- echoed back here so timeseries_signals.py can
            compute a % change without needing the original DataFrame.
        past_covariates: optional darts.TimeSeries of covariates
            (indicator/pattern/sentiment columns) covering the forecast
            horizon, forwarded to model.predict() unchanged.

    Returns:
        dict:
            task_type: "timeseries"
            forecast: np.ndarray, the n predicted future close prices,
                in chronological order
            last_known_close: float, echoed back
            n_predictions: int, len(forecast)
    """
    forecast = model.predict(n=n, past_covariates=past_covariates)
    logger.info(f"Generated {len(forecast)}-step timeseries forecast from last_known_close={last_known_close}")
    return {
        "task_type": "timeseries",
        "forecast": forecast,
        "last_known_close": last_known_close,
        "n_predictions": len(forecast),
    }