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
doesn't apply. Two timeseries equivalents below, same standardized-dict
philosophy, different shape because the input shape is genuinely
different, not because timeseries is treated as a lesser case:

    generate_timeseries_predictions()   -- single anchored n-step
        forecast (task_type="timeseries_regression" or
        "timeseries_classification" depending on which family the
        trained model belongs to -- see ml/timeseries/registry.py's
        TS_REGRESSORS/TS_CLASSIFIERS).
    generate_timeseries_historical_predictions() -- many forecasts
        walking forward across a series (PDF heading 10's evaluation
        needs this shape, not just one anchor point) -- wraps
        model.historical_forecasts(), covering both the one-step
        (forecast_horizon=1, stride=1) and fixed-window
        (train_length=N) modes.
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


def generate_timeseries_predictions(
    model,
    n: int,
    last_known_close: Optional[float] = None,
    past_covariates=None,
    future_covariates=None,
) -> dict:
    """
    Generate a standardized prediction result for a single anchored
    timeseries forecast (ml/timeseries/*, e.g. NBEATSTimeseriesModel,
    TCNTimeseriesModel, StatsForecastTimeseriesModel,
    SKLearnClassifierTimeseriesModel).

    Args:
        model: a trained BaseTimeseriesModel or BaseTimeseriesClassifier
            (ml/timeseries/registry.py's TS_REGRESSORS/TS_CLASSIFIERS)
            exposing .predict(n, past_covariates, future_covariates),
            and .predict_proba()/.classes_ if it's a classifier.
        n: int, how many steps ahead to forecast (this project uses
            output_chunk_length=1, forecast_horizon=1 -- see
            ml/config.yaml).
        last_known_close: float, the close price the forecast is
            anchored from (the last row of whatever series train() was
            fit on) -- echoed back here so timeseries_signals.py can
            compute a % change without needing the original DataFrame.
            Only meaningful for regression forecasters; None for a
            classifier (there's no "% change from anchor" concept for a
            discrete label).
        past_covariates / future_covariates: optional darts.TimeSeries,
            forwarded to model.predict() unchanged.

    Returns:
        dict:
            task_type: "timeseries_regression" or "timeseries_classification",
                based on whether `model` is a TS_REGRESSORS or
                TS_CLASSIFIERS instance (checked via TASK_TYPE, set on
                every concrete model class -- see
                base_timeseries_model.py / base_timeseries_classifier.py).
            forecast: np.ndarray, the n predicted future close prices
                (regression) or class labels (classification), in
                chronological order
            probabilities: np.ndarray or None -- shape (n, n_classes)
                for a classifier, None for a regressor
            classes: np.ndarray or None -- classifier's class label set,
                None for a regressor
            last_known_close: float or None, echoed back
            n_predictions: int, len(forecast)
    """
    is_classifier = getattr(model, "TASK_TYPE", "regression") == "classification"

    forecast = model.predict(n=n, past_covariates=past_covariates, future_covariates=future_covariates)

    probabilities = None
    classes = None
    if is_classifier:
        probabilities = model.predict_proba(n=n, past_covariates=past_covariates, future_covariates=future_covariates)
        classes = np.asarray(model.classes_)

    task_type = "timeseries_classification" if is_classifier else "timeseries_regression"
    logger.info(
        f"Generated {len(forecast)}-step {task_type} forecast"
        + (f" from last_known_close={last_known_close}" if last_known_close is not None else "")
    )
    return {
        "task_type": task_type,
        "forecast": forecast,
        "probabilities": probabilities,
        "classes": classes,
        "last_known_close": last_known_close,
        "n_predictions": len(forecast),
    }


def generate_timeseries_historical_predictions(
    model,
    series,
    past_covariates=None,
    future_covariates=None,
    forecast_horizon: int = 1,
    stride: int = 1,
    retrain: bool = False,
    train_length: Optional[int] = None,
) -> dict:
    """
    Generate a standardized prediction result for WALK-FORWARD
    timeseries forecasting (PDF heading 10 -- many forecasts across a
    series, not one anchor point), via model.historical_forecasts().
    Both forecasting modes this project uses are just different
    argument combinations here:

        one-step forecasting:     forecast_horizon=1, stride=1 (defaults)
        fixed window forecasting: train_length=N (with retrain=True to
            actually refit on each rolling window; retrain=False slides
            the window without refitting)

    Args:
        model: a trained BaseTimeseriesModel or BaseTimeseriesClassifier.
        series: darts.TimeSeries to walk forward over -- typically the
            full (train+test) series so historical_forecasts() has
            enough leading history to produce a forecast at the start
            of the test period.
        past_covariates / future_covariates: optional darts.TimeSeries
            covering `series`'s full span.
        forecast_horizon, stride, retrain, train_length: forwarded to
            model.historical_forecasts() (see that method's docstring
            on base_timeseries_model.py / base_timeseries_classifier.py
            for the full explanation of each).

    Returns:
        dict:
            task_type: "timeseries_regression" or "timeseries_classification"
            forecast: np.ndarray, one predicted value per historical
                forecast point (last_points_only=True is always used
                here, matching this project's forecast_horizon=1 setup)
            n_predictions: int, len(forecast)
    """
    is_classifier = getattr(model, "TASK_TYPE", "regression") == "classification"

    forecast = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=retrain,
        train_length=train_length,
        last_points_only=True,
    )

    task_type = "timeseries_classification" if is_classifier else "timeseries_regression"
    logger.info(
        f"Generated {len(forecast)} historical {task_type} forecasts "
        f"(forecast_horizon={forecast_horizon}, stride={stride}, retrain={retrain}, "
        f"train_length={train_length})"
    )
    return {
        "task_type": task_type,
        "forecast": forecast,
        "n_predictions": len(forecast),
    }