# crypto_pipeline/ml/preprocessing/preprocessing_pipeline.py

"""
preprocessing_pipeline.py
--------------------------
Data Preprocessing stage (PDF heading 4).

Chains together the scaling and stationarity methods from scalers.py /
stationarity.py in the ORDER given by config, e.g. the PDF's own example:

    Raw Features -> Fractional Differencing -> RobustScaler -> Training Data

Only preprocessing parameters learned from the TRAINING split may be
applied to the test split -- this module takes train_df and test_df
separately (already split by train_test_split.py) and enforces that:
each step fits on train_df only, then applies the SAME fitted transform
to test_df via the returned fit_info / sklearn object. Nothing here ever
re-fits on test_df.

Every fitted preprocessing object is returned alongside the transformed
data so it can be persisted with the model later (PDF heading 11).
"""

import logging
from typing import Callable, Dict, List

import pandas as pd

from crypto_pipeline.ml.preprocessing.scalers import PREPROCESSING_SCALERS
from crypto_pipeline.ml.preprocessing.stationarity import PREPROCESSING_STATIONARITY

logger = logging.getLogger(__name__)

# One combined registry -- config just names a method, this module doesn't
# care whether it came from scalers.py or stationarity.py.
PREPROCESSING_REGISTRY: Dict[str, Callable] = {
    **PREPROCESSING_SCALERS,
    **PREPROCESSING_STATIONARITY,
}


def run_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: List[str], ml_config: dict) -> dict:
    """
    Apply the configured chain of preprocessing methods to train_df and
    test_df's feature columns only (never datetime, never target).

    Args:
        train_df: chronologically-first split from train_test_split.py
        test_df: chronologically-last split from train_test_split.py
        feature_columns: resolved list from feature_selector.select_features()
        ml_config: ml/config.yaml dict. Expects a "preprocessing" section:

            preprocessing:
              steps:
                - method: fractional_differencing
                  params: {d: 0.2, threshold: 1e-3}
                - method: robust_scaler
                  params: {}

    Returns:
        dict with keys:
            train_df: pd.DataFrame, transformed (datetime/target untouched)
            test_df: pd.DataFrame, transformed with train-fitted params
            fit_objects: list of {method, fit_info} in the order applied --
                this is what gets persisted alongside the trained model
            dropped_rows: {"train": int, "test": int} -- rows removed due
                to leading NaNs from stationarity methods
    """
    preprocessing_config = ml_config.get("preprocessing", {})
    steps = preprocessing_config.get("steps", [])

    if not steps:
        logger.info("No preprocessing steps configured -- passing data through unchanged")
        return {
            "train_df": train_df,
            "test_df": test_df,
            "fit_objects": [],
            "dropped_rows": {"train": 0, "test": 0},
        }

    # No .copy() -- the first thing each loop iteration does is
    # drop(columns=...) + concat(), both of which already return new
    # DataFrames rather than mutating train_df/test_df in place.
    train_out = train_df
    test_out = test_df
    fit_objects = []

    train_initial_rows = len(train_out)
    test_initial_rows = len(test_out)

    for step in steps:
        method_name = step["method"]
        params = step.get("params", {}) or {}

        if method_name not in PREPROCESSING_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing method '{method_name}'. "
                f"Available: {list(PREPROCESSING_REGISTRY.keys())}"
            )
        transform_fn = PREPROCESSING_REGISTRY[method_name]

        # YAML loads [0, 1] as a list, but sklearn's MinMaxScaler requires
        # a tuple for feature_range.
        if "feature_range" in params:
            params = {**params, "feature_range": tuple(params["feature_range"])}

        # Fit on train rows only (fit_mask=None here means "fit on
        # everything passed in", and what's passed in is exactly
        # train_out -- test never enters the fit).
        transformed_train, fit_info = transform_fn(train_out[feature_columns], **params)

        # Re-apply the SAME fitted transform to test. Methods that persist
        # a fitted sklearn object under fit_info["_sklearn_object"] use it
        # directly; purely backward-looking/causal methods (the
        # stationarity methods, and the row-wise Normalizer) are simply
        # re-run against test_out, since they have nothing data-driven to
        # leak from train in the first place.
        if "_sklearn_object" in fit_info:
            fitted_scaler = fit_info["_sklearn_object"]
            transformed_test = pd.DataFrame(
                fitted_scaler.transform(test_out[feature_columns].values),
                columns=feature_columns,
                index=test_out.index,
            )
        else:
            transformed_test, _ = transform_fn(test_out[feature_columns], **params)

        train_out = train_out.drop(columns=feature_columns)
        train_out = pd.concat([train_out, transformed_train], axis=1)[train_df.columns]

        test_out = test_out.drop(columns=feature_columns)
        test_out = pd.concat([test_out, transformed_test], axis=1)[test_df.columns]

        fit_objects.append({"method": method_name, "fit_info": fit_info})
        logger.info(f"Applied preprocessing step: {method_name} (params={params})")

    # Stationarity methods create leading NaN rows by construction -- drop
    # them now, once, after the full chain has run.
    train_out = train_out.dropna(subset=feature_columns)
    test_out = test_out.dropna(subset=feature_columns)

    dropped_rows = {
        "train": train_initial_rows - len(train_out),
        "test": test_initial_rows - len(test_out),
    }
    if dropped_rows["train"] or dropped_rows["test"]:
        logger.info(
            f"Dropped rows with leading NaNs after preprocessing: "
            f"train={dropped_rows['train']} ({dropped_rows['train']/train_initial_rows*100:.1f}%), "
            f"test={dropped_rows['test']} ({dropped_rows['test']/test_initial_rows*100:.1f}%)"
        )

    return {
        "train_df": train_out,
        "test_df": test_out,
        "fit_objects": fit_objects,
        "dropped_rows": dropped_rows,
    }