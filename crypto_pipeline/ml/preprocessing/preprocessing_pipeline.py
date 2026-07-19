# crypto_pipeline/ml/preprocessing/preprocessing_pipeline.py

"""
preprocessing_pipeline.py
--------------------------
Data Preprocessing stage (PDF heading 4).

Chains together the scaling and stationarity methods from scalers.py /
stationarity.py in the ORDER given by config, e.g. the PDF's own example:

    Raw Features -> Fractional Differencing -> RobustScaler -> Training Data

Only preprocessing parameters learned from the TRAINING split may be
applied to the val/test splits -- this module takes train_df and test_df
separately (already split by train_test_split.py), plus an optional
val_df, and enforces that: each step fits on train_df only, then applies
the SAME fitted transform to val_df and test_df via the returned
fit_info / sklearn object. Nothing here ever re-fits on val_df or
test_df -- fitting a scaler/transform on validation data would leak
information the model isn't supposed to have yet, same as it would for
test.

Every fitted preprocessing object is returned alongside the transformed
data so it can be persisted with the model later (PDF heading 11).
"""

import logging
from typing import Callable, Dict, List, Optional

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


def run_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    ml_config: dict,
    val_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Apply the configured chain of preprocessing methods to train_df,
    val_df (if given), and test_df's feature columns only (never
    datetime, never target).

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

        val_df: optional chronologically-middle split from
            train_test_split.split_dataset() (split_info["val_df"], None
            if split.val_size wasn't set in config). When given, every
            step's fitted transform (fit on train_df only, same as test)
            is also applied to val_df -- val is never part of the fit.

    Returns:
        dict with keys:
            train_df: pd.DataFrame, transformed (datetime/target untouched)
            val_df: pd.DataFrame or None, transformed with train-fitted
                params (only present if val_df was given)
            test_df: pd.DataFrame, transformed with train-fitted params
            fit_objects: list of {method, fit_info} in the order applied --
                this is what gets persisted alongside the trained model
            dropped_rows: {"train": int, "val": int, "test": int} -- rows
                removed due to leading NaNs from stationarity methods
                ("val" is 0 if val_df wasn't given)
    """
    preprocessing_config = ml_config.get("preprocessing", {})
    steps = preprocessing_config.get("steps", [])
    has_val = val_df is not None

    if not steps:
        logger.info("No preprocessing steps configured -- passing data through unchanged")
        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "fit_objects": [],
            "dropped_rows": {"train": 0, "val": 0, "test": 0},
        }

    # No .copy() -- the first thing each loop iteration does is
    # drop(columns=...) + concat(), both of which already return new
    # DataFrames rather than mutating train_df/val_df/test_df in place.
    train_out = train_df
    val_out = val_df if has_val else None
    test_out = test_df
    fit_objects = []

    train_initial_rows = len(train_out)
    val_initial_rows = len(val_out) if has_val else 0
    test_initial_rows = len(test_out)

    def _apply_fitted(source_df: pd.DataFrame, fit_info: dict, transform_fn: Callable, params: dict) -> pd.DataFrame:
        """
        Re-apply an already-fitted (on train only) transform to some
        other split (val or test) -- shared so val and test go through
        the exact same logic and can never accidentally diverge.
        """
        if "_sklearn_object" in fit_info:
            fitted_scaler = fit_info["_sklearn_object"]
            return pd.DataFrame(
                fitted_scaler.transform(source_df[feature_columns].values),
                columns=feature_columns,
                index=source_df.index,
            )
        elif "lower_bounds" in fit_info and "upper_bounds" in fit_info:
            lower = pd.Series(fit_info["lower_bounds"])[feature_columns]
            upper = pd.Series(fit_info["upper_bounds"])[feature_columns]
            return source_df[feature_columns].clip(lower=lower, upper=upper, axis=1)
        else:
            # Purely backward-looking/causal methods (the stationarity
            # methods, and the row-wise Normalizer) have nothing
            # data-driven to leak from train, so they're just re-run.
            transformed, _ = transform_fn(source_df[feature_columns], **params)
            return transformed

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
        # train_out -- val/test never enter the fit).
        transformed_train, fit_info = transform_fn(train_out[feature_columns], **params)

        # Re-apply the SAME fitted transform to val (if present) and
        # test -- neither ever refits its own params (that would leak
        # val/test-set statistics into what's supposed to be a
        # train-only fit).
        transformed_test = _apply_fitted(test_out, fit_info, transform_fn, params)
        transformed_val = _apply_fitted(val_out, fit_info, transform_fn, params) if has_val else None

        train_out = train_out.drop(columns=feature_columns)
        train_out = pd.concat([train_out, transformed_train], axis=1)[train_df.columns]

        test_out = test_out.drop(columns=feature_columns)
        test_out = pd.concat([test_out, transformed_test], axis=1)[test_df.columns]

        if has_val:
            val_out = val_out.drop(columns=feature_columns)
            val_out = pd.concat([val_out, transformed_val], axis=1)[val_df.columns]

        fit_objects.append({"method": method_name, "fit_info": fit_info})
        logger.info(f"Applied preprocessing step: {method_name} (params={params})")

    # Stationarity methods create leading NaN rows by construction -- drop
    # them now, once, after the full chain has run.
    train_out = train_out.dropna(subset=feature_columns)
    test_out = test_out.dropna(subset=feature_columns)
    if has_val:
        val_out = val_out.dropna(subset=feature_columns)

    dropped_rows = {
        "train": train_initial_rows - len(train_out),
        "val": (val_initial_rows - len(val_out)) if has_val else 0,
        "test": test_initial_rows - len(test_out),
    }
    if dropped_rows["train"] or dropped_rows["val"] or dropped_rows["test"]:
        val_pct = f", val={dropped_rows['val']} ({dropped_rows['val']/val_initial_rows*100:.1f}%)" if has_val else ""
        logger.info(
            f"Dropped rows with leading NaNs after preprocessing: "
            f"train={dropped_rows['train']} ({dropped_rows['train']/train_initial_rows*100:.1f}%)"
            f"{val_pct}, "
            f"test={dropped_rows['test']} ({dropped_rows['test']/test_initial_rows*100:.1f}%)"
        )

    return {
        "train_df": train_out,
        "val_df": val_out,
        "test_df": test_out,
        "fit_objects": fit_objects,
        "dropped_rows": dropped_rows,
    }