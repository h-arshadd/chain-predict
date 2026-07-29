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


def apply_fitted_preprocessing(
    source_df: pd.DataFrame,
    feature_columns: List[str],
    fit_info: dict,
    transform_fn: Callable,
    params: dict,
    preceding_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Re-apply ONE already-fitted (fit on train only, at training time)
    preprocessing step to some other DataFrame -- val/test during
    run_preprocessing() below, or brand-new live data during inference
    (see apply_saved_preprocessing()). Never re-fits anything; every
    branch here only uses fit_info's already-learned parameters (a
    fitted sklearn scaler object, saved clip bounds, etc.) or, for the
    purely backward-looking methods, no fitted parameters at all.

    Args:
        source_df: the data to transform (val, test, or a live tail).
        feature_columns: same list used at fit time, same order.
        fit_info: one entry's fit_info, as returned by a
            crypto_pipeline.ml.preprocessing.scalers/stationarity
            transform function and persisted in
            {run_dir}/preprocessing.joblib (see model_loader.load_run()'s
            "fit_objects" return value).
        transform_fn: the same registered method (PREPROCESSING_REGISTRY[method])
            used at fit time.
        params: that step's configured params (fractional_differencing's
            d/threshold, etc.) -- needed because the backward-looking
            branches below call transform_fn() again rather than only
            reading fit_info.
        preceding_df: whichever data comes immediately before source_df
            in time -- only used to seed differencing's warm-up window
            (see the weight_length/order branch below); ignored by
            every other branch. For inference, this is simply the extra
            leading history rows already fetched alongside source_df
            (see apply_saved_preprocessing()), not a separate train/val
            split.

    Returns:
        pd.DataFrame, feature_columns only, same index as source_df.
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
    elif "weight_length" in fit_info or "order" in fit_info:
        # Fractional/simple differencing (stationarity.py) -- purely
        # backward-looking, but each call independently computes its own
        # leading warm-up window (weight_length-1 rows for fractional,
        # `order` rows for simple) from whatever DataFrame it's given.
        # Calling it on source_df ALONE would produce a warm-up window
        # with no real history behind it, generating a fresh block of
        # leading NaNs right at the start of source_df. Fixed by
        # prepending preceding_df's tail (enough rows to cover the
        # warm-up window) as real history, transforming that combined
        # frame, then slicing the prepended rows back off -- source_df's
        # own first rows end up computed from real preceding values
        # instead of restarting cold. Nothing here leaks any FITTED
        # (data-driven) train statistic since these methods have none --
        # only raw feature values (already-public market data) are reused.
        warmup = fit_info.get("weight_length", fit_info.get("order", 1) + 1) - 1
        warmup = max(warmup, 0)
        if warmup == 0:
            transformed, _ = transform_fn(source_df[feature_columns], **params)
            return transformed
        history = preceding_df[feature_columns].iloc[-warmup:]
        combined = pd.concat([history, source_df[feature_columns]])
        transformed, _ = transform_fn(combined, **params)
        return transformed.iloc[warmup:].set_axis(source_df.index)
    else:
        # Purely backward-looking/causal methods with no separate
        # warm-up window (row-wise Normalizer, rolling_zscore) have
        # nothing data-driven to leak from train and don't create a
        # leading gap, so they're just re-run directly.
        transformed, _ = transform_fn(source_df[feature_columns], **params)
        return transformed


def apply_saved_preprocessing(
    df: pd.DataFrame,
    feature_columns: List[str],
    fit_objects: List[dict],
) -> pd.DataFrame:
    """
    Apply a run's already-fitted preprocessing chain (as persisted at
    training time and read back by
    crypto_pipeline.ml.persistence.model_loader.load_run()'s
    "fit_objects" return value) to a NEW DataFrame -- the entry point
    run_preprocessing()'s own docstring says doesn't exist yet ("there is
    no separate apply-an-already-fitted-transform-to-new-data entry
    point in this codebase"). This is that entry point, for live
    inference (crypto_pipeline.ml.inference.live_inference): never
    re-fits anything, only replays fit_objects's already-learned
    parameters in the exact order they were originally applied.

    df must already include enough LEADING history before the rows you
    actually want signals for to cover every step's warm-up window (the
    same role preceding_df plays in run_preprocessing() for val/test) --
    e.g. an EMA_50 feature or a fractional-differencing step both need
    real prior bars, not just the newest candle. live_inference.py
    handles fetching that extra lookback; this function only walks the
    preprocessing chain across whatever df it's given, using df itself
    as its own "preceding" history for the warm-up window (there is no
    separate earlier split to draw from here, unlike train->val->test).

    Args:
        df: feature dataframe (feature_columns present, extra leading
            rows for warm-up included), NOT yet preprocessed.
        feature_columns: same list/order the run was trained with
            (model_loader.load_run()'s "feature_columns").
        fit_objects: the run's persisted preprocessing chain, in
            original application order -- each entry
            {"method": str, "fit_info": dict}.

    Returns:
        pd.DataFrame, same shape/index as df, with feature_columns
        replaced by their preprocessed values. Non-feature columns
        (datetime, close, etc.) pass through untouched. Leading rows
        that are still NaN after the chain (not enough real history was
        provided to fully cover every step's warm-up) are left as NaN
        rather than silently dropped -- callers decide what to do with
        an incomplete tail.
    """
    if not fit_objects:
        return df

    out = df
    for entry in fit_objects:
        method_name = entry["method"]
        fit_info = entry["fit_info"]
        if method_name not in PREPROCESSING_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing method '{method_name}' in saved fit_objects. "
                f"Available: {list(PREPROCESSING_REGISTRY.keys())}"
            )
        transform_fn = PREPROCESSING_REGISTRY[method_name]
        # fit_info mixes real transform_fn kwargs (d, threshold, window,
        # norm, ...) with bookkeeping this module added when it was first
        # written (method, note, weight_length, last_values, bound
        # dicts, the fitted sklearn object itself) -- none of the
        # transform_fn implementations in scalers.py/stationarity.py
        # accept **kwargs, so every bookkeeping key has to be stripped
        # before calling transform_fn(**params) again for the
        # backward-looking branches in apply_fitted_preprocessing().
        # order/window/d/threshold/n_quantiles/norm/etc. are exactly the
        # keys each apply_*() function's own signature defines beyond
        # df/fit_mask -- everything else here is metadata, not a param.
        _NON_PARAM_KEYS = {
            "method", "note", "weight_length", "last_values",
            "lower_bounds", "upper_bounds", "_sklearn_object",
        }
        params = {k: v for k, v in fit_info.items() if k not in _NON_PARAM_KEYS}
        transformed = apply_fitted_preprocessing(
            out, feature_columns, fit_info, transform_fn, params, preceding_df=out,
        )
        out = out.drop(columns=feature_columns)
        out = pd.concat([out, transformed], axis=1)[df.columns]

    return out


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

    def _apply_fitted(source_df: pd.DataFrame, fit_info: dict, transform_fn: Callable, params: dict, preceding_df: pd.DataFrame) -> pd.DataFrame:
        return apply_fitted_preprocessing(source_df, feature_columns, fit_info, transform_fn, params, preceding_df)

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
        # train-only fit). preceding_df is whichever split sits
        # immediately before in time, used only to seed differencing's
        # warm-up window (see _apply_fitted) -- test's preceding split
        # is val when val exists (train -> val -> test), else train.
        transformed_val = _apply_fitted(val_out, fit_info, transform_fn, params, preceding_df=train_out) if has_val else None
        test_preceding = val_out if has_val else train_out
        transformed_test = _apply_fitted(test_out, fit_info, transform_fn, params, preceding_df=test_preceding)

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