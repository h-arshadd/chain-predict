# preprocessing_lab/run_experiment.py

"""
run_experiment.py
-----------------
Entry point for running preprocessing methods end-to-end:
    1. reads preprocessing_lab/config.yaml (which methods + which target
       types + which columns -- methods and target_types are lists, so
       you can run one or run all of them in one go)
    2. for each target_type, calls ml_module.run_ml_pipeline() once to get
       a fresh, live dataset with that target attached (reused across all
       methods for that target_type, not refetched per method)
    3. for each method, looks it up in the registry and applies it to
       feature columns only (never datetime, never target)
    4. saves to outputs/<method_name>/<target_type>/transformed.csv

FIXES APPLIED:
- Added dropna logic for methods that create leading NaN rows (fractional/simple differencing)
- Logs how many rows were dropped
"""

import os
import json
import yaml
import datetime as dt
import inspect
import pandas as pd

from crypto_pipeline.ml_module.main import run_ml_pipeline
from crypto_pipeline.ml_module.target_pipeline import generate_target
from crypto_pipeline.preprocessing_lab.registry import PREPROCESSING_REGISTRY


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_clean_dataset(ml_config_path: str, target_type: str) -> pd.DataFrame:
    """
    Calls ml_module.run_ml_pipeline() to get features + OHLCV, then
    (re)generates the target using target_type (regression/classification)
    instead of whatever model_type is written in ml_module/config.yaml.
    This lets preprocessing_lab produce either target on demand.

    Also applies the SAME NaN-dropping logic that ml_module/main.py's
    __main__ block applies -- required here because run_ml_pipeline()
    as an importable function does NOT drop NaNs itself.
    """
    ml_config = load_config(ml_config_path)
    ml_config["model_type"] = target_type

    df = run_ml_pipeline(ml_config_path)

    # run_ml_pipeline already generated a target using ml_config's own
    # model_type -- drop it and regenerate with the requested target_type.
    df = df.drop(columns=["target"], errors="ignore")
    df = generate_target(df, ml_config)

    # sentiment columns (sen_*) are allowed to be NaN -- a missing/no-post
    # sentiment value for an hour shouldn't discard an otherwise valid
    # OHLCV + feature row. Every other column must be non-NaN.
    sentiment_cols = [c for c in df.columns if c.startswith("sen_")]
    required_cols = [c for c in df.columns if c not in sentiment_cols]
    df = df.dropna(subset=required_cols)

    return df


def run_one(config: dict, base_dir: str, df: pd.DataFrame, method_name: str, target_type: str) -> pd.DataFrame:
    """Apply one preprocessing method to an already-fetched dataset and save it."""

    initial_rows = len(df)

    # pick method from registry
    if method_name not in PREPROCESSING_REGISTRY:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {list(PREPROCESSING_REGISTRY.keys())}"
        )
    transform_fn = PREPROCESSING_REGISTRY[method_name]

    # apply it only to feature columns (never datetime, never target).
    # some methods can't handle every column (e.g. log_transform needs
    # positive values, MACD columns are signed) -- check for a per-method
    # override before falling back to the default feature_columns list.
    method_overrides = config.get("method_feature_columns", {}) or {}
    feature_cols = method_overrides.get(method_name, config["data"]["feature_columns"])
    all_params = config.get("params", {}) or {}

    # Filter params to only include those accepted by this specific method
    # Different methods accept different parameters
    sig = inspect.signature(transform_fn)
    valid_params = {k: v for k, v in all_params.items() if k in sig.parameters}

    # YAML loads [0, 1] as a list, but sklearn's MinMaxScaler requires a
    # tuple for feature_range -- coerce it here rather than in scalers.py
    if "feature_range" in valid_params:
        valid_params["feature_range"] = tuple(valid_params["feature_range"])

    transformed_features, fit_info = transform_fn(df[feature_cols], **valid_params)

    # reassemble: datetime + transformed features + target, untouched
    result = df.drop(columns=feature_cols).copy()
    result = pd.concat([result, transformed_features], axis=1)
    result = result[df.columns]

    # DROP ROWS WITH NaN IN FEATURE COLUMNS (from differencing/rolling methods)
    # This is important: fractional/simple differencing create leading NaN rows
    rows_before_drop = len(result)
    result = result.dropna(subset=feature_cols)
    rows_after_drop = len(result)
    dropped_rows = rows_before_drop - rows_after_drop

    if dropped_rows > 0:
        print(f"⚠️  Dropped {dropped_rows} rows with NaN features ({dropped_rows/initial_rows*100:.1f}% of original)")
        print(f"   ({fit_info.get('note', 'N/A')})")
        print(f"   Remaining rows: {rows_after_drop} (from {initial_rows} original)")

    # save under outputs/<method_name>/<target_type>/ so regression and
    # classification runs of the same method never overwrite each other
    out_dir = os.path.join(base_dir, config["output"]["dir"], method_name, target_type)
    os.makedirs(out_dir, exist_ok=True)

    data_out_path = os.path.join(out_dir, "transformed.csv")
    result.to_csv(data_out_path, index=False)

    # save fit_info too (minus any non-serializable sklearn object) so
    # every run's exact parameters are recorded, not just the data
    printable_fit_info = {k: v for k, v in fit_info.items() if k != "_sklearn_object"}
    fit_info_path = os.path.join(out_dir, "fit_info.json")
    with open(fit_info_path, "w") as f:
        json.dump(printable_fit_info, f, indent=2, default=str)

    print(f"\n✅ Method: {method_name} | Target type: {target_type}")
    print(f"Fit info: {printable_fit_info}")
    print(f"Saved transformed dataset to: {data_out_path}")
    print(f"Saved fit info to: {fit_info_path}")
    print(f"Final shape: {result.shape}")

    return result


def run(config_path: str) -> dict:
    """
    Runs every method in config['methods'] against every target_type in
    config['target_types']. Both accept either a list or a single string
    (so config.yaml can list many methods, or just one).

    Returns dict keyed by (method_name, target_type) -> transformed df.
    """
    base_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    methods = config["methods"]
    if isinstance(methods, str):
        methods = [methods]

    target_types = config.get("target_types", ["regression"])
    if isinstance(target_types, str):
        target_types = [target_types]

    ml_config_path = os.path.join(base_dir, "..", "ml_module", "config.yaml")

    results = {}
    saved_paths = []
    for target_type in target_types:
        # fetch once per target_type, reuse across all methods for that target
        df = get_clean_dataset(ml_config_path, target_type)

        for method_name in methods:
            result = run_one(config, base_dir, df, method_name, target_type)
            results[(method_name, target_type)] = result
            saved_paths.append(
                os.path.join(base_dir, config["output"]["dir"], method_name, target_type, "transformed.csv")
            )

    print(f"\n{'='*60}")
    print(f"Done. {len(methods)} method(s) x {len(target_types)} target_type(s) = {len(results)} run(s)")
    for p in saved_paths:
        print(f"  {p}")

    return results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "config.yaml")
    run(config_path)