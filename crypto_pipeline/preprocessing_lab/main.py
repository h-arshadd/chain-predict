# preprocessing_lab/run_experiment.py

"""
run_experiment.py
-----------------
Entry point for running ONE preprocessing method end-to-end:
    1. reads preprocessing_lab/config.yaml (which method + which columns)
    2. calls ml_module.run_ml_pipeline() to get a fresh, live dataset
       (same NaN-cleaning logic ml_module/main.py normally applies when
       run standalone -- replicated here since run_ml_pipeline() itself
       does NOT drop NaNs, only ml_module's __main__ block does)
    3. looks up the chosen method in the registry
    4. applies it to feature columns only (never datetime, never target)
    5. saves to outputs/<method_name>/<timestamp>/transformed.csv
       -- timestamped so re-running the same method never overwrites a
       previous run's results. Every experiment run is kept.

Rename note: this file used to be called main.py. Renamed to
run_experiment.py because "main" is meaningless once there are multiple
entry points in this project (this one runs a single preprocessing
experiment; later we'll add separate entry points for stationarity
analysis, model training, and backtesting).
"""

import os
import json
import yaml
import datetime as dt
import pandas as pd

from crypto_pipeline.ml_module.main import run_ml_pipeline
from crypto_pipeline.preprocessing_lab.registry import PREPROCESSING_REGISTRY


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_clean_dataset(ml_config_path: str) -> pd.DataFrame:
    """
    Calls ml_module.run_ml_pipeline() and applies the SAME NaN-dropping
    logic that ml_module/main.py's __main__ block applies -- required
    here because run_ml_pipeline() as an importable function does NOT
    drop NaNs itself; that cleanup only happens in ml_module's own
    script-only block, which we never trigger when importing the
    function directly.
    """
    df = run_ml_pipeline(ml_config_path)

    # sentiment columns (sen_*) are allowed to be NaN -- a missing/no-post
    # sentiment value for an hour shouldn't discard an otherwise valid
    # OHLCV + feature row. Every other column must be non-NaN.
    sentiment_cols = [c for c in df.columns if c.startswith("sen_")]
    required_cols = [c for c in df.columns if c not in sentiment_cols]
    df = df.dropna(subset=required_cols)

    return df


def run(config_path: str) -> pd.DataFrame:
    base_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    # 1. get a clean, live dataset from ml_module
    ml_config_path = os.path.join(base_dir, "..", "ml_module", "config.yaml")
    df = get_clean_dataset(ml_config_path)

    # 2. pick method from registry
    method_name = config["method"]
    if method_name not in PREPROCESSING_REGISTRY:
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Available: {list(PREPROCESSING_REGISTRY.keys())}"
        )
    transform_fn = PREPROCESSING_REGISTRY[method_name]

    # 3. apply it only to feature columns (never datetime, never target)
    feature_cols = config["data"]["feature_columns"]
    params = config.get("params", {}) or {}

    transformed_features, fit_info = transform_fn(df[feature_cols], **params)

    # 4. reassemble: datetime + transformed features + target, untouched
    result = df.drop(columns=feature_cols).copy()
    result = pd.concat([result, transformed_features], axis=1)
    result = result[df.columns]

    # 5. save to a TIMESTAMPED folder -- never overwrites a previous run
    out_dir = os.path.join(base_dir, config["output"]["dir"], method_name)
    os.makedirs(out_dir, exist_ok=True)

    data_out_path = os.path.join(out_dir, "transformed.csv")
    result.to_csv(data_out_path, index=False)

    # save fit_info too (minus any non-serializable sklearn object) so
    # every run's exact parameters are recorded, not just the data
    printable_fit_info = {k: v for k, v in fit_info.items() if k != "_sklearn_object"}
    fit_info_path = os.path.join(out_dir, "fit_info.json")
    with open(fit_info_path, "w") as f:
        json.dump(printable_fit_info, f, indent=2, default=str)

    print(f"Method: {method_name}")
    print(f"Fit info: {printable_fit_info}")
    print(f"Saved transformed dataset to: {data_out_path}")
    print(f"Saved fit info to: {fit_info_path}")
    print(f"Shape: {result.shape}")

    return result


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "config.yaml")
    run(config_path)