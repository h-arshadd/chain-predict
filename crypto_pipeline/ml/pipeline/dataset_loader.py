# crypto_pipeline/ml/pipeline/dataset_loader.py

"""
dataset_loader.py
------------------
Dataset Loading stage (PDF heading 1).

Does NOT fetch from Postgres itself --
crypto_pipeline.ml.data_prep.main.run_ml_pipeline() already does that
(OHLCV + features + sentiment + target). This module just takes that
output, validates it, and optionally writes a debug CSV for manual
inspection (never read back in by any pipeline step).

exchange / symbol / model_type are read from ml/data_prep/config.yaml,
not duplicated in ml/config.yaml, so there is one source of truth.
"""

import logging
import os

import pandas as pd
import yaml

from crypto_pipeline.ml.data_prep.main import run_ml_pipeline

logger = logging.getLogger(__name__)


def load_dataset(ml_config_path: str, data_prep_config_path: str) -> pd.DataFrame:
    """
    Run the data_prep pipeline and validate the resulting dataset.

    Args:
        ml_config_path: path to ml/config.yaml (output/debug settings)
        data_prep_config_path: path to ml/data_prep/config.yaml (passed
            through to run_ml_pipeline, and read for exchange/symbol/model_type)

    Returns:
        pd.DataFrame: validated dataset, ready for feature selection.
    """

    ml_config = _load_yaml(ml_config_path)
    data_prep_config = _load_yaml(data_prep_config_path)

    df = run_ml_pipeline(data_prep_config_path)

    _validate_dataset(df)

    output_config = ml_config.get("output", {})
    if output_config.get("save_debug_csv", False):
        _save_debug_csv(df, data_prep_config, output_config)

    logger.info(f"Dataset loaded and validated: {df.shape}")
    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset before it moves further down the pipeline.
    Fails loudly -- does not silently fix or drop anything here.

    Covers the PDF's 5 required checks:
        - Missing values
        - Duplicate timestamps
        - Timestamp ordering
        - Feature consistency
        - Target availability

    Note on missing values: data_prep's own DB/collection layer is
    responsible for filling OHLCV gaps (ffill/bfill) before this pipeline
    ever sees the data. This function does NOT fill anything -- it only
    verifies that promise held. sen_* columns are the one expected
    exception (a missing sentiment reading for an hour is valid, per
    data_prep/main.py's own dropna logic), everything else must be clean
    by the time it reaches here.
    """

    if df.empty:
        raise ValueError("Dataset is empty")

    if "datetime" not in df.columns:
        raise ValueError("Dataset must have a 'datetime' column")

    if "target" not in df.columns:
        raise ValueError("Dataset must have a 'target' column")

    if df["datetime"].duplicated().any():
        n_dupes = df["datetime"].duplicated().sum()
        raise ValueError(f"Dataset has {n_dupes} duplicate timestamps")

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Dataset timestamps are not sorted ascending")

    if df["target"].isna().any():
        n_nan = df["target"].isna().sum()
        raise ValueError(
            f"Dataset has {n_nan} NaN target rows. "
            f"target_pipeline.generate_target() should have dropped these already."
        )

    fully_empty_cols = [col for col in df.columns if df[col].isna().all()]
    if fully_empty_cols:
        raise ValueError(f"Dataset has fully-empty columns: {fully_empty_cols}")

    # --- Feature consistency -------------------------------------------------
    # No duplicate column names (e.g. a bad merge/concat silently doubling a
    # column), and no unnamed/blank column labels (e.g. an unintended index
    # column leaking in from a CSV round-trip).
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {dupes}")

    blank_cols = [c for c in df.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    if blank_cols:
        raise ValueError(f"Dataset has unnamed/blank column labels: {blank_cols}")

    # Every column except datetime must be numeric -- catches a bad parse or
    # merge upstream (e.g. a price column that accidentally came through as
    # strings) before it silently breaks preprocessing/model training later.
    non_numeric_cols = [
        c for c in df.columns
        if c != "datetime" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric_cols:
        raise ValueError(f"Dataset has non-numeric feature/target columns: {non_numeric_cols}")

    # --- Missing values -------------------------------------------------------
    # sen_* columns are allowed NaN (no post that hour is a valid state).
    # Every other column (OHLCV, indicators, patterns, target) must be
    # fully populated by the time it reaches the ML module.
    sentiment_cols = [c for c in df.columns if c.startswith("sen_")]
    required_cols = [c for c in df.columns if c not in sentiment_cols]
    cols_with_nans = [c for c in required_cols if df[c].isna().any()]
    if cols_with_nans:
        nan_counts = {c: int(df[c].isna().sum()) for c in cols_with_nans}
        raise ValueError(
            f"Dataset has missing values in required (non-sentiment) columns: "
            f"{nan_counts}. These should already be filled upstream in data_prep "
            f"(e.g. ffill/bfill at collection time) -- this indicates an upstream "
            f"regression, not something to silently patch here."
        )

    logger.info("Dataset validation passed")


def _save_debug_csv(df: pd.DataFrame, data_prep_config: dict, output_config: dict) -> None:
    """
    Save a debug-only CSV for manual inspection.
    Path: {base_dir}/{exchange}/{symbol}/{model_type}/dataset_{model_type}.csv
    """

    exchange = data_prep_config["data"]["exchange"]
    symbol = data_prep_config["data"]["symbol"]
    model_type = data_prep_config["model_type"]

    base_dir = output_config.get("base_dir", "outputs")
    out_dir = os.path.join(base_dir, exchange, symbol, model_type)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"dataset_{model_type}.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Debug CSV saved to {out_path}")


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)