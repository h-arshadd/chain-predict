# crypto_pipeline/ml/pipeline/train_test_split.py

"""
train_test_split.py
--------------------
Train / Test Split stage (PDF heading 3).

This is TIME SERIES data -- there is no shuffling, no random_state, no
sklearn train_test_split. A single chronological cutoff point is chosen
and everything before it is train, everything from it onward is test.
Splitting any other way (e.g. random row sampling) would let the model
train on rows that come AFTER some of its test rows, which is a direct
lookahead/leakage bug for time series.

config-driven: `split.test_size` in ml/config.yaml controls what fraction
of rows (by time order) are held out for test, taken from the end of the
dataset. Nothing here is hardcoded.

Per the PDF spec, every experiment must record:
    - Training start date
    - Training end date
    - Test start date
    - Test end date
This module returns that alongside the two DataFrames so it can be saved
with the rest of the experiment metadata later.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_dataset(df: pd.DataFrame, ml_config: dict, timestamp_column: str = "datetime") -> dict:
    """
    Chronologically split a time-ordered DataFrame into train/test.

    Args:
        df: dataset from dataset_loader.load_dataset() (already validated
            as sorted ascending by timestamp_column, no duplicates)
        ml_config: ml/config.yaml dict
        timestamp_column: name of the datetime column (from
            feature_selector.select_features()'s resolved config, so this
            stays config-driven rather than hardcoded twice)

    Returns:
        dict with keys:
            train_df: pd.DataFrame, chronologically first portion
            test_df: pd.DataFrame, chronologically last portion
            train_start: pd.Timestamp
            train_end: pd.Timestamp
            test_start: pd.Timestamp
            test_end: pd.Timestamp
            test_size: float, the fraction used (as read from config)
    """

    if timestamp_column not in df.columns:
        raise ValueError(f"timestamp_column '{timestamp_column}' not found in dataset")

    split_config = ml_config.get("split", {})
    test_size = split_config.get("test_size", 0.2)

    if not 0 < test_size < 1:
        raise ValueError(f"split.test_size must be between 0 and 1, got {test_size}")

    n = len(df)
    n_test = int(round(n * test_size))

    if n_test == 0:
        raise ValueError(
            f"test_size={test_size} on {n} rows produces 0 test rows -- increase test_size"
        )
    if n_test == n:
        raise ValueError(
            f"test_size={test_size} on {n} rows produces 0 train rows -- decrease test_size"
        )

    cutoff_idx = n - n_test

    # df is assumed already sorted ascending by timestamp_column (this is
    # enforced upstream in dataset_loader._validate_dataset) -- a plain
    # positional cutoff by row order IS the chronological cutoff here.
    # No re-sort happens in this function: if the caller passes in an
    # unsorted df, that's a bug to fix at the source, not paper over here.
    # No .copy() here -- these slices are only read from downstream
    # (preprocessing_pipeline.run_preprocessing() builds new frames via
    # drop()/concat(), never mutates train_df/test_df in place).
    train_df = df.iloc[:cutoff_idx]
    test_df = df.iloc[cutoff_idx:]

    train_start = train_df[timestamp_column].iloc[0]
    train_end = train_df[timestamp_column].iloc[-1]
    test_start = test_df[timestamp_column].iloc[0]
    test_end = test_df[timestamp_column].iloc[-1]

    logger.info(
        f"Chronological split (test_size={test_size}): "
        f"train {train_start} -> {train_end} ({len(train_df)} rows), "
        f"test {test_start} -> {test_end} ({len(test_df)} rows)"
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "test_size": test_size,
    }