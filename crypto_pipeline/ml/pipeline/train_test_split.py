# crypto_pipeline/ml/pipeline/train_test_split.py

"""
train_test_split.py
--------------------
Train / Test (/ Validation) Split stage (PDF heading 3).

This is TIME SERIES data -- there is no shuffling, no random_state, no
sklearn train_test_split. Chronological cutoff points are chosen and the
dataset is sliced in time order: train (earliest) -> validation (middle,
optional) -> test (latest). Splitting any other way (e.g. random row
sampling) would let the model train on rows that come AFTER some of its
validation/test rows, which is a direct lookahead/leakage bug for time
series. The validation slice sits chronologically BEFORE test (never
between train and "the future") for the same reason: it must be usable
for early stopping / model selection without ever peeking at data more
recent than what test will evaluate on.

config-driven: `split.test_size` in ml/config.yaml controls what fraction
of rows (by time order) are held out for test, taken from the end of the
dataset. `split.val_size` is optional and controls a validation slice
taken immediately before the test slice, out of the same remaining rows
-- if it's 0/absent, no validation split is produced and the pipeline
behaves exactly as before (train/test only). Nothing here is hardcoded.

Per the PDF spec, every experiment must record:
    - Training start date
    - Training end date
    - Test start date
    - Test end date
(and, when a validation split exists, its own start/end dates). This
module returns that alongside the DataFrames so it can be saved with the
rest of the experiment metadata later.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_dataset(df: pd.DataFrame, ml_config: dict, timestamp_column: str = "datetime") -> dict:
    """
    Chronologically split a time-ordered DataFrame into train/(val)/test.

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
            val_df: pd.DataFrame or None, chronologically middle portion
                (only present if split.val_size > 0 in config)
            test_df: pd.DataFrame, chronologically last portion
            train_start / train_end: pd.Timestamp
            val_start / val_end: pd.Timestamp or None
            test_start / test_end: pd.Timestamp
            test_size: float, the fraction used (as read from config)
            val_size: float, the fraction used (0.0 if no validation split)
    """

    if timestamp_column not in df.columns:
        raise ValueError(f"timestamp_column '{timestamp_column}' not found in dataset")

    split_config = ml_config.get("split", {})
    test_size = split_config.get("test_size", 0.2)
    val_size = split_config.get("val_size", 0.0) or 0.0

    if not 0 < test_size < 1:
        raise ValueError(f"split.test_size must be between 0 and 1, got {test_size}")
    if not 0 <= val_size < 1:
        raise ValueError(f"split.val_size must be between 0 (disabled) and 1, got {val_size}")
    if val_size > 0 and test_size + val_size >= 1:
        raise ValueError(
            f"split.test_size ({test_size}) + split.val_size ({val_size}) must be < 1 "
            f"so at least some rows are left for training -- they currently add up to "
            f"{test_size + val_size}."
        )

    # train_size is optional and purely a readable label in config -- it
    # does not drive the split (test_size/val_size alone do, train =
    # whatever's left). If it's set, it must actually add up with
    # test_size (+ val_size) so the config doesn't silently say something
    # untrue (e.g. test_size=0.2, train_size=0.9 would be internally
    # inconsistent).
    train_size = split_config.get("train_size")
    if train_size is not None:
        if not 0 < train_size < 1:
            raise ValueError(f"split.train_size must be between 0 and 1, got {train_size}")
        if abs((train_size + test_size + val_size) - 1.0) > 1e-6:
            raise ValueError(
                f"split.train_size ({train_size}) + split.val_size ({val_size}) + "
                f"split.test_size ({test_size}) must add up to 1.0 -- they currently "
                f"add up to {train_size + val_size + test_size}. train_size is just a "
                f"label (test_size/val_size are what actually drive the split), but it "
                f"must be consistent with them or remove it from config."
            )

    n = len(df)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size)) if val_size > 0 else 0

    if n_test == 0:
        raise ValueError(
            f"test_size={test_size} on {n} rows produces 0 test rows -- increase test_size"
        )
    if val_size > 0 and n_val == 0:
        raise ValueError(
            f"val_size={val_size} on {n} rows produces 0 validation rows -- increase "
            f"val_size or disable it (set split.val_size to 0)"
        )
    if n_test + n_val >= n:
        raise ValueError(
            f"test_size={test_size} + val_size={val_size} on {n} rows leaves 0 train "
            f"rows -- decrease test_size and/or val_size"
        )

    test_cutoff_idx = n - n_test
    val_cutoff_idx = test_cutoff_idx - n_val

    # df is assumed already sorted ascending by timestamp_column (this is
    # enforced upstream in dataset_loader._validate_dataset) -- a plain
    # positional cutoff by row order IS the chronological cutoff here.
    # No re-sort happens in this function: if the caller passes in an
    # unsorted df, that's a bug to fix at the source, not paper over here.
    # No .copy() here -- these slices are only read from downstream
    # (preprocessing_pipeline.run_preprocessing() builds new frames via
    # drop()/concat(), never mutates train_df/val_df/test_df in place).
    # Order is train -> val -> test, all strictly non-overlapping and in
    # time order, so val never leaks into (or lags behind) test.
    train_df = df.iloc[:val_cutoff_idx]
    val_df = df.iloc[val_cutoff_idx:test_cutoff_idx] if n_val > 0 else None
    test_df = df.iloc[test_cutoff_idx:]

    train_start = train_df[timestamp_column].iloc[0]
    train_end = train_df[timestamp_column].iloc[-1]
    test_start = test_df[timestamp_column].iloc[0]
    test_end = test_df[timestamp_column].iloc[-1]

    if val_df is not None:
        val_start = val_df[timestamp_column].iloc[0]
        val_end = val_df[timestamp_column].iloc[-1]
        logger.info(
            f"Chronological split (test_size={test_size}, val_size={val_size}): "
            f"train {train_start} -> {train_end} ({len(train_df)} rows), "
            f"val {val_start} -> {val_end} ({len(val_df)} rows), "
            f"test {test_start} -> {test_end} ({len(test_df)} rows)"
        )
    else:
        val_start = None
        val_end = None
        logger.info(
            f"Chronological split (test_size={test_size}, no validation split): "
            f"train {train_start} -> {train_end} ({len(train_df)} rows), "
            f"test {test_start} -> {test_end} ({len(test_df)} rows)"
        )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "test_size": test_size,
        "val_size": val_size,
        "train_size": train_size if train_size is not None else round(1.0 - test_size - val_size, 10),
    }