# crypto_pipeline/ml/preprocessing/feature_selector.py

"""
feature_selector.py
--------------------
Feature Selection stage (PDF heading 2).

Feature columns, target column, and timestamp column are all defined
through ml/config.yaml -- never hardcoded here. Prediction horizon is
NOT set separately: it comes from ml_config["target"]["horizon"], the
same value target_pipeline.py used to generate the target in the first
place, so there is exactly one place horizon is ever set.

Two ways to select feature columns in config:
  1. Explicit list: features.feature_columns: [ind_RSI_14, ind_EMA_20, ...]
  2. Prefix-based:   features.include_prefixes: [ind_, pat_, sen_]
If feature_columns is given, it wins and is used exactly as listed (order
preserved). Otherwise include_prefixes is resolved against the df's
actual columns at runtime, sorted for a stable, reproducible order.

Either way, features.feature_columns_extra (if set) is appended AFTER
whichever of the above resolved -- e.g. for raw OHLCV columns like
close/volume that don't share the ind_/pat_/sen_ prefixes and so would
never be picked up by include_prefixes, and that you also wouldn't want
to have to spell out by hand alongside every entry in an explicit
feature_columns list.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def select_features(df: pd.DataFrame, ml_config: dict) -> dict:
    """
    Resolve feature columns, target column, timestamp column, and
    prediction horizon against the actual dataset.

    Args:
        df: dataset from dataset_loader.load_dataset()
        ml_config: ml/config.yaml dict

    Returns:
        dict with keys:
            feature_columns: list[str], order preserved
            target_column: str
            timestamp_column: str
            horizon: int
    """

    features_config = ml_config.get("features", {})

    timestamp_column = features_config.get("timestamp_column", "datetime")
    target_column = features_config.get("target_column", "target")
    horizon = ml_config.get("target", {}).get("horizon")

    if horizon is None:
        raise ValueError("target.horizon must be set in ml/config.yaml")

    if timestamp_column not in df.columns:
        raise ValueError(f"timestamp_column '{timestamp_column}' not found in dataset")

    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset")

    feature_columns = features_config.get("feature_columns")

    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"feature_columns not found in dataset: {missing}")
        logger.info(f"Using explicit feature_columns from config: {len(feature_columns)} columns")
    else:
        include_prefixes = features_config.get("include_prefixes")
        if not include_prefixes:
            raise ValueError(
                "ml/config.yaml features section must set either "
                "'feature_columns' (explicit list) or 'include_prefixes' (e.g. [ind_, pat_, sen_])"
            )
        excluded = {timestamp_column, target_column}
        feature_columns = sorted(
            col for col in df.columns
            if col not in excluded and any(col.startswith(p) for p in include_prefixes)
        )
        logger.info(f"Resolved feature_columns from prefixes {include_prefixes}: {len(feature_columns)} columns")

    if not feature_columns:
        raise ValueError("No feature columns resolved -- check config's feature_columns/include_prefixes")

    feature_columns_extra = features_config.get("feature_columns_extra") or []
    if feature_columns_extra:
        missing_extra = [c for c in feature_columns_extra if c not in df.columns]
        if missing_extra:
            raise ValueError(f"feature_columns_extra not found in dataset: {missing_extra}")

        # Preserve order, skip anything already present (e.g. an explicit
        # feature_columns list that already names one of these columns)
        # rather than duplicating it.
        already_selected = set(feature_columns)
        new_extras = [c for c in feature_columns_extra if c not in already_selected]
        feature_columns = list(feature_columns) + new_extras
        logger.info(f"Appended feature_columns_extra: {new_extras}")

    logger.info(f"Feature selection: {len(feature_columns)} features, target='{target_column}', horizon={horizon}")

    return {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "timestamp_column": timestamp_column,
        "horizon": horizon,
    }