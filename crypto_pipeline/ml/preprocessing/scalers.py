# crypto_pipeline/ml/preprocessing/scalers.py

"""
scalers.py
----------
Scaling methods for the ML module's Preprocessing stage (PDF heading 4).

Production version -- method selection/comparison already happened in
preprocessing_lab, this is the trimmed, non-exploratory implementation
used by the real training pipeline.

Every function follows the same contract:
    transformed_df, fit_info = some_func(df, fit_mask=None, **kwargs)

- df: DataFrame of FEATURE COLUMNS ONLY (never datetime/target)
- fit_mask: boolean mask selecting which ROWS to fit params on. In the
  real pipeline this is ALWAYS the train-set mask -- per the PDF spec,
  "Only pre-processing parameters learned from the training dataset may
  be applied to validation and test datasets" (prevents data leakage).
  fit_mask=None (fit on everything given) only makes sense for ad-hoc/
  exploratory use, never for an actual train/test run.
- fit_info: dict of fitted parameters, persisted alongside the model so
  the exact same transform can be re-applied at inference time.
"""

from typing import Callable, Dict

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
)


def _fit_slice(df: pd.DataFrame, fit_mask) -> pd.DataFrame:
    return df if fit_mask is None else df.loc[fit_mask]


# ---------------------------------------------------------------------
# Required scalers
# ---------------------------------------------------------------------
def apply_standard_scaler(df: pd.DataFrame, fit_mask=None):
    """Z-score scaling: (x - mean) / std, fit on train rows only."""
    fit_df = _fit_slice(df, fit_mask)
    scaler = StandardScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "standard_scaler",
        "mean": dict(zip(df.columns, scaler.mean_)),
        "std": dict(zip(df.columns, scaler.scale_)),
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_standard_scaler(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    mean = pd.Series(fit_info["mean"])[transformed_df.columns]
    std = pd.Series(fit_info["std"])[transformed_df.columns]
    return transformed_df * std + mean


def apply_minmax_scaler(df: pd.DataFrame, fit_mask=None, feature_range=(0, 1)):
    """Scales each column to feature_range, fit on train rows only."""
    fit_df = _fit_slice(df, fit_mask)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "minmax_scaler",
        "min": dict(zip(df.columns, scaler.data_min_)),
        "max": dict(zip(df.columns, scaler.data_max_)),
        "feature_range": feature_range,
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_minmax_scaler(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    lo, hi = fit_info["feature_range"]
    dmin = pd.Series(fit_info["min"])[transformed_df.columns]
    dmax = pd.Series(fit_info["max"])[transformed_df.columns]
    scale = (dmax - dmin) / (hi - lo)
    return (transformed_df - lo) * scale + dmin


def apply_robust_scaler(df: pd.DataFrame, fit_mask=None):
    """Scales using median/IQR (outlier-robust), fit on train rows only."""
    fit_df = _fit_slice(df, fit_mask)
    scaler = RobustScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "robust_scaler",
        "median": dict(zip(df.columns, scaler.center_)),
        "iqr": dict(zip(df.columns, scaler.scale_)),
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_robust_scaler(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    median = pd.Series(fit_info["median"])[transformed_df.columns]
    iqr = pd.Series(fit_info["iqr"])[transformed_df.columns]
    return transformed_df * iqr + median


def apply_maxabs_scaler(df: pd.DataFrame, fit_mask=None):
    """Scales by max absolute value, preserving sign/zero. Fit on train only."""
    fit_df = _fit_slice(df, fit_mask)
    scaler = MaxAbsScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "maxabs_scaler",
        "max_abs": dict(zip(df.columns, scaler.max_abs_)),
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_maxabs_scaler(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    max_abs = pd.Series(fit_info["max_abs"])[transformed_df.columns]
    return transformed_df * max_abs


def apply_quantile_transformer(df: pd.DataFrame, fit_mask=None, n_quantiles=1000,
                                output_distribution="uniform", random_state=42):
    """Maps each column to output_distribution via empirical CDF, fit on train only."""
    fit_df = _fit_slice(df, fit_mask)
    n_q = min(n_quantiles, len(fit_df))
    scaler = QuantileTransformer(
        n_quantiles=n_q,
        output_distribution=output_distribution,
        random_state=random_state,
    )
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "quantile_transformer",
        "n_quantiles": n_q,
        "output_distribution": output_distribution,
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_quantile_transformer(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    scaler = fit_info["_sklearn_object"]
    inv = scaler.inverse_transform(transformed_df.values)
    return pd.DataFrame(inv, columns=transformed_df.columns, index=transformed_df.index)


# ---------------------------------------------------------------------
# Extra scalers (kept from preprocessing_lab -- config-selectable, not
# required by the PDF, but proven useful in the lab's crypto-specific
# testing; remove from the registry below if they end up unused)
# ---------------------------------------------------------------------
def apply_power_transformer(df: pd.DataFrame, fit_mask=None, method="yeo-johnson"):
    """Learned power transform (Yeo-Johnson) to reduce skew, fit on train only."""
    fit_df = _fit_slice(df, fit_mask)
    scaler = PowerTransformer(method=method, standardize=True)
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "power_transformer",
        "pt_method": method,
        "lambdas": dict(zip(df.columns, scaler.lambdas_)),
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_power_transformer(transformed_df: pd.DataFrame, fit_info: dict) -> pd.DataFrame:
    scaler = fit_info["_sklearn_object"]
    inv = scaler.inverse_transform(transformed_df.values)
    return pd.DataFrame(inv, columns=transformed_df.columns, index=transformed_df.index)


def apply_normalizer(df: pd.DataFrame, fit_mask=None, norm="l2"):
    """
    Row-wise unit-norm scaling. Stateless per row (no fit_mask needed, no
    leakage possible by construction) -- fit_mask accepted for interface
    consistency but unused.
    """
    normalizer = Normalizer(norm=norm)
    transformed = normalizer.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {"method": "normalizer", "norm": norm, "note": "row-wise, not reversible, no fit_mask used"}
    return out, fit_info


def apply_rolling_zscore(df: pd.DataFrame, fit_mask=None, window=168):
    """
    Z-score over a rolling window. NOTE: fit_mask is unused here by
    design -- this is a purely backward-looking, causal transform (each
    point only uses its own past `window` rows), so there's no separate
    "fit on train" step the way static scalers need. Applying it to
    df=train_df and df=test_df separately is fine as long as test_df's
    rolling windows only look at rows genuinely before them in time
    (true if the caller passes contiguous, correctly-ordered data).
    """
    rolling_mean = df.rolling(window=window, min_periods=window).mean()
    rolling_std = df.rolling(window=window, min_periods=window).std()
    out = (df - rolling_mean) / rolling_std
    fit_info = {
        "method": "rolling_zscore",
        "window": window,
        "note": f"first {window} rows are NaN by construction (not enough history yet)",
    }
    return out, fit_info


PREPROCESSING_SCALERS: Dict[str, Callable] = {
    "standard_scaler": apply_standard_scaler,
    "minmax_scaler": apply_minmax_scaler,
    "robust_scaler": apply_robust_scaler,
    "maxabs_scaler": apply_maxabs_scaler,
    "quantile_transformer": apply_quantile_transformer,
    "power_transformer": apply_power_transformer,
    "normalizer": apply_normalizer,
    "rolling_zscore": apply_rolling_zscore,
}