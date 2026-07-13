# preprocessing_lab/preprocessing/scalers.py

"""
scalers.py
----------
Column-wise scaling / distribution-reshaping methods.

Every function follows the same contract:
    transformed_df, fit_info = some_func(df, fit_mask=None, **kwargs)

- df: DataFrame of FEATURE COLUMNS ONLY (never datetime/target)
- fit_mask: optional boolean mask selecting which ROWS to fit params on
            (e.g. training rows only). None = fit on all rows given
            (fine for single-method demos; mandatory to set once you do
            train/val/test splits, or you leak future info into the fit).
- fit_info: dict of fitted parameters, so the exact same transform can be
            re-applied to new data later without re-fitting, and so it
            can be reversed where the method is reversible.
"""

import numpy as np
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


def _fit_slice(df, fit_mask):
    return df if fit_mask is None else df.loc[fit_mask]


# ---------------------------------------------------------------------
# 1. StandardScaler (Z-score)
# ---------------------------------------------------------------------
def apply_standard_scaler(df: pd.DataFrame, fit_mask=None):
    """
    Z-score scaling: (x - mean) / std, per column.

    Purpose: puts every feature on the same scale (mean 0, std 1) so
    features with large natural magnitude (volume, in thousands) don't
    dominate features with small magnitude (RSI, 0-100) purely due to
    units.

    Advantages:
        - Simple, standard, works well when data is roughly Gaussian.
        - Helps gradient-based / distance-based models train better.

    Disadvantages:
        - Very sensitive to outliers -- a single price spike blows out
          mean/std and squashes normal values near 0. Crypto has this
          problem constantly.
        - Does not fix non-stationarity -- if the mean drifts over the
          year (which BTC price does hugely), the scaler's fitted mean
          becomes stale for later data.

    Suitable models: Linear Regression, Logistic Regression, distance /
    gradient-based models. No effect on tree models (XGBoost/LightGBM)
    since they split on thresholds, not distances.

    Preserves trend: YES (linear transform, shape unchanged).
    Improves stationarity: NO (only rescales, doesn't detrend).
    Reversible: YES exactly -- x = x_scaled * std + mean.
    """
    fit_df = _fit_slice(df, fit_mask)
    scaler = StandardScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "standard_scaler",
        "mean": dict(zip(df.columns, scaler.mean_)),
        "std": dict(zip(df.columns, scaler.scale_)),
    }
    return out, fit_info


def inverse_standard_scaler(transformed_df, fit_info):
    mean = pd.Series(fit_info["mean"])[transformed_df.columns]
    std = pd.Series(fit_info["std"])[transformed_df.columns]
    return transformed_df * std + mean


# ---------------------------------------------------------------------
# 2. MinMaxScaler
# ---------------------------------------------------------------------
def apply_minmax_scaler(df: pd.DataFrame, fit_mask=None, feature_range=(0, 1)):
    """
    Scales each column to a fixed range (default [0, 1]):
        x_scaled = (x - min) / (max - min)

    Purpose: bound every feature into a known fixed range. Useful when a
    model or algorithm expects/benefits from bounded inputs.

    Advantages:
        - Guarantees bounded output range, easy to interpret.
        - Preserves the exact shape of the original distribution (no
          reshaping, just a linear squeeze).

    Disadvantages:
        - EXTREMELY sensitive to outliers -- one huge spike sets the max,
          crushing all other values toward 0. Worse than StandardScaler
          for this. Crypto flash-spikes make this risky on raw OHLCV.
        - A single new extreme value outside the fitted [min,max] at
          inference time will produce values outside [0,1] (or you clip
          and lose information).
        - Does not fix non-stationarity.

    Suitable models: Neural nets (bounded activations), KNN, anything
    distance-based needing bounded ranges. Not needed for tree models.

    Preserves trend: YES (linear transform).
    Improves stationarity: NO.
    Reversible: YES exactly -- x = x_scaled * (max-min) + min.
    """
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
    }
    return out, fit_info


def inverse_minmax_scaler(transformed_df, fit_info):
    lo, hi = fit_info["feature_range"]
    dmin = pd.Series(fit_info["min"])[transformed_df.columns]
    dmax = pd.Series(fit_info["max"])[transformed_df.columns]
    scale = (dmax - dmin) / (hi - lo)
    return (transformed_df - lo) * scale + dmin


# ---------------------------------------------------------------------
# 3. RobustScaler
# ---------------------------------------------------------------------
def apply_robust_scaler(df: pd.DataFrame, fit_mask=None):
    """
    Scales using median and IQR instead of mean/std:
        x_scaled = (x - median) / IQR

    Purpose: same goal as StandardScaler (put features on comparable
    scale) but using statistics that ignore outliers.

    Advantages:
        - Much more robust to outliers than Standard/MinMax, since median
          and IQR (25th-75th percentile) aren't dragged around by a few
          extreme spikes. Well suited to crypto's spiky OHLCV data.

    Disadvantages:
        - Output isn't bounded to a fixed range like MinMax.
        - Still doesn't fix non-stationarity (only outlier-robust scaling
          of the CURRENT distribution shape, not detrending).

    Suitable models: Linear/Logistic Regression, any model sensitive to
    feature scale, especially when data is known to have outliers.

    Preserves trend: YES (linear transform).
    Improves stationarity: NO.
    Reversible: YES exactly -- x = x_scaled * IQR + median.
    """
    fit_df = _fit_slice(df, fit_mask)
    scaler = RobustScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "robust_scaler",
        "median": dict(zip(df.columns, scaler.center_)),
        "iqr": dict(zip(df.columns, scaler.scale_)),
    }
    return out, fit_info


def inverse_robust_scaler(transformed_df, fit_info):
    median = pd.Series(fit_info["median"])[transformed_df.columns]
    iqr = pd.Series(fit_info["iqr"])[transformed_df.columns]
    return transformed_df * iqr + median


# ---------------------------------------------------------------------
# 4. MaxAbsScaler
# ---------------------------------------------------------------------
def apply_maxabs_scaler(df: pd.DataFrame, fit_mask=None):
    """
    Scales each column by its max absolute value:
        x_scaled = x / max(|x|)
    Result range: [-1, 1], and (importantly) does NOT shift the data --
    zero stays at zero.

    Purpose: scale magnitude while preserving sign and sparsity (zero
    stays zero). Useful for features like MACD or MACD histogram which
    are naturally centered around 0 and can be positive or negative.

    Advantages:
        - Preserves zero (doesn't destroy sparsity, unlike MinMax which
          shifts everything).
        - Simple, cheap, no data-driven centering needed.

    Disadvantages:
        - Still sensitive to outliers (single max spike sets the whole
          scale).
        - Doesn't fix non-stationarity.

    Suitable models: Good specifically for signed, zero-centered features
    (MACD, MACD histogram) rather than strictly positive ones like
    volume. Also common for sparse data.

    Preserves trend: YES (linear transform).
    Improves stationarity: NO.
    Reversible: YES exactly -- x = x_scaled * max_abs.
    """
    fit_df = _fit_slice(df, fit_mask)
    scaler = MaxAbsScaler()
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "maxabs_scaler",
        "max_abs": dict(zip(df.columns, scaler.max_abs_)),
    }
    return out, fit_info


def inverse_maxabs_scaler(transformed_df, fit_info):
    max_abs = pd.Series(fit_info["max_abs"])[transformed_df.columns]
    return transformed_df * max_abs


# ---------------------------------------------------------------------
# 5. QuantileTransformer (uniform output)
# ---------------------------------------------------------------------
def apply_quantile_transformer(df: pd.DataFrame, fit_mask=None, n_quantiles=1000,
                                output_distribution="uniform", random_state=42):
    """
    Maps each column to a uniform [0,1] distribution based on rank
    (percentile) within the fitted data, using empirical CDF.

    Purpose: makes every feature follow the SAME distribution shape
    (uniform), completely removing the original distribution's shape,
    ignoring outlier magnitude entirely (only rank/order matters).

    Advantages:
        - Extremely robust to outliers -- an extreme value just becomes
          "the highest rank," it can't distort other values at all.
        - Makes heavily skewed features (like volume, which has a long
          right tail) look well-behaved.

    Disadvantages:
        - NON-LINEAR transform -- destroys the actual magnitude
          relationships between points. Two points close together in
          value can end up far apart after transform if density is high
          there, and vice versa. This can hide real magnitude signal.
        - Loses the ability to interpret "how much bigger" -- only rank
          survives.
        - Only approximately reversible (via stored quantiles), not exact
          like linear scalers.

    Suitable models: Tree-based or rank-sensitive models (XGBoost,
    LightGBM can definitely benefit here, since the whole point is
    monotonic transform which doesn't hurt splits). Less ideal for linear
    models that rely on real magnitude relationships.

    Preserves trend: PARTIALLY -- preserves ORDER (higher stays higher)
    but NOT magnitude/shape of the trend, since it's non-linear.
    Improves stationarity: SOMEWHAT -- bounding to [0,1] limits variance
    explosion, but the underlying rank-drift over time is untouched.
    Reversible: Approximately (sklearn stores quantiles for inverse_transform).
    """
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
        "_sklearn_object": scaler,  # kept in-memory only, for inverse_transform
    }
    return out, fit_info


def inverse_quantile_transformer(transformed_df, fit_info):
    scaler = fit_info["_sklearn_object"]
    inv = scaler.inverse_transform(transformed_df.values)
    return pd.DataFrame(inv, columns=transformed_df.columns, index=transformed_df.index)


# ---------------------------------------------------------------------
# 6. PowerTransformer (Yeo-Johnson) -- EXTRA #1
# ---------------------------------------------------------------------
def apply_power_transformer(df: pd.DataFrame, fit_mask=None, method="yeo-johnson"):
    """
    Applies a learned power transform (Yeo-Johnson, works with negative
    and zero values too, unlike Box-Cox) to make each column's
    distribution more Gaussian-shaped, then standardizes it (mean 0,
    std 1) by default in sklearn.

    WHY ADDED (beyond the 5 required scalers): crypto features like
    'volume' and MACD histogram are heavily skewed / heavy-tailed. Unlike
    Standard/MinMax/Robust/MaxAbs (which only shift+scale, keeping the
    ORIGINAL shape), PowerTransformer actually RESHAPES the distribution
    itself to reduce skew. This is a fundamentally different category of
    transform, useful to contrast against pure linear scalers in the
    report.

    Advantages:
        - Reduces skewness and heavy tails, which helps models (esp.
          linear ones) that assume roughly-normal residuals/inputs.
        - Automatically finds the best power parameter (lambda) per
          column via maximum likelihood, no manual tuning needed.

    Disadvantages:
        - NON-LINEAR -- like QuantileTransformer, changes the actual
          shape of relationships, not just scale.
        - More opaque: the transform parameter (lambda) isn't as
          intuitive as "mean and std."
        - Can behave oddly on features that are already close to
          symmetric/Gaussian (unnecessary reshaping).

    Suitable models: Linear/Logistic Regression and any model whose
    performance depends on approximately-normal input distributions.
    Less relevant for tree models.

    Preserves trend: PARTIALLY -- order preserved (monotonic transform)
    but shape/magnitude changed.
    Improves stationarity: NO directly -- addresses distribution SHAPE
    (skew), not the drifting MEAN over time.
    Reversible: YES via inverse_transform (sklearn stores lambda per column).
    """
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


def inverse_power_transformer(transformed_df, fit_info):
    scaler = fit_info["_sklearn_object"]
    inv = scaler.inverse_transform(transformed_df.values)
    return pd.DataFrame(inv, columns=transformed_df.columns, index=transformed_df.index)


# ---------------------------------------------------------------------
# 7. Normalizer (row-wise L2) -- EXTRA #2
# ---------------------------------------------------------------------
def apply_normalizer(df: pd.DataFrame, fit_mask=None, norm="l2"):
    """
    Scales each ROW to unit norm (default L2: sqrt(sum of squares) = 1).
    NOTE: This is fundamentally different from every other scaler here --
    those scale each COLUMN using statistics across time/history.
    Normalizer scales each ROW using only that row's own values, so
    there's nothing to "fit" on training data (no fit_mask needed, no
    leakage possible by construction).

    WHY ADDED: included specifically to give a clear row-vs-column
    scaling contrast case for the report. It changes what the model
    "sees" -- rather than each feature's absolute level over time, the
    model sees the RELATIVE pattern across features at one timestamp.

    Advantages:
        - Cannot leak future information into row scaling (no fitting
          across time at all).
        - Useful for models that care about direction/pattern across
          features rather than absolute magnitude.

    Disadvantages:
        - Destroys the absolute magnitude/level of any single feature
          over time -- e.g. you lose "is volume high today vs last
          week," you only keep "how does volume compare to RSI/MACD
          right now, in this row." Likely BAD fit for this dataset's
          columns, which are mixed units (price, volume, RSI 0-100,
          MACD) -- normalizing a row containing both $97,000 and RSI=58
          together is not very meaningful.
        - Not reversible without storing the row's original norm.

    Suitable models: More common in text/embeddings context (e.g. TF-IDF
    vectors) where every column is the same kind of quantity. Included
    here mainly as an instructive contrast, not because it's expected to
    perform well on mixed-unit OHLCV+indicator data -- the report should
    say this explicitly.

    Preserves trend: NO in the usual sense -- there is no "time" axis
    involved in the transform at all, so the notion of trend across rows
    isn't preserved or destroyed by definition, it's simply not
    addressed by this transform (each row stands alone).
    Improves stationarity: NOT APPLICABLE -- this doesn't touch the
    across-time relationship.
    Reversible: NO (row norm isn't stored by default here).
    """
    normalizer = Normalizer(norm=norm)
    transformed = normalizer.transform(df.values)  # no meaningful "fit" -- stateless per row
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {"method": "normalizer", "norm": norm, "note": "row-wise, not reversible, no fit_mask used"}
    return out, fit_info


# ---------------------------------------------------------------------
# 8. Rolling Z-score -- EXTRA #3
# ---------------------------------------------------------------------
def apply_rolling_zscore(df: pd.DataFrame, fit_mask=None, window=168):
    """
    Z-score computed over a ROLLING window (default 168 hours = 1 week for
    hourly data) instead of one fixed mean/std for the whole series:
        x_scaled[t] = (x[t] - rolling_mean[t]) / rolling_std[t]

    WHY ADDED: this is the most "finance-native" of the extras. Crypto is
    non-stationary -- volatility regimes change (calm periods vs violent
    swings). A single global mean/std (StandardScaler) gets stale a month
    into the data. A rolling window adapts: "how extreme is this value
    relative to the PAST `window` hours," not relative to the whole
    year. This directly targets the "changing volatility" and
    "non-stationarity" problems named in the task doc.

    Advantages:
        - Adapts to changing volatility regimes over time -- exactly
          what static scalers can't do.
        - No global-fit leakage concern in the traditional sense, because
          each point only uses PAST data in its own window (if computed
          correctly with only backward-looking windows, which this
          implementation does).

    Disadvantages:
        - First `window` rows have no full window yet -- NaN by
          construction until enough history accumulates (this is
          expected and correct, not a bug -- there's genuinely no valid
          rolling stat yet).
        - Choice of window size is a hyperparameter that meaningfully
          changes behavior -- too short is noisy, too long behaves like
          a static scaler again.
        - Not reversible in a simple closed form (rolling mean/std at
          each point would all need to be stored to invert).

    Suitable models: Especially relevant for any strategy or model meant
    to adapt across different volatility regimes -- this is common
    practice in real trading systems specifically because it doesn't
    treat January's volatility and December's volatility the same way.

    Preserves trend: PARTIALLY -- LOCAL trend is preserved / even
    emphasized (a rise within the window shows up clearly), but LONG-TERM
    trend across the whole series is actively removed, since the
    reference point itself moves with the window. This is a genuinely
    useful contrast to log/static-scaled trend preservation in the
    report.
    Improves stationarity: YES, this is the main point of the method --
    directly addresses changing-volatility non-stationarity by
    continuously re-centering/re-scaling.
    Reversible: NO (not implemented here; would require storing every
    window's mean/std).
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