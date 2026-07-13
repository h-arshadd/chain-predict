"""
distribution.py
----------------
Methods targeting DISTRIBUTION SHAPE / outliers / heavy tails, distinct
from scaling (magnitude) and stationarity (trend/drift) methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer


def apply_winsorization(df: pd.DataFrame, fit_mask=None, lower_pct=0.01, upper_pct=0.99):
    """
    Winsorization: clips extreme values to a fixed percentile bound,
    rather than removing them. Values below the `lower_pct` percentile
    are set TO that percentile value; values above `upper_pct` are set
    to that percentile value. Everything in between is untouched.

    Purpose: neutralizes the influence of extreme outliers (flash spikes,
    data glitches, liquidation cascades) WITHOUT deleting the row
    entirely -- unlike dropping outlier rows, winsorization keeps every
    row in the dataset (which matters for time series -- you can't just
    delete an hour, it'd break the time index continuity that later
    steps like rolling windows / triple-barrier labeling depend on).

    Advantages:
        - Very simple to understand and communicate.
        - Directly bounds the impact of extreme values without losing
          data points.
        - Cheap to compute.

    Disadvantages:
        - The clip percentiles are fit on the training window's
          percentiles -- must use fit_mask correctly or this leaks future
          extremes into past scaling. We support fit_mask here for that
          reason.
        - Somewhat arbitrary threshold choice (1%/99% here) -- different
          thresholds meaningfully change results, should be treated as a
          tunable hyperparameter and reported as such.
        - Distorts the true tail values in exchange for stability --
          might remove genuinely important signal from real large moves
          (e.g. real crash / real breakout candles are exactly the kind
          of value this clips).

    Suitable models: Any model, but especially helps linear models &
    anything sensitive to outlier leverage. Good to pair with a scaler
    afterward (winsorize first, then scale) as a combined pipeline --
    worth testing in the comparative report.

    Preserves trend: PARTIALLY -- the overall shape/trend is preserved
    for the bulk of the data (99% of it, untouched), but the most extreme
    trend moves (the actual spikes) are flattened at the clip boundary.
    Improves stationarity: SOMEWHAT -- reduces variance spikes which can
    help some stationarity tests, but does not address a drifting mean.
    Reversible: NO -- clipped values lose their original magnitude
    permanently (that's the whole point), so the exact original values
    for clipped points cannot be recovered.
    """
    fit_df = df if fit_mask is None else df.loc[fit_mask]
    lower_bounds = fit_df.quantile(lower_pct)
    upper_bounds = fit_df.quantile(upper_pct)

    out = df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    fit_info = {
        "method": "winsorization",
        "lower_pct": lower_pct,
        "upper_pct": upper_pct,
        "lower_bounds": lower_bounds.to_dict(),
        "upper_bounds": upper_bounds.to_dict(),
    }
    return out, fit_info


def apply_log_transform(df: pd.DataFrame, fit_mask=None, offset=1e-8):
    """
    Natural log transform: x_scaled = log(x + offset).
    `offset` guards against log(0) or log(negative) errors -- columns
    with zero/negative values (e.g. MACD, MACD histogram, MACD signal,
    which are signed and can be negative) will need special handling
    (see note below); this base version assumes non-negative input like
    price/volume.

    Purpose: compresses large values much more than small ones --
    directly tackles heavy-tailed, exponential-like growth patterns
    (price trending from $20k to $100k over a year is a huge absolute
    range, but a much smaller range in log-space). Extremely standard in
    finance -- log returns/log prices are practically the default in
    quant research.

    Advantages:
        - Turns multiplicative relationships into additive ones (e.g. a
          10% move looks the same size in log-space whether price is
          $1,000 or $100,000) -- very natural fit for price data that
          grows/shrinks by percentage, not by fixed dollar amounts.
        - Reduces the influence of large-magnitude outliers/heavy tails
          without deleting or clipping any data (unlike winsorization).
        - Simple, well-understood, cheap, and its inverse (exp) is exact.

    Disadvantages:
        - Cannot be applied directly to zero or negative values (MACD,
          MACD histogram, and MACD signal in this dataset ARE signed --
          this function will raise/produce NaN/inf on those columns
          unless you route only price/volume-like columns to it via
          config's feature_columns selection per method, or use a
          signed-log variant, which is NOT implemented here -- flagged as
          a known limitation to note in the report and to be careful
          about at config-selection time).
        - Does not by itself guarantee stationarity (log-price still
          trends upward over time -- log RETURNS, i.e. log-diff, would be
          closer to stationary, but that is a different transform, not
          implemented in this function).

    Suitable models: Especially useful before Linear Regression / any
    model assuming additive, roughly-normal-ish residuals on financial
    price/volume series.

    Preserves trend: YES -- the overall upward/downward trend direction
    is preserved (monotonic transform), though the trend's SHAPE in
    log-space differs from raw-space (large absolute moves late in an
    uptrend look smaller relative to the whole log-scaled range).
    Improves stationarity: PARTIALLY -- reduces heteroskedasticity
    (variance that grows with price level) which helps some stationarity
    tests, but does not remove a trending mean by itself.
    Reversible: YES, exactly -- x = exp(x_scaled) - offset.
    """
    if (df <= 0).any().any():
        cols_with_nonpositive = df.columns[(df <= 0).any()].tolist()
        raise ValueError(
            f"apply_log_transform: columns contain zero/negative values, "
            f"log() is undefined there: {cols_with_nonpositive}. "
            f"Route signed columns (e.g. MACD-family) to a different "
            f"method via config, or handle them separately."
        )
    out = np.log(df + offset)
    fit_info = {"method": "log_transform", "offset": offset}
    return out, fit_info


def inverse_log_transform(transformed_df, fit_info):
    return np.exp(transformed_df) - fit_info["offset"]


def apply_gaussian_quantile_transform(df: pd.DataFrame, fit_mask=None, n_quantiles=1000, random_state=42):
    """
    Same mechanism as the QuantileTransformer in scalers.py, but with
    output_distribution="normal" instead of "uniform" -- maps each
    column's empirical distribution to a standard Gaussian (mean 0, std
    1, bell-shaped) via rank/percentile mapping.

    Purpose: forces every feature to look Gaussian-distributed, which
    many classical statistical models and tests implicitly assume.
    Distinct from PowerTransformer -- PowerTransformer LEARNS a smooth
    parametric power function per column; this method uses the
    NON-PARAMETRIC empirical rank mapping (no functional form assumed at
    all, just "what percentile is this point at, map that percentile to
    the same percentile of a normal curve").

    Advantages:
        - Output is (by construction) very close to perfectly Gaussian --
          stronger normality guarantee than PowerTransformer, which only
          approximately reduces skew.
        - Extremely robust to outliers, same as the uniform-output
          version -- outliers just become extreme ranks, not extreme
          magnitudes.

    Disadvantages:
        - Same non-linearity concerns as any quantile-based transform:
          destroys real magnitude/distance information, keeps only rank
          order.
        - With limited data (n_quantiles vs. actual row count), quantile
          estimates in the tails can be noisy/unstable -- worth checking
          n_quantiles isn't larger than available fit rows (we cap this
          automatically below, mirroring the uniform version in
          scalers.py).

    Suitable models: Particularly useful for any classical statistical
    method or model with a Gaussian assumption baked in (e.g. some linear
    model diagnostics, certain hypothesis tests used in the Stationarity
    Analysis step). Less meaningful for tree models, which don't care
    about distribution shape at all.

    Preserves trend: PARTIALLY -- order/rank preserved (monotonic), shape
    and magnitude changed substantially.
    Improves stationarity: SOMEWHAT -- bounds variance strongly (Gaussian
    output has fixed shape), but the RANK of a value can still drift
    across regimes if fit on a static window, so it doesn't fully resolve
    non-stationarity from mean drift.
    Reversible: Approximately (sklearn stores quantiles for inverse_transform,
    same mechanism as the uniform QuantileTransformer above).
    """
    fit_df = df if fit_mask is None else df.loc[fit_mask]
    n_q = min(n_quantiles, len(fit_df))
    scaler = QuantileTransformer(
        n_quantiles=n_q,
        output_distribution="normal",
        random_state=random_state,
    )
    scaler.fit(fit_df.values)
    transformed = scaler.transform(df.values)
    out = pd.DataFrame(transformed, columns=df.columns, index=df.index)
    fit_info = {
        "method": "gaussian_quantile_transform",
        "n_quantiles": n_q,
        "_sklearn_object": scaler,
    }
    return out, fit_info


def inverse_gaussian_quantile_transform(transformed_df, fit_info):
    scaler = fit_info["_sklearn_object"]
    inv = scaler.inverse_transform(transformed_df.values)
    return pd.DataFrame(inv, columns=transformed_df.columns, index=transformed_df.index)