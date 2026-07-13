# preprocessing_lab/preprocessing/distribution.py

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

# add to imports at top of distribution.py:
# from scipy import stats

# ---------------------------------------------------------------------
# Extra distribution methods (alongside Winsorization, Log Transform,
# Gaussian Quantile Transform)
# ---------------------------------------------------------------------

def apply_box_cox(df: pd.DataFrame, fit_mask=None):
    """
    Box-Cox transform: a classic power transform (older cousin of
    PowerTransformer's Yeo-Johnson method in scalers.py), finds the best
    power parameter (lambda) per column to make the distribution more
    Gaussian. UNLIKE Yeo-Johnson, Box-Cox REQUIRES strictly positive
    values -- fails on zero or negative inputs.

    Purpose: same broad goal as PowerTransformer -- reshape a skewed
    distribution toward Gaussian -- but using the original, simpler
    classic method it's based on. Good direct comparison point in the
    report: Box-Cox (positive-only) vs Yeo-Johnson (handles negatives).

    Advantages:
        - Well-established, classic statistical method, easy to explain
          and cite.
        - Effective at reducing skew for strictly positive, right-tailed
          data like price and volume.

    Disadvantages:
        - REQUIRES strictly positive values -- fails outright on
          MACD-family columns (signed, can be zero/negative), same
          restriction as Log Transform and Log Returns. Only route
          price/volume-like columns to this method via config.
        - Non-linear -- same interpretability trade-off as any power/
          quantile transform (changes shape, not just scale).

    Suitable models: Same use case as PowerTransformer -- linear models
    or anything assuming roughly-Gaussian inputs, applied to strictly
    positive financial columns (price, volume).

    Preserves trend: PARTIALLY -- monotonic (order preserved), shape
    changed.
    Improves stationarity: NO directly -- targets distribution SHAPE
    (skew), not the drifting mean over time.
    Reversible: YES via scipy's inv_boxcox, given the fitted lambda per
    column (stored in fit_info below).
    """
    from scipy import stats

    if (df <= 0).any().any():
        cols_with_nonpositive = df.columns[(df <= 0).any()].tolist()
        raise ValueError(
            f"apply_box_cox: columns contain zero/negative values, "
            f"Box-Cox requires strictly positive input: {cols_with_nonpositive}. "
            f"Route signed columns (e.g. MACD-family) to a different method."
        )

    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    lambdas = {}
    for col in df.columns:
        transformed, fitted_lambda = stats.boxcox(df[col].values)
        out[col] = transformed
        lambdas[col] = fitted_lambda

    fit_info = {"method": "box_cox", "lambdas": lambdas}
    return out, fit_info


def inverse_box_cox(transformed_df, fit_info):
    from scipy.special import inv_boxcox
    out = pd.DataFrame(index=transformed_df.index, columns=transformed_df.columns, dtype=float)
    for col in transformed_df.columns:
        out[col] = inv_boxcox(transformed_df[col].values, fit_info["lambdas"][col])
    return out


def apply_sqrt_transform(df: pd.DataFrame, fit_mask=None, offset=0.0):
    """
    Square root transform: x_scaled = sqrt(x + offset)

    Purpose: compresses large values, same broad idea as Log Transform,
    but MUCH more gently -- log compresses aggressively (a 10x increase
    in price becomes a small additive step in log-space), sqrt compresses
    more mildly. Useful as a "medium strength" option between doing
    nothing and full log compression.

    Advantages:
        - Gentler than log transform -- keeps more of the original
          magnitude differences visible, useful when full log compression
          feels too aggressive for a given column.
        - Requires only non-negative values (offset=0 default), less
          restrictive than log (which needs strictly positive) or Box-Cox.
        - Simple, fast, easy to explain.

    Disadvantages:
        - Weaker outlier/heavy-tail compression than log transform --
          less effective if the actual problem is severe skew.
        - Still fails on negative values without an offset (MACD-family
          columns need offset tuning or should be routed elsewhere).
        - Does not address stationarity/drift at all, same as log
          transform.

    Suitable models: Similar use case to Log Transform, chosen when a
    milder compression is preferred -- useful direct comparison in the
    report ("does the strength of compression matter, or just having
    some compression at all?").

    Preserves trend: YES -- monotonic, and much closer to the original
    shape than log transform since compression is gentler.
    Improves stationarity: PARTIALLY, weaker than log transform for the
    same reason (milder compression = milder heteroskedasticity fix).
    Reversible: YES, exactly -- x = (x_scaled)^2 - offset.
    """
    if (df + offset < 0).any().any():
        cols_with_negative = df.columns[(df + offset < 0).any()].tolist()
        raise ValueError(
            f"apply_sqrt_transform: columns would be negative under sqrt "
            f"after offset: {cols_with_negative}. Increase offset or route "
            f"signed columns elsewhere."
        )
    out = np.sqrt(df + offset)
    fit_info = {"method": "sqrt_transform", "offset": offset}
    return out, fit_info


def inverse_sqrt_transform(transformed_df, fit_info):
    return (transformed_df ** 2) - fit_info["offset"]


def apply_sigma_clipping(df: pd.DataFrame, fit_mask=None, n_sigma=3):
    """
    Sigma clipping: clips values beyond n_sigma standard deviations from
    the mean, per column:
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std

    Purpose: same broad goal as Winsorization (neutralize extreme
    outliers without deleting rows) but using a DIFFERENT rule for what
    counts as "extreme" -- standard deviations from the mean, instead of
    percentile rank. Good direct comparison case in the report:
    percentile-based clipping (Winsorization) vs std-based clipping
    (this method).

    Advantages:
        - Statistically grounded in the normal distribution (n_sigma=3
          corresponds to a well-known ~99.7% coverage IF the data were
          truly Gaussian).
        - Simple, fast, one intuitive parameter (n_sigma).

    Disadvantages:
        - The mean/std themselves are sensitive to the very outliers
          you're trying to clip -- a circular weakness Winsorization's
          percentile approach avoids (percentiles are far more robust to
          extreme values than mean/std are). This is the key trade-off to
          highlight against Winsorization in the report.
        - Assumes roughly-Gaussian data for the "3 sigma = 99.7%"
          intuition to hold; crypto data is heavy-tailed, so this
          assumption is shakier here than for well-behaved data.
        - Must use fit_mask correctly (fit mean/std only on training
          rows) to avoid leaking future extremes into past clipping
          bounds.

    Suitable models: Same use case as Winsorization -- any model
    sensitive to outlier leverage, especially linear models.

    Preserves trend: PARTIALLY, same framing as Winsorization -- bulk of
    data untouched, most extreme moves flattened at the clip boundary.
    Improves stationarity: SOMEWHAT -- reduces variance spikes, doesn't
    address drifting mean.
    Reversible: NO -- clipped values lose their original magnitude
    permanently, same limitation as Winsorization.
    """
    fit_df = df if fit_mask is None else df.loc[fit_mask]
    mean = fit_df.mean()
    std = fit_df.std()
    lower_bounds = mean - n_sigma * std
    upper_bounds = mean + n_sigma * std

    out = df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    fit_info = {
        "method": "sigma_clipping",
        "n_sigma": n_sigma,
        "lower_bounds": lower_bounds.to_dict(),
        "upper_bounds": upper_bounds.to_dict(),
    }
    return out, fit_info


def apply_iqr_clipping(df: pd.DataFrame, fit_mask=None, k=1.5):
    """
    Tukey's IQR clipping: clips values beyond k * IQR from Q1/Q3, the
    classic statistical outlier rule (k=1.5 is the standard "mild
    outlier" threshold; k=3.0 is sometimes used for "extreme outliers"):
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

    Purpose: a THIRD way of defining "what counts as an outlier,"
    alongside Winsorization (fixed percentile) and Sigma Clipping
    (std-based). This one is quantile-based like Winsorization, but uses
    a different rule -- IQR spread rather than a fixed percentile cutoff
    -- so it adapts to how spread-out the middle of the data actually is,
    rather than always cutting at exactly the 1st/99th percentile
    regardless of shape.

    Advantages:
        - The classic, most widely taught statistical outlier definition
          (this is what "outlier" means in a standard boxplot).
        - More robust than sigma clipping (uses median-adjacent
          quantiles, not mean/std, so less distorted by the very outliers
          being clipped) -- similar robustness advantage to Winsorization.
        - Adapts to the actual spread of the middle 50% of data, unlike
          Winsorization's fixed percentile cutoffs which don't adjust to
          how tight or wide that middle chunk is.

    Disadvantages:
        - k is a real hyperparameter (1.5 vs 3.0 changes results
          meaningfully) -- should be reported as tunable, same caveat as
          Winsorization's percentile choice.
        - Like all clipping methods, permanently loses true tail
          magnitude in exchange for stability.

    Suitable models: Same use case as Winsorization/Sigma Clipping -- any
    outlier-sensitive model. Good three-way comparison across all three
    clipping philosophies in the final report.

    Preserves trend: PARTIALLY, same framing as the other two clipping
    methods.
    Improves stationarity: SOMEWHAT, same as Winsorization/Sigma Clipping.
    Reversible: NO, same limitation as the other two clipping methods.
    """
    fit_df = df if fit_mask is None else df.loc[fit_mask]
    q1 = fit_df.quantile(0.25)
    q3 = fit_df.quantile(0.75)
    iqr = q3 - q1
    lower_bounds = q1 - k * iqr
    upper_bounds = q3 + k * iqr

    out = df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    fit_info = {
        "method": "iqr_clipping",
        "k": k,
        "lower_bounds": lower_bounds.to_dict(),
        "upper_bounds": upper_bounds.to_dict(),
    }
    return out, fit_info